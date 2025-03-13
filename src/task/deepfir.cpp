
#include "deepfir.h"

// 四维张量容器，优化内存布局
struct Tensor4D {
    int batch;    // 批次维度（固定为1）
    int height;   // 频率轴（129）
    int width;    // 时间轴（16）
    int channel;  // 通道维度（固定为1）
    std::vector<float> data; // 数据存储（行优先顺序）

    // 预分配内存
    void reserve_space(int total_batches) {
        data.reserve(batch * height * width * channel);
    }
};



// 构造函数
DeepFIR::DeepFIR()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
}
// 析构函数
DeepFIR::~DeepFIR()
{
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    for (auto &tensor : output_tensors_)
    {
        free(tensor.data);
        tensor.data = nullptr;
    }
}



// 加载模型，获取输入输出属性
nn_error_e DeepFIR::LoadModel(const char *model_path)
{

    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolo load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolo input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.attr.size = input_tensor_.attr.n_elems * nn_tensor_type_to_size(input_tensor_.attr.type); //nn_tensor_type_to_size(input_tensor_.attr.type)
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;//output_shapes[i].n_dims
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        if (output_shapes[i].type != NN_TENSOR_FLOAT16)
        {
            NN_LOG_ERROR("yolo output tensor type is not FP16, but %d", output_shapes[i].type);
            return NN_RKNN_OUTPUT_ATTR_ERROR;
        }
        tensor.attr.type = output_shapes[i].type;
        tensor.attr.index = i;
        tensor.attr.layout = output_shapes[i].layout;
        tensor.attr.zp = output_shapes[i].zp;
        tensor.attr.scale = output_shapes[i].scale;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        // tensor.attr.size = output_shapes[i].n_elems;
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
    return NN_SUCCESS;
}

// 运行模型
nn_error_e DeepFIR::Run(const char *audio_file, STFTResult &result, std::vector<Detection> &objects)
{
    Preprocess(audio_file,result);
    Inference(result,input_tensor_);               // 推理
    return NN_SUCCESS;
}
// 图像预处理
nn_error_e DeepFIR::Preprocess(const char *audio_file, STFTResult &result)
{
    // 将预处理后的结果放入input_tensor_中
    compute_stft(audio_file,result);
    return NN_SUCCESS;
}

// 推理
nn_error_e DeepFIR::Inference(STFTResult &result, tensor_data_s &tensor)
{
    std::vector<tensor_data_s> inputs;
    // // 步骤1：准备输入数据
    // prepare_inputs(stft_result, inputs);
    std::vector<std::vector<__fp16>> result_matrix;  // 假设数据类型为float
    
    const int total_frames = result.magnitude.size();
    const int freq_bins = result.magnitude[0].size();
    const int batch_size = 16;
    inputs.clear();
    // 1. 创建连续内存容器存储数据
    const int num_time = 16;    // 时间维度大小
    const int num_freq = 129;   // 频率维度大小
    std::vector<__fp16> Push(num_time * num_freq);  // 根据实际数据类型调整（如float/int）
    const int total_batches = total_frames / batch_size;
    result_matrix.resize(total_batches*16, std::vector<__fp16>(129));
    for (int batch_idx = 0; batch_idx < total_batches; ++batch_idx) {
        tensor_data_s input;
        // 分配内存并填充数据（转置操作）
        input.data = new float[129 * 16];
        float* dst = static_cast<float*>(input.data);
        const int frame_start = batch_idx * batch_size;
        int index = 0;
        for (int f = 0; f < 129; ++f) {     // 频率维度
            for (int t = 0; t < 16; ++t) {  // 时间维度
                // 计算一维索引（行优先）
                // int index = t * num_freq + f;
                Push[index] = 1; //result.magnitude[frame_start + t][f]
                index++;
            }
        }

        // 步骤 1：将 FP32 转换为 FP16
        std::vector<__fp16> fp16_data(num_time * num_freq);
        for (size_t i = 0; i < Push.size(); ++i) {
            fp16_data[i] = static_cast<__fp16>(Push[i]); // 显式转换为 FP16
        }

        // 步骤 2：计算正确的字节数
        size_t fp16_bytes = fp16_data.size() * sizeof(__fp16);

        // 步骤 3：拷贝数据到 tensor.data
        memcpy(tensor.data, fp16_data.data(), fp16_bytes);
        // memcpy(tensor.data, Push.data(), tensor.attr.size);
        inputs.push_back(tensor);
        inputs[0].attr.zp = 0;
        inputs[0].attr.scale = 1;
        output_tensors_[0].attr.layout = NN_TENSOR_NHWC;
        engine_->Run(inputs, output_tensors_, false);
        
        
        
        // 把输出和输入相乘
        // 遍历频率维度 (0~128) 和时间维度 (0~15)
        for (int freq = 0; freq < 129; ++freq) {
            for (int time = 0; time < 16; ++time) {
                // 获取 inputs 中对应位置的值
                __fp16 input_val = *((__fp16*)inputs[0].data + freq * 16 + time);
                // 获取 output_tensors_ 中对应位置的值（忽略第4维）
                __fp16 output_val = *((__fp16*)output_tensors_[0].data + freq * 16 + time);
                // 计算乘积
                result_matrix[batch_idx*16 + time][freq] = input_val * output_val;
            }
        }
        // for (int time = 0; time < 16; ++time) {
        //     for (int freq = 0; freq < 129; ++freq) {
        //         // 获取 inputs 中对应位置的值
        //         __fp16 input_val = *((__fp16*)inputs[0].data + freq);
        //         // 获取 output_tensors_ 中对应位置的值（忽略第4维）
        //         __fp16 output_val = *((__fp16*)output_tensors_[0].data + freq);
        //         // 计算乘积
        //         result_matrix[batch_idx*16 + time][freq] = input_val * output_val;
        //     }
        // }




        inputs.clear();
    }
    save_spectrogramFP16(result_matrix, "spectrogram_Final_BN.jpg");

    // for (size_t i = 0; i < inputs.size(); ++i) {
    //     // 运行模型0
    
    //     engine_->Run(inputs, output_tensors_, false);
    // }
    // // 步骤4：处理输出结果
    // process_outputs(inputs, output_tensors_, stft_result);
    // // 存储成图片
    // save_spectrogram(stft_result.magnitude, "spectrogram_Final.jpg");
    return NN_SUCCESS;   
}



// void process_outputs(const std::vector<tensor_data_s>& inputs,
//     std::vector<tensor_data_s>& outputs,
//     STFTResult& result) 
// {
//     assert(inputs.size() == outputs.size() && "Input/Output size mismatch");

//     for (size_t i = 0; i < outputs.size(); ++i) {
//         const auto& input = inputs[i];
//         const auto& output = outputs[i];

//         // 验证输出维度
//         assert(output.attr.dims[0] == 1 && 
//         output.attr.dims[1] == 129 &&
//         output.attr.dims[2] == 16 &&
//         "Invalid output dimensions");

//         const int frame_start = i * 16;
//         const float* input_data = static_cast<float*>(input.data);
//         const float* output_data = static_cast<float*>(output.data);

//         // 矩阵相乘操作（逐元素乘积）
//         for (int t = 0; t < 16; ++t) {
//             for (int f = 0; f < 129; ++f) {
//                 // 输入内存布局：f * 16 + t
//                 // 输出内存布局：f * 16 + t
//                 const float product = input_data[f * 16 + t] * 
//                                     output_data[f * 16 + t];
//                 result.magnitude[frame_start + t][f] = product;
//             }
//         }

//         // 释放内存
//         delete[] input_data;
//         delete[] output_data;
//     }
// }

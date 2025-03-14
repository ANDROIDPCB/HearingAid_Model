
#include "yolov5.h"

#include <memory>

#include "utils/logging.h"
#include "process/preprocess.h"
#include "process/postprocess.h"

#include <ctime>

void DetectionGrp2DetectionArray(yolov5::detect_result_group_t &det_grp, std::vector<Detection> &objects)
{
    // 根据当前系统时间生成随机数种子
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < det_grp.count; i++)
    {
        Detection det;
        det.className = det_grp.results[i].name;

        det.box = cv::Rect(det_grp.results[i].box.left,
                           det_grp.results[i].box.top,
                           det_grp.results[i].box.right - det_grp.results[i].box.left,
                           det_grp.results[i].box.bottom - det_grp.results[i].box.top);

        det.confidence = det_grp.results[i].prop;
        det.class_id = 0;
        // generate random cv::Scalar color
        det.color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        objects.push_back(det);
    }
}

// 构造函数
Yolov5::Yolov5()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
}
// 析构函数
Yolov5::~Yolov5()
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
nn_error_e Yolov5::LoadModel(const char *model_path)
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
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be float32
        if (output_shapes[i].type != NN_TENSOR_INT8)
        {
            NN_LOG_ERROR("yolo output tensor type is not int8, but %d", output_shapes[i].type);
            return NN_RKNN_OUTPUT_ATTR_ERROR;
        }
        tensor.attr.type = output_shapes[i].type;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
    return NN_SUCCESS;
}

// 图像预处理
nn_error_e Yolov5::Preprocess(const cv::Mat &image)
{
    // 将预处理后的结果放入input_tensor_中
    cvimg2tensor(image, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    return NN_SUCCESS;
}

// 推理
nn_error_e Yolov5::Inference()
{
    std::vector<tensor_data_s> inputs;
    // 将input_tensor_放入inputs中
    inputs.push_back(input_tensor_);
    // 运行模型
    engine_->Run(inputs, output_tensors_, false);
    return NN_SUCCESS;
}

// 运行模型
nn_error_e Yolov5::Run(const cv::Mat &img, std::vector<Detection> &objects)
{
    Preprocess(img);           // 图像预处理
    Inference();               // 推理
    Postprocess(img, objects); // 后处理
    return NN_SUCCESS;
}

// 后处理
nn_error_e Yolov5::Postprocess(const cv::Mat &img, std::vector<Detection> &objects)
{
    int height = input_tensor_.attr.dims[1];
    int width = input_tensor_.attr.dims[2];
    float scale_w = height * 1.f / img.cols;
    float scale_h = width * 1.f / img.rows;

    yolov5::detect_result_group_t detections;

    yolov5::post_process((int8_t *)output_tensors_[0].data,
                         (int8_t *)output_tensors_[1].data,
                         (int8_t *)output_tensors_[2].data,
                         height, width,
                         BOX_THRESH, NMS_THRESH,
                         scale_w, scale_h,
                         out_zps_, out_scales_,
                         &detections);

    DetectionGrp2DetectionArray(detections, objects);
    return NN_SUCCESS;
}

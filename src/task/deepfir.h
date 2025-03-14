#ifndef RK3588_DEMO_DEEPFIR_H
#define RK3588_DEMO_DEEPFIR_H

#include "types/yolo_datatype.h"
#include "engine/engine.h"
#include "process/preprocess_deepfir.h"
class DeepFIR
{
public:
    DeepFIR();
    ~DeepFIR();

    nn_error_e LoadModel(const char *model_path);                        // 加载模型
    nn_error_e Run(const char *audio_file, STFTResult &result, std::vector<Detection> &objects); // 运行模型

private:
    nn_error_e Preprocess(const char *audio_file, STFTResult &result);                                   // 图像预处理
    nn_error_e Inference(STFTResult &result,tensor_data_s &tensor);                                                      // 推理
    nn_error_e Postprocess(const cv::Mat &img, std::vector<Detection> &objects); // 后处理

    // void process_outputs(const std::vector<tensor_data_s>& inputs,
    //     std::vector<tensor_data_s>& outputs,
    //     STFTResult& result);
    
    tensor_data_s input_tensor_;
    std::vector<tensor_data_s> output_tensors_;
    std::vector<int32_t> out_zps_;
    std::vector<float> out_scales_;
    std::shared_ptr<NNEngine> engine_;
};

#endif // RK3588_DEMO_DeepFIR_H
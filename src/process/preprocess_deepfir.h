#ifndef RK3588_DEMO_PREPROCESS_DEEPFIR_H
#define RK3588_DEMO_PREPROCESS_DEEPFIR_H


#include <vector>
#include <string>    // 必须包含必要头文件
#include <cmath>
#include <stdexcept>
#include <sndfile.h>
#include <kissfft/kiss_fft.h>
#include "types/datatype.h"

struct STFTResult {
    std::vector<std::vector<float>> magnitude;
    std::vector<std::vector<float>> phase;
};

// 函数声明（默认参数只能出现在声明中）
STFTResult compute_stft(const char* filename, STFTResult &result);
void save_spectrogramFP16(const std::vector<std::vector<__fp16>>& magnitude,
    const std::string& output_path,
    int img_width = 800,
    int img_height = 600,
    bool use_log_scale = true);
void save_spectrogram(const std::vector<std::vector<float>>& magnitude,
    const std::string& output_path,
    int img_width = 800,
    int img_height = 600,
    bool use_log_scale = true);
void prepare_inputs(const STFTResult& result, 
        std::vector<tensor_data_s>& inputs);
#endif // RK3588_DEMO_PREPROCESS_H

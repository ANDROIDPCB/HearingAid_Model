#ifndef RK3588_DEMO_PREPROCESS_DEEPFIR_H
#define RK3588_DEMO_PREPROCESS_DEEPFIR_H


#include <vector>
#include <string>    // 必须包含必要头文件
#include <cmath>
#include <stdexcept>
#include <sndfile.h>
#include <kissfft/kiss_fft.h>

struct STFTResult {
    std::vector<std::vector<float>> magnitude;
    std::vector<std::vector<float>> phase;
};

// 函数声明（默认参数只能出现在声明中）
STFTResult compute_stft(const char* filename);
void save_spectrogram(const std::vector<std::vector<float>>& magnitude,
    const std::string& output_path,
    int img_width = 800,
    int img_height = 600,
    bool use_log_scale = true);
#endif // RK3588_DEMO_PREPROCESS_H

#ifndef AUDIO_RT_FFT_H
#define AUDIO_RT_FFT_H

#include <vector>
#include <fftw3.h>
#include <cmath>
#include "utool_AudioRT/Config.h"



constexpr size_t NUM_FRAMES = NUM_TIME;
constexpr size_t FRAME_SIZE = WINDOW_SIZE;
constexpr size_t FFT_BINS = FRAME_SIZE / 2 + 1;

class FFTProcessor {
public:
    FFTProcessor();
    ~FFTProcessor();

    // 禁用拷贝和移动
    FFTProcessor(const FFTProcessor&) = delete;
    FFTProcessor& operator=(const FFTProcessor&) = delete;

    void compute_spectrum(
        const std::vector<std::vector<__fp16>>& time_data,
        std::vector<std::vector<__fp16>>& mag,
        std::vector<std::vector<__fp16>>& phase
    );
    void compute_spectrum_One(
        const std::vector<std::vector<__fp16>>& time_data,
        std::vector<std::vector<__fp16>>& mag,
        std::vector<std::vector<__fp16>>& phase);
    void reconstruct_signal_Only(
        const std::vector<std::vector<__fp16>>& mag,
        const std::vector<std::vector<__fp16>>& phase,
        std::vector<std::vector<__fp16>>& time_data);
    void reconstruct_signal(
        const std::vector<std::vector<__fp16>>& mag,
        const std::vector<std::vector<__fp16>>& phase,
        std::vector<std::vector<__fp16>>& time_data
    );



private:
    // FFTW资源
    fftwf_plan r2c_plan_;
    fftwf_plan c2r_plan_;
    float* fft_in_;
    fftwf_complex* fft_out_;
    float* ifft_out_;

    // 初始化方法
    void initialize_fftw();
    void cleanup_fftw();
};




#endif
#include "Audio_RT_FFT.h"

FFTProcessor::FFTProcessor() {
    initialize_fftw();
}

FFTProcessor::~FFTProcessor() {
    cleanup_fftw();
}

void FFTProcessor::initialize_fftw() {
    // 内存分配（对齐内存）
    fft_in_ = (float*)fftwf_malloc(sizeof(float) * FRAME_SIZE);
    fft_out_ = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * FFT_BINS);
    ifft_out_ = (float*)fftwf_malloc(sizeof(float) * FRAME_SIZE);

    // 创建优化计划
    r2c_plan_ = fftwf_plan_dft_r2c_1d(FRAME_SIZE, fft_in_, fft_out_, FFTW_MEASURE);
    c2r_plan_ = fftwf_plan_dft_c2r_1d(FRAME_SIZE, fft_out_, ifft_out_, FFTW_MEASURE);
}

void FFTProcessor::cleanup_fftw() {
    fftwf_destroy_plan(r2c_plan_);
    fftwf_destroy_plan(c2r_plan_);
    fftwf_free(fft_in_);
    fftwf_free(fft_out_);
    fftwf_free(ifft_out_);
}

void FFTProcessor::compute_spectrum(
    const std::vector<std::vector<__fp16>>& time_data,
    std::vector<std::vector<__fp16>>& mag,
    std::vector<std::vector<__fp16>>& phase) 
{
    mag.resize(NUM_FRAMES, std::vector<__fp16>(FFT_BINS));
    phase.resize(NUM_FRAMES, std::vector<__fp16>(FFT_BINS));

    for (size_t i = 0; i < NUM_FRAMES; ++i) {
        // 将fp16转换为float并拷贝到对齐内存
        for (size_t j = 0; j < FRAME_SIZE; ++j) {
            fft_in_[j] = static_cast<float>(time_data[i][j]);
        }
        // 执行FFT
        fftwf_execute(r2c_plan_);
        // 计算幅度和相位
        for (size_t k = 0; k < FFT_BINS; ++k) {
            float real = fft_out_[k][0];
            float imag = fft_out_[k][1];
            // 幅度（转换为dB并量化到fp16）
            float magnitude = std::sqrt(real*real + imag*imag);
            mag[i][k] = static_cast<__fp16>(magnitude);
            // 相位（归一化到[-π, π]）
            phase[i][k] = static_cast<__fp16>(std::atan2(imag, real) / M_PI);
        }
    }
}


// 进行单独一帧的FFT，以适应新数据帧的移入
void FFTProcessor::compute_spectrum_One(
    const std::vector<std::vector<__fp16>>& time_data,
    std::vector<std::vector<__fp16>>& mag,
    std::vector<std::vector<__fp16>>& phase) 
{
    // 将fp16转换为float并拷贝到对齐内存
    for (size_t j = 0; j < FRAME_SIZE; ++j) {
        fft_in_[j] = static_cast<float>(time_data[NUM_FRAMES-1][j]); // 只处理最后一行
    }
    // 执行FFT
    fftwf_execute(r2c_plan_);
    // 计算幅度和相位
    for (size_t k = 0; k < FFT_BINS; ++k) {
        float real = fft_out_[k][0];
        float imag = fft_out_[k][1];
        // 幅度（转换为dB并量化到fp16）
        float magnitude = std::sqrt(real*real + imag*imag);
        mag[NUM_FRAMES-1][k] = static_cast<__fp16>(magnitude);
        // 相位（归一化到[-π, π]）
        phase[NUM_FRAMES-1][k] = static_cast<__fp16>(std::atan2(imag, real) / M_PI);
    }
}

// 只对mag和phase的最后两行进行IFFT
void FFTProcessor::reconstruct_signal_Only(
    const std::vector<std::vector<__fp16>>& mag,
    const std::vector<std::vector<__fp16>>& phase,
    std::vector<std::vector<__fp16>>& time_data)
{
    time_data.resize(2, std::vector<__fp16>(FRAME_SIZE)); // 固定就是最后2帧的IFFT
    for (size_t i = NUM_FRAMES - 2; i < NUM_FRAMES; ++i) {
        // 重建复数谱
        for (size_t k = 0; k < FFT_BINS; ++k) {
            // 反量化
            float magnitude = static_cast<float>(mag[i][k]);
            float angle = static_cast<float>(phase[i][k]) * M_PI;

            fft_out_[k][0] = magnitude * std::cos(angle);
            fft_out_[k][1] = magnitude * std::sin(angle);
        }

        // 执行逆变换
        fftwf_execute(c2r_plan_);

        // 缩放并转换回fp16
        const float scale = 1.0f / FRAME_SIZE;
        for (size_t j = 0; j < FRAME_SIZE; ++j) {
            time_data[i - NUM_FRAMES + 2][j] = static_cast<__fp16>(ifft_out_[j] * scale);
        }
    }

}


void FFTProcessor::reconstruct_signal(
    const std::vector<std::vector<__fp16>>& mag,
    const std::vector<std::vector<__fp16>>& phase,
    std::vector<std::vector<__fp16>>& time_data)
{
    time_data.resize(NUM_FRAMES, std::vector<__fp16>(FRAME_SIZE));
    for (size_t i = 0; i < NUM_FRAMES; ++i) {
        // 重建复数谱
        for (size_t k = 0; k < FFT_BINS; ++k) {
            // 反量化
            float magnitude = static_cast<float>(mag[i][k]);
            float angle = static_cast<float>(phase[i][k]) * M_PI;

            fft_out_[k][0] = magnitude * std::cos(angle);
            fft_out_[k][1] = magnitude * std::sin(angle);
        }

        // 执行逆变换
        fftwf_execute(c2r_plan_);

        // 缩放并转换回fp16
        const float scale = 1.0f / FRAME_SIZE;
        for (size_t j = 0; j < FRAME_SIZE; ++j) {
            time_data[i][j] = static_cast<__fp16>(ifft_out_[j] * scale);
        }
    }
}


// 使用示例
int main() {
    // 初始化数据
    std::vector<std::vector<__fp16>> time_data(NUM_FRAMES, 
        std::vector<__fp16>(FRAME_SIZE, __fp16(0.0f)));

    // 创建处理器
    FFTProcessor processor;

    // 正向变换
    std::vector<std::vector<__fp16>> mag, phase;
    processor.compute_spectrum(time_data, mag, phase);

    // 逆向变换
    std::vector<std::vector<__fp16>> reconstructed;
    processor.reconstruct_signal(mag, phase, reconstructed);

    return 0;
}
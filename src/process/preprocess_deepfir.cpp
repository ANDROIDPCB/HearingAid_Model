#include "preprocess_deepfir.h"

STFTResult compute_stft(const char* filename, 
                       int n_fft = 256,
                       int hop_length = 16,
                       int win_length = 256) 
{
    // 初始化返回结构
    STFTResult result;

    // 1. 读取音频文件
    SF_INFO sfinfo = {0};
    SNDFILE* file = sf_open(filename, SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failed to open audio file");
    }

    // 2. 读取音频数据
    std::vector<float> buffer(sfinfo.frames * sfinfo.channels);
    sf_read_float(file, buffer.data(), buffer.size());
    sf_close(file);

    // 3. 转换为单声道
    std::vector<float> mono(sfinfo.frames);
    if (sfinfo.channels > 1) {
        for (sf_count_t i = 0; i < sfinfo.frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < sfinfo.channels; ++c) {
                sum += buffer[i * sfinfo.channels + c];
            }
            mono[i] = sum / sfinfo.channels;
        }
    } else {
        mono = std::move(buffer);
    }

    // 4. 生成汉明窗
    std::vector<float> hamming_window(win_length);
    for (int i = 0; i < win_length; ++i) {
        hamming_window[i] = 0.54f - 0.46f * cos(2 * M_PI * i / (win_length - 1));
    }

    // 5. 信号填充（中心对齐）
    std::vector<float> padded(mono.size() + n_fft, 0.0f);
    std::copy(mono.begin(), mono.end(), padded.begin() + n_fft/2);

    // 6. 分帧处理
    std::vector<std::vector<float>> frames;
    for (size_t start = 0; start + win_length <= padded.size(); start += hop_length) {
        std::vector<float> frame(win_length);
        for (int i = 0; i < win_length; ++i) {
            frame[i] = padded[start + i] * hamming_window[i];
        }
        frames.emplace_back(std::move(frame));
    }

    // 7. FFT配置
    kiss_fft_cfg cfg = kiss_fft_alloc(n_fft, 0, NULL, NULL);
    if (!cfg) {
        throw std::runtime_error("Failed to initialize FFT");
    }

    // 8. 预分配结果空间
    result.magnitude.resize(frames.size(), std::vector<float>(n_fft/2 + 1));
    result.phase.resize(frames.size(), std::vector<float>(n_fft/2 + 1));

    // 9. 处理每帧
    for (size_t i = 0; i < frames.size(); ++i) {
        kiss_fft_cpx in[n_fft], out[n_fft];
        
        // 填充FFT输入
        for (int j = 0; j < n_fft; ++j) {
            in[j].r = j < win_length ? frames[i][j] : 0.0f;
            in[j].i = 0.0f;
        }

        // 执行FFT
        kiss_fft(cfg, in, out);

        // 计算幅度和相位
        for (int j = 0; j <= n_fft/2; ++j) {
            result.magnitude[i][j] = std::hypot(out[j].r, out[j].i);
            result.phase[i][j] = std::atan2(out[j].i, out[j].r);
        }
    }

    // 10. 清理资源
    kiss_fft_free(cfg);

    return result;
}
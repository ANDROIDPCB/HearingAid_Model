#include <portaudio.h>
#include <iostream>
#include "utool_AudioRT/Audio_RT_Base.h"
#include "utool_AudioRT/Config.h"



// PortAudio 回调函数：实时读取输入并播放输出
static int audioCallback(const void *inputBuffer, void *outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData) {
    // 将输入缓冲区转换为浮点型指针
    const float *in = (const float*)inputBuffer;
    float *out = (float*)outputBuffer;

    // 直接复制输入到输出（直通模式）
    for (unsigned int i = 0; i < framesPerBuffer * NUM_CHANNELS; i++) {
    *out++ = *in++;  // 可以在此处添加音频处理逻辑（如滤波、增益等）
    }
    return paContinue;  // 持续运行
}

int main(int argc, char **argv)
{
    PaError err;
    PaStream* stream;
    PaStreamParameters inputParameters;
    PaStreamParameters outputParameters;

    // 初始化PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return -1;
    }
    // 列出所有音频设备
    list_audio_devices();
    // 获取用户输入设备索引
    int inputDeviceIndex, outputDeviceIndex;
    std::cout << "Enter input device index: ";
    std::cin >> inputDeviceIndex;
    std::cout << "Enter output device index: ";
    std::cin >> outputDeviceIndex;
    // 配置输入参数
    inputParameters.device = inputDeviceIndex;
    inputParameters.channelCount = NUM_CHANNELS;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
    inputParameters.hostApiSpecificStreamInfo = nullptr;
    // 配置输出参数
    outputParameters.device = outputDeviceIndex;
    outputParameters.channelCount = NUM_CHANNELS;
    outputParameters.sampleFormat = paFloat32;
    outputParameters.suggestedLatency = Pa_GetDeviceInfo(outputParameters.device)->defaultLowOutputLatency;
    outputParameters.hostApiSpecificStreamInfo = nullptr;

    std::cout << "打开音频流" << std::endl;

    // 打开音频流
    err = Pa_OpenStream(&stream,
        &inputParameters,
        &outputParameters,
        SAMPLE_RATE,
        HOP_SIZE,
        paClipOff,
        audioCallback,
        nullptr);

    std::cout << "开始录音并播放...按 Ctrl+C 可以停止" << std::endl;

    // 循环等待用户中断
    try {
        while (true) {
            Pa_Sleep(1000);
        }
    } catch (const std::exception& e) {
        std::cout << "停止录音并播放" << std::endl;
    }

    // 停止和关闭流
    err = Pa_StopStream(stream);
    if (err != paNoError) {
        std::cerr << "Error stopping stream: " << Pa_GetErrorText(err) << std::endl;
    }

    err = Pa_CloseStream(stream);
    if (err != paNoError) {
        std::cerr << "Error closing stream: " << Pa_GetErrorText(err) << std::endl;
    }

    // 终止PortAudio
    Pa_Terminate();
    return 0;
}
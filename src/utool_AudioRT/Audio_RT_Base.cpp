#include <portaudio.h>
#include <iostream>

// 音频实时播放的基础函数
void list_audio_devices() {
    PaError err;
    int numDevices;
    const PaDeviceInfo *deviceInfo;

    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return;
    }

    numDevices = Pa_GetDeviceCount();
    for (int i = 0; i < numDevices; ++i) {
        deviceInfo = Pa_GetDeviceInfo(i);
        std::cout << "Device " << i << ": " << deviceInfo->name << std::endl;
    }

    Pa_Terminate();
}
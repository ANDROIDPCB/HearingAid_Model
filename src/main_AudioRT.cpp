#include <portaudio.h>
#include <iostream>
#include "utool_AudioRT/Audio_RT_Base.h"
#include "utool_AudioRT/Config.h"
#include <vector>
#include <deque>
#include <algorithm>
#include "utool_AudioRT/Audio_RT_Process.h"

// 定义时间缓存队列
std::deque<std::deque<__fp16>> Time_Cache(
    NUM_TIME,                     // 行数
    std::deque<__fp16>(WINDOW_SIZE, __fp16(0.0f)) // 每行初始化为包含 WINDOW_SIZE 个元素的 deque
);
// 
std::vector<std::vector<__fp16>> Time_Cache_Matrix; // 进行数组操作的原型数组
std::vector<__fp16> Anly_Windows; // 分析窗口
std::vector<__fp16> Sys_Windows; // 分析窗口
std::vector<__fp16> result_AddSys; // 最后处理两帧叠加综合窗的结果


// 用于指定当前要进行音频塞入的行数
int input_num = 0; 
int hopsize_num = WINDOW_SIZE / HOP_SIZE; // 每一行的Hopsize数
int max_input_num = NUM_TIME + hopsize_num - 1;
int index_num = 0; //用于指定的行数
int move_up_flage = 0; //用于指定是否整体数组上移
int matrix_copy_flage = 0; //用于判断是否已经从队列中复制了数组数据

// PortAudio 回调函数：实时读取输入并播放输出
static int audioCallback(const void *inputBuffer, void *outputBuffer,
    unsigned long sampleNum,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData) {
    // 将输入缓冲区转换为浮点型指针
    audio_sample* input = (audio_sample*)inputBuffer;
    audio_sample* output = (audio_sample*)outputBuffer;
    // 调整索引
    input_num++;
    if(input_num > max_input_num){input_num = max_input_num;move_up_flage = 1;}
    index_num = std::max(input_num - hopsize_num, 0);

    if(move_up_flage == 1){ // 数组整体上移
        deque_move_up(Time_Cache); 
    }
    if(index_num >= 1){ // 已经可以复制了
        deque_copy_start_tar(Time_Cache, index_num - 1, index_num);
    }// 完成当前指定帧的复制

    // 将样本数据进行push
    for (unsigned int i = 0; i < sampleNum; i++) {
        if(index_num == 0){
            //删除开头的点
            Time_Cache[index_num].pop_front();
        }
        else{
            Time_Cache[index_num].pop_back();
        }
        //新的样本点推入末尾
        Time_Cache[index_num].push_back(static_cast<__fp16>(input[i].left * 100));
    }
    // 表示缓存数据已经满了，可以进行输出了
    if(input_num == 33){
        // 在这里面进行音频的算法流处理
        // 1.先复制上面的二维队列到数组中以方便操作
        if(matrix_copy_flage == 0){ // 还没复制队列数据，要开始复制队列数据
            
            Time_Cache_Matrix = deque_to_matrix(Time_Cache);
            matrix_copy_flage = 1;
            // 添加窗函数，每一帧都添加分析窗函数
            Time_Cache_Matrix = apply_window_multiply(Time_Cache_Matrix,Anly_Windows);
        }
        else{ 
            shift_matrix_up(Time_Cache_Matrix,Time_Cache); // 把数组整体上移一位，并且把队列的最后一行数据放到数组最后一行中
            // 添加合成窗函数(只对最后一行)
            multiply_last_n_rows_inplace(Time_Cache_Matrix,Sys_Windows,1);
        }
        // 最后两行乘上综合窗并进行叠加还原
        result_AddSys = Generate_SysResult(Time_Cache_Matrix,Sys_Windows,2);


        // 将数据写入输出缓冲区
        for (unsigned int i = 0; i < HOP_SIZE; ++i) {
            //取出开头样本点输出到左右声道
            output[i].left = result_AddSys[i];
            output[i].right = result_AddSys[i];
        }
    }
    return paContinue;  // 持续运行
}

int main(int argc, char **argv)
{
    PaError err;
    PaStream* stream;
    PaStreamParameters inputParameters;
    PaStreamParameters outputParameters;
    // 生成分析窗(输入FFT的时域加窗)
    Anly_Windows = asymmetric_Analy_windows(HOP_SIZE,WINDOW_SIZE,10);
    // 生成综合窗(IFFT输出的时域结果加窗)
    Sys_Windows = asymmetric_Sys_windows(HOP_SIZE,WINDOW_SIZE,10);
    // 初始化PortAudio
    err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return -1;
    }
    // 列出所有音频设备
    list_audio_devices();
    // 获取用户输入设备索引-
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



    if (err != paNoError) {
        std::cerr << "Error opening stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return -1;
    }

    // 启动流
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Error starting stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return -1;
    }
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
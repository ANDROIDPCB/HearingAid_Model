#ifndef CONFIG_H
#define CONFIG_H



// 采样率
#define SAMPLE_RATE   (16000)

//通道数
#define NUM_CHANNELS    (2)

//帧长
#define WINDOW_SIZE      (256)  

//帧移
#define HOP_SIZE         (64)   

// 存储缓存帧的长度
#define NUM_TIME    (30)

// 存储缓FFT频点数量
#define NUM_FREQ    (129)

//音频采样结构结构体
typedef struct {
    float left;
    float right;
} audio_sample;




#endif
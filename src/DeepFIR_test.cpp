
#include "task/deepfir.h"
#include "process/preprocess_deepfir.h"
// #include "utils/logging.h"

int main(int argc, char **argv)
{
    // model file path
    const char *model_file = argv[1];
    const char *audio_file = argv[2];
    // 初始化
    DeepFIR deepfir;
    // 加载模型
    deepfir.LoadModel(model_file);

    // 运行模型
    std::vector<Detection> objects;
    STFTResult result;
    deepfir.Run(audio_file, result, objects);

    return 0;
}
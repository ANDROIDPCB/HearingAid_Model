
#include "task/deepfir.h"
// #include "utils/logging.h"

int main(int argc, char **argv)
{
    // model file path
    const char *model_file = argv[1];
    // 初始化
    DeepFIR deepfir;
    // 加载模型
    deepfir.LoadModel(model_file);

    // 运行模型
    std::vector<Detection> objects;
    // deepfir.Run(img, objects);

    return 0;
}
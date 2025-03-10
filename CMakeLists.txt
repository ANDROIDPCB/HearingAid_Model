# 设置最低版本号
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
# 设置项目名称
project(rk3588-demo VERSION 0.0.1 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置库架构
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

#  rknn_api 文件夹路径
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
#  rknn_api include 路径
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
#  rknn_api lib 路径
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# 寻找本地的opencv
find_package(OpenCV REQUIRED)
# 输出OpenCV信息
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")


# 用来搜索头文件的目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RGA_DIR}/include
)


# 构建DEEPFIR的预处理库
# deepfir_lib
add_library(deepfir_process SHARED src/process/preprocess_deepfir.cpp)
# 链接库
target_link_libraries(deepfir_process

)

# 构建自定义封装API库
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
# 链接库
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)


# deepfir_lib
add_library(deepfir_lib SHARED src/task/deepfir.cpp)
# 链接库
target_link_libraries(deepfir_lib
    rknn_engine
)


# 测试deepfir
add_executable(deepfir src/DeepFIR_test.cpp)
# 链接库
target_link_libraries(deepfir
        deepfir_lib
)

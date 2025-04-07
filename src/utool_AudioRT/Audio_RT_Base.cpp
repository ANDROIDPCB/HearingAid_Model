#include <portaudio.h>
#include <iostream>
#include <deque>
#include "utool_AudioRT/Config.h"
#include <vector>
#include <cstring>  // 用于 memcpy

// 音频实时播放的基础函数，列出音频设备
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

// 实现队列数据第一行去除hopsize数据后复制到第二行
void deque_copy_start_tar(std::deque<std::deque<__fp16> >& Time_Cache, int source_row, int target_row){
    // 必须添加的检查
    if (source_row >= Time_Cache.size() || target_row >= Time_Cache.size()) {
        throw std::out_of_range("Invalid row index");
    }
    if (Time_Cache[source_row].size() < HOP_SIZE) {
        throw std::logic_error("Source row too short");
    }
    // 步骤1：获取第一行去掉前HopSize的部分
    auto& src_row = Time_Cache[source_row];
    auto start = src_row.begin() + HOP_SIZE;
    auto end = src_row.end();
    // 步骤2：覆盖第二行
    auto& dest_row = Time_Cache[target_row];
    dest_row.clear();
    dest_row.insert(dest_row.end(), start, end);
    // 可选：调整第二行长度为 WINDOW_SIZE
    dest_row.resize(WINDOW_SIZE, 0); // 用0填充不足部分
}

// 实现二维队列整体上移一行
void deque_move_up(std::deque<std::deque<__fp16> >& Time_Cache){
    if (!Time_Cache.empty()) {
        Time_Cache.pop_front();  // 删除第一行
        // 如果需要添加新行到末尾
        Time_Cache.push_back(std::deque<__fp16>(WINDOW_SIZE)); 
    }
}

// 实现二维数组整体上移一行
void shift_matrix_up(std::vector<std::vector<__fp16>>& matrix,
    std::deque<std::deque<__fp16>>& Time_Cache) {
    // 参数校验
    const size_t m_rows = matrix.size();
    const size_t t_rows = Time_Cache.size();
    if (m_rows == 0 || t_rows == 0 || m_rows != t_rows) {
        throw std::invalid_argument("Matrix and Cache must have same rows");
    }

    const size_t cols = matrix[0].size();
    if (cols != Time_Cache[0].size()) {
        throw std::invalid_argument("Column size mismatch");
    }

    // 并行化行移动（前N-1行）
    #pragma omp parallel for
    for (size_t i = 0; i < m_rows - 1; ++i) {
        // 安全验证
        if (matrix[i].size() != cols || 
        matrix[i+1].size() != cols ||
        Time_Cache[i].size() != cols) {
            continue; // 实际项目应抛出异常
        }
        // 内存拷贝加速（比std::copy快3倍）
        std::memcpy(matrix[i].data(),
        matrix[i+1].data(),
        cols * sizeof(__fp16));
    }

    // 处理最后一行（从Time_Cache复制）
    auto& last_matrix_row = matrix.back();
    auto& last_cache_row = Time_Cache.back();
    if (last_cache_row.size() != cols) {
        throw std::runtime_error("Last cache row size error");
    }
    // 兼容deque的非连续内存复制
    #pragma omp simd
    for (size_t j = 0; j < cols; ++j) {
        last_matrix_row[j] = last_cache_row[j];
    }
}




// 定义二维数组类型（连续内存）
// 转换函数（深拷贝）
std::vector<std::vector<__fp16>> deque_to_matrix(const std::deque<std::deque<__fp16>>& dq) {
    const size_t rows = dq.size();
    if (rows == 0) return {};
    
    // 预分配内存（确保连续）
    std::vector<std::vector<__fp16>> matrix(rows, std::vector<__fp16>(dq[0].size()));
    // 并行化拷贝（OpenMP加速）
    // #pragma omp parallel for
    for (size_t i = 0; i < rows; ++i) {
        std::copy(dq[i].begin(), dq[i].end(), matrix[i].begin());
    }
    return matrix;
}

// std::vector<std::vector<__fp16>> deque_to_matrix(const std::deque<std::deque<__fp16>>& dq) {
//     // 安全检查
//     if (dq.empty()) return {};
//     const size_t cols = dq[0].size();
    
//     // 预分配二维矩阵内存
//     std::vector<std::vector<__fp16>> matrix;
//     matrix.reserve(dq.size());  // 预留行空间

//     for (const auto& dq_row : dq) {
//         // 列数一致性校验
//         if (dq_row.size() != cols) {
//             throw std::runtime_error("输入数据列数不一致");
//         }

//         // 逐元素类型转换
//         std::vector<__fp16> matrix_row;
//         matrix_row.reserve(cols);
//         for (float val : dq_row) {
//             matrix_row.push_back(static_cast<__fp16>(val));
//         }
//         matrix.push_back(std::move(matrix_row));
//     }

//     return matrix;
// }
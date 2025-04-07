#include <vector>
#include <cmath>
#include <cassert>
#include "utool_AudioRT/Config.h"

// 创建Sys窗函数
std::vector<__fp16> asymmetric_Sys_windows(int hop_length, int win_length, int zero_d) {
    const int zero_s = win_length - 2 * hop_length;
    const int Bottom_C = win_length - hop_length - zero_d;

    // Generate hanning_window_analy_Half_2
    std::vector<__fp16> hanning_window_analy_Half_2;
    hanning_window_analy_Half_2.reserve(hop_length);
    for (int i = hop_length; i < 2 * hop_length; ++i) {
        const float x = static_cast<float>(i);
        const float val = std::sqrt(0.5f * (1.0f - std::cos(M_PI * x / hop_length)));
        hanning_window_analy_Half_2.push_back(static_cast<__fp16>(val));
    }
    // Generate A_N_Array and compute A_N
    const int A_N_start = win_length - 2 * hop_length - zero_d;
    const int A_N_end = win_length - hop_length - zero_d;
    std::vector<__fp16> A_N;
    A_N.reserve(hop_length);
    for (int i = A_N_start; i < A_N_end; ++i) {
        const float x = static_cast<float>(i);
        const float val = std::sqrt(0.5f * (1.0f - std::cos(M_PI * x / Bottom_C)));
        A_N.push_back(static_cast<__fp16>(val));
    }
    // Compute hanning_window_Sys_Ahead
    std::vector<__fp16> hanning_window_Sys_Ahead;
    hanning_window_Sys_Ahead.reserve(hop_length);
    for (int i = 0; i < hop_length; ++i) {
        const float sys_val = 0.5f * (1.0f - std::cos(M_PI * i / hop_length));
        const float a_n_val = static_cast<float>(A_N[i]);
        const float result = sys_val / a_n_val;
        hanning_window_Sys_Ahead.push_back(static_cast<__fp16>(result));
    }
    // Concatenate and pad
    std::vector<__fp16> hanning_window_Sys;
    hanning_window_Sys.reserve(hanning_window_Sys_Ahead.size() + hanning_window_analy_Half_2.size() + zero_s);
    hanning_window_Sys.insert(hanning_window_Sys.end(), hanning_window_Sys_Ahead.begin(), hanning_window_Sys_Ahead.end());
    hanning_window_Sys.insert(hanning_window_Sys.end(), hanning_window_analy_Half_2.begin(), hanning_window_analy_Half_2.end());
    hanning_window_Sys.insert(hanning_window_Sys.begin(), zero_s, static_cast<__fp16>(0));
    return hanning_window_Sys;
}

// 创建Analy窗函数
std::vector<__fp16> asymmetric_Analy_windows(int hop_length, int win_length, int zero_d) {
    const int Bottom_C = win_length - hop_length - zero_d;

    // Generate hanning_window_analy_Half
    const int Analy_Half_len = win_length - zero_d - hop_length;
    std::vector<__fp16> hanning_window_analy_Half;
    hanning_window_analy_Half.reserve(Analy_Half_len);
    for (int i = 0; i < Analy_Half_len; ++i) {
        const float x = static_cast<float>(i);
        const float val = std::sqrt(0.5f * (1.0f - std::cos(M_PI * x / Bottom_C)));
        hanning_window_analy_Half.push_back(static_cast<__fp16>(val));
    }

    // Generate hanning_window_analy_Half_2
    std::vector<__fp16> hanning_window_analy_Half_2;
    hanning_window_analy_Half_2.reserve(hop_length);
    for (int i = hop_length; i < 2 * hop_length; ++i) {
        const float x = static_cast<float>(i);
        const float val = std::sqrt(0.5f * (1.0f - std::cos(M_PI * x / hop_length)));
        hanning_window_analy_Half_2.push_back(static_cast<__fp16>(val));
    }

    // Concatenate and pad
    std::vector<__fp16> hanning_window_analy;
    hanning_window_analy.reserve(hanning_window_analy_Half.size() + hanning_window_analy_Half_2.size() + zero_d);
    hanning_window_analy.insert(hanning_window_analy.end(), hanning_window_analy_Half.begin(), hanning_window_analy_Half.end());
    hanning_window_analy.insert(hanning_window_analy.end(), hanning_window_analy_Half_2.begin(), hanning_window_analy_Half_2.end());
    hanning_window_analy.insert(hanning_window_analy.begin(), zero_d, static_cast<__fp16>(0));

    return hanning_window_analy;
}

// 应用窗口
std::vector<std::vector<__fp16>> apply_window_multiply(
    const std::vector<std::vector<__fp16>>& Time_Cache_Matrix,
    const std::vector<__fp16>& Anly_Windows)
{
    // 输入有效性检查
    assert(!Time_Cache_Matrix.empty());
    assert(!Anly_Windows.empty());
    assert(Time_Cache_Matrix[0].size() == Anly_Windows.size());

    // 预分配结果内存
    std::vector<std::vector<__fp16>> result(Time_Cache_Matrix.size(),std::vector<__fp16>(Time_Cache_Matrix[0].size()));
    // result.resize(Time_Cache_Matrix.size(),std::vector<__fp16>(Time_Cache_Matrix[0].size()));
    
    // 并行友好的逐元素乘法
    for (const auto& row : Time_Cache_Matrix) {
        std::vector<__fp16> multiplied_row;
        multiplied_row.resize(row.size());
        
        for (size_t i = 0; i < row.size(); ++i) {
            // 使用 float 中间值避免精度损失
            const float val = static_cast<float>(row[i]) * static_cast<float>(Anly_Windows[i]);
            multiplied_row.push_back(static_cast<__fp16>(val));
        }
        
        result.push_back(std::move(multiplied_row));
    }
    
    return result;
}

// 原位处理最后一行数据
void multiply_last_n_rows_inplace(
    std::vector<std::vector<__fp16>>& matrix,
    const std::vector<__fp16>& window,
    size_t n_rows)
{
    const size_t start = matrix.size() > n_rows ? matrix.size() - n_rows : 0;
    for (size_t i = start; i < matrix.size(); ++i) {
        for (size_t j = 0; j < window.size(); ++j) {
            matrix[i][j] = static_cast<__fp16>(
                static_cast<float>(matrix[i][j]) * 
                static_cast<float>(window[j]));
        }
    }
}



// 应用分析窗口结果，只处理末尾的两帧，并且只特定处理最后一帧倒数第二个Hop块，以及倒数第二帧最后一个Hop块，把两个块相加
std::vector<__fp16> Generate_SysResult(
    const std::vector<std::vector<__fp16>>& matrix,
    const std::vector<__fp16>& window,
    size_t n_rows)
{
    std::vector<std::vector<__fp16>> result;
    const size_t start_row = matrix.size() > n_rows ? matrix.size() - n_rows : 0;

    std::vector<__fp16> row_result;
    row_result.resize(HOP_SIZE);
    for (size_t j = 0; j < HOP_SIZE; ++j) {
        const float val_end2 = static_cast<float>(matrix[start_row][window.size() - HOP_SIZE - 1 + j]) 
                        * static_cast<float>(window[window.size() - HOP_SIZE - 1 + j]);
        const float val_end = static_cast<float>(matrix[start_row][window.size() - 2*HOP_SIZE - 1 + j]) 
        * static_cast<float>(window[window.size() - 2*HOP_SIZE - 1 + j]);
        row_result.push_back(static_cast<__fp16>(val_end2 + val_end));
    }

    
    return row_result;
}

void test_corruption() {
    std::vector<std::vector<__fp16>> clean_matrix(30);
    std::vector<std::vector<__fp16>> result;
    result.resize(clean_matrix.size()); // 仅测试此处是否崩溃
}


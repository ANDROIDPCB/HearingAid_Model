#ifndef AUDIO_RT_PROCESS_H
#define AUDIO_RT_PROCESS_H

#include <vector>
#include <cmath>


std::vector<__fp16> asymmetric_Analy_windows(int hop_length, int win_length, int zero_d);
std::vector<__fp16> asymmetric_Sys_windows(int hop_length, int win_length, int zero_d);
std::vector<std::vector<__fp16>> apply_window_multiply(
    const std::vector<std::vector<__fp16>>& Time_Cache_Matrix,
    const std::vector<__fp16>& Anly_Windows);
void multiply_last_n_rows_inplace(
    std::vector<std::vector<__fp16>>& matrix,
    const std::vector<__fp16>& window,
    size_t n_rows);
std::vector<__fp16> Generate_SysResult(
const std::vector<std::vector<__fp16>>& matrix,
const std::vector<__fp16>& window,
size_t n_rows);
void test_corruption();


#endif
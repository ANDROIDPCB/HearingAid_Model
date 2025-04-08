#ifndef AUDIO_RT_BASE_H
#define AUDIO_RT_BASE_H
#include <vector>
#include <deque>  // 必须添加此行
void list_audio_devices();
void deque_copy_start_tar(std::deque<std::deque<__fp16> >& Time_Cache, int source_row, int target_row);
void deque_move_up(std::deque<std::deque<__fp16> >& Time_Cache);
std::vector<std::vector<__fp16>> deque_to_matrix(const std::deque<std::deque<__fp16>>& dq);
void shift_matrix_up(std::vector<std::vector<__fp16>>& matrix, std::deque<std::deque<__fp16> >& Time_Cache);
void saveToWav(const std::string& filename, const std::vector<float>& data);
#endif
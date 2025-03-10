#include <iostream>

int main(){
    // std::cout << "Hell World"<< std::endl;

    int favorites_num;
    std::cout << "请输入0~10中你最喜欢的数字:";
    std::cin >> favorites_num;
    std::cout << favorites_num <<"也是我最喜欢的数字"<< std::endl;
    return 0;
}
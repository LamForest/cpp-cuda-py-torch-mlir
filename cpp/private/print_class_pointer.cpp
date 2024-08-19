#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

struct MyClass {
// private: //除了模板参数，其他方式无法拿到私有成员的类成员指针；
    int a;
    char ch0;
    char ch1;
    double b;
    double c;
    double d;
    std::string str;
    std::vector<double> vec;
};

int main() {
    int MyClass::*ptrToInt = &MyClass::a;
    char MyClass::*pchar0 = &MyClass::ch0;
    char MyClass::*pchar1 = &MyClass::ch1;

    double MyClass::*ptrToDouble = &MyClass::b;
    double MyClass::*ptrToDouble2 = &MyClass::c;
    double MyClass::*ptrToDouble3 = &MyClass::d;
    std::string MyClass::*ptrToStr = &MyClass::str;
    std::vector<double> MyClass::*ptrToVec = &MyClass::vec;

    // 将成员指针转换为足够大的整数类型来查看它的值
    // 注意：这是不可移植的，并且仅用于示教目的

    std::cout << "======== with reinterpret_cast ========" << std::endl;
    std::cout << "The internal value of ptrToInt: " << reinterpret_cast<std::uintptr_t&>(ptrToInt) << std::endl;
    std::cout << "The internal value of pchar0: " << reinterpret_cast<std::uintptr_t&>(pchar0) << std::endl;
    std::cout << "The internal value of pchar1: " << reinterpret_cast<std::uintptr_t&>(pchar1) << std::endl;
    std::cout << "The internal value of ptrToDouble: " << reinterpret_cast<std::uintptr_t&>(ptrToDouble) << std::endl;
    std::cout << "The internal value of ptrToDouble2: " << reinterpret_cast<std::uintptr_t&>(ptrToDouble2) << std::endl;
    std::cout << "The internal value of ptrToDouble3: " << reinterpret_cast<std::uintptr_t&>(ptrToDouble3) << std::endl;
    std::cout << "The internal value of ptrToStr: " << reinterpret_cast<std::uintptr_t&>(ptrToStr) << std::endl;
    std::cout << "The internal value of ptrToVec: " << reinterpret_cast<std::uintptr_t&>(ptrToVec) << std::endl;

    std::cout << "======== without reinterpret_cast ========" << std::endl;

    std::cout << "The internal value of ptrToInt: " << ptrToInt << std::endl;
    std::cout << "The internal value of pchar0: " << pchar0 << std::endl;
    std::cout << "The internal value of pchar1: " << pchar1 << std::endl;
    std::cout << "The internal value of ptrToDouble: " << ptrToDouble << std::endl;
    std::cout << "The internal value of ptrToDouble2: " << ptrToDouble2 << std::endl;
    std::cout << "The internal value of ptrToDouble3: " << ptrToDouble3 << std::endl;
    std::cout << "The internal value of ptrToStr: " << ptrToStr << std::endl;
    std::cout << "The internal value of ptrToVec: " << ptrToVec << std::endl;

    return 0;
}
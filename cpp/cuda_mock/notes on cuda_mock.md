# 

plt拿到的name是fullpath，并且可能有so.1 so.2的后缀，所以这里用contains
```cpp
bool targetLib(const char* name) {
    return adt::StringRef(name).contain("libcuda.so");
}
```

而符号名则可以精确匹配，所以用 `==`
```cpp
bool targetSym(const char* name) {
    return adt::StringRef(name) == "__printf_chk"
}
```
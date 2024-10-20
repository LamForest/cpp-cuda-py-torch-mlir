# 如何访问private成员 - 终极模板大法
http://jiangzhuti.me/posts/C++%E5%A5%87%E6%8A%80%E6%B7%AB%E5%B7%A7%E4%B9%8B%E8%AE%BF%E9%97%AEprivate%E6%88%90%E5%91%98

利用C++规范的feature，类成员指针作为模板参数时，不会检查访问权限

但是类型要是public的，如果这个类是定义是Private的，那么这个办法也不可行

# 类成员指针 - 成员在类内的偏移
https://zhuanlan.zhihu.com/p/367373924

Note: 不能直接打印，需要`reinterpret_cast<std::uintptr_t&>`，否则只能打印出1


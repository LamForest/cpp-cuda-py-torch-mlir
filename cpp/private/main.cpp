#include <iostream>

struct safe {
    private:
        int data = 42;
};

template<int safe::* MEMBER_INT_PTR>
struct GenerateThiefFunction {
    friend void steal_from(safe& victim_object) {
        victim_object.*MEMBER_INT_PTR = 100;
        std::cout << victim_object.*MEMBER_INT_PTR << std::endl;
    }
};

template struct GenerateThiefFunction<&safe::data>;

void steal_from(safe& victim_object);

int main() {

    safe the_one;
    steal_from(the_one);

    return 0;
}

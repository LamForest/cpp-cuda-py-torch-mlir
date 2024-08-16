#include <stdio.h>
#include <stdint.h>

void updatePtrs(int32_t *ptrA, int32_t *ptrB, int32_t *val)
{
    *ptrA += *val;
    *ptrB += *val;
}

void updatePtrs_restrict(int32_t * __restrict__ ptrA, int32_t * __restrict__ ptrB, int32_t * __restrict__ val)
{
    *ptrA += *val;
    *ptrB += *val;
}

int main(){

    int32_t a = 1;
    int32_t b = 10;
    int32_t c = 100;
    updatePtrs(&a, &b, &c);
    printf("updatePtrs                             a: %d, b: %d, c: %d\n", a, b, c);


    a = 1;
    b = 10;
    c = 100;
    updatePtrs_restrict(&a, &a, &a);
    printf("updatePtrs_restrict                    a: %d, b: %d, c: %d\n", a, b, c);



    a = 10;
    updatePtrs_restrict(&a, &a, &a);
    printf("updatePtrs_restrict(aliasing pointers) a: %d\n", a);

    a = 10;
    updatePtrs(&a, &a, &a);
    printf("updatePtrs         (aliasing pointers) a: %d\n", a);

    return 0;
}
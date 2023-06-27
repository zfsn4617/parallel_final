#include "functions.h"
__global__ void scalar_mul(int64 *ret, int64 *a, int64 *b)
{
    int64 *vec_1 = a + blockIdx.x * 256;
    int64 *vec_2 = b + blockIdx.x * 256;
    int64 *vec_ret = ret + blockIdx.x * 256;
    int idx = threadIdx.x;
    vec_ret[idx] = vec_1[idx] * vec_2[idx] % 8380417;
}
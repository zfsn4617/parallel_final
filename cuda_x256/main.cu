#include <stdio.h>
#include "functions.h"
__global__ void hello_from_gpu()
{
    printf("hello word from the gpu %d, %d!\n", blockIdx.x, threadIdx.x);
}

int main()
{
    Initialize_NTT();
    Initialize_iNTT();

    int64 *a = new int64[50000 * 256];
    int64 *b = new int64[50000 * 256];
    int64 *z = new int64[50000 * 256];

    int64 *gpu_a;
    int64 *gpu_b;
    int64 *gpu_a_ntt;
    int64 *gpu_b_ntt;
    int64 *gpu_z_ntt;
    int64 *gpu_z;

    cudaMalloc(&gpu_a, 50000 * 256 * sizeof(int64));
    cudaMalloc(&gpu_b, 50000 * 256 * sizeof(int64));
    cudaMalloc(&gpu_a_ntt, 50000 * 256 * sizeof(int64));
    cudaMalloc(&gpu_b_ntt, 50000 * 256 * sizeof(int64));
    cudaMalloc(&gpu_z_ntt, 50000 * 256 * sizeof(int64));
    cudaMalloc(&gpu_z, 50000 * 256 * sizeof(int64));

    for (int i = 0; i < 50000; i++)
    {
        for (int j = 0; j < 256; j++)
        {
            a[i * 256 + j] = i+1;
            b[i * 256 + j] = i+1;
        }
    }

    auto ckpt0 = high_resolution_clock::now();
    cudaMemcpy(gpu_a, a, 50000 * 256 * sizeof(int64), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, 50000 * 256 * sizeof(int64), cudaMemcpyHostToDevice);
    auto ckpt1 = high_resolution_clock::now();

    NTT<<<50000, 32>>>(gpu_a, gpu_a_ntt);
    NTT<<<50000, 32>>>(gpu_b, gpu_b_ntt);
    cudaDeviceSynchronize();
    cudaError_t err;
    err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));
    auto ckpt2 = high_resolution_clock::now();

    scalar_mul<<<50000, 256>>>(gpu_z_ntt, gpu_a_ntt, gpu_b_ntt);
    cudaDeviceSynchronize();
    auto ckpt3 = high_resolution_clock::now();

    iNTT<<<50000, 32>>>(gpu_z_ntt, gpu_z);
    cudaDeviceSynchronize();
    auto ckpt4 = high_resolution_clock::now();

    cudaMemcpy(z, gpu_z, 50000 * 256 * sizeof(int64), cudaMemcpyDeviceToHost);
    auto ckpt5 = high_resolution_clock::now();
    for (int i = 256 * 100; i < 256 * 101; i++)
    {
        cout << setw(8) << z[i];
        if (i % 8 == 7)
            cout << endl;
    }
    double t0 = duration_cast<microseconds>(ckpt1 - ckpt0).count() / 50000.0;
    double t1 = duration_cast<microseconds>(ckpt2 - ckpt1).count() / 50000.0;
    double t2 = duration_cast<microseconds>(ckpt3 - ckpt2).count() / 50000.0;
    double t3 = duration_cast<microseconds>(ckpt4 - ckpt3).count() / 50000.0;
    double t4 = duration_cast<microseconds>(ckpt5 - ckpt4).count() / 50000.0;
    cout << "计算时间0：" << t0 << "微秒/次" << endl;
    cout << "计算时间1：" << t1 << "微秒/次" << endl;
    cout << "计算时间2：" << t2 << "微秒/次" << endl;
    cout << "计算时间3：" << t3 << "微秒/次" << endl;
    cout << "计算时间4：" << t4 << "微秒/次" << endl;
    cout << "NTT乘法计算时间：" << t1 + t2 + t3 << "微秒/次" << endl;
    cout << "总计算时间：" << t0 + t1 + t2 + t3 + t4 << "微秒/次" << endl;
    cudaDeviceSynchronize();

    return 0;
}

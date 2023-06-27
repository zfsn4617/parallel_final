#include <iostream>
using std::cout;  
using std::endl;  
#include <chrono>
#include "ntt.cuh"
#include "util.h"
typedef std::chrono::high_resolution_clock Clock;

#define check 1
int main()
{
    unsigned n = 65536*(1<<9);
    int size_array = sizeof(unsigned) * n;
    int size = sizeof(unsigned);
    unsigned q = 469762049, psi = 4782969, psiinv = 392193156;
    // unsigned q = 8380417, psi=1921994, psiinv=527981
    // s = 13, t = 1023 
    unsigned int q_bit = 29;
    // unsinged int q_bit = 23
    unsigned* psiTable = (unsigned*)malloc(size_array);
    unsigned* psiinvTable = (unsigned*)malloc(size_array);
    unsigned* psi_powers, * psiinv_powers;
    fillTablePsi64(psi, q, psiinv, psiTable, psiinvTable, n);
    
    cudaMalloc(&psi_powers, size_array);
    cudaMalloc(&psiinv_powers, size_array);
    
    cudaMemcpy(psi_powers, psiTable, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(psiinv_powers, psiinvTable, size_array, cudaMemcpyHostToDevice);
    auto t1 = Clock::now();
    
    cout << "q = " << q << endl;
    cout << "root of unity = " << psi << endl;

   
    unsigned int bit_length = q_bit;
    double mu1 = powl(2, 2 * bit_length);
    unsigned mu = mu1 / q;

    unsigned* a;
    cudaMallocHost(&a, sizeof(unsigned) * n);
    randomArray64(a, n, q);
    unsigned* res_a;
    cudaMallocHost(&res_a, sizeof(unsigned) * n);

    unsigned* d_a;
    cudaMalloc(&d_a, size_array);

    cudaMemcpyAsync(d_a, a, size_array, cudaMemcpyHostToDevice, 0);

    // block num 1
    // thread num 1024
    // shared memery 2048*sizeof(unsigned)
    CTBasedNTTInnerSingle<8*(1<<9), 65536*(1<<9)><<<8*(1<<9), 65536*(1<<9), 8192 * sizeof(unsigned), 0>>>(d_a, q, mu, bit_length, psi_powers);
    unsigned* mid_a;
    cudaMallocHost(&mid_a, sizeof(unsigned) * n);
    cudaMemcpyAsync(mid_a, d_a, size_array, cudaMemcpyDeviceToHost, 0);
    cudaDeviceSynchronize(); 
    GSBasedINTTInnerSingle<8*(1<<9), 65536*(1<<9)><<<8*(1<<9), 65536*(1<<9), 8192 * sizeof(unsigned), 0>>>(d_a, q, mu, bit_length, psiinv_powers);

    cudaMemcpyAsync(res_a, d_a, size_array, cudaMemcpyDeviceToHost, 0);  // do this in async
    cudaDeviceSynchronize();  // CPU being a gentleman, and waiting for GPU to finish it's job
    auto t2 = Clock::now();
    printf("Device FFT took %ld \n",
           std::chrono::duration_cast<
                   std::chrono::milliseconds>(t2 - t1)
                   .count());
    bool correct = 1;
    if (check) //check the correctness of results
    {
        for (int i = 0; i < n; i++)
        {
            if (a[i] != res_a[i])
            {
                correct = 0;
                break;
            }
        }
    }

    if (correct)
        cout << "\nNTT and INTT are working correctly." << endl;
    else
        cout << "\nNTT and INTT are not working correctly." << endl;

    cudaFreeHost(a); cudaFreeHost(res_a);
    cudaFree(d_a);
    return 0;
}



#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <stdio.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;
typedef long long int64;

//void bit_reverse(int* a, int bits);
int mod_pow(int a, int n, int p);
__global__ void NTT(int64* a, int64* out);
__global__ void iNTT(int64* a, int64* out);
__global__ void scalar_mul(int64 *ret, int64 *a, int64 *b);
const int omega = 1921994;
const int omega_inv = 527981;
const int mod = 8380417;
const int times = 256;
const int log_times = 8;
extern int omega_pow[times * 2 + 1];
extern int omega_inv_pow[times * 2 + 1];
extern int input_index[times];

void Initialize_NTT();
void Initialize_iNTT();

#endif
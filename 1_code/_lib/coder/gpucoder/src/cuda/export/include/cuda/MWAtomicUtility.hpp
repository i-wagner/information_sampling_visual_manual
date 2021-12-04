/* Copyright 2017 The MathWorks, Inc. */

#ifndef __MW_ATOMIC_UTILS_H__
#define __MW_ATOMIC_UTILS_H__

/********** AtomicAdd ***********/

float                   cpu_float_atomicAdd(float* u1, float u2);
double                 cpu_double_atomicAdd(double* u1, double u2);
int                     cpu_int32_atomicAdd(int* u1, int u2);
unsigned int           cpu_uint32_atomicAdd(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicAdd(unsigned long long int* u1, unsigned long long int u2);
unsigned long int      cpu_uint64_atomicAdd(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicAdd(unsigned long long int* u1, unsigned long long int u2);
__device__  unsigned long int     gpu_uint64_atomicAdd(unsigned long int* u1, unsigned long int u2);

/********** AtomicSub ***********/

int           cpu_int32_atomicSub(int* u1, int u2);
unsigned int cpu_uint32_atomicSub(unsigned int* u1, unsigned int u2);

/********** AtomicExch ***********/

int                    cpu_int32_atomicExch(int* u1, int u2);
unsigned int           cpu_uint32_atomicExch(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicExch(unsigned long long int* u1, unsigned long long int u2);
unsigned long int      cpu_uint64_atomicExch(unsigned long int* u1, unsigned long int u2);
float                  cpu_float_atomicExch(float* u1, float u2);

__device__ unsigned long long int gpu_uint64_atomicExch(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicExch(unsigned long int* u1, unsigned long int u2);

/********** AtomicMin ***********/

int cpu_int32_atomicMin(int* u1, int u2);
unsigned int cpu_uint32_atomicMin(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicMin(unsigned long long int* u1, unsigned long long int u2);
unsigned long int cpu_uint64_atomicMin(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicMin(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicMin(unsigned long int* u1, unsigned long int u2);

/********** AtomicMax ***********/

int cpu_int32_atomicMax(int* u1, int u2);
unsigned int cpu_uint32_atomicMax(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicMax(unsigned long long int* u1, unsigned long long int u2);
unsigned long int cpu_uint64_atomicMax(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicMax(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicMax(unsigned long int* u1, unsigned long int u2);

/********** AtomicInc ***********/

unsigned int cpu_uint32_atomicInc(unsigned int* u1, unsigned int u2);

/********** AtomicDec ***********/

unsigned int cpu_uint32_atomicDec(unsigned int* u1, unsigned int u2);

/********** AtomicAnd ***********/

int cpu_int32_atomicAnd(int* u1, int u2);
unsigned int cpu_uint32_atomicAnd(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicAnd(unsigned long long int* u1, unsigned long long int u2);
unsigned long int cpu_uint64_atomicAnd(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicAnd(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicAnd(unsigned long int* u1, unsigned long int u2);

/********** AtomicOr ***********/

int cpu_int32_atomicOr(int* u1, int u2);
unsigned int cpu_uint32_atomicOr(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicOr(unsigned long long int* u1, unsigned long long int u2);
unsigned long int cpu_uint64_atomicOr(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicOr(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicOr(unsigned long int* u1, unsigned long int u2);

/********** AtomicXor ***********/

int cpu_int32_atomicXor(int* u1, int u2);
unsigned int cpu_uint32_atomicXor(unsigned int* u1, unsigned int u2);
unsigned long long int cpu_uint64_atomicXor(unsigned long long int* u1, unsigned long long int u2);
unsigned long int cpu_uint64_atomicXor(unsigned long int* u1, unsigned long int u2);

__device__ unsigned long long int gpu_uint64_atomicXor(unsigned long long int* u1, unsigned long long int u2);
__device__ unsigned long int gpu_uint64_atomicXor(unsigned long int* u1, unsigned long int u2);

/********** AtomicCAS ***********/

int cpu_int32_atomicCAS(int* u1, int u2, int u3);
unsigned int cpu_uint32_atomicCAS(unsigned int* u1, unsigned int u2, unsigned int u3);
unsigned long long int cpu_uint64_atomicCAS(unsigned long long int* u1, unsigned long long int u2, unsigned long long int u3);
unsigned long int cpu_uint64_atomicCAS(unsigned long int* u1, unsigned long int u2, unsigned long int u3);

__device__ unsigned long long int gpu_uint64_atomicCAS(unsigned long long int* u1, unsigned long long int u2, unsigned long long int u3);
__device__ unsigned long int gpu_uint64_atomicCAS(unsigned long int* u1, unsigned long int u2, unsigned long int u3);

#endif

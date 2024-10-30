#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <iostream>

#define THREADS 32


template <unsigned int blockSize>
__device__ 
void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64 && tid < 32) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32 && tid < 16) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16 && tid < 8)  sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8  && tid < 4)  sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4  && tid < 2)  sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2  && tid < 1)  sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ 
void dotProduct(const float *w, const float *x, float *g_odata, unsigned int n) {
    extern __shared__ float cache[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockSize * 2 + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float temp_sum = 0.0f;

    while (idx < n) {
        temp_sum += w[idx] * x[idx];
        if (idx + blockSize < n)
            temp_sum += w[idx + blockSize] * x[idx + blockSize];
        idx += gridSize;
    }

    cache[tid] = temp_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = cache[0];
}

// Aquired from NVIDIA documentation modified slightly
// Templates are used to avoid being set as constant during compilation
template <unsigned int blockSize>
__global__ void reduce(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    // Load data into shared memory
    unsigned int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// Example test
int main() {
    unsigned int n = 1000;
    float b = 1.0f;       

    // Allocate and initialize host memory
    float *h_w = new float[n];
    float *h_x = new float[n];
    for (unsigned int i = 0; i < n; ++i) {
        h_w[i] = 300.0f;  // Example initialization
        h_x[i] = 3.0f;
    }

    // Allocate device memory
    float *d_w, *d_x, *d_partialSums;
    cudaMalloc(&d_w, n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_w, h_w, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate number of blocks
    unsigned int numBlocks = (n + THREADS * 2 - 1) / (THREADS * 2);

    // Allocate memory for partial sums
    cudaMalloc(&d_partialSums, numBlocks * sizeof(float));

    // Launch dotProduct kernel
    dotProduct<THREADS><<<numBlocks, THREADS, THREADS * sizeof(float)>>>(d_w, d_x, d_partialSums, n);

    // Sum the partial results
    float finalResult = 0.0f;

    if (numBlocks > 1) {
        float *d_finalResult;
        cudaMalloc(&d_finalResult, sizeof(float));

        // Perform reduction on partial sums
        reduce<THREADS><<<numBlocks, THREADS, THREADS * sizeof(float)>>>(d_partialSums, d_finalResult, numBlocks);

        // Copy the result back to host
        cudaMemcpy(&finalResult, d_finalResult, sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_finalResult);
    } else {
        // Only one block, copy result directly
        cudaMemcpy(&finalResult, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost);
    }

    // Incorporate the bias
    finalResult += b;

    std::cout << "Dot Product Result: " << finalResult << std::endl;

    // Free memory
    delete[] h_w;
    delete[] h_x;
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_partialSums);

    return 0;
}
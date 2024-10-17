#include "../include/fc_layer.cuh"
#include "../include/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void init_curand_states(curandState* state, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void init_weights(float* weights, int size, float scale, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState localState = states[idx];
        float r = sqrtf(6.0f / size);
        weights[idx] = (curand_uniform(&localState) * 2.0f - 1.0f) * r;
        states[idx] = localState;
    }
}

FCLayer::FCLayer(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {

    // 分配并初始化权重和偏置
    cudaMalloc(&weights, input_size * output_size * sizeof(float));
    cudaMalloc(&bias, output_size * sizeof(float));
    cudaMalloc(&grad_weights, input_size * output_size * sizeof(float));
    cudaMalloc(&grad_bias, output_size * sizeof(float));

    // 初始化 cuBLAS
    cublasCreate(&cublas_handle);

    // 初始化 cuRAND
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 1234ULL);

    // 初始化 cuRAND 状态
    curandState* d_states;
    cudaMalloc(&d_states, input_size * output_size * sizeof(curandState));
    int blockSize = 256;
    int numBlocks = (input_size * output_size + blockSize - 1) / blockSize;
    init_curand_states << <numBlocks, blockSize >> > (d_states, 1234ULL, input_size * output_size);

    // 初始化权重
    init_weights << <numBlocks, blockSize >> > (weights, input_size * output_size, sqrtf(2.0f / input_size), d_states);

    // 初始化偏置为0
    cudaMemset(bias, 0, output_size * sizeof(float));

    // 清理临时内存
    cudaFree(d_states);
}

FCLayer::~FCLayer() {
    cudaFree(weights);
    cudaFree(bias);
    cudaFree(grad_weights);
    cudaFree(grad_bias);
    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);
}

void FCLayer::forward(const float* input, float* output, int batch_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 计算 output = weights^T * input + bias
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        output_size, batch_size, input_size,
        &alpha, weights, input_size, input, input_size,
        &beta, output, output_size);

    // 添加偏置
    for (int i = 0; i < batch_size; ++i) {
        cudaMemcpy(output + i * output_size, bias, output_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void FCLayer::backward(const float* input, const float* grad_output, float* grad_input, int batch_size) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 计算输入梯度：grad_input = weights * grad_output
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        input_size, batch_size, output_size,
        &alpha, weights, input_size, grad_output, output_size,
        &beta, grad_input, input_size);

    // 计算权重梯度：grad_weights = input * grad_output^T
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
        input_size, output_size, batch_size,
        &alpha, input, input_size, grad_output, output_size,
        &beta, grad_weights, input_size);

    // 计算偏置梯度
    for (int i = 0; i < batch_size; ++i) {
        cublasSaxpy(cublas_handle, output_size, &alpha,
            grad_output + i * output_size, 1, grad_bias, 1);
    }
}

__global__ void update_params_kernel(float* params, float* grads, int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * grads[idx];
    }
}

void FCLayer::update_params(float learning_rate) {
    int blockSize = 256;
    int numBlocks = (input_size * output_size + blockSize - 1) / blockSize;

    // 更新权重
    update_params_kernel << <numBlocks, blockSize >> > (weights, grad_weights, input_size * output_size, learning_rate);

    // 更新偏置
    numBlocks = (output_size + blockSize - 1) / blockSize;
    update_params_kernel << <numBlocks, blockSize >> > (bias, grad_bias, output_size, learning_rate);
}
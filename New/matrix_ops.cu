#include "../include/matrix_ops.cuh"
#include "../include/cuda_utils.cuh"

// CUDA核函数：执行矩阵乘法
__global__ void matrix_multiply_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // 计算当前线程负责的输出元素的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程不会越界
    if (row < M && col < N)
    {
        float sum = 0.0f;
        // 计算点积
        for (int i = 0; i < K; ++i)
        {
            sum += A[row * K + i] * B[i * N + col];
        }
        // 将结果存入输出矩阵
        C[row * N + col] = sum;
    }
}
// 主机端函数：启动矩阵乘法核函数
void matrix_multiply(float *A, float *B, float *C, int M, int N, int K, cudaStream_t stream)
{
    // 定义线程块和网格的维度
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // 启动核函数
    matrix_multiply_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// CUDA核函数：执行矩阵加法
__global__ void matrix_add_kernel(float *A, float *B, float *C, int size)
{
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 确保线程不会越界
    if (idx < size)
    {
        C[idx] = A[idx] + B[idx];
    }
}
// 主机端函数：启动矩阵加法核函数
void matrix_add(float *A, float *B, float *C, int size, cudaStream_t stream)
{
    // 定义线程块和网格的维度
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
#include "../include/activation_functions.cuh"

namespace ActivationFunctions {

    __global__ void relu_forward_kernel(float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            output[idx] = fmaxf(input[idx], 0.0f);
        }
    }

    __global__ void relu_backward_kernel(float* grad_output, float* input, float* grad_input, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
        }
    }

    void relu_forward(float* input, float* output, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        relu_forward_kernel << <numBlocks, blockSize >> > (input, output, size);
    }

    void relu_backward(float* grad_output, float* input, float* grad_input, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        relu_backward_kernel << <numBlocks, blockSize >> > (grad_output, input, grad_input, size);
    }

}  // namespace ActivationFunctions
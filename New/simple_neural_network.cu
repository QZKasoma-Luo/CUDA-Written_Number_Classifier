#include "../include/simple_neural_network.cuh"
#include "../include/activation_functions.cuh"
#include "../include/cuda_utils.cuh"
#include <cmath>

// 定义 subtract 函数
__global__ void subtract(float* a, const float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

SimpleNeuralNetwork::SimpleNeuralNetwork(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    hidden_layer = new FCLayer(input_size, hidden_size);
    output_layer = new FCLayer(hidden_size, output_size);

    cudaMalloc(&hidden_output, hidden_size * sizeof(float));
    cudaMalloc(&hidden_grad, hidden_size * sizeof(float));
}

SimpleNeuralNetwork::~SimpleNeuralNetwork() {
    delete hidden_layer;
    delete output_layer;
    cudaFree(hidden_output);
    cudaFree(hidden_grad);
}

void SimpleNeuralNetwork::forward(const float* input, float* output, int batch_size) {
    // Hidden layer forward
    hidden_layer->forward(input, hidden_output, batch_size);
    ActivationFunctions::relu_forward(hidden_output, hidden_output, hidden_size * batch_size);

    // Output layer forward
    output_layer->forward(hidden_output, output, batch_size);
}

void SimpleNeuralNetwork::backward(const float* input, const float* target, int batch_size) {
    // 计算输出层的梯度
    float* output_grad;
    cudaMalloc(&output_grad, output_size * batch_size * sizeof(float));

    // 假设使用均方误差损失函数
    // dZ[2] = A[2] - Y
    subtract << <(output_size * batch_size + 255) / 256, 256 >> > (output_layer->get_weights(), target, output_grad, output_size * batch_size);

    // 输出层反向传播
    output_layer->backward(hidden_output, output_grad, hidden_grad, batch_size);

    // 隐藏层的激活函数梯度
    ActivationFunctions::relu_backward(hidden_grad, hidden_output, hidden_grad, hidden_size * batch_size);

    // 隐藏层反向传播
    float* input_grad;
    cudaMalloc(&input_grad, input_size * batch_size * sizeof(float));
    hidden_layer->backward(input, hidden_grad, input_grad, batch_size);

    cudaFree(output_grad);
    cudaFree(input_grad);
}

void SimpleNeuralNetwork::update_params(float learning_rate) {
    hidden_layer->update_params(learning_rate);
    output_layer->update_params(learning_rate);
}
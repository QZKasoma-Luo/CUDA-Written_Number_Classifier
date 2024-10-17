#include "../include/simple_neural_network.cuh"
#include "../include/mnist_loader.cuh"
#include "../include/cuda_utils.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>

// 计算交叉熵损失
__global__ void compute_loss_kernel(float* predictions, float* targets, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float batch_loss = 0;
        for (int c = 0; c < num_classes; ++c) {
            float pred = fmaxf(fminf(predictions[idx * num_classes + c], 1 - 1e-7f), 1e-7f);
            batch_loss -= targets[idx * num_classes + c] * logf(pred);
        }
        atomicAdd(loss, batch_loss);
    }
}

float compute_loss(float* predictions, float* targets, int batch_size, int num_classes) {
    float* d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    compute_loss_kernel << <num_blocks, threads_per_block >> > (predictions, targets, d_loss, batch_size, num_classes);

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / batch_size;
}

// 评估函数
float evaluate(SimpleNeuralNetwork& nn, float* images, float* labels, int num_samples, int batch_size, int input_size, int num_classes) {
    int correct = 0;
    for (int i = 0; i < num_samples; i += batch_size) {
        int current_batch_size = std::min(batch_size, num_samples - i);
        float* output;
        cudaMalloc(&output, num_classes * current_batch_size * sizeof(float));

        nn.forward(images + i * input_size, output, current_batch_size);

        int* h_predictions = new int[current_batch_size];
        int* h_labels = new int[current_batch_size];

        // 获取预测结果和实际标签
        for (int j = 0; j < current_batch_size; ++j) {
            float max_val = -INFINITY;
            int max_idx = 0;
            for (int c = 0; c < num_classes; ++c) {
                float val;
                cudaMemcpy(&val, &output[j * num_classes + c], sizeof(float), cudaMemcpyDeviceToHost);
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            h_predictions[j] = max_idx;
            cudaMemcpy(&h_labels[j], &labels[i + j], sizeof(float), cudaMemcpyDeviceToHost);
        }

        // 计算正确预测的数量
        for (int j = 0; j < current_batch_size; ++j) {
            if (h_predictions[j] == static_cast<int>(h_labels[j])) {
                correct++;
            }
        }

        delete[] h_predictions;
        delete[] h_labels;
        cudaFree(output);
    }

    return static_cast<float>(correct) / num_samples;
}

int main() {
    const int input_size = 784;  // 28x28
    const int hidden_size = 128;
    const int output_size = 10;  // 10 digits
    const int batch_size = 64;
    const float learning_rate = 0.01f;
    const int num_epochs = 1000;

    SimpleNeuralNetwork nn(input_size, hidden_size, output_size);

    // 加载 MNIST 数据
    MNISTData mnist_data = load_mnist("../data/train_mnist/MNIST/raw");

    std::cout << "Starting training...\n";

    // 训练循环
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        for (int i = 0; i < mnist_data.num_train; i += batch_size) {
            int current_batch_size = std::min(batch_size, mnist_data.num_train - i);

            // 前向传播
            float* output;
            cudaMalloc(&output, output_size * current_batch_size * sizeof(float));
            nn.forward(mnist_data.train_images + i * input_size, output, current_batch_size);

            // 计算损失
            float loss = compute_loss(output, mnist_data.train_labels + i * output_size, current_batch_size, output_size);
            total_loss += loss;

            // 反向传播
            nn.backward(mnist_data.train_images + i * input_size, mnist_data.train_labels + i * output_size, current_batch_size);

            // 更新参数
            nn.update_params(learning_rate);

            cudaFree(output);
        }

        float avg_loss = total_loss / (mnist_data.num_train / batch_size);
        std::cout << "Epoch " << epoch + 1 << " Loss: " << avg_loss << std::endl;

        // 在测试集上评估
        float accuracy = evaluate(nn, mnist_data.test_images, mnist_data.test_labels, mnist_data.num_test, batch_size, input_size, output_size);
        std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;
    }

    std::cout << "Training completed.\n";

    // 清理内存
    free_mnist(mnist_data);

    return 0;
}
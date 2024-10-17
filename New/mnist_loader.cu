#include "../include/mnist_loader.cuh"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// 辅助函数：读取大端序整数
int read_int(std::ifstream& file) {
    int value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int));
    return (value << 24) | ((value & 0x0000FF00) << 8) | ((value & 0x00FF0000) >> 8) | (value >> 24);;
}

MNISTData load_mnist(const std::string& path) {
    MNISTData data;

    // 加载训练图像
    std::ifstream train_images("D:/CSC_485B_GPU_Computing/New_Direction/mnist_cuda_nn/data/train_mnist/MNIST/raw/train-images-idx3-ubyte", std::ios::binary);
    if (!train_images) throw std::runtime_error("Cannot open train-images file");

    int magic = read_int(train_images);
    data.num_train = read_int(train_images);
    int rows = read_int(train_images);
    int cols = read_int(train_images);

    std::vector<float> train_image_data(data.num_train * rows * cols);
    for (int i = 0; i < data.num_train * rows * cols; ++i) {
        unsigned char temp = 0;
        train_images.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        train_image_data[i] = static_cast<float>(temp) / 255.0f;
    }

    // 加载训练标签
    std::ifstream train_labels("D:/CSC_485B_GPU_Computing/New_Direction/mnist_cuda_nn/data/train_mnist/MNIST/raw/train-labels-idx1-ubyte", std::ios::binary);
    if (!train_labels) throw std::runtime_error("Cannot open train-labels file");

    magic = read_int(train_labels);
    int num_labels = read_int(train_labels);
    if (num_labels != data.num_train) throw std::runtime_error("Labels file has incorrect number of items");

    std::vector<float> train_label_data(data.num_train * 10, 0);
    for (int i = 0; i < data.num_train; ++i) {
        unsigned char temp = 0;
        train_labels.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        train_label_data[i * 10 + temp] = 1.0f;  // One-hot encoding
    }

    // 加载测试图像
    std::ifstream test_images("D:/CSC_485B_GPU_Computing/New_Direction/mnist_cuda_nn/data/test_mnist/MNIST/raw/t10k-images-idx3-ubyte");
    if (!test_images) throw std::runtime_error("Cannot open test-images file");

    magic = read_int(test_images);
    data.num_test = read_int(test_images);
    rows = read_int(test_images);
    cols = read_int(test_images);

    std::vector<float> test_image_data(data.num_test * rows * cols);
    for (int i = 0; i < data.num_test * rows * cols; ++i) {
        unsigned char temp = 0;
        test_images.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        test_image_data[i] = static_cast<float>(temp) / 255.0f;
    }

    // 加载测试标签
    std::ifstream test_labels("D:/CSC_485B_GPU_Computing/New_Direction/mnist_cuda_nn/data/test_mnist/MNIST/raw/t10k-labels-idx1-ubyte");
    if (!test_labels) throw std::runtime_error("Cannot open test-labels file");

    magic = read_int(test_labels);
    num_labels = read_int(test_labels);
    if (num_labels != data.num_test) throw std::runtime_error("Test labels file has incorrect number of items");

    std::vector<float> test_label_data(data.num_test * 10, 0);
    for (int i = 0; i < data.num_test; ++i) {
        unsigned char temp = 0;
        test_labels.read(reinterpret_cast<char*>(&temp), sizeof(temp));
        test_label_data[i * 10 + temp] = 1.0f;  // One-hot encoding
    }

    // 分配 GPU 内存并复制数据
    cudaMalloc(&data.train_images, data.num_train * rows * cols * sizeof(float));
    cudaMalloc(&data.train_labels, data.num_train * 10 * sizeof(float));
    cudaMalloc(&data.test_images, data.num_test * rows * cols * sizeof(float));
    cudaMalloc(&data.test_labels, data.num_test * 10 * sizeof(float));

    cudaMemcpy(data.train_images, train_image_data.data(), data.num_train * rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data.train_labels, train_label_data.data(), data.num_train * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data.test_images, test_image_data.data(), data.num_test * rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data.test_labels, test_label_data.data(), data.num_test * 10 * sizeof(float), cudaMemcpyHostToDevice);

    return data;
}

void free_mnist(MNISTData& data) {
    cudaFree(data.train_images);
    cudaFree(data.train_labels);
    cudaFree(data.test_images);
    cudaFree(data.test_labels);
}
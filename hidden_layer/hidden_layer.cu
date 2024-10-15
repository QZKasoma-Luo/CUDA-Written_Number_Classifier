#include "hidden_layer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

struct HiddenLayer
{
    int inputSize;
    int hiddenSize;
    float *d_weights;
    float *d_biases;
    float *d_output;
    cublasHandle_t cublasHandle;
};

// Function declarations

/**
 * @brief CUDA kernel function to adjust the data range
 * @param data Array of data to be adjusted
 * @param size Size of the array
 */
__global__ void adjustRange(float *data, int size);

/**
 * @brief CUDA kernel function for forward propagation
 * @param weights Weight matrix
 * @param input Input data
 * @param biases Bias vector
 * @param output Output result
 * @param inputSize Input dimension
 * @param hiddenSize Size of the hidden layer
 * @param batchSize Batch size
 */
__global__ void forwardPassKernel(float *weights, float *input, float *biases, float *output, int inputSize, int hiddenSize, int batchSize);

/**
 * @brief Perform forward propagation
 * @param layer Hidden layer structure
 * @param input Input data
 * @param batchSize Batch size
 */
void forwardPass(HiddenLayer *layer, float *input, int batchSize);

/**
 * @brief Perform backward propagation
 * @param layer Hidden layer structure
 * @param gradOutput Output gradient
 * @param input Input data
 * @param learningRate Learning rate
 * @param batchSize Batch size
 */
void backwardPass(HiddenLayer *layer, float *gradOutput, float *input, float learningRate, int batchSize);

/**
 * @brief CUDA kernel function for backward propagation
 * @param gradOutput Output gradient
 * @param input Input data
 * @param weights Weight matrix
 * @param biases Bias vector
 * @param output Output from forward propagation
 * @param gradWeights Weight gradients
 * @param gradBiases Bias gradients
 * @param inputSize Input dimension
 * @param hiddenSize Size of the hidden layer
 * @param batchSize Batch size
 * @param learningRate Learning rate
 */
__global__ void backwardPassKernel(float *gradOutput, float *input, float *weights, float *biases, float *output, float *gradWeights, float *gradBiases, int inputSize, int hiddenSize, int batchSize, float learningRate);

/**
 * @brief Get the weights of the hidden layer
 * @param layer Hidden layer structure
 * @return Pointer to the weight matrix
 */
float *getWeights(HiddenLayer *layer);

/**
 * @brief Get the biases of the hidden layer
 * @param layer Hidden layer structure
 * @return Pointer to the bias vector
 */
float *getBiases(HiddenLayer *layer);

/**
 * @brief Create a new hidden layer
 * @param inputSize Input dimension
 * @param hiddenSize Size of the hidden layer
 * @return Pointer to the newly created hidden layer structure
 */
HiddenLayer *createHiddenLayer(int inputSize, int hiddenSize);

/**
 * @brief Initialize the weights and biases of the hidden layer
 * @param layer Hidden layer structure
 */
void initializeLayer(HiddenLayer *layer);

/**
 * @brief Get the output of the hidden layer
 * @param layer Hidden layer structure
 * @param output Array to store the output
 * @param batchSize Batch size
 */
void getOutput(HiddenLayer *layer, float *output, int batchSize);

/**
 * @brief Free the resources occupied by the hidden layer
 * @param layer Hidden layer structure
 */
void destroyHiddenLayer(HiddenLayer *layer);

// Function implementations
float *getWeights(HiddenLayer *layer)
{
    return layer->d_weights;
}

float *getBiases(HiddenLayer *layer)
{
    return layer->d_biases;
}

HiddenLayer *createHiddenLayer(int inputSize, int hiddenSize)
{
    HiddenLayer *layer = (HiddenLayer *)malloc(sizeof(HiddenLayer));
    layer->inputSize = inputSize;
    layer->hiddenSize = hiddenSize;

    cudaMalloc(&layer->d_weights, inputSize * hiddenSize * sizeof(float));
    cudaMalloc(&layer->d_biases, hiddenSize * sizeof(float));
    layer->d_output = NULL; // Initialize to NULL

    cublasCreate(&layer->cublasHandle);

    return layer;
}

void initializeLayer(HiddenLayer *layer)
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

    // Use cuRAND to generate random numbers in the range [0, 1) in GPU memory
    curandGenerateUniform(gen, layer->d_weights, layer->inputSize * layer->hiddenSize);
    curandGenerateUniform(gen, layer->d_biases, layer->hiddenSize);

    // Adjust the range to [-0.5, 0.5)
    adjustRange<<<(layer->inputSize * layer->hiddenSize + 255) / 256, 256>>>(
        layer->d_weights, layer->inputSize * layer->hiddenSize);
    adjustRange<<<(layer->hiddenSize + 255) / 256, 256>>>(
        layer->d_biases, layer->hiddenSize);

    curandDestroyGenerator(gen);
}

__global__ void adjustRange(float *data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread ID
    if (idx < size)                                  // Ensure the thread is within the data range
    {
        data[idx] = data[idx] - 0.5f;
    }
}

void forwardPass(HiddenLayer *layer, float *input, int batchSize)
{
    dim3 blockSize(256); // 256 threads per block for T4
    dim3 gridSize((layer->hiddenSize * batchSize + blockSize.x - 1) / blockSize.x);

    float *d_input;
    cudaMalloc(&d_input, layer->inputSize * batchSize * sizeof(float));
    cudaMemcpy(d_input, input, layer->inputSize * batchSize * sizeof(float), cudaMemcpyHostToDevice);

    if (layer->d_output != NULL)
    {
        cudaFree(layer->d_output);
    }
    cudaMalloc(&layer->d_output, layer->hiddenSize * batchSize * sizeof(float));

    forwardPassKernel<<<gridSize, blockSize>>>(
        d_input,
        layer->d_weights,
        layer->d_biases,
        layer->d_output,
        layer->inputSize,
        layer->hiddenSize,
        batchSize);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error in forwardpass: %s\n", cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    cudaFree(d_input);

    // Debug: Check if d_output contains valid data
    /*
    float *debug_output = (float *)malloc(layer->hiddenSize * batchSize * sizeof(float));
    cudaMemcpy(debug_output, layer->d_output, layer->hiddenSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Debug: First few values of d_output after forwardPass:\n");
    for (int i = 0; i < 6 && i < layer->hiddenSize * batchSize; i++)
    {
        printf("%f ", debug_output[i]);
    }
    printf("\n");
    free(debug_output);
    */
}

__global__ void forwardPassKernel(float *input, float *weights, float *biases, float *output, int inputSize, int hiddenSize, int batchSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuronIdx = idx / batchSize;
    int batchIdx = idx % batchSize;

    if (neuronIdx < hiddenSize)
    {
        float sum = 0.0f;
        for (int i = 0; i < inputSize; ++i)
        {
            sum += weights[neuronIdx * inputSize + i] * input[batchIdx * inputSize + i];
            // Debug output
            /*
            if (neuronIdx == 0 && batchIdx == 0)
            {
                printf("Debug: weight[%d]=%f, input[%d]=%f\n", i, weights[neuronIdx * inputSize + i], i, input[batchIdx * inputSize + i]);
            }
            */
        }
        sum += biases[neuronIdx];

        // Debug output
        // printf("Debug: neuron %d, batch %d, sum before ReLU: %f\n", neuronIdx, batchIdx, sum);

        output[batchIdx * hiddenSize + neuronIdx] = fmaxf(sum, 0.0f); // ReLU activation

        // Debug output
        // printf("Debug: neuron %d, batch %d, output after ReLU: %f\n", neuronIdx, batchIdx, output[batchIdx * hiddenSize + neuronIdx]);
    }
}

void backwardPass(HiddenLayer *layer, float *gradOutput, float *input, float learningRate, int batchSize)
{
    float *d_gradWeights, *d_gradBiases;
    cudaMalloc(&d_gradWeights, layer->inputSize * layer->hiddenSize * sizeof(float));
    cudaMalloc(&d_gradBiases, layer->hiddenSize * sizeof(float));

    dim3 blockSize(256);
    dim3 gridSize((layer->hiddenSize * batchSize + blockSize.x - 1) / blockSize.x);

    backwardPassKernel<<<gridSize, blockSize>>>(
        gradOutput, input, layer->d_weights, layer->d_biases, layer->d_output,
        d_gradWeights, d_gradBiases, layer->inputSize, layer->hiddenSize, batchSize, learningRate);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in backwardPass: %s\n", cudaGetErrorString(error));
    }

    cudaDeviceSynchronize();
    cudaFree(d_gradWeights);
    cudaFree(d_gradBiases);
}

__global__ void backwardPassKernel(float *gradOutput, float *input, float *weights, float *biases, float *output, float *gradWeights, float *gradBiases, int inputSize, int hiddenSize, int batchSize, float learningRate)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuronIdx = idx / batchSize;
    int batchIdx = idx % batchSize;

    if (neuronIdx < hiddenSize)
    {
        float gradActivation = (output[batchIdx * hiddenSize + neuronIdx] > 0) ? 1.0f : 0.0f; // ReLU derivative
        float grad = gradOutput[batchIdx * hiddenSize + neuronIdx] * gradActivation;

        atomicAdd(&gradBiases[neuronIdx], grad);

        for (int i = 0; i < inputSize; ++i)
        {
            float gradWeight = grad * input[batchIdx * inputSize + i];
            atomicAdd(&gradWeights[neuronIdx * inputSize + i], gradWeight);
        }

        for (int i = 0; i < inputSize; ++i)
        {
            weights[neuronIdx * inputSize + i] -= learningRate * gradWeights[neuronIdx * inputSize + i] / batchSize;
        }
        biases[neuronIdx] -= learningRate * gradBiases[neuronIdx] / batchSize;
    }
}

void getOutput(HiddenLayer *layer, float *output, int batchSize)
{
    if (layer->d_output == NULL)
    {
        fprintf(stderr, "Error: d_output is not allocated\n");
        return;
    }

    // Debug output
    // printf("Debug: hiddenSize = %d, batchSize = %d\n", layer->hiddenSize, batchSize);

    float *h_temp_output = (float *)malloc(layer->hiddenSize * batchSize * sizeof(float));
    if (h_temp_output == NULL)
    {
        fprintf(stderr, "Error: Failed to allocate host memory for temporary output\n");
        return;
    }

    cudaError_t error = cudaMemcpy(h_temp_output, layer->d_output, layer->hiddenSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in getOutput (device to host): %s\n", cudaGetErrorString(error));
        free(h_temp_output);
        return;
    }

    memcpy(output, h_temp_output, layer->hiddenSize * batchSize * sizeof(float));

    // Debug output
    /*
    printf("Debug: First few values of output on CPU after transfer:\n");
    for (int i = 0; i < 6 && i < layer->hiddenSize * batchSize; i++)
    {
        printf("%f ", output[i]);
    }
    printf("\n");
    */

    free(h_temp_output);
}

void destroyHiddenLayer(HiddenLayer *layer)
{
    cudaFree(layer->d_weights);
    cudaFree(layer->d_biases);
    cudaFree(layer->d_output);
    cublasDestroy(layer->cublasHandle);
    free(layer);
}
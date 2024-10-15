






#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <dirent.h>  
#include <sys/stat.h>    
#include <string.h>      
#include <errno.h>
#include <chrono>
#include <iostream>


#define WIDTH 28;
#define HEIGHT 28;
#define IMGSIZE 784


// CPU baseline


float* read_image_file_to_vector(const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Unable to open file %s!\n", filename);
        return NULL;
    }

    float *vector = (float*)malloc(IMGSIZE * sizeof(float));
    if (!vector) {
        printf("Memory allocation failed!\n");
        fclose(file);
        return NULL;
    }

    // Read pixel data and normalize
    for (int i = 0; i < IMGSIZE; i++) {
        unsigned char pixel;
        size_t read_items = fread(&pixel, sizeof(unsigned char), 1, file);
        if (read_items != 1) {
            printf("Error reading pixel data from %s!\n", filename);
            free(vector);
            fclose(file);
            return NULL;
        }
        vector[i] = pixel / 255.0f;
    }

    fclose(file);
    return vector;
}

int cpu_parse(const char* foldername) {

    DIR *dir;
    struct dirent *entry;
    struct stat file_stat;


    if ((dir = opendir(foldername)) == NULL) {
        perror("opendir");
        return 1;
    }

    while ((entry = readdir(dir)) != NULL) {
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", foldername, entry->d_name);

        // Skip directories
        if (stat(filepath, &file_stat) == -1 || S_ISDIR(file_stat.st_mode)) {
            continue;
        }

        // assume all files in the folder are images
        // Process each image file
        float* vector = read_image_file_to_vector(filepath);
        if (vector) {
            // printf("Successfully processed image file: %s\n", filepath);

            // Example: Print the first 10 pixel values
            // for (int i = 0; i < 10; i++) {
            //     printf("%f ", vector[i]);
            // }
            // printf("\n");

            // Free the allocated vector memory
            free(vector);
        }
    }

    closedir(dir);
    return 0;
}

// CUDA kernel 
__global__ void normalize(unsigned char* d_pixels, float* d_vector) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // normalize pixel values
    if (idx < IMGSIZE) {
        d_vector[idx] = d_pixels[idx] / 255.0f;
    }
}

// Function to read all images from a folder
// Sends each pixel to a kernal to be normalized
// Further processing to be added
int process_images(const char* foldername) {
    DIR *dir;
    struct dirent *entry;
    struct stat file_stat;

    // Open the directory
    if ((dir = opendir(foldername)) == NULL) {
        perror("opendir");
        return 1;
    }

    // Allocate memory for pixel data on CPU
    unsigned char *pixels = (unsigned char*)malloc(IMGSIZE * sizeof(unsigned char));
    if (!pixels) {
        printf("Memory allocation failed!\n");
        closedir(dir);
        return 1;
    }

    // Allocate memory on GPU
    unsigned char *d_pixels;
    float *d_vector;
    cudaError_t err;

    err = cudaMalloc((void**)&d_pixels, IMGSIZE * sizeof(unsigned char));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_pixels: %s\n", cudaGetErrorString(err));
        free(pixels);
        closedir(dir);
        return 1;
    }

    err = cudaMalloc((void**)&d_vector, IMGSIZE * sizeof(float));
    if (err != cudaSuccess) {
        printf("CUDA malloc failed for d_vector: %s\n", cudaGetErrorString(err));
        cudaFree(d_pixels);
        free(pixels);
        closedir(dir);
        return 1;
    }

    // Iterate over the files in the directory
    while ((entry = readdir(dir)) != NULL) {
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", foldername, entry->d_name);

        // Skip directories
        if (stat(filepath, &file_stat) == -1 || S_ISDIR(file_stat.st_mode)) {
            continue;
        }

        // Read the image file
        FILE *file = fopen(filepath, "rb");
        if (!file) {
            printf("Unable to open file %s: %s\n", filepath, strerror(errno));
            cudaFree(d_pixels);
            cudaFree(d_vector);
            free(pixels);
            closedir(dir);
            return 1;
        }

        // Read pixel data from the image file
        size_t read_items = fread(pixels, sizeof(unsigned char), IMGSIZE, file);
        if (read_items != IMGSIZE) {
            printf("Error reading pixel data from %s!\n", filepath);
            fclose(file);
            cudaFree(d_pixels);
            cudaFree(d_vector);
            free(pixels);
            closedir(dir);
            return 1;
        }

        fclose(file);

        // Copy data from CPU to GPU
        err = cudaMemcpy(d_pixels, pixels, IMGSIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("CUDA memcpy failed from Host to Device for d_pixels: %s\n", cudaGetErrorString(err));
            cudaFree(d_pixels);
            cudaFree(d_vector);
            free(pixels);
            closedir(dir);
            return 1;
        }

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (IMGSIZE + threadsPerBlock - 1) / threadsPerBlock;

        normalize<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_vector);

        // Synchronize to ensure kernel execution is complete
        cudaDeviceSynchronize();

        // Check for kernel launch errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            cudaFree(d_pixels);
            cudaFree(d_vector);
            free(pixels);
            closedir(dir);
            return 1;
        }
    }

    closedir(dir);

    // Free GPU memory
    cudaFree(d_pixels);
    cudaFree(d_vector);

    free(pixels);

    return 0;
}

int main() {
    const char* foldername = "/content/drive/MyDrive/idx3_images/";

    auto const cpu_start = std::chrono::high_resolution_clock::now();
    cpu_parse("/content/drive/MyDrive/idx3_images/");
    auto const cpu_end = std::chrono::high_resolution_clock::now();

    std::cout << "CPU Baseline time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
              << " ns" << std::endl;


    auto const kernel_start = std::chrono::high_resolution_clock::now();
    int status = process_images(foldername);
    auto const kernel_end = std::chrono::high_resolution_clock::now();

    std::cout << "GPU Solution time: "
      << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
      << " ns" << std::endl;


    return 0;
}


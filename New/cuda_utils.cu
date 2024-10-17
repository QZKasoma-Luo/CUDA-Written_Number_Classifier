#include "../include/cuda_utils.cuh"

void check_cuda_error(cudaError_t result, const char *func, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
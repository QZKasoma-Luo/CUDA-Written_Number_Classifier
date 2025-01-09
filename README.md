# GPU-Accelerated KNN Implementation

A CUDA-optimized K-Nearest Neighbors algorithm implementation, designed specifically for the MNIST dataset. This implementation achieves high performance through GPU acceleration and various optimization techniques. This is Google colab version, if you want to run it on your local machine, nesscary modifcation required.

## Features

- CUDA-accelerated KNN implementation with:
  - Multi-stream parallel processing
  - Optimized bitonic sort algorithm
  - Efficient memory management
  - Vectorized computations
- Performance enhancements:
  - 96.65% accuracy on MNIST (Before Optimzing GPU Computation)
  - Significantly faster than CPU implementations

## Running on Google Colab

1. Open the Google Colab notebook version of the code
2. The notebook includes all necessary setup and MNIST dataset downloading
3. Simply run all cells in order - no additional setup required

Both versions will output:

- Loading progress
- Processing progress
- Final accuracy
- Detailed timing breakdown for each kernel
- Overall execution time

_Note: The Colab version is completely self-contained and ready to run. It automatically handles all dependencies and dataset management._

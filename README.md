# GPU-Accelerated KNN Implementation

A CUDA-optimized K-Nearest Neighbors algorithm implementation, designed specifically for the MNIST dataset. This implementation achieves high performance through GPU acceleration and various optimization techniques.

## Features

- CUDA-accelerated KNN implementation with:
  - Multi-stream parallel processing
  - Optimized bitonic sort algorithm
  - Efficient memory management
  - Vectorized computations
- Performance enhancements:
  - 96.65% accuracy on MNIST
  - Significantly faster than CPU implementations
  - Efficient memory usage with page-locked memory

## Running on Local Machine

1. Prerequisites:

```
- CUDA Toolkit (11.0 or later)
- C++ compiler with C++14 support
- NVIDIA GPU with CUDA support
```

2. Clone and Build:

```bash
git clone [repository-url]
cd [project-directory]
nvcc -O3 main.cu -o knn
```

3. Prepare your MNIST dataset with the following structure:

```
./
├── train_mnist/
│   └── MNIST/
│       └── raw/
│           ├── train-images-idx3-ubyte
│           └── train-labels-idx1-ubyte
└── test_mnist/
    └── MNIST/
        └── raw/
            ├── t10k-images-idx3-ubyte
            └── t10k-labels-idx1-ubyte
```

4. Run the program:

```bash
./knn
```

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

#include <stdio.h>
#include <iostream> 
#include <stdlib.h>

using namespace std;

__global__ void matrixMul_square_only(int *a, int *b, int *c, int M_size) {
    //expect int pointer to sequence of elements in one matrix 
    //Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * M_size + col] = 0;
    for (int k = 0; k < M_size; k++) {
        c[row * M_size + col] += a[row * M_size + k] * b[k * M_size + col];
        }
}
__global__ void transpose_parallel_row_square_only(int *from, int *to, int M_size){
    //expect int pointer to sequence of elements in one matrix 
    int k = threadIdx.x;
    for (int i = 0; i < M_size; i++){
        to[i + k * M_size] = from[k + i * M_size];
    }
    
}



void init_matrix(int* m, int N){
    for(int i = 0; i < N; i++){
        m[i] = rand() % 100;
    }
}
int main(){
    int N = 16;
    int colums = 4;
    int rows = 4;
    size_t bytes = N * N * sizeof(int);

    int *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    init_matrix(a, N);
    init_matrix(b, N);

    int*d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, a, bytes, cudaMemcpyHostToDevice);
    transpose_parallel_row_square_only<<<128,32>>>(d_in,d_out,max(colums,rows));
    cudaMemcpy(b, d_out, bytes, cudaMemcpyDeviceToHost);
      cout << "matrix A\n";
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) { 
        cout << a[i*colums+j] << " "; 
    }
    cout << endl;
    }
      cout << "matrix A transpose\n";
    for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) { 
        cout << b[i*colums+j] << " "; 
    }
    cout << endl;
    }


    return 0;
}
// int main() {
//     // const int n = 10;
//     int* d_A, *d_B;
//     int A[] = {1,1,1,2,2,2,3,3,3};
//     int* B;
//     cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
//     transpose_parallel_row<<<128,32>>>(d_A, d_B, 3);
//     cudaMemcpy(B, d_B, sizeof(d_B), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < 9; i++) { 
//         cout << B[i] << " "; 
//         cout << endl; 
//     } 
//     for (int i = 0; i < 9; i++) { 
//         cout << A[i] << " "; 
//         cout << endl; 
//     }

//     return 0;
// }
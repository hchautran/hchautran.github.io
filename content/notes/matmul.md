---
title: Matrix multiplication (Version 1)
draft: false
tags:
  - CUDA
---
# Simple Matrix Multiplication with CUDA: A Beginner's Guide

Matrix multiplication is one of the most fundamental operations in linear algebra and forms the backbone of many computational applications, from machine learning to scientific computing. While CPUs can handle matrix multiplication, GPUs excel at this task due to their parallel architecture. In this post, we'll explore how to implement the simplest version of matrix multiplication using CUDA.

## Why Use GPU for Matrix Multiplication?

Matrix multiplication involves computing the dot product of rows and columns, which can be done independently for each element of the result matrix. This makes it an ideal candidate for parallel processing. A GPU can execute thousands of threads simultaneously, making it much faster than a CPU for large matrices.

## The Mathematical Foundation

Before diving into code, let's recall matrix multiplication:

For matrices A (m×k) and B (k×n), the result matrix C (m×n) is computed as:

$$C_{ij}=\sum_{k=0}^{K} A_{ik} * B_{kj}$$



Each element $C_{ij}$ requires K multiply-add operations, and since we have m×n elements to compute, the total complexity is O(m×n×k).

## The Naive CUDA Implementation

Let's start with the simplest possible CUDA implementation. We'll assign one thread to compute each element of the result matrix.

### CUDA Kernel

```c++
__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int K) {
    // Calculate the row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundaries
    if (row < M && col < N) {
        float sum = 0.0f;

        // Compute dot product for C[row][col]
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

### Complete Host Code

```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        for (int k = 0; k < K; k++) {
            C[row*N+ col] += A[row * K + k] * B[k * N + col];
        }
    }
}

void initializeMatrix(std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    // Matrix dimensions
    const int M = 1024;  // Rows in A and C
    const int N = 1024;  // Columns in B and C
    const int K = 1024;  // Columns in A, rows in B

    // Host matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    // Initialize matrices with random values
    srand(time(nullptr));
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Device matrices
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize(ceil( M/ (float)blockSize.x), ceil( N/ (float)blockSize.y));
    // Launch kernel and measure time
    auto start = std::chrono::high_resolution_clock::now();
    matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));

    std::cout << "Matrix multiplication completed in " << duration.count() << " ms" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

## Understanding the Implementation

### Thread Organization

- Each thread computes one element of the result matrix C
- We use a 2D grid of 2D blocks (16×16 threads per block)
- Thread coordinates map directly to matrix indices

### Memory Access Pattern

- Each thread reads one row from matrix A and one column from matrix B
- This creates a lot of redundant memory accesses (not optimal, but simple)
- Global memory is accessed without optimization

### Grid and Block Configuration

- Block size: 16×16 = 256 threads (good for most GPUs)
- Grid size calculated to cover the entire result matrix
- Boundary checking ensures we don't go out of bounds

## Compilation and Running

To compile and run this code:

```bash
nvcc -o matrix_mul matrix_mul.cu
./matrix_mul
```

Make sure you have CUDA toolkit installed and a CUDA-capable GPU.

## Performance Characteristics

This naive implementation will work but isn't optimized. For 1024×1024 matrices, you might see:

- Significant speedup over CPU (10-50x depending on your hardware)
- But much slower than optimized libraries like cuBLAS

## Limitations and Next Steps

This simple implementation has several limitations:

1. **Memory Bandwidth**: We're not utilizing shared memory or memory coalescing
2. **Thread Divergence**: All threads follow the same execution path (actually good here)
3. **Arithmetic Intensity**: Low ratio of computation to memory access

Future optimizations could include:

- Tiled matrix multiplication using shared memory
- Memory coalescing optimization
- Using tensor cores on modern GPUs
- Comparing with cuBLAS performance

## Conclusion

This simple CUDA matrix multiplication implementation demonstrates the basic concepts of GPU programming: thread organization, memory management, and kernel execution. While not optimized, it provides a solid foundation for understanding how GPUs can accelerate mathematical computations.

The beauty of this approach lies in its simplicity - one thread per result element makes the mapping straightforward and the code easy to understand. As you progress in CUDA programming, you'll learn more sophisticated techniques, but this naive version will always remain a valuable reference point for understanding the fundamentals.


---
## Reference
Hwu, W.-M. W., Kirk, D. B., & El Hajj, I. (2022). Programming massively parallel processors (4th ed.). doi:10.1016/c2020-0-02969-5

---
title: Matrix multiplication (Version 2) - Tiled Matmul
draft: false
tags:
  - CUDA
  - notes
---

# Tiled Matrix Multiplication: Optimizing for Memory Hierarchy

Building on our [naive matrix multiplication implementation](matmul), we'll now explore **tiled matrix multiplication** - a crucial optimization technique that dramatically improves performance by leveraging the GPU's memory hierarchy. This approach addresses the memory bandwidth limitations we discussed in our [CUDA memory model](cuda_mem) post.

> [!note] Why Tiling Matters
> The naive approach suffers from poor **compute-to-memory access ratio** (arithmetic intensity). Each element is loaded multiple times from global memory, making the kernel memory-bound. Tiling solves this by reusing data in fast on-chip memories.

## The Problem with Naive Matrix Multiplication

Our previous [naive implementation](matmul) has a fundamental limitation: each element of matrices A and B is loaded from global memory multiple times. For a matrix multiplication C = A × B:

- Each element of A is loaded **n times** (once for each column of B)
- Each element of B is loaded **m times** (once for each row of A)
- This results in very low arithmetic intensity: **0.25 FLOP/B**

> [!example] Memory Access Analysis
> For a 1024×1024 matrix multiplication:
> - **Naive approach**: ~8.4M memory loads, 2.1B FLOPs → 0.25 FLOP/B
> - **Tiled approach**: ~0.5M memory loads, 2.1B FLOPs → 4+ FLOP/B
> 
> This 16x improvement in arithmetic intensity can make the difference between memory-bound and compute-bound performance!

## The Tiling Strategy

Tiling divides the computation into smaller **tiles** that fit in fast on-chip memory (shared memory). The key insight is:

1. **Load tiles** of A and B into shared memory
2. **Compute partial results** using the tiles
3. **Reuse the loaded data** for multiple computations
4. **Accumulate results** in registers

### Mathematical Foundation

For matrices A (M×K), B (K×N), and C (M×N), we partition them into tiles:

- A is divided into tiles of size TILE_SIZE × TILE_SIZE
- B is divided into tiles of size TILE_SIZE × TILE_SIZE  
- C is computed tile by tile

Each tile of C is computed as:
$$C_{tile} = \sum_{k=0}^{K/TILE\_SIZE} A_{tile,k} \times B_{tile,k}$$

## CUDA Implementation

### Kernel Design

```c++
__global__ void tiledMatMul(float* A, float* B, float* C, int M, int N, int K) {
    // Allowcate Shared memory for tiles
    
    
    // Iterate over tiles
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of A into shared memory
        
        // Load tile of B into shared memory
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial result using shared memory
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
}
```

### Memory Access Pattern

> [!example] Memory Access Optimization
> 
> **Before tiling** (naive):
> - Each thread loads A[i][k] and B[k][j] for every k
> - Total loads: M×N×K elements from A + M×N×K elements from B
> 
> **After tiling** (TILE_SIZE = 16):
> - Each tile is loaded once and reused for 16×16 = 256 computations
> - Total loads: (M×K + K×N) / 16 elements
> - **16x reduction** in memory traffic!

### Thread Block Configuration

```c++
// Launch configuration
dim3 blockSize(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads per block
dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
              (M + TILE_SIZE - 1) / TILE_SIZE);

tiledMatMul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
```

## Handling Out-of-Bounds Matrix Access

When matrices don't divide evenly by the tile size, we need to handle boundary conditions carefully. Let's examine a concrete example:

> [!example] Out-of-Bounds Example
> 
> **Matrix dimensions**: A(1000×800), B(800×1200), C(1000×1200)
> **Tile size**: 16×16
> 
> **Grid configuration**:
> - Grid X: (1200 + 16 - 1) / 16 = 75 blocks
> - Grid Y: (1000 + 16 - 1) / 16 = 63 blocks
> - Total: 75 × 63 = 4,725 blocks
> 
> **Boundary cases**:
> - **Rightmost tiles**: Columns 1184-1199 (partial tile)
> - **Bottom tiles**: Rows 992-999 (partial tile)
> - **Last tile in K dimension**: K=800, tile covers 784-799

### Boundary Condition Analysis

```c++
// Example: Thread at (tx=8, ty=12) in block (bx=74, by=62)
int row = by * TILE_SIZE + ty;  // 62*16 + 12 = 1004
int col = bx * TILE_SIZE + tx;  // 74*16 + 8 = 1192

// For matrix A (1000×800):
if (row < M && (tile * TILE_SIZE + tx) < K) {
    // row = 1004 >= 1000 → FALSE, so As[ty][tx] = 0.0f
    As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
} else {
    As[ty][tx] = 0.0f;  // ← This executes
}

// For matrix B (800×1200):
if ((tile * TILE_SIZE + ty) < K && col < N) {
    // col = 1192 < 1200 → TRUE, but depends on tile
    Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
} else {
    Bs[ty][tx] = 0.0f;  // ← May execute for last tile
}
```

### Visual Representation

```
Matrix A (1000×800) with 16×16 tiles:
┌─────────────────────────────────────────┐
│ 62×16 = 992 rows                        │
├─────────────────────────────────────────┤
│ 8 rows (992-999) ← Partial bottom tile  │
└─────────────────────────────────────────┘

Matrix B (800×1200) with 16×16 tiles:
┌─────────────────────────────────────────────────────────┐
│ 74×16 = 1184 columns                                    │
├─────────────────────────────────────────────────────────┤
│ 16 columns (1184-1199) ← Partial rightmost tile        │
└─────────────────────────────────────────────────────────┘
```

### Why Zero Padding Works

> [!tip] Zero Padding Strategy
> 
> Setting out-of-bounds elements to 0.0f is safe because:
> - **Multiplication by zero**: 0 × anything = 0
> - **Addition of zero**: sum + 0 = sum (no effect)
> - **Preserves correctness**: Final result is unchanged
> 
> **Alternative approaches**:
> - Skip computation for out-of-bounds threads
> - Use separate kernels for boundary tiles
> - Pad matrices to tile-aligned dimensions

### Complete Kernel With Boundary Handling

```c++
__global__ void tiledMatMulSafe(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile with bounds checking
        int A_col = tile * TILE_SIZE + tx;
        if (row < M && A_col < K) {
            As[ty][tx] = A[row * K + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile with bounds checking  
        int B_row = tile * TILE_SIZE + ty;
        if (B_row < K && col < N) {
            Bs[ty][tx] = B[B_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute with bounds checking
        for (int k = 0; k < TILE_SIZE; k++) {
            int A_k = tile * TILE_SIZE + k;
            if (A_k < K) {  // Only multiply if within bounds
                sum += As[ty][k] * Bs[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result with bounds checking
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

## Performance Analysis

### Arithmetic Intensity Improvement

> [!example] Arithmetic Intensity Calculation
> 
> **Tiled approach** (TILE_SIZE = 16):
> - Memory loads: 2 × (M×K + K×N) / 16 bytes
> - FLOPs: 2 × M × N × K
> - Arithmetic intensity: 16 FLOP/B (vs 0.25 FLOP/B for naive)
> 
> **On A100 GPU** (1555 GB/s bandwidth):
> - Memory-bound limit: 1555 × 16 = 24,880 GFLOPS
> - This approaches the compute peak (~19,500 GFLOPS)!

### Memory Hierarchy Utilization

> [!tip] Memory Usage Breakdown
> - **Global memory**: Initial tile loads (coalesced access)
> - **Shared memory**: Tile storage and reuse (fast on-chip access)
> - **Registers**: Accumulation variable `sum` (fastest access)
> - **Constant memory**: Could store TILE_SIZE if needed



### 3. Warp-Level Optimizations

Exploit warp-level primitives for better memory coalescing and computation.

## Comparison with Naive Implementation

| Aspect | Naive | Tiled |
|--------|-------|-------|
| **Arithmetic Intensity** | 0.25 FLOP/B | 16+ FLOP/B |
| **Memory Traffic** | 2×M×N×K | 2×(M×K + K×N)/16 |
| **Shared Memory** | Not used | TILE_SIZE² × 2 |
| **Performance** | Memory-bound | Near compute-bound |
| **Code Complexity** | Simple | Moderate |

> [!example] Performance Results
> 
> **1024×1024 matrix multiplication on A100**:
> - **Naive**: ~400 GFLOPS (memory-bound)
> - **Tiled (16×16)**: ~15,000 GFLOPS (near compute-bound)
> - **Speedup**: ~37x improvement!

## Best Practices

> [!tip] Implementation Guidelines
> 1. **Choose appropriate tile size**: Balance shared memory usage vs occupancy
> 2. **Ensure memory coalescing**: Threads should access consecutive memory locations
> 3. **Minimize bank conflicts**: Avoid multiple threads accessing same shared memory bank
> 4. **Use appropriate synchronization**: `__syncthreads()` at the right points
> 5. **Handle boundary conditions**: Check array bounds for partial tiles

## When to Use Tiling

> [!note] Tiling is most beneficial when:
> - Matrices are large enough to benefit from data reuse
> - Memory bandwidth is the bottleneck (not compute)
> - You have sufficient shared memory per block
> - The algorithm has regular access patterns

## Next Steps

This tiled implementation forms the foundation for even more advanced optimizations:

- **Tensor Core utilization** for mixed-precision operations
- **Multi-GPU scaling** for very large matrices
- **Automatic tuning** of tile sizes and thread configurations
- **Integration with cuBLAS** for production use

## Summary

Tiled matrix multiplication demonstrates the power of understanding GPU memory hierarchy. By strategically using shared memory to reuse data, we can:

- **Increase arithmetic intensity** from 0.25 to 16+ FLOP/B
- **Reduce memory traffic** by 16x
- **Achieve near-peak performance** on modern GPUs
- **Transform memory-bound kernels** into compute-bound ones

The key insight is that **data reuse** is often more valuable than raw computational power. This principle applies to many other GPU algorithms beyond matrix multiplication.

---
## References
Hwu, W.-M. W., Kirk, D. B., & El Hajj, I. (2022). Programming massively parallel processors (4th ed.). doi:10.1016/c2020-0-02969-5
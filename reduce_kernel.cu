/*************************************************************************************************************************
* Name: Malik Freeman
* Date: 3/2/2025
* Course: SSE 662 - Design, Maintenance, and Quality
* Assignment: Module 4 Programming Task
* File Name: reduce_kernal.cu
* Description: This file implements a CUDA-based parallel reduction kernel to compute the sum of an array.
*              The kernel performs a block-wise reduction using shared memory and atomic operations to accumulate
*              partial sums efficiently. The program launches the reduction kernel with configurable block sizes
*              and synchronizes execution to ensure correctness. The kernel iterates through the array, computes
*              partial sums per thread, performs intra-block reduction using shared memory, and writes the final
*              sum using an atomic operation. The launchReduceKernel function manages kernel execution.
*
*************************************************************************************************************************/


#include <cuda_runtime.h>

__global__ void reduceKernel(float *inputArray, float *output, int arraySize) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    //  Each thread computes its partial sum over the array
    float partialSum = 0.0f;
    while (idx < arraySize) {
        partialSum += inputArray[idx];
        idx += stride;
    }
    sdata[tid] = partialSum;
    __syncthreads();

    //  Reduce within the block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    //  Block leader writes result to output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Host function to launch the kernel
void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks) {
    // Launch with dynamic shared memory (blockSize * sizeof(float))
    reduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, arraySize);
    cudaDeviceSynchronize();
}
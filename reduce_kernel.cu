#include <cuda_runtime.h>

__global__ void reduceKernel(float *inputArray, float *output, int arraySize) {
    extern __shared__ float shared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float partialSum = 0.0f;

    while (idx < arraySize) {
        partialSum += inputArray[idx];
        idx += stride;
    }

    int localId = threadIdx.x;
    shared[localId] = partialSum;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i /= 2) {
        if (localId < i) {
            shared[localId] += shared[localId + i];
        }
        __syncthreads();
    }

    if (localId == 0) {
        atomicAdd(output, shared[0]);
    }
}
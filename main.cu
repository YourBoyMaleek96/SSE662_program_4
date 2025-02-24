#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

__global__ void reduceKernel(float *inputArray, float *output, int arraySize); // Kernel prototype

uint64_t xoroshiro128plus(uint64_t s[2]) {
    uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    s[1] = (s1 << 36) | (s1 >> 28);
    return result;
}

void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks) {
    reduceKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, arraySize);
    cudaDeviceSynchronize();
}

int main() {
    int dataSize = 4096;
    std::vector<float> h_data(dataSize);

    uint64_t seed[2] = {12345, 67890};
    for (int i = 0; i < dataSize; ++i) {
        h_data[i] = static_cast<float>(xoroshiro128plus(seed)) / UINT64_MAX;
    }

    float *d_data, *d_output;
    cudaMalloc(&d_data, dataSize * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_data, h_data.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice);

    // CPU sum calculation
    float h_result_cpu = 0.0f;
    for (float val : h_data) {
        h_result_cpu += val;
    }

    std::vector<int> blockSizes = {128, 256, 512, 1024};
    std::vector<int> numBlocksVec;

    for (int blockSize : blockSizes) {
        numBlocksVec.push_back((dataSize + blockSize - 1) / blockSize);
    }

    // CUDA event for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << std::setw(15) << "Block Size" << std::setw(15) << "Num Blocks"
              << std::setw(15) << "CUDA Time (ms)" << std::setw(15) << "CUDA Result" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Measure performance for each block size and number of blocks
    for (int blockSize : blockSizes) {
        int numBlocks = (dataSize + blockSize - 1) / blockSize;
        cudaMemset(d_output, 0, sizeof(float));

        cudaEventRecord(start);
        launchReduceKernel(d_data, d_output, dataSize, blockSize, numBlocks);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float h_result_cuda;
        cudaMemcpy(&h_result_cuda, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        std::cout << std::setw(15) << blockSize << std::setw(15) << numBlocks
                  << std::setw(15) << elapsedTime << std::setw(15) << h_result_cuda << std::endl;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "CPU Result: " << h_result_cpu << std::endl;

    cudaFree(d_data);
    cudaFree(d_output);
    cudaDeviceReset();

    return 0;
}

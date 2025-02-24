/*************************************************************************************************************************
* Name: Malik Freeman
* Date: 3/2/2025
* Course: SSE 662 - Design, Maintenance, and Quality
* Assignment: Module 4 Programming Task
* File Name: main.cu
* Description: This file implements a CUDA-based parallel reduction algorithm to compute the sum of an array.
*              The program initializes an array of floating-point numbers, transfers data to the GPU, and launches
*              a reduction kernel with varying block sizes to measure performance. The results are compared against
*              a CPU-based reduction implementation for accuracy. Performance metrics such as execution time,
*              percentage error, and speedup are displayed in a tabular format. The CUDA Runtime API is used for
*              memory management, kernel execution, and performance timing.
*************************************************************************************************************************/





#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

extern void launchReduceKernel(float *d_input, float *d_output, int arraySize, int blockSize, int numBlocks);

// CPU reduction for verification
float cpuReduce(float *data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

int main() {
    const int arraySize = 32768;
    const int arrayBytes = arraySize * sizeof(float);

    // Host memory
    float *h_input = new float[arraySize];
    float *h_output = new float(1);

    // Seed and generate random data (0 to 1)
    srand(42);
    for (int i = 0; i < arraySize; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, arrayBytes);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, arrayBytes, cudaMemcpyHostToDevice);

    // Experiment configurations
    int blockSizes[] = {128, 256, 512, 1024};
    const int numTests = 4;

    // CPU reduction
    auto cpuStart = std::chrono::high_resolution_clock::now();
    float cpuResult = cpuReduce(h_input, arraySize);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    float cpuTime = std::chrono::duration<float, std::milli>(cpuEnd - cpuStart).count();

    // Table header
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "----------------------------------------------------------------------------------------------------\n";
    std::cout << "| Block Size | CPU Result | GPU Result | % Error   | CPU Time (ms) | GPU Time (ms) | Speedup    |\n";
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    // Test block sizes
    for (int i = 0; i < numTests; i++) {
        int blockSize = blockSizes[i];
        int numBlocks = (arraySize + blockSize - 1) / blockSize;

        // Reset output
        cudaMemset(d_output, 0, sizeof(float));

        // Warm-up run
        launchReduceKernel(d_input, d_output, arraySize, blockSize, numBlocks);

        // Time GPU
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        launchReduceKernel(d_input, d_output, arraySize, blockSize, numBlocks);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpuTime = 0;
        cudaEventElapsedTime(&gpuTime, start, stop);

        // Get GPU result
        cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
        float gpuResult = *h_output;

        // % Error
        float error = (cpuResult != 0) ? fabs((gpuResult - cpuResult) / cpuResult) * 100.0f : 0.0f;

        // Speedup
        float speedup = cpuTime / gpuTime;

        // Print row
        std::cout << "| " << std::setw(10) << blockSize << " | "
                  << std::setw(10) << cpuResult << " | "
                  << std::setw(10) << gpuResult << " | "
                  << std::setw(9) << error << " | "
                  << std::setw(13) << cpuTime << " | "
                  << std::setw(13) << gpuTime << " | "
                  << std::setw(10) << speedup << " |\n";

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    std::cout << "----------------------------------------------------------------------------------------------------\n";

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
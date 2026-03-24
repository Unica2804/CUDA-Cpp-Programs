#include <iostream>
#include <cuda_runtime.h>

// Macro to wrap CUDA calls for immediate failure detection
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void helloCuda() {
    printf("Thread %d executing on RTX 3050 (sm_86)\n", threadIdx.x);
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found. Check Podman CDI mapping." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Detected " << deviceCount << " CUDA device(s). C++20 standard active." << std::endl;

    // Launch a simple kernel with 1 block and 5 threads
    helloCuda<<<1, 5>>>();

    // Block the host until the device has completed all preceding tasks
    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
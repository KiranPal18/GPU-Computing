
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define BINS 256
#define THREADS 256

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        return 1; \
    } \
} while (0)

__global__ void histogram(const unsigned char* input, int N, unsigned int* hist) {
    __shared__ unsigned int localHist[BINS];

    int tid = threadIdx.x;

    for (int i = tid; i < BINS; i += blockDim.x) {
        localHist[i] = 0;
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        atomicAdd(&localHist[input[i]], 1);
    }

    __syncthreads();

    for (int i = tid; i < BINS; i += blockDim.x) {
        atomicAdd(&hist[i], localHist[i]);
    }
}

int main() {
    int N = 1 << 20;

    std::vector<unsigned char> input(N);
    std::vector<unsigned int> hist(BINS, 0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < N; i++) {
        input[i] = dist(rng);
    }

    unsigned char* d_input;
    unsigned int* d_hist;

    CHECK(cudaMalloc(&d_input, N * sizeof(unsigned char)));
    CHECK(cudaMalloc(&d_hist, BINS * sizeof(unsigned int)));

    CHECK(cudaMemcpy(d_input, input.data(),
                     N * sizeof(unsigned char),
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemset(d_hist, 0, BINS * sizeof(unsigned int)));

    int blocks = (N + THREADS - 1) / THREADS;

    histogram<<<blocks, THREADS>>>(d_input, N, d_hist);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(hist.data(), d_hist,
                     BINS * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));

    for (int i = 0; i < BINS; i++) {
        std::cout << i << " : " << hist[i] << '\n';
    }

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_hist));

    return 0;
}

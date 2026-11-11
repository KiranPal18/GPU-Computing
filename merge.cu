
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

#define THREADS 256

#define CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        return 1; \
    } \
} while (0)

__device__ int lowerBound(const int* arr, int l, int r, int x) {
    while (l < r) {
        int mid = l + (r - l) / 2;

        if (arr[mid] < x)
            l = mid + 1;
        else
            r = mid;
    }

    return l;
}

__device__ int upperBound(const int* arr, int l, int r, int x) {
    while (l < r) {
        int mid = l + (r - l) / 2;

        if (arr[mid] <= x)
            l = mid + 1;
        else
            r = mid;
    }

    return l;
}

__global__ void mergePass(const int* src, int* dst, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    int pairWidth = 2 * width;
    int start = (idx / pairWidth) * pairWidth;

    int mid = start + width;
    if (mid > n) mid = n;

    int end = start + pairWidth;
    if (end > n) end = n;

    int x = src[idx];
    int pos;

    if (idx < mid) {
        int smaller = lowerBound(src, mid, end, x);

        pos = start + (idx - start) + (smaller - mid);
    }
    else {
        int smallerOrEqual = upperBound(src, start, mid, x);

        pos = start + (idx - mid) + (smallerOrEqual - start);
    }

    dst[pos] = x;
}

int main() {
    int N = 1 << 20;

    std::vector<int> input(N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 10000);

    for (int i = 0; i < N; i++) {
        input[i] = dist(rng);
    }

    int *d_a, *d_b;

    CHECK(cudaMalloc((void**)&d_a, N * sizeof(int)));
    CHECK(cudaMalloc((void**)&d_b, N * sizeof(int)));

    CHECK(cudaMemcpy(d_a, input.data(),
                     N * sizeof(int),
                     cudaMemcpyHostToDevice));

    int blocks = (N + THREADS - 1) / THREADS;

    int* d_src = d_a;
    int* d_dst = d_b;

    for (int width = 1; width < N; width *= 2) {

        mergePass<<<blocks, THREADS>>>(d_src, d_dst, N, width);

        CHECK(cudaGetLastError());

        std::swap(d_src, d_dst);
    }

    CHECK(cudaDeviceSynchronize());

    std::vector<int> output(N);

    CHECK(cudaMemcpy(output.data(), d_src,
                     N * sizeof(int),
                     cudaMemcpyDeviceToHost));

    std::vector<int> expected = input;
    std::sort(expected.begin(), expected.end());

    std::cout << "Correct: "
              << (output == expected) << std::endl;

    for (int i = 0; i < 20 && i < N; i++) {
        std::cout << output[i] << " ";
    }

    std::cout << std::endl;

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));

    return 0;
}

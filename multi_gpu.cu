#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <thread>

using namespace std;

// CUDA Kernel for vector addition
__global__ void vecadd(int *a, int *b, int *c, int n) {
    // Calculate the global index of the thread across the entire grid
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Boundary check to ensure the thread does not access memory out of bounds
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Work done by one CPU thread: add its chunk [start, start+len) on GPU `dev`
void run_on_gpu(int dev, int *a, int *b, int *c, int start, int len) {
    // Every CPU thread must select its own GPU, the current device is per thread
    CUDA_CHECK(cudaSetDevice(dev));

    // Pointers for GPU device memory
    int *d_a, *d_b, *d_c;

    // Allocate memory on this GPU for its chunk only
    CUDA_CHECK(cudaMalloc(&d_a, len*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, len*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, len*sizeof(int)));

    // Transfer this chunk from Host (CPU) to Device (GPU)
    CUDA_CHECK(cudaMemcpy(d_a, a + start, len*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b + start, len*sizeof(int), cudaMemcpyHostToDevice));

    // Define execution configuration: block size and grid size
    int block = 256;
    int grid = (len + block - 1) / block;

    // Launch the kernel on this GPU
    vecadd<<<grid, block>>>(d_a, d_b, d_c, len);
    CUDA_CHECK(cudaGetLastError());

    // Transfer the result back into this chunk of the host vector
    CUDA_CHECK(cudaMemcpy(c + start, d_c, len*sizeof(int), cudaMemcpyDeviceToHost));

    // Free GPU memory to avoid leaks
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

int main() {
    int gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus));
    cout << "GPUs: " << gpus << "\n";

    int n=1000;
    vector<int> a(n), b(n), c(n);

    // Initialize host vectors with sample data
    for (int i=0; i<n; i++) {
        a[i]=i*i;
        b[i]=i;
    }

    // Split the vector into one chunk per GPU, the last GPU takes the remainder
    int chunk = n / gpus;
    vector<thread> workers;
    for (int dev=0; dev<gpus; dev++) {
        int start = dev * chunk;
        int len = (dev == gpus-1) ? n - start : chunk;
        workers.emplace_back(run_on_gpu, dev, a.data(), b.data(), c.data(), start, len);
    }

    // Wait for every GPU thread to finish
    for (auto &t : workers) {
        t.join();
    }

    // Display the result
    for (int i=0; i<n; i++) {
        cout << c[i] << " ";
    }

    cout << "\n";

    return 0;
}
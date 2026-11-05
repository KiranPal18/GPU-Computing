#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

// Simple reduction kernel using global memory
__global__ void CudaSum(int *a, int n, int * ans) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;  // Calculate global thread index
    int active = blockDim.x;  // Number of values still being reduced
    while (active > 1) {
        int next = (active+1)/2;
        if (threadIdx.x < active/2 && i+next < n) {
            a[i] += a[i+next];
        }
        __syncthreads();
        active = next;
    }
    int start = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0 && start < n) {
        atomicAdd(ans, a[start]); // Each block contributes its own first element
    }
}

// Optimized reduction kernel using shared memory (tiling)
__global__ void Tile_CudaSum(int *a, int n, int *ans) {
    extern __shared__ int tile[]; // Dynamic shared memory allocation
    int idx = threadIdx.x;
    int localSum = 0;
    
    // Grid-stride loop to handle arrays larger than grid size
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        localSum += a[i];
    }
    tile[idx] = localSum; // Store partial sum in shared memory
    __syncthreads();

    // Perform reduction in shared memory
    int active = blockDim.x;
    while (active > 1) {
        int next = (active+1)/2;
        if (idx < active/2) {
            tile[idx] += tile[idx+next];
        }
        __syncthreads();
        active = next;
    }
    if (idx == 0) {
        atomicAdd(ans, tile[0]); // Add block's shared sum to global total
    }
}

// Optimized reduction kernel using shared memory and warp shuffles
__global__ void Warp_Tile_CudaSum(int *a, int n, int *ans) {
    extern __shared__ int tile[]; // Dynamic shared memory allocation
    int idx = threadIdx.x;
    int localSum = 0;
    
    // Grid-stride loop to handle arrays larger than grid size
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        localSum += a[i];
    }
    tile[idx] = localSum; // Store partial sum in shared memory
    __syncthreads();

    // Perform reduction in shared memory until only one warp (32 threads) remains
    int active = blockDim.x;
    while (active > 32) {
        int next = (active+1)/2;
        if (idx < active/2) {
            tile[idx] += tile[idx+next];
        }
        __syncthreads();
        active = next;
    }
    int val = 0;
    // Only participating lanes belong to the shuffle mask, including short blocks.
    unsigned mask = __ballot_sync(__activemask(), idx < active);
    if (idx < active) {
        val = tile[idx];
        for (int offset = 16; offset > 0; offset /= 2) {
            int other = __shfl_down_sync(mask, val, offset);
            if (idx+offset < active) val += other;
        }
    }
    if (idx == 0) {
        atomicAdd(ans, val); // Add block's final sum to global total
    }
}

int main() {
    int n = 17;
    int sum = 0;

    // Initialize host data
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        int val = i % 7;
        a[i] = (val == 0) ? 1 : val; 
    }

    // Calculate expected sum on CPU for verification
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }

    // Allocate GPU memory
    int *d_a, *ans;
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ans, sizeof(int)));
    
    CUDA_CHECK(cudaMemset(ans, 0, sizeof(int)));

    // Copy data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    // Define execution configuration
    dim3 block(16);
    dim3 grid((n+block.x-1) / block.x);
    
    // Launch kernel with dynamic shared memory size
    Warp_Tile_CudaSum<<<grid, block, 16*sizeof(int)>>>(d_a, n, ans);
    CUDA_CHECK(cudaGetLastError());

    int gans=0;
    CUDA_CHECK(cudaDeviceSynchronize());
    // Copy result back from Device to Host
    CUDA_CHECK(cudaMemcpy(&gans, ans, sizeof(int), cudaMemcpyDeviceToHost));

    cout << "GPU is: " << gans << "\n";
    cout << "expected is: " << sum << "\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(ans));
    return gans == sum ? 0 : 1;
}

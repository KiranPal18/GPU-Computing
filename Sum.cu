#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// Simple reduction kernel using global memory
__global__ void CudaSum(int *a, int n, int * ans) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;  // Calculate global thread index
    int stride = blockDim.x / 2;  // Initial stride for tree-based reduction
    while (stride > 0) {
        if (threadIdx.x < stride && i+stride<n) {
            a[i] += a[i+stride]; // Sum elements in pairs
        }
        __syncthreads(); // Sync threads before reducing stride
        stride /= 2;
    }
    if (threadIdx.x == 0) {
        atomicAdd(ans, a[0]); // Add block result to global total
    }
}

// Optimized reduction kernel using shared memory (tiling)
__global__ void Tile_CudaSum(int *a, int n, int *ans) {
    extern __shared__ int tile[]; // Dynamic shared memory allocation
    int idx = threadIdx.x;
    float localSum = 0.0f;
    
    // Grid-stride loop to handle arrays larger than grid size
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        localSum += a[i];
    }
    tile[idx] = localSum; // Store partial sum in shared memory
    __syncthreads();

    // Perform reduction in shared memory
    int stride = blockDim.x / 2;
    while (stride > 0) {
        if (idx < stride) {
            tile[idx] += tile[idx + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    if (idx == 0) {
        atomicAdd(ans, tile[0]); // Add block's shared sum to global total
    }
}

// Optimized reduction kernel using shared memory and warp shuffles
__global__ void Warp_Tile_CudaSum(int *a, int n, int *ans) {
    extern __shared__ int tile[]; // Dynamic shared memory allocation
    int idx = threadIdx.x;
    float localSum = 0.0f;
    
    // Grid-stride loop to handle arrays larger than grid size
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        localSum += a[i];
    }
    tile[idx] = localSum; // Store partial sum in shared memory
    __syncthreads();

    // Perform reduction in shared memory until only one warp (32 threads) remains
    int stride = blockDim.x / 2;
    while (stride >= 32) {
        if (idx < stride) {
            tile[idx] += tile[idx + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    int val = 0;
    if (idx < 32) {
        // Load remaining values from shared memory into registers
        val = tile[idx]; 
        
        // Use warp shuffle for final reduction (no __syncthreads needed within a warp)
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
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
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&ans, sizeof(int));
    
    // Copy data from Host to Device
    cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Define execution configuration
    dim3 block(16);
    dim3 grid((n+block.x-1) / block.x);
    
    // Launch kernel with dynamic shared memory size
    Warp_Tile_CudaSum<<<grid, block, 16*sizeof(int)>>>(d_a, n, ans);

    int gans=0;
    cudaDeviceSynchronize();
    // Copy result back from Device to Host
    cudaMemcpy(&gans, ans, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "GPU is: " << gans << "\n";
    cout << "expected is: " << sum << "\n";

    cudaFree(d_a);
    return 0;
}
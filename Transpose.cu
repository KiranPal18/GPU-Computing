#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

/**
 * CUDA Kernel for Matrix Transposition using Shared Memory.
 * This implementation uses a tiling approach to ensure global memory coalescing
 * for both read and write operations and employs padding to avoid shared memory bank conflicts.
 */
__global__ void Transpose(float *a, float *b, const int n, const int m) {
    // Dynamic shared memory allocation.
    extern __shared__ float tile[];

    // Padding the tile width by 1 to avoid shared memory bank conflicts.
    // When accessing columns of the tile, this ensures that elements in the same column
    // map to different banks.
    int tile_width = blockDim.x + 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 1. Read from global memory into shared memory (Coalesced Read)
    // Threads read a contiguous block of memory from 'a' and store it in the shared tile.
    if (x < m && y < n) {
        tile[threadIdx.y * tile_width + threadIdx.x] = a[y * m + x];
    }
    
    // Wait for all threads in the block to finish loading the tile into shared memory
    __syncthreads();

    // Square blocks: exchange local coordinates and the output tile position
    y = blockIdx.y * blockDim.y + threadIdx.x;
    x = blockIdx.x * blockDim.x + threadIdx.y;
    
    // 2. Write from shared memory back to global memory (Coalesced Write)
    // By reading from the tile in a transposed manner (swapping threadIdx.x/y),
    // we can write the output to 'b' in a coalesced fashion.
    if (y < n && x < m) {
        b[x * n + y] = tile[threadIdx.x * tile_width + threadIdx.y];
    }
}

int main () {
    int n = 7, m = 5;
    vector<float> a(n*m);
    for (int i=0; i<n*m; i++) {
        a[i] = i+1;
    }
    
    cout << "A:\n";
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            cout << a[i*m+j] << " ";
        }
        cout << "\n";
    }


    vector<float> expected(n*m);
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) expected[j*n+i]=a[i*m+j];
    }

    float *d_a, *d_b;
    CUDA_CHECK(cudaMallocManaged(&d_a, n*m*sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_b, n*m*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n*m*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((m + 31) / 32, (n + 31) / 32);

    Transpose<<<grid, block, 32 * 33 * sizeof(float)>>>(d_a, d_b, n, m);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(a.data(), d_b, n*m*sizeof(float), cudaMemcpyDeviceToHost));

    cout << "A Transpose: \n";
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            cout << a[i*n+j] << " ";
        }
        cout << "\n";
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return a == expected ? 0 : 1;
}

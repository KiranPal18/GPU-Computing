#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

// Reduce absolute values inside each block; no shared global-variable race.
__global__ void L_InfNorm(const double *a, int n, double *b) {
    extern __shared__ double tile[];
    int t = threadIdx.x;
    double value = 0.0;
    for (int i=blockIdx.x*blockDim.x+t; i<n; i+=blockDim.x*gridDim.x) {
        value = fmax(value, fabs(a[i]));
    }
    tile[t] = value;
    __syncthreads();

    int active = blockDim.x;
    while (active > 1) {
        int next = (active+1)/2;
        if (t < active/2) tile[t] = fmax(tile[t], tile[t+next]);
        __syncthreads();
        active = next;
    }
    if (t == 0) b[blockIdx.x] = tile[0];
}

int main() {
    int n=100003;
    vector<double> a(n);
    double expected=0.0;
    for (int i=0; i<n; i++) {
        a[i] = (i%357-178)/10.0;
        expected = max(expected, fabs(a[i]));
    }

    double *d_a, *d_partial, *d_b;
    int block, minimum_grid;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minimum_grid, &block, L_InfNorm, 0, 256));
    int grid = min((n+block-1)/block, minimum_grid);
    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_partial, grid*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n*sizeof(double), cudaMemcpyHostToDevice));

    L_InfNorm<<<grid, block, block*sizeof(double)>>>(d_a, n, d_partial);
    CUDA_CHECK(cudaGetLastError());
    // A second launch supplies the grid-wide synchronization that a block barrier cannot.
    L_InfNorm<<<1, block, block*sizeof(double)>>>(d_partial, grid, d_b);
    CUDA_CHECK(cudaGetLastError());

    double answer;
    CUDA_CHECK(cudaMemcpy(&answer, d_b, sizeof(double), cudaMemcpyDeviceToHost));
    cout << "GPU norm: " << answer << "\nExpected: " << expected << "\n";
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_b));
    return fabs(answer-expected) < 1e-12 ? 0 : 1;
}

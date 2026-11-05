#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

#define tile_dim 16

// General matrix multiplication: C = alpha*A*B + beta*C.
__global__ void gemm(const float *a, const float *b, float *c,
                     int n, int k, int m, float alpha, float beta) {
    __shared__ float tile_a[tile_dim][tile_dim];
    __shared__ float tile_b[tile_dim][tile_dim];
    int tx=threadIdx.x, ty=threadIdx.y;
    int row=blockIdx.y*tile_dim+ty, col=blockIdx.x*tile_dim+tx;
    float sum=0.0f;

    for (int phase=0; phase<(k+tile_dim-1)/tile_dim; phase++) {
        int x=phase*tile_dim;
        tile_a[ty][tx] = row<n && x+tx<k ? a[row*k+x+tx] : 0.0f;
        tile_b[ty][tx] = x+ty<k && col<m ? b[(x+ty)*m+col] : 0.0f;
        __syncthreads();
        for (int i=0; i<tile_dim; i++) sum += tile_a[ty][i]*tile_b[i][tx];
        __syncthreads();
    }
    if (row<n && col<m) {
        // beta=0 must also work when C has not been initialized.
        c[row*m+col] = alpha*sum + (beta == 0.0f ? 0.0f : beta*c[row*m+col]);
    }
}

int main() {
    int n=19, k=23, m=17;
    float alpha=0.5f, beta=0.25f;
    vector<float> a(n*k), b(k*m), c(n*m, 2.0f), expected=c;
    for (int i=0; i<n*k; i++) a[i]=(i%11-5)/5.0f;
    for (int i=0; i<k*m; i++) b[i]=(i%7-3)/3.0f;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            float sum=0.0f;
            for (int x=0; x<k; x++) sum += a[i*k+x]*b[x*m+j];
            expected[i*m+j] = alpha*sum+beta*c[i*m+j];
        }
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, a.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, c.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), c.size()*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(tile_dim, tile_dim);
    dim3 grid((m+tile_dim-1)/tile_dim, (n+tile_dim-1)/tile_dim);
    gemm<<<grid, block>>>(d_a, d_b, d_c, n, k, m, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size()*sizeof(float), cudaMemcpyDeviceToHost));

    float error=0.0f;
    bool correct=true;
    for (int i=0; i<n*m; i++) {
        if (!isfinite(c[i])) correct=false;
        error=max(error, fabs(c[i]-expected[i]));
    }
    cout << "GEMM maximum error: " << error << "\n";
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return correct && error < 1e-4f ? 0 : 1;
}

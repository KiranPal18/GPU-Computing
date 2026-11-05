#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <cuda_pipeline.h>
#include "attention_example.h"
#include "flash_attention_common.h"

using namespace std;

// Educational FA3-inspired pipeline, NOT the complete Hopper FlashAttention-3.
// Two K/V buffers overlap asynchronous copies with current-tile computation.
// Real FA3 also uses Hopper TMA, WGMMA, warp specialization, and low precision.
// Compile with -arch=sm_80 or newer for cp.async; older targets copy synchronously.
__device__ void prefetch_kv(const float *k, const float *v, float *tile_k,
                            float *tile_v, int start, int n, int d) {
    for (int i=threadIdx.x; i<fa_key_tile*fa_max_dim; i+=blockDim.x) {
        int key=start+i/fa_max_dim, feature=i%fa_max_dim;
        bool valid=key<n && feature<d;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // An invalid element copies zero bytes and zero-fills the shared value.
        int source=valid ? key*d+feature : 0;
        __pipeline_memcpy_async(tile_k+i, k+source, sizeof(float), valid ? 0 : sizeof(float));
        __pipeline_memcpy_async(tile_v+i, v+source, sizeof(float), valid ? 0 : sizeof(float));
#else
        tile_k[i]=valid ? k[key*d+feature] : 0.0f;
        tile_v[i]=valid ? v[key*d+feature] : 0.0f;
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __pipeline_commit();
#endif
}

__global__ void flash_attention_3(const float *q, const float *k, const float *v,
                                 float *output, int n, int d, bool causal) {
    __shared__ float tile_k[2][fa_key_tile*fa_max_dim];
    __shared__ float tile_v[2][fa_key_tile*fa_max_dim];
    int t=threadIdx.x, lane=t%32;
    int row=blockIdx.x*fa_query_rows+t/32;
    float query[2]={0.0f}, result[2]={0.0f};
    for (int x=0; x<2; x++) {
        if (row<n && lane+32*x<d) query[x]=q[row*d+lane+32*x];
    }
    float maximum=-CUDART_INF_F, denominator=0.0f;
    prefetch_kv(k, v, tile_k[0], tile_v[0], 0, n, d);

    for (int start=0, phase=0; start<n; start+=fa_key_tile, phase++) {
        int current=phase%2;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __pipeline_wait_prior(0);
#endif
        // Wait for every thread's copies before any warp reads the current tile.
        __syncthreads();
        if (start+fa_key_tile<n) {
            prefetch_kv(k, v, tile_k[1-current], tile_v[1-current], start+fa_key_tile, n, d);
        }
        flash_warp_update(tile_k[current], tile_v[current], query, result, maximum,
                          denominator, row, start, n, d, causal);
        // All readers finish before a buffer can be reused two iterations later.
        __syncthreads();
    }
    if (row<n) {
        for (int x=0; x<2; x++) {
            if (lane+32*x<d) output[row*d+lane+32*x]=result[x]/denominator;
        }
    }
}

void launch_flash_3(const float *q, const float *k, const float *v, float *output,
                    int n, int d, bool causal) {
    if (n<=0 || d<=0 || d>fa_max_dim) {
        cerr << "FlashAttention example requires n>0 and 1<=d<=64\n";
        exit(EXIT_FAILURE);
    }
    flash_attention_3<<<(n+fa_query_rows-1)/fa_query_rows,fa_query_rows*32>>>(
        q, k, v, output, n, d, causal);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    return attention_example(launch_flash_3);
}

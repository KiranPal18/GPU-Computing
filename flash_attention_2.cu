#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "attention_example.h"
#include "flash_attention_common.h"

using namespace std;

// Educational FlashAttention-2: split Q across warps and blocks, reuse K/V,
// keep each warp's output in registers, and delay division until the end.
// This SIMT implementation demonstrates the algorithm without Tensor Cores.
__global__ void flash_attention_2(const float *q, const float *k, const float *v,
                                 float *output, int n, int d, bool causal) {
    __shared__ float tile_k[fa_key_tile*fa_max_dim];
    __shared__ float tile_v[fa_key_tile*fa_max_dim];
    int t=threadIdx.x, lane=t%32;
    int row=blockIdx.x*fa_query_rows+t/32;
    float query[2]={0.0f}, result[2]={0.0f};
    for (int x=0; x<2; x++) {
        if (row<n && lane+32*x<d) query[x]=q[row*d+lane+32*x];
    }
    float maximum=-CUDART_INF_F, denominator=0.0f;

    for (int start=0; start<n; start+=fa_key_tile) {
        for (int i=t; i<fa_key_tile*fa_max_dim; i+=blockDim.x) {
            int key=start+i/fa_max_dim, feature=i%fa_max_dim;
            bool valid=key<n && feature<d;
            tile_k[i]=valid ? k[key*d+feature] : 0.0f;
            tile_v[i]=valid ? v[key*d+feature] : 0.0f;
        }
        __syncthreads();
        flash_warp_update(tile_k, tile_v, query, result, maximum, denominator,
                          row, start, n, d, causal);
        __syncthreads();
    }
    if (row<n) {
        for (int x=0; x<2; x++) {
            if (lane+32*x<d) output[row*d+lane+32*x]=result[x]/denominator;
        }
    }
}

void launch_flash_2(const float *q, const float *k, const float *v, float *output,
                    int n, int d, bool causal) {
    if (n<=0 || d<=0 || d>fa_max_dim) {
        cerr << "FlashAttention example requires n>0 and 1<=d<=64\n";
        exit(EXIT_FAILURE);
    }
    flash_attention_2<<<(n+fa_query_rows-1)/fa_query_rows,fa_query_rows*32>>>(
        q, k, v, output, n, d, causal);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    return attention_example(launch_flash_2);
}

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "attention_example.h"

using namespace std;

#define key_tile 16
#define max_dim 64

// Educational FlashAttention-1: tiled K/V and online softmax, no n*n matrix.
// One thread owns a query row; normalized output is updated after every tile.
// Launch with key_tile threads; this example supports 1 <= d <= max_dim.
__global__ void flash_attention_1(const float *q, const float *k, const float *v,
                                 float *output, int n, int d, bool causal) {
    __shared__ float tile_k[key_tile][max_dim];
    __shared__ float tile_v[key_tile][max_dim];
    int t=threadIdx.x, row=blockIdx.x*key_tile+t;
    float result[max_dim]={0.0f};
    float maximum=-CUDART_INF_F, denominator=0.0f;

    for (int start=0; start<n; start+=key_tile) {
        for (int x=0; x<d; x++) {
            tile_k[t][x] = start+t<n ? k[(start+t)*d+x] : 0.0f;
            tile_v[t][x] = start+t<n ? v[(start+t)*d+x] : 0.0f;
        }
        __syncthreads();
        if (row<n) {
            float scores[key_tile], tile_max=-CUDART_INF_F;
            for (int j=0; j<key_tile; j++) {
                float sum=0.0f;
                for (int x=0; x<d; x++) sum += q[row*d+x]*tile_k[j][x];
                scores[j] = start+j<n && (!causal || start+j<=row)
                            ? sum/sqrtf(float(d)) : -CUDART_INF_F;
                tile_max=fmaxf(tile_max, scores[j]);
            }
            // A future causal tile can be completely masked out.
            if (tile_max != -CUDART_INF_F) {
                float next_max=fmaxf(maximum, tile_max);
                float alpha=expf(maximum-next_max), tile_sum=0.0f;
                for (int j=0; j<key_tile; j++) {
                    scores[j]=expf(scores[j]-next_max);
                    tile_sum+=scores[j];
                }
                float next_sum=alpha*denominator+tile_sum;
                for (int x=0; x<d; x++) {
                    float sum=0.0f;
                    for (int j=0; j<key_tile; j++) sum += scores[j]*tile_v[j][x];
                    result[x]=(alpha*denominator*result[x]+sum)/next_sum;
                }
                maximum=next_max;
                denominator=next_sum;
            }
        }
        __syncthreads();
    }
    if (row<n) {
        for (int x=0; x<d; x++) output[row*d+x]=result[x];
    }
}

void launch_flash_1(const float *q, const float *k, const float *v, float *output,
                    int n, int d, bool causal) {
    if (n<=0 || d<=0 || d>max_dim) {
        cerr << "FlashAttention example requires n>0 and 1<=d<=64\n";
        exit(EXIT_FAILURE);
    }
    flash_attention_1<<<(n+key_tile-1)/key_tile,key_tile>>>(q, k, v, output, n, d, causal);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    return attention_example(launch_flash_1);
}
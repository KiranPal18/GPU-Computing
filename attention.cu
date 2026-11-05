#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "attention_example.h"

using namespace std;

// Original scaled dot-product attention: softmax(Q*K^T/sqrt(d))*V.
// This baseline explicitly stores the n*n attention matrix.
__global__ void attention_scores(const float *q, const float *k, float *scores,
                                 int n, int d, bool causal) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n && j<n) {
        float sum=0.0f;
        for (int x=0; x<d; x++) sum += q[i*d+x]*k[j*d+x];
        scores[i*n+j] = causal && j>i ? -CUDART_INF_F : sum/sqrtf(float(d));
    }
}

__global__ void attention_softmax(float *scores, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) {
        float maximum=-CUDART_INF_F, sum=0.0f;
        for (int j=0; j<n; j++) maximum=fmaxf(maximum, scores[i*n+j]);
        for (int j=0; j<n; j++) sum += (scores[i*n+j]=expf(scores[i*n+j]-maximum));
        for (int j=0; j<n; j++) scores[i*n+j] /= sum;
    }
}

__global__ void attention_values(const float *scores, const float *v, float *output, int n, int d) {
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    if (i<n && x<d) {
        float sum=0.0f;
        for (int j=0; j<n; j++) sum += scores[i*n+j]*v[j*d+x];
        output[i*d+x]=sum;
    }
}

void launch_attention(const float *q, const float *k, const float *v, float *output,
                      int n, int d, bool causal) {
    float *scores;
    CUDA_CHECK(cudaMalloc(&scores, size_t(n)*n*sizeof(float)));
    dim3 block(16,16);
    attention_scores<<<dim3((n+15)/16,(n+15)/16), block>>>(q, k, scores, n, d, causal);
    CUDA_CHECK(cudaGetLastError());
    attention_softmax<<<(n+127)/128,128>>>(scores, n);
    CUDA_CHECK(cudaGetLastError());
    attention_values<<<dim3((d+15)/16,(n+15)/16), block>>>(scores, v, output, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(scores));
}

int main() {
    return attention_example(launch_attention);
}
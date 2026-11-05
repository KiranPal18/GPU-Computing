
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <iostream>
#include <vector>
#include <cmath>

constexpr int CIN = 3;
constexpr int H = 5;
constexpr int W = 5;
constexpr int KH = 3;
constexpr int KW = 3;
constexpr int OH = H - KH + 1;
constexpr int OW = W - KW + 1;
constexpr int N = CIN * KH * KW;
constexpr int BLOCK = 32;

__global__ void conv_thread(
    const float* input,
    const float* filter,
    float* output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= OH * OW) return;

    int r = idx / OW;
    int c = idx % OW;
    float sum = 0.0f;

    #pragma unroll
    for (int ch = 0; ch < CIN; ch++) {
        #pragma unroll
        for (int i = 0; i < KH; i++) {
            #pragma unroll
            for (int j = 0; j < KW; j++) {
                int in_idx = ch * H * W + (r + i) * W + c + j;
                int f_idx = ch * KH * KW + i * KW + j;
                sum += input[in_idx] * filter[f_idx];
            }
        }
    }

    output[idx] = sum;
}

__global__ void conv_block(
    const float* input,
    const float* filter,
    float* output
) {
    int idx = blockIdx.x;
    if (idx >= OH * OW) return;

    int r = idx / OW;
    int c = idx % OW;
    int t = threadIdx.x;

    __shared__ float partial[BLOCK];

    float value = 0.0f;

    // A strided loop also handles filters with more elements than threads.
    for (int x=t; x<N; x+=BLOCK) {
        int ch = x / (KH * KW);
        int rem = x % (KH * KW);
        int i = rem / KW;
        int j = rem % KW;

        int in_idx = ch * H * W + (r + i) * W + c + j;
        int f_idx = ch * KH * KW + i * KW + j;

        value += input[in_idx] * filter[f_idx];
    }

    partial[t] = value;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            partial[t] += partial[t + stride];

        __syncthreads();
    }

    if (t == 0)
        output[idx] = partial[0];
}

int main() {
    std::vector<float> input(CIN * H * W);
    std::vector<float> filter(N, 1.0f);
    std::vector<float> out1(OH * OW);
    std::vector<float> out2(OH * OW);

    for (int ch = 0; ch < CIN; ch++)
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                input[ch * H * W + r * W + c] =
                    float(ch * H * W + r * W + c + 1);

    float *d_input, *d_filter, *d_out1, *d_out2;

    CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filter, filter.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out1, out1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, out2.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filter, filter.data(), filter.size() * sizeof(float),
               cudaMemcpyHostToDevice));

    conv_thread<<<(OH * OW + 127) / 128, 128>>>(
        d_input, d_filter, d_out1
    );
    CUDA_CHECK(cudaGetLastError());

    conv_block<<<OH * OW, BLOCK>>>(
        d_input, d_filter, d_out2
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out1.data(), d_out1, out1.size() * sizeof(float),
               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out2.data(), d_out2, out2.size() * sizeof(float),
               cudaMemcpyDeviceToHost));

    std::vector<float> expected(OH*OW, 0.0f);
    for (int r=0; r<OH; r++) {
        for (int c=0; c<OW; c++) {
            for (int ch=0; ch<CIN; ch++) {
                for (int i=0; i<KH; i++) {
                    for (int j=0; j<KW; j++) {
                        expected[r*OW+c] += input[ch*H*W+(r+i)*W+c+j]*filter[ch*KH*KW+i*KW+j];
                    }
                }
            }
        }
    }
    bool correct = true;

    for (int r = 0; r < OH; r++) {
        for (int c = 0; c < OW; c++) {
            int idx = r * OW + c;
            std::cout << out1[idx] << " ";
            if (!std::isfinite(out1[idx]) || !std::isfinite(out2[idx]) ||
                std::fabs(out1[idx]-expected[idx]) > 1e-4f ||
                std::fabs(out2[idx]-expected[idx]) > 1e-4f)
                correct = false;
        }
        std::cout << "\n";
    }

    std::cout << "Both kernels match: "
              << (correct ? "YES" : "NO") << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaFree(d_out2));

    return correct ? 0 : 1;
}

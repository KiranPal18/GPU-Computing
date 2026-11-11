#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

__global__ void conv2d(const float* input, float* output, const float* weights, const float* bias, 
                       int in_channels, int out_channels, int in_size, int kernel_size) {
    int out_size = in_size - kernel_size + 1;
    int c_out = blockIdx.x;
    int x = blockIdx.y * blockDim.x + threadIdx.x;
    int y = blockIdx.z * blockDim.y + threadIdx.y;

    if (c_out < out_channels && x < out_size && y < out_size) {
        float sum = bias[c_out];
        for (int c_in = 0; c_in < in_channels; ++c_in) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    int in_x = x + kx;
                    int in_y = y + ky;
                    sum += input[(c_in * in_size + in_y) * in_size + in_x] * weights[(((c_out * in_channels) + c_in) * kernel_size + ky) * kernel_size + kx];
                }
            }
        }
        output[(c_out * out_size + y) * out_size + x] = sum;
    }
}

__global__ void avg_pool(const float* input, float* output, const float* weights, const float* bias, 
                         int channels, int in_size) {
    int out_size = in_size / 2;
    int c = blockIdx.x;
    int x = blockIdx.y * blockDim.x + threadIdx.x;
    int y = blockIdx.z * blockDim.y + threadIdx.y;

    if (c < channels && x < out_size && y < out_size) {
        float sum = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                sum += input[(c * in_size + (y * 2 + i)) * in_size + (x * 2 + j)];
            }
        }
        output[(c * out_size + y) * out_size + x] = 0.25f * sum * weights[c] + bias[c];
    }
}

__global__ void fully_connected(const float* input, float* output, const float* weights, const float* bias, 
                                int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_features) {
        float sum = bias[i];
        for (int j = 0; j < in_features; ++j) {
            sum += input[j] * weights[i * in_features + j];
        }
        output[i] = sum;
    }
}

__global__ void tanh_activation(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = tanhf(data[i]);
    }
}

// Deterministic demo data; replace these values with trained weights for inference.
void copy_data(float *device, int n, float scale, bool constant=false) {
    vector<float> data(n);
    for (int i=0; i<n; i++) data[i]=constant ? scale : scale*(i%23-11)/11.0f;
    CUDA_CHECK(cudaMemcpy(device, data.data(), n*sizeof(float), cudaMemcpyHostToDevice));
}

int main() {
    float *d_input, *d_c1_out, *d_s2_out, *d_c3_out, *d_s4_out, *d_c5_out, *d_f6_out, *d_output;
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_w4, *d_b4, *d_w5, *d_b5, *d_w6, *d_b6, *d_w7, *d_b7;

    CUDA_CHECK(cudaMalloc(&d_input, 1 * 32 * 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1_out, 6 * 28 * 28 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s2_out, 6 * 14 * 14 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c3_out, 16 * 10 * 10 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s4_out, 16 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c5_out, 120 * 1 * 1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f6_out, 84 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, 10 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w1, 6 * 1 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, 6 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w3, 16 * 6 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w4, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4, 16 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w5, 120 * 16 * 5 * 5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5, 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w6, 84 * 120 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b6, 84 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w7, 10 * 84 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b7, 10 * sizeof(float)));

    // cudaMalloc does not initialize memory. Every input, weight, and bias must be set.
    copy_data(d_input, 32*32, 1.0f);
    copy_data(d_w1, 6*5*5, 0.1f);
    copy_data(d_b1, 6, 0.01f);
    copy_data(d_w2, 6, 1.0f, true);
    copy_data(d_b2, 6, 0.0f, true);
    copy_data(d_w3, 16*6*5*5, 0.03f);
    copy_data(d_b3, 16, 0.01f);
    copy_data(d_w4, 16, 1.0f, true);
    copy_data(d_b4, 16, 0.0f, true);
    copy_data(d_w5, 120*16*5*5, 0.02f);
    copy_data(d_b5, 120, 0.01f);
    copy_data(d_w6, 84*120, 0.03f);
    copy_data(d_b6, 84, 0.01f);
    copy_data(d_w7, 10*84, 0.03f);
    copy_data(d_b7, 10, 0.01f);

    dim3 threads(16, 16);
    
    dim3 blocks_c1(6, (28+15)/16, (28+15)/16);
    conv2d<<<blocks_c1, threads>>>(d_input, d_c1_out, d_w1, d_b1, 1, 6, 32, 5);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(6*28*28+255)/256, 256>>>(d_c1_out, 6*28*28);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_s2(6, (14+15)/16, (14+15)/16);
    avg_pool<<<blocks_s2, threads>>>(d_c1_out, d_s2_out, d_w2, d_b2, 6, 28);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(6*14*14+255)/256, 256>>>(d_s2_out, 6*14*14);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_c3(16, (10+15)/16, (10+15)/16);
    conv2d<<<blocks_c3, threads>>>(d_s2_out, d_c3_out, d_w3, d_b3, 6, 16, 14, 5);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(16*10*10+255)/256, 256>>>(d_c3_out, 16*10*10);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_s4(16, (5+15)/16, (5+15)/16);
    avg_pool<<<blocks_s4, threads>>>(d_c3_out, d_s4_out, d_w4, d_b4, 16, 10);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(16*5*5+255)/256, 256>>>(d_s4_out, 16*5*5);
    CUDA_CHECK(cudaGetLastError());

    dim3 blocks_c5(120, 1, 1);
    conv2d<<<blocks_c5, threads>>>(d_s4_out, d_c5_out, d_w5, d_b5, 16, 120, 5, 5);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(120+255)/256, 256>>>(d_c5_out, 120);
    CUDA_CHECK(cudaGetLastError());

    fully_connected<<<(84+255)/256, 256>>>(d_c5_out, d_f6_out, d_w6, d_b6, 120, 84);
    CUDA_CHECK(cudaGetLastError());
    tanh_activation<<<(84+255)/256, 256>>>(d_f6_out, 84);
    CUDA_CHECK(cudaGetLastError());

    fully_connected<<<(10+255)/256, 256>>>(d_f6_out, d_output, d_w7, d_b7, 84, 10);
    CUDA_CHECK(cudaGetLastError());

    vector<float> output(10);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, 10*sizeof(float), cudaMemcpyDeviceToHost));
    bool correct=true;
    cout << "LeNet demo logits (untrained weights):\n";
    for (float value : output) {
        cout << value << " ";
        if (!isfinite(value)) correct=false;
    }
    cout << "\n";

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_c1_out));
    CUDA_CHECK(cudaFree(d_s2_out));
    CUDA_CHECK(cudaFree(d_c3_out));
    CUDA_CHECK(cudaFree(d_s4_out));
    CUDA_CHECK(cudaFree(d_c5_out));
    CUDA_CHECK(cudaFree(d_f6_out));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_w3));
    CUDA_CHECK(cudaFree(d_b3));
    CUDA_CHECK(cudaFree(d_w4));
    CUDA_CHECK(cudaFree(d_b4));
    CUDA_CHECK(cudaFree(d_w5));
    CUDA_CHECK(cudaFree(d_b5));
    CUDA_CHECK(cudaFree(d_w6));
    CUDA_CHECK(cudaFree(d_b6));
    CUDA_CHECK(cudaFree(d_w7));
    CUDA_CHECK(cudaFree(d_b7));

    return correct ? 0 : 1;
}

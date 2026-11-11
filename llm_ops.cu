#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

// The non-attention building blocks of a LLaMA style transformer layer:
// RMSNorm -> (attention, see attention.cu) -> RoPE on Q/K -> SwiGLU feed forward.
// Every kernel is checked against a plain CPU version.

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight. One block normalizes one token.
__global__ void rmsnorm(const float *x, const float *weight, float *y, int dim, float eps) {
    __shared__ float warp_sums[32];
    const float *row = x + blockIdx.x * dim;
    float *out = y + blockIdx.x * dim;

    // Each thread sums the squares of a strided part of the row
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += row[i] * row[i];
    }

    // Reduce inside each warp with shuffles, then reduce the warp results
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) warp_sums[0] = rsqrtf(sum / dim + eps);
    }
    __syncthreads();

    float scale = warp_sums[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out[i] = row[i] * scale * weight[i];
    }
}

// RoPE: rotate every pair (x[2i], x[2i+1]) of a token by angle pos * base^(-2i/dim).
// One thread rotates one pair, in place.
__global__ void rope(float *x, int tokens, int dim, float base) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int half = dim / 2;
    if (idx < tokens * half) {
        int pos = idx / half, i = idx % half;
        float angle = pos * powf(base, -2.0f * i / dim);
        float c = cosf(angle), s = sinf(angle);
        float *p = x + pos * dim + 2 * i;
        float a = p[0], b = p[1];
        p[0] = a * c - b * s;
        p[1] = a * s + b * c;
    }
}

// SwiGLU feed forward activation: out = silu(gate) * up, silu(g) = g / (1 + e^-g)
__global__ void swiglu(const float *gate, const float *up, float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float g = gate[i];
        out[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// Largest difference between the GPU result and the CPU reference
float max_diff(const vector<float> &a, const vector<float> &b) {
    float m = 0.0f;
    for (size_t i=0; i<a.size(); i++) m = max(m, fabs(a[i]-b[i]));
    return m;
}

int main() {
    int tokens=16, dim=512, n=tokens*dim;
    float eps=1e-5f, base=10000.0f;

    // Deterministic demo data
    vector<float> x(n), w(dim), gate(n), up(n);
    for (int i=0; i<n; i++) {
        x[i] = (i % 23 - 11) / 11.0f;
        gate[i] = (i % 17 - 8) / 4.0f;
        up[i] = (i % 13 - 6) / 6.0f;
    }
    for (int i=0; i<dim; i++) w[i] = 1.0f + (i % 7) / 10.0f;

    // CPU references
    vector<float> ref_norm(n), ref_rope(x), ref_swiglu(n);
    for (int t=0; t<tokens; t++) {
        float sum = 0.0f;
        for (int i=0; i<dim; i++) sum += x[t*dim+i] * x[t*dim+i];
        float scale = 1.0f / sqrt(sum / dim + eps);
        for (int i=0; i<dim; i++) ref_norm[t*dim+i] = x[t*dim+i] * scale * w[i];
        for (int i=0; i<dim/2; i++) {
            float angle = t * pow(base, -2.0f * i / dim);
            float a = ref_rope[t*dim+2*i], b = ref_rope[t*dim+2*i+1];
            ref_rope[t*dim+2*i] = a * cos(angle) - b * sin(angle);
            ref_rope[t*dim+2*i+1] = a * sin(angle) + b * cos(angle);
        }
    }
    for (int i=0; i<n; i++) ref_swiglu[i] = gate[i] / (1.0f + exp(-gate[i])) * up[i];

    // Pointers for GPU device memory
    float *d_x, *d_w, *d_y, *d_gate, *d_up, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, dim*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gate, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_up, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), dim*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gate, gate.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, up.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    vector<float> result(n);
    int block = 256;

    // RMSNorm: one block per token, block size must be a multiple of 32
    rmsnorm<<<tokens, block>>>(d_x, d_w, d_y, dim, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.data(), d_y, n*sizeof(float), cudaMemcpyDeviceToHost));
    float e1 = max_diff(result, ref_norm);
    cout << "RMSNorm max error: " << e1 << "\n";

    // RoPE: one thread per pair, rotates d_x in place
    rope<<<(n/2 + block - 1) / block, block>>>(d_x, tokens, dim, base);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.data(), d_x, n*sizeof(float), cudaMemcpyDeviceToHost));
    float e2 = max_diff(result, ref_rope);
    cout << "RoPE max error: " << e2 << "\n";

    // SwiGLU: one thread per element
    swiglu<<<(n + block - 1) / block, block>>>(d_gate, d_up, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(result.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
    float e3 = max_diff(result, ref_swiglu);
    cout << "SwiGLU max error: " << e3 << "\n";

    // Free GPU memory to avoid leaks
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_gate));
    CUDA_CHECK(cudaFree(d_up));
    CUDA_CHECK(cudaFree(d_out));

    if (e1 > 1e-4f || e2 > 1e-4f || e3 > 1e-4f) {
        cerr << "Mismatch with CPU reference\n";
        return 1;
    }
    cout << "All kernels match the CPU reference\n";

    return 0;
}

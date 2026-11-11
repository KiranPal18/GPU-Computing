#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <thread>

using namespace std;

// Data parallel training by hand, the same idea as ddp.py:
// every GPU gets a slice of the data, computes gradients on it, the gradients are
// summed across GPUs (all-reduce) and every GPU applies the same update.
// Model: y = w*x + b, trained to learn y = 3x + 2 with MSE loss.

// Each thread adds the gradient of its sample to grad[0] (dw) and grad[1] (db)
__global__ void linear_grad(const float *x, const float *y, float w, float b, int n, int total, float *grad) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        float err = w * x[i] + b - y[i];
        // Divide by the full dataset size so summing over GPUs gives the global mean
        atomicAdd(&grad[0], 2.0f * err * x[i] / total);
        atomicAdd(&grad[1], 2.0f * err / total);
    }
}

// Work done by one CPU thread: gradient of its slice on GPU `dev`
void grad_on_gpu(int dev, float *d_x, float *d_y, float *d_grad, int len, int total, float w, float b, float *grad) {
    // Every CPU thread must select its own GPU, the current device is per thread
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMemset(d_grad, 0, 2*sizeof(float)));

    int block = 256;
    int grid = (len + block - 1) / block;
    linear_grad<<<grid, block>>>(d_x, d_y, w, b, len, total, d_grad);
    CUDA_CHECK(cudaGetLastError());

    // Copy this GPU's partial gradient back to the host
    CUDA_CHECK(cudaMemcpy(grad, d_grad, 2*sizeof(float), cudaMemcpyDeviceToHost));
}

int main() {
    int gpus=0;
    CUDA_CHECK(cudaGetDeviceCount(&gpus));
    cout << "GPUs: " << gpus << "\n";

    // Toy data: x in [-1, 1], y = 3x + 2
    int n=1024;
    vector<float> x(n), y(n);
    for (int i=0; i<n; i++) {
        x[i] = -1.0f + 2.0f * i / (n - 1);
        y[i] = 3.0f * x[i] + 2.0f;
    }

    // Split the data into one slice per GPU, the last GPU takes the remainder.
    // The slices are copied once and stay on their GPU for the whole training.
    int chunk = n / gpus;
    vector<float*> d_x(gpus), d_y(gpus), d_grad(gpus);
    vector<int> len(gpus);
    for (int dev=0; dev<gpus; dev++) {
        int start = dev * chunk;
        len[dev] = (dev == gpus-1) ? n - start : chunk;
        cout << "GPU " << dev << " gets " << len[dev] << " of " << n << " samples\n";

        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaMalloc(&d_x[dev], len[dev]*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y[dev], len[dev]*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad[dev], 2*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x[dev], x.data() + start, len[dev]*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y[dev], y.data() + start, len[dev]*sizeof(float), cudaMemcpyHostToDevice));
    }

    // Model parameters, sent to every GPU as kernel arguments (the "broadcast")
    float w = 0.0f, b = 0.0f, lr = 0.1f;
    vector<float> grads(2*gpus);

    for (int step=0; step<200; step++) {
        // Forward + backward: every GPU works on its own slice in parallel
        vector<thread> workers;
        for (int dev=0; dev<gpus; dev++) {
            workers.emplace_back(grad_on_gpu, dev, d_x[dev], d_y[dev], d_grad[dev], len[dev], n, w, b, &grads[2*dev]);
        }
        for (auto &t : workers) {
            t.join();
        }

        // All-reduce: sum the partial gradients from every GPU
        float dw = 0.0f, db = 0.0f;
        for (int dev=0; dev<gpus; dev++) {
            dw += grads[2*dev];
            db += grads[2*dev+1];
        }

        // Same update for everyone, so every GPU keeps the same model
        w -= lr * dw;
        b -= lr * db;

        if (step % 20 == 0) {
            cout << "step " << step << "  w " << w << "  b " << b << "\n";
        }
    }

    // Free GPU memory to avoid leaks
    for (int dev=0; dev<gpus; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(d_x[dev]));
        CUDA_CHECK(cudaFree(d_y[dev]));
        CUDA_CHECK(cudaFree(d_grad[dev]));
    }

    cout << "w=" << w << " b=" << b << " (target 3, 2)\n";
    if (fabs(w - 3.0f) > 1e-2f || fabs(b - 2.0f) > 1e-2f) {
        cerr << "Training did not converge\n";
        return 1;
    }

    return 0;
}

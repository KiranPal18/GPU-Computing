#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <thread>

using namespace std;

// CUDA_CHECK reports CUDA errors and stops the program if a CUDA call fails.

// Each GPU runs this same kernel on its own part of the input vectors.
__global__ void vecadd(const float *a, const float *b, float *c, int n) {
    // Find the element assigned to this CUDA thread within this GPU's input part.
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    // The final block may contain extra threads, so check the input boundary.
    if (i<n) c[i]=a[i]+b[i];
}

// This function runs in a CPU thread and manages the work for one GPU.
// start is the first element in the full vector; n is this worker's element count.
void run_gpu(int device, const float *a, const float *b, float *c, int start, int n) {
    // A worker with no assigned elements has nothing to copy or calculate.
    if (n==0) return;
    // Select the GPU for CUDA calls made by this CPU thread.
    CUDA_CHECK(cudaSetDevice(device));
    // A stream is a queue of GPU operations. Each worker creates its own queue.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // h_ buffers are on the CPU. Pin their memory so transfers can be asynchronous.
    float *h_a, *h_b, *h_c;
    size_t bytes=n*sizeof(float);
    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));
    // Copy only this GPU's assigned input elements into the pinned buffers.
    for (int i=0; i<n; i++) {
        h_a[i]=a[start+i];
        h_b[i]=b[start+i];
    }

    // d_ buffers are separate allocations in the selected GPU's memory.
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // One stream preserves copy -> kernel -> copy ordering on this GPU.
    // Other host threads submit their work to other GPUs independently.
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream));
    // Use 256 threads per block and round up to cover every assigned element.
    // The last two launch arguments mean no extra shared memory and use this stream.
    vecadd<<<(n+255)/256, 256, 0, stream>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    // Queue the result copy after the kernel in the same stream.
    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream));

    // An asynchronous call can return before the GPU finishes. Wait for this queue.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Store results in this worker's part of the full output vector.
    // Workers write different elements, so they do not overwrite each other.
    for (int i=0; i<n; i++) c[start+i]=h_c[i];

    // Release GPU buffers, pinned CPU buffers, and the worker's stream.
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    // Assume GPUs 0 and 1 exist. Change num_gpus to change the configured count.
    int num_gpus=2, n=1000003;
    // Keep the full vectors on the CPU. NaN marks output elements not yet filled.
    vector<float> a(n), b(n), c(n, NAN);
    // Initialize the inputs with small, repeatable sample values.
    for (int i=0; i<n; i++) {
        a[i]=(i%101-50)*0.25f;
        b[i]=(i%67-33)*0.5f;
    }

    // Store one CPU thread per GPU so their GPU work can run at the same time.
    vector<thread> gpus;
    for (int device=0; device<num_gpus; device++) {
        // Split [0,n) into adjacent parts, including any leftover elements.
        // 1LL uses a wider integer for the multiplication to avoid int overflow.
        int start=1LL*n*device/num_gpus;
        int end=1LL*n*(device+1)/num_gpus;
        // Start a CPU thread and pass the GPU ID, vector pointers, and assigned range.
        gpus.emplace_back(run_gpu, device, a.data(), b.data(), c.data(), start, end-start);
    }
    // Wait until every worker has copied its completed result back to c.
    for (thread &gpu : gpus) gpu.join();

    // Compare the combined GPU output with the addition expected on the CPU.
    bool correct=true;
    for (int i=0; i<n; i++) {
        if (!isfinite(c[i]) || fabs(c[i]-(a[i]+b[i]))>1e-6f) correct=false;
    }
    // Print the result and return a nonzero exit code if a value is incorrect.
    cout << "Vector addition across " << num_gpus << " GPUs: " << (correct ? "PASS" : "FAIL") << "\n";
    return 0;
}
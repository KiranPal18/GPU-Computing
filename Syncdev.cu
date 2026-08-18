#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to calculate the exponential of elements in array 'a' and store them in 'b'
__global__ void Helper(const int *a, int *b, const int n) {
    // Calculate the global index of the thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Bound check to ensure the thread does not access memory outside the array
    if (i < n) {
        b[i] = (int)expf((float)a[i]);
    }
    printf("thread: %d\n", i);
}

int main() {
    int n=100;
    vector<int> a(n), b(n);

    // Initialize array 'a'
    for (int i=0; i<n; i++) {
        a[i] = ((i+2) * (i+1) / 2);
    }

    int *d_a, *d_b;

    // Allocate memory on the GPU device
    cudaMalloc((void **) &d_a, n*sizeof(int));
    cudaMalloc((void **) &d_b, n*sizeof(int));

    // Copy data from host memory to device memory
    cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Define execution configuration (block and grid size)
    dim3 block(16);
    dim3 grid((n+block.x-1)/block.x);

    // Launch the CUDA kernel asynchronously
    Helper<<<grid, block>>>(d_a, d_b, n);

    cout << "Kernel Started...!\n";

    // Force the CPU to wait until all GPU tasks are completed.
    // Without this, the "Kernel Completed" message might print before the kernel finishes
    // because kernel launches are non-blocking.
    cudaDeviceSynchronize();
    
    cout << "Kernel Completed...!\n";

    // Copy the result back from device memory to host memory
    cudaMemcpy(b.data(), d_b, n*sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU resources
    cudaFree(d_a);
    cudaFree(d_b);

    // Print the resulting array
    for (int i=0; i<n ;i++) {
        cout << b[i] << (i!=n-1 ? " " : "\n");
    }

    return 0;
}
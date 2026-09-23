#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for vector addition
__global__ void vecadd(int *a, int *b, int *c, int n) {
    // Calculate the global index of the thread across the entire grid
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Boundary check to ensure the thread does not access memory out of bounds
    if (i < n) { 
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n=1000;
    vector<int> a(n), b(n), c(n);

    // Initialize host vectors with sample data
    for (int i=0; i<n; i++) {
        a[i]=i*i;
        b[i]=i;
    }

    // Pointers for GPU device memory
    int *d_a, *d_b, *d_c;

    // Allocate memory on the GPU device
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_c, n*sizeof(int));

    // Transfer data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Define execution configuration: block size and grid size
    int block = 256;
    int grid = (n + block - 1) / block;
    
    // Launch the kernel on the GPU
    vecadd<<<grid, block>>>(d_a, d_b, d_c, n);

    // Transfer the result back from Device (GPU) to Host (CPU)
    cudaMemcpy(c.data(), d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory to avoid leaks
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    // Display the result
    for (int i=0; i<n; i++) {
        cout << c[i] << " ";
    }

    cout << "\n";

    return 0;
}
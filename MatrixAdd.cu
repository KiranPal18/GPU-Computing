#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for matrix addition
__global__ void matadd(int *a, int *b, int *c, int h, int w) {
    // Calculate the global index for columns (x) and rows (y)
    int j = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // Boundary check to ensure threads stay within matrix dimensions
    if (i < h && j < w) { 
        // Matrices are stored as 1D arrays (row-major order)
        c[i*w+j] = a[i*w+j] + b[i*w+j];  
    }
}

int main() {
    int n=1000, m=100;

    // To ensure contiguous memory allocation for GPU transfer, 
    // we use a 1D vector to represent a 2D matrix (row-major order).
    vector<int> a(n*m), b(n*m), c(n*m);

    // Initialize matrices with sample data
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            a[i*m+j]=i*i+j;
            b[i*m+j]=i+j;
        }
    }

    // Pointers for GPU device memory
    int *d_a, *d_b, *d_c;

    // Allocate memory on the GPU device
    cudaMalloc(&d_a, n*m*sizeof(int));
    cudaMalloc(&d_b, n*m*sizeof(int));
    cudaMalloc(&d_c, n*m*sizeof(int));

    // Transfer data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, a.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);

    // Define 2D execution configuration: blocks and grids
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );
    
    // Launch the kernel on the GPU
    matadd<<<grid, block>>>(d_a, d_b, d_c, n, m);

    // Transfer the result back from Device (GPU) to Host (CPU)
    cudaMemcpy(c.data(), d_c, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory to avoid leaks
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    // Display the result
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            cout << c[i*m+j] << " ";
        }
        cout << "\n";
    }

    cout << "\n";

    return 0;
}
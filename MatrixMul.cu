#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel for matrix multiplication
__global__ void matrixmul(int *a, int *b, int *c, int h, int k, int w) {
    // Calculate global index for columns (x) and rows (y)
    int j = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // Boundary check to ensure threads are within matrix dimensions
    if (i < h && j < w) { 
        // Initialize the result cell
        c[i*w+j]=0;
        // Compute the dot product of row i from A and column j from B
        for (int x=0; x<k; x++) {
            c[i*w+j] += a[i*k+x] * b[x*w+j];
        }
    }
}

int main() {
    int n=2, k=5, m=3;

    // Using 1D arrays to represent 2D matrices for contiguous memory allocation,
    // which is required for efficient transfer to the GPU.
    vector<int> a(n*k), b(k*m), c(n*m);

    // Initialize matrices with sample data
    for (int i=0; i<n; i++) {
        for (int j=0; j<k; j++) {
            a[i*k+j]=i*i+j;
        }
    }
    for (int i=0; i<k; i++) {
        for (int j=0; j<m; j++) {
            b[i*m+j]=i*i+j;
        }
    }

    // Pointers for GPU device memory
    int *d_a, *d_b, *d_c;

    // Allocate memory on the GPU device
    cudaMalloc(&d_a, n*k*sizeof(int));
    cudaMalloc(&d_b, k*m*sizeof(int));
    cudaMalloc(&d_c, n*m*sizeof(int));

    // Transfer data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, a.data(), n*k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);

    // Define 2D execution configuration: blocks and grids
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );

    // Launch the kernel on the GPU
    matrixmul<<<grid, block>>>(d_a, d_b, d_c, n, k, m);

    // Transfer the result back from Device (GPU) to Host (CPU)
    cudaMemcpy(c.data(), d_c, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory to avoid leaks
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

    // Display results
    cout << "A\n";
    for (int i=0; i<n; i++) {
        for (int j=0; j<k; j++) {
            cout << a[i*k+j] << " ";
        }
        cout << "\n";
    }

    cout << "B\n";
    for (int i=0; i<k; i++) {
        for (int j=0; j<m; j++) {
            cout << b[i*m+j] << " ";
        }
        cout << "\n";
    }

    cout << "C\n";
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            cout << c[i*m+j] << " ";
        }
        cout << "\n";
    }

    cout << "\n";

    return 0;
}
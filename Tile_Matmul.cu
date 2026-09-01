#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

__global__ void Tiled_Matmul (const float * a, const float * b ,float *c, int n, int k, int m) {
    // Use shared memory to cache matrix tiles and reduce global memory access latency
    
    // Shared memory must have a fixed size at compile time or be declared as extern
    __shared__ float tile_a[16][16];
    
    // Extern shared memory is used for dynamic allocation. Note that all extern declarations 
    // in a block share the same memory region; thus, only one variable can be declared this way.
    extern __shared__ float tile_b[][16];

    // Calculate global and local thread indices
    int tx = threadIdx.y, ty = threadIdx.x;
    int bx = blockIdx.y, by = blockIdx.x;
    int row = bx*16+tx, col = by*16+ty;
    
    // Determine the number of tiles needed to cover the inner dimension k
    int phase = (k + 16 - 1)/ 16;
    float p=0.0;
    for (int i=0 ;i<phase; i++) {

        // Load tile from global memory to shared memory with boundary checking
        if (row < n && i * 16 + tx < k) {
            tile_a[tx][ty] = a[row*k + i*16+ ty];
        }
        else {
            tile_a[tx][ty]=0.0;
        }

        // Load tile from global memory to shared memory with boundary checking
        if (col < m && i * 16 + ty < k) {
            tile_b[tx][ty] = b[(i*16+tx)*m + col];
        }
        else {
            tile_b[tx][ty]=0.0;
        }

        // Ensure all threads have finished loading the tile before starting computation
        __syncthreads();
        for (int x=0; x<16; x++) {
            p += tile_a[tx][x] * tile_b[x][ty];
        }

        // Ensure all threads have finished computation before loading the next tile
        __syncthreads();
    }
    if (row < n && col < m){
        c[col + row * m] = p;
    }
}

int main() {
    int n=2, k=5, m=3;

    // Using 1D arrays to represent 2D matrices for contiguous memory allocation,
    // which is required for efficient transfer to the GPU.
    vector<float> a(n*k), b(k*m), c(n*m);

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
    float *d_a, *d_b, *d_c;

    // Allocate memory on the GPU device
    cudaMalloc(&d_a, n*k*sizeof(float));
    cudaMalloc(&d_b, k*m*sizeof(float));
    cudaMalloc(&d_c, n*m*sizeof(float));

    // Transfer data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, a.data(), n*k*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k*m*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*m*sizeof(float), cudaMemcpyHostToDevice);

    // Define 2D execution configuration: blocks and grids
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );

    // Launch the kernel on the GPU
    Tiled_Matmul<<<grid, block>>>(d_a, d_b, d_c, n, k, m);

    // Transfer the result back from Device (GPU) to Host (CPU)
    cudaMemcpy(c.data(), d_c, n*m*sizeof(float), cudaMemcpyDeviceToHost);

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
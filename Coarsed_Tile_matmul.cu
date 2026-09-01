#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define t_row 16 // Height of the tile for matrix C
#define t_col 32 // Width of the tile for matrix C
#define t_k   16 // Step size along the inner dimension (K)

//Coarsening allows a single thread to compute multiple output elements (across the columns).
#define coarse_factor 2 

//Instead of one thread calculating one element of C, one thread calculates 'coarse_factor' elements.
//This increases the work per thread, improves the ratio of arithmetic to memory access.

__global__ void Tiled_Matmul (const float * a, const float * b ,float *q, int n, int k, int m) {
    // Use shared memory to cache matrix tiles and reduce global memory access latency
    
    // Shared memory must have a fixed size at compile time or be declared as extern
    __shared__ float tile_a[t_row][t_k];
    
    // Extern shared memory is used for dynamic allocation. Note that all extern declarations 
    // in a block share the same memory region; thus, only one variable can be declared this way.
    // __shared__ float tile_b[t_k][t_col];
    extern __shared__ float tile_b[][t_col];

    // Calculate global and local thread indices
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by*t_row+ty;
    int col_start = bx*t_col*coarse_factor+tx;
    
    // Determine the number of tiles needed to cover the inner dimension k
    int phase = (k + t_k - 1)/ t_k;

    // Accumulators for the coarsened results. Each thread stores multiple partial sums.
    float p[coarse_factor] = {0.0f};

    for (int i=0 ;i<phase; i++) {

        if (ty < t_row && tx < t_k) {
            // Load tile from global memory to shared memory with boundary checking
            if (row < n && i * t_k + tx < k) {
                tile_a[ty][tx] = a[row*k + i*t_k+ tx];
            }
            else {
                tile_a[ty][tx]=0.0;
            }
        }

        /* 
           COARSENED LOOP:
           Instead of loading a single tile of B and computing, we repeat the process 
           'coarse_factor' times. We reuse the current tile_a for multiple different 
           tiles of B (shifting the column offset).
        */

        for (int c=0; c<coarse_factor; c++) {
            if (ty < t_k && tx < t_col) {
                // Load tile from global memory to shared memory with boundary checking
                int col = col_start + t_col*c;
                if (col < m && i * t_k + ty < k) {
                    tile_b[ty][tx] = b[(i*t_k+ty)*m + col];
                }
                else {
                    tile_b[ty][tx]=0.0;
                }
            }

            // Synchronize to ensure tile_b is fully loaded before starting the dot product.
            __syncthreads();

            if (ty < t_row && tx < t_col) {
                for (int x = 0; x < t_k; x++) {
                    p[c] += tile_a[ty][x] * tile_b[x][tx];
                }
            }

            // Ensure all threads have finished computation before loading the next tile
            // Synchronize to ensure all threads finished using tile_b before the next coarse step 
            // or the next phase loads new data into shared memory.
            __syncthreads();
        }
    }

    // Store the coarsened results back to global memory at the respective shifted column offsets.
    for (int c=0; c<coarse_factor; c++) {
        int col = col_start + t_col*c;
        if (row < n && col < m){
            q[col + row * m] = p[c];
        }
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
    dim3 block(t_col, t_row);
    dim3 grid(
        (m + (block.x*coarse_factor) - 1) / (block.x*coarse_factor),
        (n + block.y - 1) / block.y
    );

    //Size of the shared memory that will be used with extern quantifier

    size_t shared_mem = t_k * t_col * sizeof(float);

    // Launch the kernel on the GPU
    Tiled_Matmul<<<grid, block, shared_mem>>>(d_a, d_b, d_c, n, k, m);

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
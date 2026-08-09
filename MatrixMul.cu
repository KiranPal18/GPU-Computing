#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

//defining Kernel
__global__ void matrixmul(int *a, int *b, int *c, int h, int k, int w) {
    //Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;  //column
    int i = threadIdx.y + blockIdx.y * blockDim.y;  //row

    if (i < h && j < w) { //Make the extra threads ideal
        c[i*w+j]=0;
        for (int x=0; x<k; x++) {
            c[i*w+j] += a[i*k+x] * b[x*w+j];
        }
    }
}

int main() {
    int n=2, k=5, m=3;
    //cin >> n >> k >> m;


    //You might thing that doing Matrix Multiplication of is to take two matrix (2D Tensor) and then add the individual terms
    //But the main problem we will be facing if we use vector<vector<int>> will simply be that the data
    //of matrix is not stored in a contigous manner, thus making is not feasible to transfer the data to 
    //Device(GPU)
    //Therefore we use a 1D tensor to do the matrix additon but treat it as a row major linear representaion
    //of a 2D tensor and this same concept can be applied to any dimension tensor
    vector<int> a(n*k), b(k*m), c(n*m);

    //Could take any vector for simplicity I hardcoded them
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

    //We need pointer for the GPU's memory access
    int *d_a, *d_b, *d_c;

    //Now we need to allocate the memory in GPU for the vectors and transfer them from host(CPU) to device(GPU)
    
    //Allocating memory
    cudaMalloc(&d_a, n*k*sizeof(int));
    cudaMalloc(&d_b, k*m*sizeof(int));
    cudaMalloc(&d_c, n*m*sizeof(int));

    //Data transfer form host to device
    cudaMemcpy(d_a, a.data(), n*k*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), k*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);

    //Kernel launch

    //We need to define our custom kernel blocks and grid as we are doing operation in 2D tensor hence we divide the Tensor in each dimension of it
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );

    matrixmul<<<grid, block>>>(d_a, d_b, d_c, n, k, m);

    //Data transfer of the result from device to host
    cudaMemcpy(c.data(), d_c, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    //Display the result
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
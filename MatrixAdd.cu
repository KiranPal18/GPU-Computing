#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

//defining Kernel
__global__ void matadd(int *a, int *b, int *c, int h, int w) {
    //Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;  //column
    int i = threadIdx.y + blockIdx.y * blockDim.y;  //row

    if (i < h && j < w) { //Make the extra threads ideal
        c[i*w+j] = a[i*w+j] + b[i*w+j];  //Matrix is stored in linear way in memory 
    }
}

int main() {
    int n=1000, m=100;
    //cin >> n >> m;


    //You might thing that doing Matrix Addition is to take two matrix (2D Tensor) and then add the individual terms
    //But the main problem we will be facing if we use vector<vector<int>> will simply be that the data
    //of matrix is not stored in a contigous manner, thus making is not feasible to transfer the data to 
    //Device(GPU)
    //Therefore we use a 1D tensor to do the matrix additon but treat it as a row major linear representaion
    //of a 2D tensor and this same concept can be applied to any dimension tensor
    vector<int> a(n*m), b(n*m), c(n*m);

    //Could take any vector for simplicity I hardcoded them
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            a[i*m+j]=i*i+j;
            b[i*m+j]=i+j;
        }
    }

    //We need pointer for the GPU's memory access
    int *d_a, *d_b, *d_c;

    //Now we need to allocate the memory in GPU for the vectors and transfer them from host(CPU) to device(GPU)
    
    //Allocating memory
    cudaMalloc(&d_a, n*m*sizeof(int));
    cudaMalloc(&d_b, n*m*sizeof(int));
    cudaMalloc(&d_c, n*m*sizeof(int));

    //Data transfer form host to device
    cudaMemcpy(d_a, a.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*m*sizeof(int), cudaMemcpyHostToDevice);

    //Kernel launch
    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );
    matadd<<<grid, block>>>(d_a, d_b, d_c, n, m);

    //Data transfer of the result from device to host
    cudaMemcpy(c.data(), d_c, n*m*sizeof(int), cudaMemcpyDeviceToHost);

    //Display the result
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            cout << c[i*m+j] << " ";
        }
        cout << "\n";
    }

    cout << "\n";

    return 0;
}
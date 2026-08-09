#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

//defining Kernel
__global__ void vecadd(int *a, int *b, int *c, int n) {
    //Global index of the thread
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) { //Make the extra threads ideal
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n=1000;
    //cin >> n;
    vector<int> a(n), b(n), c(n);

    //Could take any vector for simplicity I hardcoded them
    for (int i=0; i<n; i++) {
        a[i]=i*i;
        b[i]=i;
    }

    //We need pointer for the GPU's memory access
    int *d_a, *d_b, *d_c;

    //Now we need to allocate the memory in GPU for the vectors and transfer them from host(CPU) to device(GPU)
    
    //Allocating memory
    cudaMalloc(&d_a, n*sizeof(int));
    cudaMalloc(&d_b, n*sizeof(int));
    cudaMalloc(&d_c, n*sizeof(int));

    //Data transfer form host to device
    cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    //Kernel launch
    int block = 256;
    int grid = (n + block - 1) / block;
    vecadd<<<grid, block>>>(d_a, d_b, d_c, n);

    //Data transfer of the result from device to host
    cudaMemcpy(c.data(), d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

    //Display the result
    for (int i=0; i<n; i++) {
        cout << c[i] << " ";
    }

    cout << "\n";

    return 0;
}
#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

// The main purpose of the __syncthreads() function is to synchronize all threads within a block,
// ensuring that no thread proceeds beyond this point until all threads in the same block have reached it.


//Doing 2 phase work on device
__global__ void Helper(const int *a, int *b,const int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    //Phase 1 of the work
    if (i < n) {
        b[i] = a[i] * (i+1);
    }


    // For the phase 2 work, we need the phase 1 of work to be completed first.
    // We call __syncthreads() outside of any conditional (if) block because all threads 
    // in the block must reach this point. If it were inside an if block, threads that 
    // do not satisfy the condition would never reach the barrier, causing a deadlock.
    __syncthreads();
    
    // Phase 2 of the work
    if (i > 0 && threadIdx.x % 2 != 0 && i < n) {
        b[i] = b[i] / b[i-1]; 
    }
}

int main() {
    int n=100;
    
    vector<int> a(n), b(n);

    for (int i=0; i<n ;i++) {
        a[i] = (int)exp((i + 10)/ 17); //Any random values for the array
    }

    int *d_a, *d_b;

    //Allocate memory on the device
    cudaMalloc((void **) &d_a, n*sizeof(int));
    cudaMalloc((void **) &d_b, n*sizeof(int));

    //Copy the data to device from the host
    cudaMemcpy(d_a, a.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    //Launch the kernel
    Helper<<<((n+32-1)/32), 32>>>(d_a, d_b, n);

    //Copy the data back to host
    cudaMemcpy(b.data(), d_b, n*sizeof(int), cudaMemcpyDeviceToHost);

    //Free up the memory allocated in the device
    cudaFree(d_a);
    cudaFree(d_b);

    //Display the results
    for (int i=0; i<n; i++) {
        cout << a[i] << " ";
    }
    cout << "\n";


    for (int i=0; i<n; i++) {
        cout << b[i] << " ";
    }
    cout << "\n";

    return 0;
}
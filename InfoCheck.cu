#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

int main() {
    // Variable to store the total number of GPUs available
    int deviceCount = 0;
    
    // Query the system to find how many CUDA-capable GPUs are installed
    cudaGetDeviceCount(&deviceCount);
    
    cout << "Number of CUDA devices: " << deviceCount << '\n' << '\n';

    // Loop through each detected GPU to extract its hardware specifications
    for (int i = 0; i < deviceCount; i++) {
        // Structure to hold detailed properties of a specific GPU
        cudaDeviceProp prop;
        
        // Fetch properties for the GPU at index 'i'
        cudaGetDeviceProperties(&prop, i);

        cout << "Device " << i << ":\n";

        // Name of the GPU
        cout << "  GPU: " << prop.name << '\n';
        
        // Number of SM; determines parallel processing power
        cout << "  SM count: " << prop.multiProcessorCount << '\n';
        
        // Total 32-bit registers available per SM; affects how many threads can run
        cout << "  Registers per SM: " << prop.regsPerMultiprocessor << '\n';
        
        // Maximum registers a single thread block can use
        cout << "  Registers per block: " << prop.regsPerBlock << '\n';
        
        // Maximum threads that can reside on one SM simultaneously
        cout << "  Max threads per SM: " << prop.maxThreadsPerMultiProcessor << '\n';
        
        // Maximum threads allowed in a single block (usually 1024)
        cout << "  Max threads per block: " << prop.maxThreadsPerBlock << '\n' << '\n';
    }

    return 0;
}
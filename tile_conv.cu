#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define filter_radius 1
#define in_tile_dim 4
#define out_tile_dim ((in_tile_dim) - 2*(filter_radius))

// declare constant memory for the filter
__constant__ float kfilter_d[2*filter_radius+1][2*filter_radius+1];

__global__ void conv2dtiledconstkernel(float *input, float *output, int width, int height, int outwidth, int outheight) {
    __shared__ float inputtile[in_tile_dim][in_tile_dim];
    
    // input tile coordinates
    int col = blockIdx.x * out_tile_dim + threadIdx.x;
    int row = blockIdx.y * out_tile_dim + threadIdx.y;
    
    if (row < height && col < width) {
        inputtile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    else {
        inputtile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // output tile coordinates
    int tilecol = threadIdx.x - filter_radius;
    int tilerow = threadIdx.y - filter_radius;

    // in a valid convolution, the output is smaller than the input
    row -= filter_radius;
    col -= filter_radius;

    if (tilecol >= 0 && tilecol < out_tile_dim && tilerow >= 0 && tilerow < out_tile_dim) {
        float sum = 0.0f;
        for (int frow = 0; frow < 2*filter_radius+1; frow++) {
            for (int fcol = 0; fcol < 2*filter_radius+1; fcol++) {
                sum += inputtile[tilerow + frow][tilecol + fcol] * kfilter_d[frow][fcol];
            }
        }
        // write to output only if we are within the valid output bounds
        if (row >= 0 && col >= 0 && row < outheight && col < outwidth) {
            output[row * outwidth + col] = sum;
        }
    }
}

int main() {
    // 1. define dimensions
    // for a filter radius of 1, the filter is 3x3.
    int width = 8;  // input width
    int height = 8; // input height
    
    int outwidth = width - 2 * filter_radius;
    int outheight = height - 2 * filter_radius;

    size_t inbytes = width * height * sizeof(float);
    size_t outbytes = outwidth * outheight * sizeof(float);

    // 2. allocate and initialize host memory
    vector<float> h_input(width * height, 1.0f); // initialize input with 1s
    for (int i=0; i<width*height; i++) {
        h_input[i]=i;
    }
    vector<float> h_output(outwidth * outheight, 0.0f);
    
    // create a 3x3 averaging filter
    float h_filter[2*filter_radius+1][2*filter_radius+1];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            h_filter[i][j] = exp(i*i+j*j)/2;
        }
    }

    // 3. copy filter to constant memory on the device
    cudaMemcpyToSymbol(kfilter_d, h_filter, sizeof(h_filter));

    // 4. allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, inbytes);
    cudaMalloc((void**)&d_output, outbytes);

    // 5. copy input data from host to device
    cudaMemcpy(d_input, h_input.data(), inbytes, cudaMemcpyHostToDevice);

    // 6. define grid and block dimensions
    // block size must match in_tile_dim (4x4)
    dim3 blocksize(in_tile_dim, in_tile_dim);
    
    // grid size depends on out_tile_dim (2x2)
    dim3 gridsize(
        (outwidth + out_tile_dim - 1) / out_tile_dim,
        (outheight + out_tile_dim - 1) / out_tile_dim
    );

    // 7. launch the kernel
    conv2dtiledconstkernel<<<gridsize, blocksize>>>(d_input, d_output, width, height, outwidth, outheight);
    
    // check for kernel launch errors
    cudaDeviceSynchronize();

    // 8. copy results back to host
    cudaMemcpy(h_output.data(), d_output, outbytes, cudaMemcpyDeviceToHost);

    // 9. verify and print results
    cout << "\noutput matrix (" << outwidth << "x" << outheight << "):\n";
    for (int r = 0; r < outheight; r++) {
        for (int c = 0; c < outwidth; c++) {
            cout << h_output[r * outwidth + c] << " ";
        }
        cout << "\n";
    }

    // 10. clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
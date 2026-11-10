#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"

using namespace std;

#define filter_radius 1
#define out_tile_dim 2
#define in_tile_dim ((out_tile_dim) + 2*(filter_radius))

// declare constant memory for the filter
__constant__ float kfilter_d[2*filter_radius+1][2*filter_radius+1];

__global__ void conv2dtiledconstkernel(float *input, float *output, int width, int height, int outwidth, int outheight) {
    __shared__ float inputtile[in_tile_dim][in_tile_dim];
    
    // Base global input coordinates for the top-left corner of this block's input tile
    int base_col = blockIdx.x * out_tile_dim;
    int base_row = blockIdx.y * out_tile_dim;

    // 1. Cooperative loading of input tile (threads load multiple elements to cover the halo)
    for (int ty = threadIdx.y; ty < in_tile_dim; ty += out_tile_dim) {
        for (int tx = threadIdx.x; tx < in_tile_dim; tx += out_tile_dim) {
            int in_row = base_row + ty;
            int in_col = base_col + tx;

            if (in_row < height && in_col < width) {
                inputtile[ty][tx] = input[in_row * width + in_col];
            } else {
                inputtile[ty][tx] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 2. Output coordinates (every thread in the block computes one output element)
    int out_col = blockIdx.x * out_tile_dim + threadIdx.x;
    int out_row = blockIdx.y * out_tile_dim + threadIdx.y;

    if (out_row < outheight && out_col < outwidth) {
        float sum = 0.0f;
        for (int frow = 0; frow < 2*filter_radius+1; frow++) {
            for (int fcol = 0; fcol < 2*filter_radius+1; fcol++) {
                sum += inputtile[threadIdx.y + frow][threadIdx.x + fcol] * kfilter_d[frow][fcol];
            }
        }
        output[out_row * outwidth + out_col] = sum;
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
    vector<float> h_input(width * height, 1.0f);
    for (int i = 0; i < width * height; i++) {
        h_input[i] = i;
    }
    vector<float> h_output(outwidth * outheight, 0.0f);
    
    // create a 3x3 averaging filter
    float h_filter[2*filter_radius+1][2*filter_radius+1];
    for(int i = 0; i < 2*filter_radius+1; i++) {
        for(int j = 0; j < 2*filter_radius+1; j++) {
            h_filter[i][j] = 1.0f / ((2*filter_radius+1)*(2*filter_radius+1));
        }
    }

    // 3. copy filter to constant memory on the device
    CUDA_CHECK(cudaMemcpyToSymbol(kfilter_d, h_filter, sizeof(h_filter)));

    // 4. allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input, inbytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, outbytes));

    // 5. copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), inbytes, cudaMemcpyHostToDevice));

    // 6. define grid and block dimensions
    // block size now matches out_tile_dim (2x2)
    dim3 blocksize(out_tile_dim, out_tile_dim);
    
    // grid size depends on out_tile_dim (2x2)
    dim3 gridsize(
        (outwidth + out_tile_dim - 1) / out_tile_dim,
        (outheight + out_tile_dim - 1) / out_tile_dim
    );

    // 7. launch the kernel
    conv2dtiledconstkernel<<<gridsize, blocksize>>>(d_input, d_output, width, height, outwidth, outheight);
    CUDA_CHECK(cudaGetLastError());
    
    // check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, outbytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int r = 0; r < outheight; r++) {
        for (int c = 0; c < outwidth; c++) {
            float expected = 0.0f;
            for (int i = 0; i < 2*filter_radius+1; i++) {
                for (int j = 0; j < 2*filter_radius+1; j++) {
                    expected += h_input[(r+i)*width+c+j] * h_filter[i][j];
                }
            }
            float value = h_output[r*outwidth+c];
            if (!isfinite(value) || fabs(value-expected) > 1e-4f) correct = false;
        }
    }

    // 9. verify and print results
    cout << "Verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    cout << "\noutput matrix (" << outwidth << "x" << outheight << "):\n";
    for (int r = 0; r < outheight; r++) {
        for (int c = 0; c < outwidth; c++) {
            cout << h_output[r * outwidth + c] << " ";
        }
        cout << "\n";
    }

    // 10. clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
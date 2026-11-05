#include "../cuda_check.h"
#include "GBlur.h"

// CUDA Kernel for Gaussian blur implemented in a separate file
__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w) {
    // Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check to stay within image dimensions
    if (i < 0 || i>= h || j < 0 || j >= w) return; 
    
    float pixval=0;
    float weight_sum=0.0f;
    
    // Iterate through the Gaussian kernel window
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            // Check boundaries for neighboring pixels
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            weight_sum += kernel[(x+k)*(2*k+1)+(y+k)];
            // Apply weighted sum using the Gaussian kernel
            pixval += (a[(x+i)*w+(y+j)] * kernel[(x+k)*(2*k+1)+(y+k)]); 
        }
    }
    // Renormalize at the edges so a constant image keeps the same brightness.
    b[i*w+j] = (unsigned char)fminf(255.0f, fmaxf(0.0f, floorf(pixval/weight_sum+0.5f)));
}

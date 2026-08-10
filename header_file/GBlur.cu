#include "GBlur.h"

// CUDA Kernel for Gaussian blur implemented in a separate file
__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w) {
    // Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check to stay within image dimensions
    if (i < 0 || i>= h || j < 0 || j >= w) return; 
    
    float pixval=0;
    int pixcnt=0;
    
    // Iterate through the Gaussian kernel window
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            // Check boundaries for neighboring pixels
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            pixcnt++;
            // Apply weighted sum using the Gaussian kernel
            pixval += (a[(x+i)*w+(y+j)] * kernel[(x+k)*(2*k+1)+(y+k)]); 
        }
        // Gaussian kernel is normalized, so no need to divide by pixcnt
        b[i*w+j] = (unsigned char)pixval;
    }
}
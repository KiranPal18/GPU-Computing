#include "GBlur.h"

__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < 0 || i>= h || j < 0 || j > w) return;  //Ideal threads
    
    float pixval=0;
    int pixcnt=0;
    
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            pixcnt++;
            pixval += (a[(x+i)*w+(y+j)] * kernel[(x+k)*(2*k+1)+(y+k)]); //pay attention to the kernel indices
        }
        //b[i*w+j] = (unsigned char)pixval / pixcnt;
        //we are not dividing by pixcnt because the Gaussian kernel is already normalized
        b[i*w+j] = (unsigned char)pixval;
    }
}
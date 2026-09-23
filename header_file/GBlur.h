#ifndef GBLUR_H
#define GBLUR_H

// Declaration of the Gaussian blur kernel for use in other source files
__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w);

#endif
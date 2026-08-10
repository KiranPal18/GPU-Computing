#ifndef GBLUR_H  //If the GBlur.h is present or not
#define GBLUR_H  //If not then define

__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w);

#endif
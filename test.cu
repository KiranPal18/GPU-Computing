#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define tile_size 32
#define coarse_factor 3

__global__ void matrixmult(float *a ,float *b, float *c, int n, int m, int k)  {
    __shared__ float tilea[tile_size][tile_size];
    __shared__ float tileb[tile_size][tile_size];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by*tile_size+ty, col_st = bx*tile_size+tx+tile_size*coarse_factor;
    vector<float> pvalue(coarse_factor, 0.0f);
    int phase = (k+tile_size-1)/tile_size;
    for (int ph = 0; ph < phase; ph++) {
        if (row < n && ph*tile_size+tx < k) {
            tilea[ty][tx] = a[row*k+ph*tile_size+tx];
        }
        else {
            tilea[ty][tx] = 0.0f;
        }
        for (int i=0; i<coarse_factor; i++) {
            int col = col_st + i*tile_size;
            if (col < k && ph*tile_size+ty < m) {
                tileb[ty][tx] = a[(ty+ph*tile_size)*m+col];
            }
            else {
                tileb[ty][tx] = 0.0f;
            }
            __syncthreads();
            for (int i=0; i<k; i++) {
                pvalue[i] += tilea[ty][i] * tileb[i][tx];
            }
            c[row*m+col] = pvalue[i];
        }
    }
}

#define FR
#define in_tile
#define out_tile

__global__ void conv2d(float* a, float *b, float *c, int n) {
    __shared__ float tile[in_tile][in_tile];
    
}


int main() {
    int n;

}
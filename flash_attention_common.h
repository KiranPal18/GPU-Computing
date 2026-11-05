#ifndef FLASH_ATTENTION_COMMON_H
#define FLASH_ATTENTION_COMMON_H

#include <cuda_runtime.h>
#include <math_constants.h>

constexpr int fa_key_tile=16;
constexpr int fa_max_dim=64;
constexpr int fa_query_rows=4;

// A warp owns one query and its output. Lanes own features lane and lane+32.
// Keep an unnormalized numerator, and divide only once after all K/V tiles.
__device__ inline void flash_warp_update(const float *tile_k, const float *tile_v,
                                         const float *query, float *result,
                                         float &maximum, float &denominator,
                                         int row, int start, int n, int d, bool causal) {
    int lane=threadIdx.x%32;
    if (row>=n) return; // Uniform for all 32 lanes of this warp
    float scores[fa_key_tile], tile_max=-CUDART_INF_F;
    for (int j=0; j<fa_key_tile; j++) {
        float dot=0.0f;
        for (int x=0; x<2; x++) {
            int feature=lane+32*x;
            if (feature<d) dot += query[x]*tile_k[j*fa_max_dim+feature];
        }
        for (int offset=16; offset>0; offset/=2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        dot=__shfl_sync(0xffffffff, dot, 0);
        scores[j] = start+j<n && (!causal || start+j<=row)
                    ? dot/sqrtf(float(d)) : -CUDART_INF_F;
        tile_max=fmaxf(tile_max, scores[j]);
    }
    if (tile_max == -CUDART_INF_F) return;
    float next_max=fmaxf(maximum, tile_max);
    float alpha=expf(maximum-next_max), tile_sum=0.0f;
    for (int j=0; j<fa_key_tile; j++) {
        scores[j]=expf(scores[j]-next_max);
        tile_sum+=scores[j];
    }
    for (int x=0; x<2; x++) {
        int feature=lane+32*x;
        if (feature<d) {
            float sum=0.0f;
            for (int j=0; j<fa_key_tile; j++) sum += scores[j]*tile_v[j*fa_max_dim+feature];
            result[x]=alpha*result[x]+sum;
        }
    }
    denominator=alpha*denominator+tile_sum;
    maximum=next_max;
}

#endif
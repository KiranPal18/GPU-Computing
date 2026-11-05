#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Stop at the first CUDA error instead of printing an uninitialized result.
inline void cuda_check(cudaError_t error, const char *call, const char *file, int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, call, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cuda_check((call), #call, __FILE__, __LINE__)

#endif

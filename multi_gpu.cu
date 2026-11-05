#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <thread>

using namespace std;

int main() {
    int n=0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    cout << n;

    return 0;
}

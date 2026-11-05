import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from pycuda.tools import DeviceMemoryPool


def main():
    # Create the CUDA context only when the example is run.
    import pycuda.autoinit

    n = 1003
    a = np.arange(n, dtype=np.float32) / np.float32(n)
    b = np.full(n, 2.0, dtype=np.float32)

    # Reuse device allocations and keep intermediate arrays on the GPU.
    pool = DeviceMemoryPool()
    d_a = gpuarray.to_gpu(a, allocator=pool.allocate)
    d_b = gpuarray.to_gpu(b, allocator=pool.allocate)
    d_c = gpuarray.empty_like(d_a)

    # SourceModule compiles a normal CUDA kernel from a Python string.
    module = SourceModule("""
    __global__ void vecadd(const float *a, const float *b, float *c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }
    """)
    vecadd = module.get_function("vecadd")
    vecadd(d_a, d_b, d_c, np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1))

    # Fuse multiply and add into one pass instead of allocating temporary arrays.
    scale = ElementwiseKernel(
        "float *a, float alpha, float bias",
        "a[i] = alpha*a[i] + bias",
        "scale_bias",
    )
    scale(d_c, np.float32(2.0), np.float32(1.0))

    # Reduce on the GPU and copy only the scalar result back to the CPU.
    reduction = ReductionKernel(
        np.float32,
        neutral="0",
        reduce_expr="a+b",
        map_expr="x[i]",
        arguments="const float *x",
    )
    answer = float(reduction(d_c).get())
    expected = np.float32(2.0) * (a + b) + np.float32(1.0)
    np.testing.assert_allclose(d_c.get(), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(answer, np.sum(expected, dtype=np.float64), rtol=1e-5)
    print("GPU sum:", answer)
    print("Expected:", float(np.sum(expected, dtype=np.float64)))
    cuda.Context.synchronize()
    del d_a, d_b, d_c
    pool.stop_holding()


if __name__ == "__main__":
    main()
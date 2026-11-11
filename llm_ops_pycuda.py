import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

# The same LLaMA style building blocks as llm_ops.cu, driven from Python:
# RMSNorm, RoPE and SwiGLU, plus greedy decoding (argmax over the vocabulary).
# Every result is checked against NumPy.


def main():
    # Create the CUDA context only when the example is run.
    import pycuda.autoinit

    tokens, dim, vocab = 16, 512, 32000
    eps, base = np.float32(1e-5), np.float32(10000.0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((tokens, dim), dtype=np.float32)
    w = (1 + rng.random(dim, dtype=np.float32) / 10).astype(np.float32)
    gate = rng.standard_normal((tokens, dim), dtype=np.float32)
    up = rng.standard_normal((tokens, dim), dtype=np.float32)
    logits = rng.standard_normal(vocab, dtype=np.float32)

    # SourceModule compiles normal CUDA kernels from a Python string.
    module = SourceModule("""
    // One block normalizes one token; block size must be a multiple of 32.
    __global__ void rmsnorm(const float *x, const float *weight, float *y, int dim, float eps) {
        __shared__ float warp_sums[32];
        const float *row = x + blockIdx.x * dim;
        float *out = y + blockIdx.x * dim;

        float sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) sum += row[i] * row[i];

        // Reduce inside each warp with shuffles, then reduce the warp results
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();
        if (warp == 0) {
            sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
            if (lane == 0) warp_sums[0] = rsqrtf(sum / dim + eps);
        }
        __syncthreads();

        float scale = warp_sums[0];
        for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = row[i] * scale * weight[i];
    }

    // One thread rotates one pair (x[2i], x[2i+1]) by angle pos * base^(-2i/dim), in place.
    __global__ void rope(float *x, int tokens, int dim, float base) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int half = dim / 2;
        if (idx < tokens * half) {
            int pos = idx / half, i = idx % half;
            float angle = pos * powf(base, -2.0f * i / dim);
            float c = cosf(angle), s = sinf(angle);
            float *p = x + pos * dim + 2 * i;
            float a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
    }
    """)
    rmsnorm = module.get_function("rmsnorm")
    rope = module.get_function("rope")

    # RMSNorm
    d_x = gpuarray.to_gpu(x)
    d_w = gpuarray.to_gpu(w)
    d_y = gpuarray.empty_like(d_x)
    rmsnorm(d_x, d_w, d_y, np.int32(dim), eps, block=(256, 1, 1), grid=(tokens, 1))
    expected = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps) * w
    np.testing.assert_allclose(d_y.get(), expected, rtol=1e-4, atol=1e-4)
    print("RMSNorm matches NumPy")

    # RoPE, applied in place to d_x
    pairs = tokens * dim // 2
    rope(d_x, np.int32(tokens), np.int32(dim), base, block=(256, 1, 1), grid=((pairs + 255) // 256, 1))
    angle = np.arange(tokens)[:, None] * base ** (-2 * np.arange(dim // 2) / dim)
    a, b = x[:, 0::2], x[:, 1::2]
    expected = np.empty_like(x)
    expected[:, 0::2] = a * np.cos(angle) - b * np.sin(angle)
    expected[:, 1::2] = a * np.sin(angle) + b * np.cos(angle)
    np.testing.assert_allclose(d_x.get(), expected, rtol=1e-4, atol=1e-4)
    print("RoPE matches NumPy")

    # SwiGLU: elementwise ops need no hand written kernel in PyCUDA.
    swiglu = ElementwiseKernel(
        "const float *gate, const float *up, float *out",
        "out[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i]",
        "swiglu",
    )
    d_gate = gpuarray.to_gpu(gate)
    d_up = gpuarray.to_gpu(up)
    d_out = gpuarray.empty_like(d_gate)
    swiglu(d_gate, d_up, d_out)
    expected = gate / (1 + np.exp(-gate)) * up
    np.testing.assert_allclose(d_out.get(), expected, rtol=1e-4, atol=1e-4)
    print("SwiGLU matches NumPy")

    # Greedy decoding: the next token is the argmax of the logits.
    # The reduction packs (logit, index) into a float2 so max keeps the index.
    argmax = ReductionKernel(
        gpuarray.vec.float2,
        neutral="make_float2(-1e30f, -1.0f)",
        reduce_expr="(a.x >= b.x) ? a : b",
        map_expr="make_float2(x[i], (float)i)",
        arguments="const float *x",
    )
    d_logits = gpuarray.to_gpu(logits)
    token = int(argmax(d_logits).get()["y"])
    assert token == int(np.argmax(logits)), (token, int(np.argmax(logits)))
    print("Next token:", token)

    cuda.Context.synchronize()
    del d_x, d_w, d_y, d_gate, d_up, d_out, d_logits


if __name__ == "__main__":
    main()

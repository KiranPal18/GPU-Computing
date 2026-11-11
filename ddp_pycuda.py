import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# Data parallel training by hand, the same idea as ddp.py and ddp.cu:
# every GPU gets a slice of the data, computes gradients on it, the gradients are
# summed across GPUs (all-reduce) and every GPU applies the same update.
# Model: y = w*x + b, trained to learn y = 3x + 2 with MSE loss.

KERNEL = """
__global__ void linear_grad(const float *x, const float *y, float w, float b, int n, int total, float *grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float err = w * x[i] + b - y[i];
        // Divide by the full dataset size so summing over GPUs gives the global mean
        atomicAdd(&grad[0], 2.0f * err * x[i] / total);
        atomicAdd(&grad[1], 2.0f * err / total);
    }
}
"""


def main():
    # One CUDA context per GPU instead of pycuda.autoinit, which only opens GPU 0.
    cuda.init()
    gpus = cuda.Device.count()
    print("GPUs:", gpus)

    # Toy data: x in [-1, 1], y = 3x + 2
    n = 1024
    x = np.linspace(-1, 1, n, dtype=np.float32)
    y = (3 * x + 2).astype(np.float32)

    # Split the data into one slice per GPU; the slices stay on their GPU.
    workers = []
    for dev, (xs, ys) in enumerate(zip(np.array_split(x, gpus), np.array_split(y, gpus))):
        ctx = cuda.Device(dev).make_context()
        print(f"GPU {dev} gets {len(xs)} of {n} samples")
        workers.append({
            "ctx": ctx,
            # The kernel must be compiled inside each GPU's own context.
            "grad_fn": SourceModule(KERNEL).get_function("linear_grad"),
            "x": gpuarray.to_gpu(xs),
            "y": gpuarray.to_gpu(ys),
            "grad": gpuarray.zeros(2, dtype=np.float32),
            "len": len(xs),
        })
        ctx.pop()

    # Model parameters, sent to every GPU as kernel arguments (the "broadcast")
    w, b, lr = np.float32(0), np.float32(0), np.float32(0.1)

    for step in range(200):
        # Forward + backward: launches are async, so all GPUs run at the same time.
        for g in workers:
            g["ctx"].push()
            g["grad"].fill(0)
            g["grad_fn"](g["x"], g["y"], w, b, np.int32(g["len"]), np.int32(n), g["grad"],
                         block=(256, 1, 1), grid=((g["len"] + 255) // 256, 1))
            g["ctx"].pop()

        # All-reduce: sum the partial gradients from every GPU
        total = np.zeros(2, dtype=np.float32)
        for g in workers:
            g["ctx"].push()
            total += g["grad"].get()
            g["ctx"].pop()

        # Same update for everyone, so every GPU keeps the same model
        w -= lr * total[0]
        b -= lr * total[1]

        if step % 20 == 0:
            print(f"step {step:3d}  w {w:.4f}  b {b:.4f}")

    # Free GPU memory inside each context, then close the context.
    for g in workers:
        g["ctx"].push()
        del g["x"], g["y"], g["grad"], g["grad_fn"]
        g["ctx"].pop()
        g["ctx"].detach()

    print(f"w={w:.4f} b={b:.4f} (target 3, 2)")
    np.testing.assert_allclose([w, b], [3, 2], atol=1e-2)


if __name__ == "__main__":
    main()

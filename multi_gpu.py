from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# One CPU worker manages one GPU and a separate part of the full vectors.
# start is its first element in the full vector; n is its number of elements.
def run_gpu(device, a, b, c, start, n):
    # Skip an empty part so we do not allocate buffers or launch an empty grid.
    if n == 0:
        return
    # A context holds this worker's GPU resources and becomes current in this thread.
    # Create it inside the worker so each thread selects its own GPU.
    context = cuda.Device(device).make_context()
    # Track allocated GPU buffers so they can be freed if an error occurs later.
    allocations = []
    # Start with no resources allocated, which also makes partial cleanup possible.
    stream = module = vecadd = None
    h_a = h_b = h_c = None
    try:
        # A stream is a queue that runs this worker's GPU operations in order.
        stream = cuda.Stream()
        # Page-locked (pinned) CPU arrays allow asynchronous CPU/GPU copies.
        h_a = cuda.pagelocked_empty(n, np.float32)
        h_b = cuda.pagelocked_empty(n, np.float32)
        h_c = cuda.pagelocked_empty(n, np.float32)
        # Fill the pinned buffers with this GPU's assigned input elements.
        h_a[:] = a[start:start + n]
        h_b[:] = b[start:start + n]

        # Allocate three GPU buffers, each with enough bytes for n float32 values.
        for _ in range(3):
            allocations.append(cuda.mem_alloc(h_a.nbytes))
        d_a, d_b, d_c = allocations

        # Compile a CUDA kernel from this string for the worker's selected GPU.
        # Each CUDA thread adds one pair; i < n protects the final partial block.
        module = SourceModule("""
        __global__ void vecadd(const float *a, const float *b, float *c, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        """)
        # Get the compiled function so Python can launch it.
        vecadd = module.get_function("vecadd")
        # Queue input copies, the kernel, and the result copy on the same stream.
        cuda.memcpy_htod_async(d_a, h_a, stream)
        cuda.memcpy_htod_async(d_b, h_b, stream)
        # Launch 256 threads per block and round the block count up to cover n.
        # np.int32 passes n with the same integer type expected by the CUDA kernel.
        vecadd(d_a, d_b, d_c, np.int32(n), block=(256, 1, 1),
               grid=((n + 255) // 256, 1), stream=stream)
        cuda.memcpy_dtoh_async(h_c, d_c, stream)
        # Wait for the copy to finish before reading the pinned result buffer.
        stream.synchronize()
        # Workers fill different output slices, so they do not overwrite each other.
        c[start:start + n] = h_c
    finally:
        # Cleanup runs on success and failure. Keep buffers alive until GPU work finishes.
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            try:
                # Free all GPU buffers allocated by this worker.
                for allocation in allocations:
                    allocation.free()
            finally:
                # Release the kernel, stream, and pinned arrays while the context is current.
                vecadd = module = stream = None
                h_a = h_b = h_c = None
                # Remove this thread's current context, then release its ownership of it.
                context.pop()
                context.detach()


def main():
    # Initialize the CUDA driver before any worker tries to select a GPU.
    cuda.init()
    # Assume GPUs 0 and 1 exist. Change num_gpus to change the configured count.
    num_gpus, n = 2, 1000003
    # Create the full input vectors on the CPU using repeatable sample values.
    values = np.arange(n, dtype=np.int32)
    a = (values % 101 - 50).astype(np.float32) * np.float32(0.25)
    b = (values % 67 - 33).astype(np.float32) * np.float32(0.5)
    # NaN marks output elements not yet filled by a worker.
    c = np.full(n, np.nan, dtype=np.float32)

    # Allow up to num_gpus CPU threads to submit work to the GPUs at the same time.
    with ThreadPoolExecutor(max_workers=num_gpus) as gpus:
        # Each future represents one worker's task, including any error it raises.
        futures = []
        for device in range(num_gpus):
            # Split the vector into adjacent parts, including any leftover elements.
            start = n * device // num_gpus
            end = n * (device + 1) // num_gpus
            # Schedule this GPU's work and keep a handle for waiting on its completion.
            futures.append(gpus.submit(run_gpu, device, a, b, c, start, end - start))
        for future in futures:
            # Wait for this task and report its error in the main thread if it failed.
            future.result()

    # Compare the combined GPU output with NumPy's CPU result, allowing a small error.
    np.testing.assert_allclose(c, a + b, rtol=0, atol=1e-6)
    print("Vector addition across", num_gpus, "GPUs: PASS")


# Importing this file defines the functions; executing it starts the GPU work.
if __name__ == "__main__":
    main()
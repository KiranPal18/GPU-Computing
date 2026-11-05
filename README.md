# GPU Computing
A hands-on workspace for learning, experimenting, and building with GPU programming using NVIDIA CUDA.

This repository serves as a growing collection of CUDA C/C++ (and eventually Python/PyCUDA) programs. The goal is to build an understanding of GPU computing from the ground up—implementing concepts from scratch rather than relying on high-level abstractions.

## Repository Organization

The repository is organized by complexity and structure:

*   **Standalone Scripts (`*.cu`):** Single-file implementations focusing on specific concepts (e.g., vector math, matrix operations, naive kernels).
*   **Modular Projects (Directories):** Multi-file projects demonstrating how to split host logic (`main.cu`), kernel implementations (`*.cu`), and declarations (`*.h`) for larger applications.
*   **Image Processing:** Scripts utilizing OpenCV alongside CUDA to demonstrate parallel convolution and filtering.

## Prerequisites

To compile and run the programs in this repository, you will need:
*   An NVIDIA GPU
*   NVIDIA CUDA Toolkit (`nvcc` compiler)
*   A compatible C++ compiler
*   **OpenCV 4** (Required only for image processing scripts)
*   **Python 3 & PyCUDA** (For upcoming Python-based GPU experiments)

*Verify your installations:*
```bash
nvcc --version
pkg-config --modversion opencv4
```

## How to Build and Run
Instead of listing commands for every single file, use the following compilation patterns based on the type of program you are running.

### 1. Standard Single-File CUDA Programs
For basic matrix and vector operations:

```bash
nvcc <filename>.cu -o <output_name>
./<output_name>
```

### 2. Programs Requiring OpenCV
For any script dealing with image processing (like blurs or convolutions):

```bash
nvcc <filename>.cu -o <output_name> $(pkg-config --cflags --libs opencv4)
./<output_name>
```

### 3. Multi-File Projects
For structured projects (navigate into the project directory first):

```bash
nvcc main.cu <kernel_file>.cu -o <output_name> $(pkg-config --cflags --libs opencv4)
./<output_name>
```

### 4. PyCUDA Scripts

```bash
python <filename>.py
```

## Concepts Explored
This repository maps out a continuous learning progression. Scripts and projects added here will fall into one of the following architectural categories:

- **The Fundamentals:** Writing `__global__` kernels, understanding threads vs. blocks vs. grids.
- **Memory Management:** Moving data between the CPU (host) and GPU (device) using `cudaMalloc` and `cudaMemcpy`.
- **Parallel Mathematics:** Mapping 2D grids and blocks to row-major matrices for vector and matrix operations.
- **Applied Parallelism (Image Processing):** Utilizing GPU convolution for operations like Gaussian Blurs.
- **Advanced Optimization:** Exploring Shared memory, tiled matrix multiplication, memory coalescing, CUDA streams, and kernel synchronization.

## Multiple GPUs with host threads

`multi_gpu.cu` and `multi_gpu.py` split a vector addition between GPUs 0 and 1.
Both examples assume those device IDs exist; they do not query the number of GPUs.
Change `num_gpus` in each file to change the configured worker count.

Each worker selects its device, allocates pinned host buffers and device buffers,
then queues input copies, the kernel, and an output copy on its own stream.
Operations in one stream execute in order; different GPUs can make progress
independently. The worker waits for its stream before reading or freeing buffers.
The main thread waits for all workers before comparing against the CPU result.
The PyCUDA version creates and releases a context inside each worker thread.

```bash
nvcc -std=c++17 multi_gpu.cu -Xcompiler=-pthread -o /tmp/multi_gpu
/tmp/multi_gpu
python3 multi_gpu.py
```

## Distributed training with or without CUDA

The vector examples demonstrate independent GPU work. To train one shared model
with data parallelism, workers also exchange gradients:

1. Start every worker with the same model parameters.
2. Give each worker a different shard of the training data.
3. Compute a forward pass, loss, and backward pass locally.
4. Use an all-reduce to combine gradients across workers.
5. Average the gradients and apply the same optimizer update on every worker.

With equally sized local batches, the shared gradient is
`(gradient_0 + gradient_1 + ... + gradient_p_minus_1) / p`.
A barrier or `thread.join()` waits for work; gradient reduction actually combines it.

| Approach | Computation | Communication |
| --- | --- | --- |
| Handwritten CUDA training | Write forward, backward, and optimizer kernels | NCCL all-reduce on GPU gradient buffers; coordinate processes across machines with a launcher/MPI |
| GPU training without writing CUDA kernels | PyTorch autograd and optimizers on NVIDIA GPUs | DistributedDataParallel (DDP) with NCCL; CUDA is still used internally |
| Training without CUDA or GPUs | PyTorch operations on CPUs | DDP with Gloo; a CPU-only PyTorch installation is sufficient |

For handwritten CUDA, initialize an NCCL communicator for every GPU, compute local
gradients, call `ncclAllReduce(..., ncclSum, communicator, stream)`, and launch an
optimizer kernel on that stream using the reduced gradient divided by the number
of workers. NCCL communicates the gradients; you still implement the model,
derivatives, and optimizer. See the [NCCL examples](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html).

`distributed_training.py` is a complete small DDP example. It learns the linear
relationship `y = x @ [1, -2, 0.5, 3] + 0.7`, shards its synthetic dataset with
`DistributedSampler`, synchronizes gradients during `loss.backward()`, and reports
global mean training loss. Each worker runs in a separate process. The sampler
reshuffles each epoch and drops any final samples needed to keep shard sizes equal.
It requires PyTorch; the vector examples require CUDA/PyCUDA instead.

```bash
# Two CPU processes on one machine; no GPU or CUDA required.
torchrun --standalone --nproc-per-node=2 distributed_training.py --device=cpu

# Two processes, each using one NVIDIA GPU; no handwritten CUDA kernels.
torchrun --standalone --nproc-per-node=2 distributed_training.py --device=cuda
```

For two machines, run the same script and environment on each machine. Use the
first machine's reachable IP for `--master-addr` in both commands; the address
below is an example. The master port and worker communication must be reachable.

```bash
# Machine 0
torchrun --nnodes=2 --nproc-per-node=2 --node-rank=0 --master-addr=10.0.0.1 --master-port=29500 distributed_training.py --device=cpu

# Machine 1
torchrun --nnodes=2 --nproc-per-node=2 --node-rank=1 --master-addr=10.0.0.1 --master-port=29500 distributed_training.py --device=cpu
```

Switch both commands to `--device=cuda` to use two NVIDIA GPUs per machine.
For models larger than one GPU's memory, explore parameter sharding, tensor
parallelism, or pipeline parallelism after learning DDP. See the
[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html),
[backend documentation](https://docs.pytorch.org/docs/stable/distributed.html), and
[torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html).

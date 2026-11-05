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
- **Advanced Optimizations (Ongoing):** Exploring Shared memory, tiled matrix multiplication, memory coalescing, CUDA streams, and kernel synchronization.
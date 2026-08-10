#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// CUDA Kernel for Gaussian blurring of an image
__global__ void imgblur(const unsigned char *a, unsigned char *b, float *kernel, const int k, const int h, const int w) {
    // Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check to stay within image dimensions
    if (i < 0 || i>= h || j < 0 || j >= w) return; 
    
    float pixval=0;
    int pixcnt=0;
    
    // Iterate through the Gaussian kernel window
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            // Check boundaries for neighboring pixels
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            pixcnt++;
            // Apply weight from the Gaussian kernel
            pixval += (a[(x+i)*w+(y+j)] * kernel[(x+k)*(2*k+1)+(y+k)]); 
        }
        // The Gaussian kernel is already normalized, so we don't divide by pixcnt
        b[i*w+j] = (unsigned char)pixval;
    }
}

int main() {
    // Read input image in grayscale using OpenCV
    Mat image = imread("image.png", IMREAD_GRAYSCALE);

    Mat small;
    // Resize image for consistent processing size
    resize(
        image,
        small,
        cv::Size(224, 224)
    );

    // Check if the image memory is contiguous for CUDA transfer
    if (small.isContinuous()) {
        cout << "YES\n";
    }

    cout << "Width: " << small.cols << "\n";
    cout << "Height: " << small.rows << "\n";
    
    imshow("Image", small);
    waitKey(0); 

    // Define and create a normalized Gaussian kernel on the Host
    int k=5;
    vector<float> kernel((2*k+1)*(2*k+1));
    float sigma=k, sum=0.0f;

    for (int i=-k; i<=k; i++) {
        for (int j=-k; j<=k; j++) {
            float val = exp(-(i*i + j*j) / (2.0f*sigma*sigma));
            kernel[(i+k)*(2*k+1)+(j+k)] = val;
            sum += val;
        }
    }

    // Normalize kernel so sum of all elements is 1.0
    for (int i = 0; i < 2*k+1; i++) {
        for (int j = 0; j < 2*k+1; j++) {
            kernel[i * (2*k+1) + j] /= sum;
        }
    }

    // Pointers for GPU device memory
    unsigned char *d_a, *d_b;
    float *d_kernel;
    cudaMalloc((void **)&d_a, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_b, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_kernel, (2*k+1)*(2*k+1)*sizeof(float));

    // Transfer image and kernel data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, small.data, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), (2*k+1)*(2*k+1)*sizeof(float), cudaMemcpyHostToDevice);
    
    // Define execution configuration: 16x16 blocks
    dim3 block(16, 16);
    dim3 grid(
        (small.cols+block.x-1)/block.x,
        (small.rows+block.y-1)/block.y
    );
    
    // Launch the Gaussian blur kernel on the GPU
    imgblur<<<grid, block>>>(d_a, d_b, d_kernel, 5, small.rows, small.cols);

    // Transfer the processed image back to Host (CPU)
    cudaMemcpy(small.data, d_b, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_kernel);
    cudaFree(d_a);
    cudaFree(d_b);

    // Display the Gaussian blurred result
    imshow("Image", small);
    waitKey(0);

    return 0;
}
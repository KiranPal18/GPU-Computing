#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// CUDA Kernel for basic image blurring (averaging)
__global__ void imgblur(const unsigned char *a, unsigned char *b, const int k, const int h, const int w) {
    // Global index of the thread
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    // Boundary check to stay within image dimensions
    if (i < 0 || i>= h || j < 0 || j >= w) return; 
    
    int pixval=0, pixcnt=0;
    
    // Iterate through a square window of size (2*k + 1) centered at the pixel
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            // Check boundaries for neighboring pixels
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            pixcnt++;
            pixval += a[(x+i)*w+(y+j)];
        }
    }
    // Compute average value for the blur effect
    b[i*w+j] = pixval / pixcnt;
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

    // Pointers for GPU device memory
    unsigned char *d_a, *d_b;
    cudaMalloc((void **)&d_a, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_b, small.cols*small.rows*sizeof(unsigned char));

    // Transfer image data from Host (CPU) to Device (GPU)
    cudaMemcpy(d_a, small.data, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // Define execution configuration: 16x16 blocks
    dim3 block(16, 16);
    dim3 grid(
        (small.cols+block.x-1)/block.x,
        (small.rows+block.y-1)/block.y
    );
    
    // Launch the blurring kernel with a radius of 5
    imgblur<<<grid, block>>>(d_a, d_b, 5, small.rows, small.cols);

    // Transfer the processed image back to Host (CPU)
    cudaMemcpy(small.data, d_b, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);

    // Display the blurred result
    imshow("Image", small);
    waitKey(0);
    return 0;
}
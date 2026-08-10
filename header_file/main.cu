#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "GBlur.h"

using namespace std;
using namespace cv;


int main() {
    Mat image = imread("image.png", IMREAD_GRAYSCALE);  //reading image as input using opencv

    Mat small;
    resize(
        image,
        small,
        cv::Size(224, 224)
    );

    if (small.isContinuous()) {  //checking whether the image tensor is stored in a contigous manner or not
        cout << "YES\n";
    }

    cout << "Width: " << small.cols << "\n";
    cout << "Height: " << small.rows << "\n";
    
    imshow("Image", small);  //Display of image
    waitKey(0);  // waiting for an keyboard input to process next steps

    // for (int i = 0; i < small.rows; i++) {
    //     for (int j = 0; j < small.cols; j++) {
    //         cout << (int)small.at<unsigned char>(i, j) << " ";
    //     }
    //     cout << "\n";
    // }


    //Defining Kernel/Filter
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

    for (int i = 0; i < 2*k+1; i++) {
        for (int j = 0; j < 2*k+1; j++) {
            kernel[i * (2*k+1) + j] /= sum;
        }
    }

    unsigned char *d_a, *d_b;
    float *d_kernel;
    cudaMalloc((void **)&d_a, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_b, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_kernel, (2*k+1)*(2*k+1)*sizeof(float));

    cudaMemcpy(d_a, small.data, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), (2*k+1)*(2*k+1)*sizeof(float), cudaMemcpyHostToDevice);
    

    //kernel Launching
    dim3 block(16, 16);
    dim3 grid(
        (small.cols+block.x-1)/block.x,
        (small.rows+block.y-1)/block.y
    );
    
    imgblur<<<grid, block>>>(d_a, d_b, d_kernel, 5, small.rows, small.cols);

    cudaMemcpy(small.data, d_b, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    

    cudaFree(d_kernel);
    cudaFree(d_a);
    cudaFree(d_b);


    //Final output image
    imshow("Image", small);
    waitKey(0);

    // for (int i = 0; i < small.rows; i++) {
    //     for (int j = 0; j < small.cols; j++) {
    //         cout << (int)small.at<unsigned char>(i, j) << " ";
    //     }
    //     cout << "\n";
    // }
    return 0;
}
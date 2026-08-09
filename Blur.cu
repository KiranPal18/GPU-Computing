#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

__global__ void imgblur(const unsigned char *a, unsigned char *b, const int k, const int h, const int w) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < 0 || i>= h || j < 0 || j > w) return;
    int pixval=0, pixcnt=0;
    for (int x=-k; x<=k; x++ ) {
        for (int y=-k; y<=k; y++) {
            if (i+x < 0 || i+x >= h || j+y < 0 || j+y >= w) continue;
            pixcnt++;
            pixval += a[(x+i)*w+(y+j)];
        }
        b[i*w+j] = pixval / pixcnt;
    }
}

int main() {
    Mat image = imread("image.png", IMREAD_GRAYSCALE);

    Mat small;
    resize(
        image,
        small,
        cv::Size(224, 224)
    );

    if (small.isContinuous()) {
        cout << "YES\n";
    }

    cout << "Width: " << small.cols << "\n";
    cout << "Height: " << small.rows << "\n";
    imshow("Image", small);
    waitKey(0);

    // for (int i = 0; i < small.rows; i++) {
    //     for (int j = 0; j < small.cols; j++) {
    //         cout << (int)small.at<unsigned char>(i, j) << " ";
    //     }
    //     cout << "\n";
    // }

    unsigned char *d_a, *d_b;
    cudaMalloc((void **)&d_a, small.cols*small.rows*sizeof(unsigned char));
    cudaMalloc((void **)&d_b, small.cols*small.rows*sizeof(unsigned char));

    cudaMemcpy(d_a, small.data, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid(
        (small.cols+block.x-1)/block.x,
        (small.rows+block.y-1)/block.y
    );
    
    imgblur<<<grid, block>>>(d_a, d_b, 5, small.rows, small.cols);

    cudaMemcpy(small.data, d_b, small.cols*small.rows*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    imshow("Image", small);
    waitKey(0);
    return 0;
}
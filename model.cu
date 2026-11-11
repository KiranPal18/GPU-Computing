#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;


#define inp_size 16
#define n_samples 53
#define epochs 100

#define l_r 0.1f

struct samples {
    int image[inp_size];
    int label;
};

__global__ void hebbupdate(const int* images, const int* labels, float* weights, float* bias, int nsamples, float lr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inp_size) {
        float grad = 0.0f;
        for (int j = 0; j < nsamples; j++) {
            float x = (images[j * inp_size + i] == 0) ? -1.0f : 1.0f;  
	    float y = (labels[j] == 0) ? -1.0f : 1.0f;
            grad += lr * x * y;
        }
        weights[i] += grad;
    }
    if (i == 0) {
        float b_upd = 0.0f;
        for (int k = 0; k < nsamples; k++) {
            float z = (labels[k]==0) ? -1.0f : 1.0f;
            b_upd += lr * z;
        }
        *bias += b_upd;
    }
}

int predict(const int* input, const float* weights, const float bias) {
    float sum = bias;
    for (int i = 0; i < inp_size; i++) {
        sum += input[i] * weights[i];
    }
    if (sum >= 0.0f) {
	    return 1;
    }
    else{
	    return 0;
    }
}


int main() {

    samples dataset[n_samples] = {
        {{1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1}, 0},
        {{1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0}, 1},
        {{0,1,1,0,1,0,0,0,1,0,0,1,0,1,1,0}, 0},
        {{1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0}, 1},
        {{1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0}, 1},
        {{0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1}, 0},
        {{0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,1}, 1},
        {{0,1,1,0,1,0,0,1,1,0,0,1,0,1,0,0}, 0},
        {{0,1,1,0,1,0,0,1,1,0,0,0,0,1,1,0}, 1},
        {{0,1,1,1,0,1,0,0,0,1,0,0,1,1,1,1}, 1},
        {{0,1,1,1,0,0,0,1,0,0,1,1,0,0,1,1}, 0},
        {{0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,1}, 1},
        {{1,1,1,0,0,0,1,0,0,0,1,0,1,1,1,0}, 1},
        {{0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0}, 0},
        {{1,1,1,0,1,0,0,0,1,0,0,1,1,1,1,0}, 1},
        {{0,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0}, 0},
        {{0,0,1,0,1,0,0,0,1,0,0,0,0,1,1,0}, 1},
        {{1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1}, 0},
        {{0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1}, 0},
        {{1,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0}, 0},
        {{0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,1}, 0},
        {{1,0,0,0,1,0,0,0,1,0,0,0,1,1,1,0}, 1},
        {{0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0}, 0},
        {{0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1}, 1},
        {{0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,1}, 1},
        {{1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,0}, 0},
        {{1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,0}, 0},
        {{0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1}, 1},
        {{1,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0}, 0},
        {{1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1}, 1},
        {{0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1}, 0},
        {{1,1,1,0,1,0,0,0,1,0,0,0,1,1,1,0}, 1},
        {{1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1}, 0},
        {{1,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0}, 1},
        {{0,1,1,0,1,0,0,0,1,0,0,0,0,1,1,0}, 1},
        {{0,1,0,1,0,0,1,0,0,1,0,1,0,0,1,0}, 0},
        {{1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,1}, 1},
        {{0,1,1,0,1,0,0,1,1,0,0,0,0,1,1,0}, 0},
        {{1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1}, 0},
        {{1,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0}, 1},
        {{1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0}, 1},
        {{0,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1}, 1},
        {{1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1}, 0},
        {{1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1}, 1},
        {{0,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1}, 0},
        {{1,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0}, 0},
        {{0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,1}, 0},
        {{1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1}, 1},
        {{0,0,1,1,0,1,0,0,0,1,0,0,0,0,1,1}, 0},
        {{0,1,1,1,0,1,0,0,0,1,0,0,0,1,1,1}, 1},
        {{1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,1}, 0},
        {{0,1,1,1,0,0,0,1,0,0,0,1,0,1,1,1}, 1},
        {{1,0,0,1,0,1,1,0,0,1,1,1,1,0,0,1}, 0}
    };

    int* h_images = new int[n_samples * inp_size];
    int* h_labels = new int[n_samples];

    float h_weights[inp_size] = {0.0f};
    float h_bias = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        h_labels[i] = dataset[i].label;

        for (int j = 0; j < inp_size; j++) {
            h_images[i * inp_size + j] = dataset[i].image[j];
        }
    }

    int* d_images;
    int* d_labels;

    float* d_weights;
    float* d_bias;


    cudaMalloc(&d_images, n_samples * inp_size * sizeof(int));
    cudaMalloc(&d_labels, n_samples * sizeof(int));
    cudaMalloc(&d_weights, inp_size * sizeof(float));

    cudaMalloc(&d_bias, sizeof(float));
    cudaMemcpy(d_images,h_images,n_samples * inp_size * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels,h_labels,n_samples * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, inp_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &h_bias, sizeof(float), cudaMemcpyHostToDevice);

    cout << "Training";
    int threads = 16;
    int blocks = 1;

    for (int epoch = 0; epoch < epochs; epoch++) {
        hebbupdate<<<blocks, threads>>>(d_images,d_labels,d_weights,d_bias, n_samples,l_r);
        cudaDeviceSynchronize();
        cout << "Epoch "<< epoch + 1<< " completed\n";
    }

    cudaMemcpy(h_weights, d_weights,inp_size * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bias,d_bias,sizeof(float),cudaMemcpyDeviceToHost);
    cout << "final params\n";
    for (int i = 0; i < inp_size; i++) {
        cout << "W[" << i << "] = " << h_weights[i] << "\n";
    }

    cout << "Bias = " << h_bias << endl;

    cout << " Test\n";
    int correct = 0;

    for (int i = 0; i < n_samples; i++) {
        int prediction = predict(dataset[i].image,h_weights,h_bias);
        cout << "Sample " << i + 1 << " | Actual = " << dataset[i].label << " | Predicted = " << prediction;
        if (prediction == dataset[i].label) {
            cout << " | Correct\n";
            correct++;
        }
        else {
            cout << " | Wrong\n";
        }
    }


    float accuracy = ((float)correct / n_samples) * 100.0f;

    cout << "Metric\n";

    cout << "Correct : " << correct << "/" << n_samples << endl;

    cout << "Accuracy: " << accuracy << "%\n";

    cudaFree(d_images);
    cudaFree(d_labels);
    cudaFree(d_weights);
    cudaFree(d_bias);

    delete[] h_images;
    delete[] h_labels;

    return 0;
}

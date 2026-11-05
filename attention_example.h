#ifndef ATTENTION_EXAMPLE_H
#define ATTENTION_EXAMPLE_H

#include <bits/stdc++.h>
#include <math_constants.h>
#include "cuda_check.h"

using namespace std;

// Single head, no batch, row-major Q/K/V of shape [n,d].
// Float inputs; causal=true prevents query i from attending to keys j>i.
using attention_launcher = void (*)(const float*, const float*, const float*, float*, int, int, bool);

inline vector<float> attention_cpu(const vector<float> &q, const vector<float> &k,
                                    const vector<float> &v, int n, int d, bool causal) {
    vector<float> output(n*d);
    vector<double> scores(n);
    for (int i=0; i<n; i++) {
        double maximum=-numeric_limits<double>::infinity();
        int end=causal ? i+1 : n;
        for (int j=0; j<end; j++) {
            double sum=0.0;
            for (int x=0; x<d; x++) sum += double(q[i*d+x])*k[j*d+x];
            scores[j]=sum/sqrt(double(d));
            maximum=max(maximum, scores[j]);
        }
        double denominator=0.0;
        for (int j=0; j<end; j++) denominator += (scores[j]=exp(scores[j]-maximum));
        for (int x=0; x<d; x++) {
            double sum=0.0;
            for (int j=0; j<end; j++) sum += scores[j]*v[j*d+x];
            output[i*d+x]=sum/denominator;
        }
    }
    return output;
}

inline int attention_example(attention_launcher launch) {
    // Non-multiple tiles, short rows, partial warps, and large softmax logits.
    for (auto shape : {pair<int,int>{1,1}, {5,3}, {17,16}, {35,33}, {65,64}}) {
        int n=shape.first, d=shape.second;
        for (bool causal : {false, true}) {
            for (float scale : {1.0f, 20.0f}) {
                vector<float> q(n*d), k(n*d), v(n*d), output(n*d);
                for (int i=0; i<n*d; i++) {
                    q[i]=scale*(i%13-6)/7.0f;
                    k[i]=scale*(i%17-8)/9.0f;
                    v[i]=(i%19-9)/10.0f;
                }
                vector<float> expected=attention_cpu(q, k, v, n, d, causal);
                float *d_q, *d_k, *d_v, *d_output;
                size_t bytes=n*d*sizeof(float);
                CUDA_CHECK(cudaMalloc(&d_q, bytes));
                CUDA_CHECK(cudaMalloc(&d_k, bytes));
                CUDA_CHECK(cudaMalloc(&d_v, bytes));
                CUDA_CHECK(cudaMalloc(&d_output, bytes));
                CUDA_CHECK(cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_k, k.data(), bytes, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice));
                launch(d_q, d_k, d_v, d_output, n, d, causal);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
                bool correct=true;
                float error=0.0f;
                for (int i=0; i<n*d; i++) {
                    if (!isfinite(output[i])) correct=false;
                    error=max(error, fabs(output[i]-expected[i]));
                }
                CUDA_CHECK(cudaFree(d_q));
                CUDA_CHECK(cudaFree(d_k));
                CUDA_CHECK(cudaFree(d_v));
                CUDA_CHECK(cudaFree(d_output));
                cout << "n=" << n << " d=" << d << " causal=" << causal
                     << " scale=" << scale << " error=" << error << "\n";
                if (!correct || error > 5e-4f) return 1;
            }
        }
    }
    return 0;
}

#endif

#include <bits/stdc++.h>
#include "../cuda_check.h"

// Include the existing kernels while giving their standalone demos unique names.
#define main vector_example_main
#include "../VectorAdd.cu"
#undef main
#define main matrix_add_example_main
#include "../MatrixAdd.cu"
#undef main
#define main matrix_mul_example_main
#include "../MatrixMul.cu"
#undef main
#define main tiled_example_main
#include "../Tile_Matmul.cu"
#undef main
#define main coarsed_example_main
#define Tiled_Matmul Coarsed_Matmul
#include "../Coarsed_Tile_matmul.cu"
#undef Tiled_Matmul
#undef main
#define main sum_example_main
#define tile reduction_tile
#include "../Sum.cu"
#undef tile
#undef main
#define main transpose_example_main
#define tile transpose_tile
#include "../Transpose.cu"
#undef tile
#undef main
#define main norm_example_main
#define tile norm_tile
#include "../DynamicBlock.cu"
#undef tile
#undef main
#define main gemm_example_main
#include "../gemm.cu"
#undef main
#define main lenet_example_main
#include "../lenet.cu"
#undef main
#define main conv_example_main
#include "../tile_conv.cu"
#undef main
#define main unrolled_example_main
#include "../unroliing_conv.cu"
#undef main
#define CUDA_KERNEL_TEST
#include "../Blur.cu"
#include "../GaussBlur.cu"
#undef CUDA_KERNEL_TEST
#define imgblur modular_imgblur
#include "../header_file/GBlur.cu"
#undef imgblur

using namespace std;

template<typename T>
T *upload(const vector<T> &data) {
    T *device;
    size_t bytes=max(size_t(1), data.size())*sizeof(T);
    CUDA_CHECK(cudaMalloc(&device, bytes));
    if (!data.empty()) {
        CUDA_CHECK(cudaMemcpy(device, data.data(), data.size()*sizeof(T), cudaMemcpyHostToDevice));
    }
    return device;
}

template<typename T>
vector<T> download(const T *device, size_t n) {
    vector<T> data(n);
    CUDA_CHECK(cudaMemcpy(data.data(), device, n*sizeof(T), cudaMemcpyDeviceToHost));
    return data;
}

template<typename T>
void check(const vector<T> &actual, const vector<T> &expected, const string &name, double tolerance=0.0) {
    for (size_t i=0; i<expected.size(); i++) {
        double value=actual[i], reference=expected[i];
        if (!isfinite(value) || fabs(value-reference)>tolerance*(1.0+fabs(reference))) {
            throw runtime_error(name+" mismatch at "+to_string(i)+": "+to_string(value)+
                                " expected "+to_string(reference));
        }
    }
}

void check_addition() {
    int n=19, m=35;
    vector<int> a(n*m), b(n*m), expected(n*m), initial(n*m, -999);
    for (int i=0; i<n*m; i++) {
        a[i]=i%17-8;
        b[i]=i%11-5;
        expected[i]=a[i]+b[i];
    }
    int *d_a=upload(a), *d_b=upload(b), *d_c=upload(initial);
    vecadd<<<(n*m+127)/128,128>>>(d_a, d_b, d_c, n*m);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_c,n*m), expected, "vector addition");
    CUDA_CHECK(cudaMemcpy(d_c, initial.data(), n*m*sizeof(int), cudaMemcpyHostToDevice));
    matadd<<<dim3((m+15)/16,(n+15)/16),dim3(16,16)>>>(d_a, d_b, d_c, n, m);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_c,n*m), expected, "matrix addition");
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

void check_reductions() {
    for (int n : {0, 1, 17, 33, 1003}) {
        for (int block : {1, 7, 16, 32, 33, 64, 127, 256}) {
            vector<int> a(n);
            for (int i=0; i<n; i++) a[i]=i%19-9;
            // This catches converting an integer sum through a float accumulator.
            if (n>0) a[0]=16777217;
            int expected=accumulate(a.begin(),a.end(),0);
            int *d_a=upload(a), *d_ans=upload(vector<int>{0});
            for (int variant=0; variant<3; variant++) {
                if (n>0) CUDA_CHECK(cudaMemcpy(d_a,a.data(),n*sizeof(int),cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_ans,0,sizeof(int)));
                int grid=max(1,(n+block-1)/block);
                if (variant==0) CudaSum<<<grid,block>>>(d_a,n,d_ans);
                else if (variant==1) Tile_CudaSum<<<grid,block,block*sizeof(int)>>>(d_a,n,d_ans);
                else Warp_Tile_CudaSum<<<grid,block,block*sizeof(int)>>>(d_a,n,d_ans);
                CUDA_CHECK(cudaGetLastError());
                check(download(d_ans,1),vector<int>{expected},"reduction "+to_string(variant));
            }
            CUDA_CHECK(cudaFree(d_a));
            CUDA_CHECK(cudaFree(d_ans));
        }
    }
}

void check_transpose() {
    for (auto shape : {pair<int,int>{7,5},{35,67},{65,3}}) {
        int n=shape.first, m=shape.second;
        vector<float> a(n*m), expected(n*m), initial(n*m, NAN);
        iota(a.begin(),a.end(),1.0f);
        for (int i=0; i<n; i++) {
            for (int j=0; j<m; j++) expected[j*n+i]=a[i*m+j];
        }
        float *d_a=upload(a), *d_b=upload(initial);
        for (int block : {8,16,32}) {
            CUDA_CHECK(cudaMemcpy(d_b,initial.data(),n*m*sizeof(float),cudaMemcpyHostToDevice));
            Transpose<<<dim3((m+block-1)/block,(n+block-1)/block),dim3(block,block),
                        block*(block+1)*sizeof(float)>>>(d_a,d_b,n,m);
            CUDA_CHECK(cudaGetLastError());
            check(download(d_b,n*m),expected,"transpose");
        }
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
    }
}

void check_matmul() {
    for (auto shape : {array<int,3>{1,1,1},{2,5,3},{3,0,5},{17,19,67}}) {
        int n=shape[0], k=shape[1], m=shape[2];
        vector<int> a(n*k), b(k*m), expected(n*m,0);
        for (int i=0; i<n*k; i++) a[i]=i%11-5;
        for (int i=0; i<k*m; i++) b[i]=i%7-3;
        for (int i=0; i<n; i++) {
            for (int j=0; j<m; j++) {
                for (int x=0; x<k; x++) expected[i*m+j]+=a[i*k+x]*b[x*m+j];
            }
        }
        int *d_a=upload(a), *d_b=upload(b), *d_c=upload(vector<int>(n*m,-999));
        matrixmul<<<dim3((m+15)/16,(n+15)/16),dim3(16,16)>>>(d_a,d_b,d_c,n,k,m);
        CUDA_CHECK(cudaGetLastError());
        check(download(d_c,n*m),expected,"naive matmul");
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));

        vector<float> fa(a.begin(),a.end()), fb(b.begin(),b.end()), reference(expected.begin(),expected.end());
        float *f_a=upload(fa), *f_b=upload(fb), *f_c=upload(vector<float>(n*m,NAN));
        Tiled_Matmul<<<dim3((m+t_col-1)/t_col,(n+t_row-1)/t_row),dim3(t_col,t_row),
                       t_k*t_col*sizeof(float)>>>(f_a,f_b,f_c,n,k,m);
        CUDA_CHECK(cudaGetLastError());
        check(download(f_c,n*m),reference,"tiled matmul");
        vector<float> initial(n*m,NAN);
        CUDA_CHECK(cudaMemcpy(f_c,initial.data(),n*m*sizeof(float),cudaMemcpyHostToDevice));
        Coarsed_Matmul<<<dim3((m+t_col*coarse_factor-1)/(t_col*coarse_factor),(n+t_row-1)/t_row),
                         dim3(t_col,t_row),t_k*t_col*sizeof(float)>>>(f_a,f_b,f_c,n,k,m);
        CUDA_CHECK(cudaGetLastError());
        check(download(f_c,n*m),reference,"coarsened matmul");
        for (float beta : {0.0f,0.25f,-0.75f}) {
            vector<float> before(n*m,beta==0.0f ? NAN : 2.0f), result(n*m);
            CUDA_CHECK(cudaMemcpy(f_c,before.data(),n*m*sizeof(float),cudaMemcpyHostToDevice));
            gemm<<<dim3((m+15)/16,(n+15)/16),dim3(16,16)>>>(f_a,f_b,f_c,n,k,m,0.5f,beta);
            CUDA_CHECK(cudaGetLastError());
            for (int i=0; i<n*m; i++) result[i]=0.5f*reference[i]+(beta==0.0f ? 0.0f : beta*before[i]);
            check(download(f_c,n*m),result,"GEMM",1e-5);
        }
        CUDA_CHECK(cudaFree(f_a));
        CUDA_CHECK(cudaFree(f_b));
        CUDA_CHECK(cudaFree(f_c));
    }
}

void check_norm() {
    for (int n : {1,17,1003}) {
        vector<double> a(n);
        double expected=0.0;
        for (int i=0; i<n; i++) {
            a[i]=(i%31-15)*1.25;
            expected=max(expected,fabs(a[i]));
        }
        double *d_a=upload(a), *partial=upload(vector<double>(7,NAN)), *answer=upload(vector<double>(1,NAN));
        L_InfNorm<<<7,33,33*sizeof(double)>>>(d_a,n,partial);
        CUDA_CHECK(cudaGetLastError());
        L_InfNorm<<<1,33,33*sizeof(double)>>>(partial,7,answer);
        CUDA_CHECK(cudaGetLastError());
        check(download(answer,1),vector<double>{expected},"infinity norm");
        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(partial));
        CUDA_CHECK(cudaFree(answer));
    }
}

void check_blur() {
    int h=5, w=7, radius=2;
    vector<float> filter(25);
    float total=0.0f;
    for (int i=-radius; i<=radius; i++) {
        for (int j=-radius; j<=radius; j++) total+=(filter[(i+radius)*5+j+radius]=expf(-(i*i+j*j)/2.0f));
    }
    for (float &x : filter) x/=total;
    float *d_filter=upload(filter);
    for (bool constant : {false,true}) {
        vector<unsigned char> image(h*w), box(h*w), gaussian(h*w);
        for (int i=0; i<h*w; i++) image[i]=constant ? 127 : (i*17)%256;
        for (int r=0; r<h; r++) {
            for (int c=0; c<w; c++) {
                int sum=0, count=0;
                double weighted=0.0, weights=0.0;
                for (int i=-radius; i<=radius; i++) {
                    for (int j=-radius; j<=radius; j++) {
                        if (r+i<0 || r+i>=h || c+j<0 || c+j>=w) continue;
                        int value=image[(r+i)*w+c+j];
                        sum+=value;
                        count++;
                        double weight=filter[(i+radius)*5+j+radius];
                        weighted+=weight*value;
                        weights+=weight;
                    }
                }
                box[r*w+c]=sum/count;
                gaussian[r*w+c]=floor(weighted/weights+0.5);
            }
        }
        unsigned char *input=upload(image), *output=upload(vector<unsigned char>(h*w,0));
        dim3 grid((w+15)/16,(h+15)/16), block(16,16);
        imgblur<<<grid,block>>>(input,output,radius,h,w);
        CUDA_CHECK(cudaGetLastError());
        check(download(output,h*w),box,"box blur");
        imgblur<<<grid,block>>>(input,output,d_filter,radius,h,w);
        CUDA_CHECK(cudaGetLastError());
        // Allow a one-level rounding difference between float GPU and double CPU.
        vector<unsigned char> actual=download(output,h*w);
        for (int i=0; i<h*w; i++) {
            if (abs(int(actual[i])-int(gaussian[i]))>(constant ? 0 : 1)) throw runtime_error("Gaussian blur");
        }
        CUDA_CHECK(cudaMemset(output,0,h*w*sizeof(unsigned char)));
        modular_imgblur<<<grid,block>>>(input,output,d_filter,radius,h,w);
        CUDA_CHECK(cudaGetLastError());
        check(download(output,h*w),actual,"modular Gaussian blur");
        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
    }
    CUDA_CHECK(cudaFree(d_filter));
}

void check_lenet_layers() {
    int in_channels=2, out_channels=3, size=7, kernel=3, out_size=5;
    vector<float> input(in_channels*size*size), weights(out_channels*in_channels*kernel*kernel), bias(out_channels);
    vector<float> expected(out_channels*out_size*out_size);
    for (size_t i=0; i<input.size(); i++) input[i]=(int(i)%13-6)/7.0f;
    for (size_t i=0; i<weights.size(); i++) weights[i]=(int(i)%11-5)/9.0f;
    for (int i=0; i<out_channels; i++) bias[i]=(i-1)*0.1f;
    for (int ch=0; ch<out_channels; ch++) {
        for (int r=0; r<out_size; r++) {
            for (int c=0; c<out_size; c++) {
                double sum=bias[ch];
                for (int x=0; x<in_channels; x++) {
                    for (int i=0; i<kernel; i++) {
                        for (int j=0; j<kernel; j++) {
                            sum+=double(input[(x*size+r+i)*size+c+j])*weights[((ch*in_channels+x)*kernel+i)*kernel+j];
                        }
                    }
                }
                expected[(ch*out_size+r)*out_size+c]=sum;
            }
        }
    }
    float *d_input=upload(input), *d_weights=upload(weights), *d_bias=upload(bias);
    float *d_output=upload(vector<float>(expected.size(),NAN));
    conv2d<<<dim3(out_channels,2,2),dim3(4,4)>>>(d_input,d_output,d_weights,d_bias,
                                                              in_channels,out_channels,size,kernel);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,expected.size()),expected,"LeNet convolution",1e-5);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    // Odd input sizes, trainable channel scale/bias, and actual average pooling.
    size=5;
    out_size=size/2;
    input.resize(out_channels*size*size);
    iota(input.begin(),input.end(),-10.0f);
    weights={1.0f,0.5f,-1.0f};
    bias={0.1f,-0.2f,0.3f};
    expected.assign(out_channels*out_size*out_size,0.0f);
    for (int ch=0; ch<out_channels; ch++) {
        for (int r=0; r<out_size; r++) {
            for (int c=0; c<out_size; c++) {
                float sum=0.0f;
                for (int i=0; i<2; i++) {
                    for (int j=0; j<2; j++) sum+=input[(ch*size+2*r+i)*size+2*c+j];
                }
                expected[(ch*out_size+r)*out_size+c]=0.25f*sum*weights[ch]+bias[ch];
            }
        }
    }
    d_input=upload(input);
    d_weights=upload(weights);
    d_bias=upload(bias);
    d_output=upload(vector<float>(expected.size(),NAN));
    avg_pool<<<dim3(out_channels,1,1),dim3(4,4)>>>(d_input,d_output,d_weights,d_bias,out_channels,size);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,expected.size()),expected,"LeNet average pooling",1e-6);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));

    input={-2.0f,1.0f,3.0f,0.5f};
    weights={1,2,3,4, -1,0,1,2, 0.5f,-0.5f,1,-1};
    bias={0.1f,0.2f,0.3f};
    expected=bias;
    for (int i=0; i<3; i++) {
        for (int j=0; j<4; j++) expected[i]+=input[j]*weights[i*4+j];
    }
    d_input=upload(input);
    d_weights=upload(weights);
    d_bias=upload(bias);
    d_output=upload(vector<float>(3,NAN));
    fully_connected<<<1,32>>>(d_input,d_output,d_weights,d_bias,4,3);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,3),expected,"LeNet fully connected",1e-6);
    tanh_activation<<<1,32>>>(d_output,3);
    CUDA_CHECK(cudaGetLastError());
    for (float &value : expected) value=tanhf(value);
    check(download(d_output,3),expected,"LeNet activation",1e-6);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
}

void check_convolution() {
    // Asymmetric coefficients catch swapped axes and unnormalized filters.
    int width=9, height=7, outwidth=width-2, outheight=height-2;
    vector<float> input(width*height), expected(outwidth*outheight,0.0f);
    float filter[3][3]={{-1,2,0.5f},{3,-2,4},{1,0,-3}};
    for (int i=0; i<width*height; i++) input[i]=(i%17-8)*0.25f;
    for (int r=0; r<outheight; r++) {
        for (int c=0; c<outwidth; c++) {
            for (int i=0; i<3; i++) {
                for (int j=0; j<3; j++) expected[r*outwidth+c]+=input[(r+i)*width+c+j]*filter[i][j];
            }
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(kfilter_d,filter,sizeof(filter)));
    float *d_input=upload(input), *d_output=upload(vector<float>(expected.size(),NAN));
    conv2dtiledconstkernel<<<dim3((outwidth+1)/2,(outheight+1)/2),dim3(4,4)>>>(
        d_input,d_output,width,height,outwidth,outheight);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,expected.size()),expected,"tiled convolution",1e-5);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    input.resize(CIN*H*W);
    vector<float> unrolled_filter(N);
    expected.assign(OH*OW,0.0f);
    for (int i=0; i<CIN*H*W; i++) input[i]=(i%17-8)*0.25f;
    for (int i=0; i<N; i++) unrolled_filter[i]=(i%7-3)*0.5f;
    for (int r=0; r<OH; r++) {
        for (int c=0; c<OW; c++) {
            for (int ch=0; ch<CIN; ch++) {
                for (int i=0; i<KH; i++) {
                    for (int j=0; j<KW; j++) {
                        expected[r*OW+c]+=input[ch*H*W+(r+i)*W+c+j]*unrolled_filter[ch*KH*KW+i*KW+j];
                    }
                }
            }
        }
    }
    d_input=upload(input);
    float *d_filter=upload(unrolled_filter);
    d_output=upload(vector<float>(expected.size(),NAN));
    conv_thread<<<1,128>>>(d_input,d_filter,d_output);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,expected.size()),expected,"thread convolution",1e-5);
    vector<float> initial(expected.size(),NAN);
    CUDA_CHECK(cudaMemcpy(d_output,initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice));
    conv_block<<<OH*OW,BLOCK>>>(d_input,d_filter,d_output);
    CUDA_CHECK(cudaGetLastError());
    check(download(d_output,expected.size()),expected,"block convolution",1e-5);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    try {
        check_addition();
        check_reductions();
        check_transpose();
        check_matmul();
        check_norm();
        check_blur();
        check_lenet_layers();
        check_convolution();
        cout << "All kernel regression checks passed.\n";
        return 0;
    }
    catch (const exception &error) {
        cerr << error.what() << "\n";
        return 1;
    }
}

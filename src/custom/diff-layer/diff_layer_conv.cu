#include <cmath>
#include <iostream>
#include "gpu_diff_layer.h"
#define TILE_WIDTH1 16
#define TILE_WIDTH2 24
__constant__ float Mask[6000];

__global__ void conv_forward_kernel1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // int numblock_eachcolumn = (H_out - 1)/TILE_WIDTH + 1;
    int numblock_eachrow = (W_out - 1)/TILE_WIDTH1 + 1;
    int w_out = TILE_WIDTH1 * (bz % numblock_eachrow) + tx;
    int h_out = TILE_WIDTH1 * (bz/numblock_eachrow) + ty;
    int b_out = bx;
    int m_out = by;

    if (h_out < H_out && w_out < W_out)
    {
        float result = 0;
        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    result += x4d(b_out, c, h_out + p, w_out + q) * k4d(m_out, c, p, q);
                }
            }
        }
        y4d(b_out, m_out, h_out, w_out) = result;
    }
    





#undef y4d
#undef x4d
#undef k4d
}

__global__ void conv_forward_kernel2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // int numblock_eachcolumn = (H_out - 1)/TILE_WIDTH + 1;
    int numblock_eachrow = (W_out - 1)/TILE_WIDTH2 + 1;
    int w_out = TILE_WIDTH2 * (bz % numblock_eachrow) + tx;
    int h_out = TILE_WIDTH2 * (bz/numblock_eachrow) + ty;
    int b_out = bx;
    int m_out = by;

    if (h_out < H_out && w_out < W_out)
    {
        float result = 0;
        for (int c = 0; c < C; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    result += x4d(b_out, c, h_out + p, w_out + q) * k4d(m_out, c, p, q);
                }
            }
        }
        y4d(b_out, m_out, h_out, w_out) = result;
    }
    





#undef y4d
#undef x4d
#undef k4d
}

	
__host__ void GPUDiffLayerSizeInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{

    cudaMalloc((void **) device_y_ptr, (B * M * (H - K + 1) * (W - K + 1))*sizeof(float));
    cudaMalloc((void **) device_x_ptr, (B * C * H * W)*sizeof(float));
    //cudaMalloc((void **) device_k_ptr, (M * C * K * K)*sizeof(float));
    cudaMemcpy(*device_x_ptr, host_x, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(*device_k_ptr, host_k, (M * C * K * K)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_k, (M * C * K * K)*sizeof(float));

}


__host__ void GPUDiffLayerSizeInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    if(M/24){
        dim3 dimBlock(TILE_WIDTH2, TILE_WIDTH2, 1);
        dim3 dimGrid(B, M, ceil((float)(H - K + 1)/TILE_WIDTH2)*ceil((float)(W - K + 1)/TILE_WIDTH2));
        conv_forward_kernel2<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    }else{
        dim3 dimGrid(B, M, ceil((float)(H - K + 1)/TILE_WIDTH1)*ceil((float)(W - K + 1)/TILE_WIDTH1));
        dim3 dimBlock(TILE_WIDTH1, TILE_WIDTH1, 1);
        conv_forward_kernel1<<<dimGrid, dimBlock>>>(device_y, device_x, device_k, B, M, C, H, W, K);
    }
}


__host__ void GPUDiffLayerSizeInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, (B * M * (H - K + 1) * (W - K + 1))*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    //cudaFree(device_k);
}


__host__ void GPUDiffLayerSizeInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
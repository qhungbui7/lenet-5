#include <cmath>
#include <iostream>
#include "gpu_reduction_tree.h"
#define TILE_WIDTH 16
__constant__ float Mask[6000];

using namespace std;


#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

__host__ void GPUReductionTreeInterface::get_device_properties() {
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");
}
__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    extern __shared__ float SM[];

    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define k4d(i3, i2, i1, i0) Mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define tree(i2, i1, i0) SM[i2 * TILE_WIDTH * C + i1 * C + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    // int numblock_eachcolumn = (H_out - 1)/TILE_WIDTH + 1;
    int numblock_eachrow = (W_out - 1)/TILE_WIDTH + 1;
    int w_out = TILE_WIDTH * (bz % numblock_eachrow) + tx;
    int h_out = TILE_WIDTH * (bz/numblock_eachrow) + ty;
    int b_out = bx;
    int m_out = by;


    if (h_out < H_out && w_out < W_out)
    {
        float result = 0;

        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                result += x4d(b_out, tz, h_out + p, w_out + q) * k4d(m_out, tz, p, q);
            }
        }
        tree(ty, tx, tz) = result;
        

        // tree reduction
        for (int stride = 1; stride < C; stride *= 2)
        {
            __syncthreads();
            if ((tz % (2 * stride) == 0) && (tz + stride < C))
            {
                    tree(ty, tx, tz) += tree(ty, tx, tz + stride);
                
            }
        }
        __syncthreads();
        if (tz == 0)
        {
            y4d(b_out, m_out, h_out, w_out) = tree(ty, tx, 0);
        }

    }


#undef y4d
#undef x4d
#undef k4d
#undef sm2d
}

	
__host__ void GPUReductionTreeInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{

    CHECK(cudaMalloc((void **) device_y_ptr, (B * M * (H - K + 1) * (W - K + 1))*sizeof(float)));
    CHECK(cudaMalloc((void **) device_x_ptr, (B * C * H * W)*sizeof(float)));
    CHECK(cudaMemcpy(*device_x_ptr, host_x, (B * C * H * W)*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(Mask, host_k, (M * C * K * K)*sizeof(float)));

}


__host__ void GPUReductionTreeInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(B, M, ceil((float)(H - K + 1)/TILE_WIDTH)*ceil((float)(W - K + 1)/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, C);
    conv_forward_kernel<<<dimGrid, dimBlock, TILE_WIDTH * TILE_WIDTH * C * sizeof(float)>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}


__host__ void GPUReductionTreeInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    CHECK(cudaMemcpy(host_y, device_y, (B * M * (H - K + 1) * (W - K + 1))*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(device_y));
    CHECK(cudaFree(device_x));
}


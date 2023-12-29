#include <cmath>
#include <stdio.h>
#include <iostream>
#include "gpu_constant_mem.h"

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


#define M_MAX   16
#define C_MAX   4
#define K_MAX   7
__constant__ float kernel[M_MAX * C_MAX * K_MAX * K_MAX];

float kernel_flatten(float* kernel, int i3, int i2, int i1, int i0, int C, int K){
  return kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0];
}

float input_flatten(float* x, int i3, int i2, int i1, int i0, int C, int H, int W){
  return  x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0];
}

void output_flatten(float* y, int i3, int i2, int i1, int i0, int H_out, int W_out, int M, int sum){
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0] = sum; 
}


__global__ void conv_forward_kernel(float* y,
                                    const float* x,
                                    const int B,
                                    const int M,
                                    const int C,
                                    const int H,
                                    const int W,
                                    const int K) {
  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire
  mini-batch The goal here is to be correct AND fast.

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

  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  const int w = blockIdx.x * blockDim.x + threadIdx.x;

  if ((h < H_out) && (w < W_out)) {
    // Naive GPU convolution kernel code
    for (int b = 0; b < B; b++) {
      for (int m = 0; m < M; m++) {
        
        float sum = 0;

        for (int c = 0; c < C; c++) {
          for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
              sum += input_flatten(x, b, c, h + p, w + q, C, H, W) * kernel_flatten(kernel, m, c, p, q, C, K);
            }
          }
        }
        output_flatten(y, b, m, h, w, H_out, W_out, M, sum);
      }
    }
  }

}

__host__ void GPUConstantMemInterface::conv_forward_gpu_prolog(const float* host_y,
                                                    const float* host_x,
                                                    const float* host_k,
                                                    float** device_y_ptr,
                                                    float** device_x_ptr,
                                                    float** device_k_ptr,
                                                    const int B,
                                                    const int M,
                                                    const int C,
                                                    const int H,
                                                    const int W,
                                                    const int K) {

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);

  // Allocate memory
  CHECK(cudaMalloc(device_y_ptr, bytes_y));
  CHECK(cudaMalloc(device_x_ptr, bytes_x));

  // Copy over the relevant data structures to the GPU
  CHECK(cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(kernel, host_k, bytes_k));
}

__host__ void GPUConstantMemInterface::conv_forward_gpu(float* device_y,
                                             const float* device_x,
                                             const float* device_k,
                                             const int B,
                                             const int M,
                                             const int C,
                                             const int H,
                                             const int W,
                                             const int K) {
  // Set the kernel dimensions
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  dim3 dim_block(32, 32);
  dim3 dim_grid(ceil((float)W_out / dim_block.x), ceil((float)H_out / dim_block.y));

  conv_forward_kernel<<<dim_grid, dim_block>>>(device_y, device_x,
                                               B, M, C, H, W, K);
  CHECK(cudaDeviceSynchronize());
}

__host__ void GPUConstantMemInterface::conv_forward_gpu_epilog(float* host_y,
                                                    float* device_y,
                                                    float* device_x,
                                                    float* device_k,
                                                    const int B,
                                                    const int M,
                                                    const int C,
                                                    const int H,
                                                    const int W,
                                                    const int K) {
  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);

  CHECK(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  CHECK(cudaFree(device_y));
  CHECK(cudaFree(device_x));
}

__host__ void GPUConstantMemInterface::get_device_properties() {
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
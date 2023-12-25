#include <cmath>
#include <iostream>
#include "gpu_constant_mem.h"

#define cudaErrChk(stmt) \
  { cudaAssert((stmt), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t error,
                       const char* file,
                       int line,
                       bool abort = true) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << ' ' << file
              << ':' << line << std::endl;
    if (abort) {
      exit(error);
    }
  }
}

#define M_MAX   16
#define C_MAX   4
#define K_MAX   7
__constant__ float kernel[M_MAX * C_MAX * K_MAX * K_MAX];

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

#define y4d(i3, i2, i1, i0) \
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) \
  x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) \
  kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

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
              sum += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
            }
          }
        }
        y4d(b, m, h, w) = sum;
      }
    }
  }

#undef y4d
#undef x4d
#undef k4d
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
  std::cout << "*** constant memory ***" << std::endl;

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  const size_t bytes_y = (B * M * H_out * W_out) * sizeof(float);
  const size_t bytes_x = (B * C * H * W) * sizeof(float);
  const size_t bytes_k = (M * C * K * K) * sizeof(float);

  // Allocate memory
  cudaErrChk(cudaMalloc(device_y_ptr, bytes_y));
  cudaErrChk(cudaMalloc(device_x_ptr, bytes_x));

  // Copy over the relevant data structures to the GPU
  cudaErrChk(
      cudaMemcpy(*device_x_ptr, host_x, bytes_x, cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpyToSymbol(kernel, host_k, bytes_k));
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

  // Call the kernel
  conv_forward_kernel<<<dim_grid, dim_block>>>(device_y, device_x,
                                               B, M, C, H, W, K);
  cudaErrChk(cudaDeviceSynchronize());
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

  // Copy the output back to host
  cudaErrChk(cudaMemcpy(host_y, device_y, bytes_y, cudaMemcpyDeviceToHost));

  // Free device memory
  cudaErrChk(cudaFree(device_y));
  cudaErrChk(cudaFree(device_x));
}

__host__ void GPUConstantMemInterface::get_device_properties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational capabilities: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem
              << std::endl;
    std::cout << "Max Constant memory size: " << deviceProp.totalConstMem
              << std::endl;
    std::cout << "Max Shared memory size per block: "
              << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0]
              << " x, " << deviceProp.maxThreadsDim[1] << " y, "
              << deviceProp.maxThreadsDim[2] << " z" << std::endl;
    std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
              << deviceProp.maxGridSize[1] << " y, "
              << deviceProp.maxGridSize[2] << " z" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
  }
}
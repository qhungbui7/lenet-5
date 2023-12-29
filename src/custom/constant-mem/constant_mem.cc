#include "constant_mem.h"
#include <math.h>
#include <iostream>
using namespace std;

void ConstantMem::initializeWeight()
{
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out = (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void ConstantMem::forward(const Matrix &bottom)
{
  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  float *x = (float *)bottom.data();
  float *y = (float *)top.data();
  float *k = (float *)weight.data();
  float *b = (float *)bias.data();

  const int B = n_sample;
  const int M = channel_out;
  const int C = channel_in;
  const int K = height_kernel;

  float *x_d;
  float *y_d;
  float *k_d;

  // Launch marker kernel to aid with student function timing
  // gpuUtils.insert_pre_barrier_kernel();

  // Data transfer CPU to GPU
  gpuInterface.conv_forward_gpu_prolog(y, x, k, &y_d, &x_d, &k_d, B, M, C, height_in, width_in, K);

  // Hand off to GPU for computation
  gpuInterface.conv_forward_gpu(y_d, x_d, k_d, B, M, C, height_in, width_in, K);
  cudaDeviceSynchronize();

  // Data transfer GPU to CPU
  gpuInterface.conv_forward_gpu_epilog(y, y_d, x_d, k_d, B, M, C, height_in, width_in, K);

  // Launch barrier kernel to aid with timing with nsight-compute
  // gpuUtils.insert_post_barrier_kernel();
}


vector<float> ConstantMem::get_parameters() const
{
  vector<float> res(weight.size() + bias.size());
  copy(weight.data(), weight.data() + weight.size(), res.begin());
  copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void ConstantMem::set_parameters(const std::vector<float> &param)
{
  copy(param.begin(), param.begin() + weight.size(), weight.data());
  copy(param.begin() + weight.size(), param.end(), bias.data());
}

vector<float> ConstantMem::get_derivatives() const
{
  vector<float> res(grad_weight.size() + grad_bias.size());
  copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  copy(grad_bias.data(), grad_bias.data() + grad_bias.size(), res.begin() + grad_weight.size());
  return res;
}


void ConstantMem::backward(const Matrix &bottom, const Matrix &grad_top){}

void ConstantMem::update(Optimizer &opt){}
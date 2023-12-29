#ifndef SRC_GPU_RESTRICT_UNROLL_H_
#define SRC_GPU_RESTRICT_UNROLL_H_
class GPUConstantMemInterface
{
public:
    void device_info();
    void get_data_from_gpu(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K);
    void forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K);
    void cpy_data_to_gpu(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K);
};

#endif
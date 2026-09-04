// RHIZOME CUDA kernels — decode GEMV (mx-dequantizing).
// Target: >= 70% of HBM bandwidth on H100 for decode GEMV (T10).
#include <cuda_runtime.h>

// y[m] = sum_k W[m,k] * x[k], W row-major, fp32. One warp per row; the
// k-loop order is fixed for determinism (R9).
extern "C" __global__ void gemv_f32(const float* __restrict__ w,
                                    const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int m,
                                    int k) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp_n = threadIdx.x >> 5;
    const float* wr = w + (long)row * k;
    float acc = 0.0f;
    // 8 warps stride the k dimension; lane-major order keeps per-lane
    // accumulation order independent of the batch (batch invariance).
    for (int i = warp_n * 32 + lane; i < k; i += 256) {
        acc += wr[i] * x[i];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) atomicAdd(&y[row], acc);
}

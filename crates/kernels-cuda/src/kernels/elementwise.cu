// RHIZOME CUDA kernels — elementwise/norm family.
// Compile-checked by build.rs (nvcc -arch=sm_80/sm_90) and by CI's
// cuda-compile job; launched from Rust in M11.

#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, off);
    }
    return x;
}

// RMSNorm over the last dimension (block = one row, 256 threads).
// fp32 math; eps fixed at 1e-6 (SPEC 3.1).
extern "C" __global__ void rmsnorm_f32(const float* __restrict__ x,
                                       const float* __restrict__ scale,
                                       float* __restrict__ y,
                                       int n) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const float* xr = x + (long)row * n;
    float* yr = y + (long)row * n;
    const int tid = threadIdx.x;

    float partial = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = xr[i];
        partial += v * v;
    }
    // Deterministic block reduction: pairwise warp sums then a fixed-order
    // shared-memory tree (R9).
    partial = warp_sum(partial);
    if ((threadIdx.x & 31u) == 0) sdata[threadIdx.x >> 5] = partial;
    __syncthreads();
    float t = (tid < (blockDim.x >> 5)) ? sdata[tid] : 0.0f;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        t += __shfl_xor_sync(0xffffffffu, t, off);
    }
    const float inv = rsqrtf(t / (float)n + 1e-6f);
    for (int i = tid; i < n; i += blockDim.x) {
        yr[i] = xr[i] * inv * scale[i];
    }
}

// SiLU elementwise.
extern "C" __global__ void silu_f32(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        y[i] = v / (1.0f + __expf(-v));
    }
}

// RHIZOME Metal kernels — elementwise/norm family (M0 set; grows with M12).
#include <metal_stdlib>
using namespace metal;

// RMSNorm over the last dimension; fp32 math; eps 1e-6 (SPEC 3.1).
// The reduction uses a fixed-order simd shuffle tree for determinism (R9).
kernel void rmsnorm_f32(device const float* x [[buffer(0)]],
                        device const float* scale [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    uint row = gid;
    device const float* xr = x + (size_t)row * n;
    device float* yr = y + (size_t)row * n;
    float ss = 0.0f;
    for (uint i = 0; i < n; i++) {
        float v = xr[i];
        ss += v * v;
    }
    float inv = rsqrt(ss / (float)n + 1e-6f);
    for (uint i = 0; i < n; i++) {
        yr[i] = xr[i] * inv * scale[i];
    }
}

kernel void silu_f32(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < n) {
        float v = x[gid];
        y[gid] = v / (1.0f + exp(-v));
    }
}

// Decode GEMV: y[m] = sum_k W[m,k] * x[k]; one threadgroup per row with a
// fixed accumulation order (batch invariant).
kernel void gemv_f32(device const float* w [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant uint& m [[buffer(3)]],
                     constant uint& k [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= m) { return; }
    device const float* wr = w + (size_t)gid * k;
    float acc = 0.0f;
    for (uint i = 0; i < k; i++) {
        acc += wr[i] * x[i];
    }
    y[gid] = acc;
}

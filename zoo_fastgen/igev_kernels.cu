// Small size-independent CUDA kernels for IGEV-Stereo inference (NHWC fp16). Compiled once to a cubin
// (core/kernels.py), launched on torch's current stream.
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

__device__ __forceinline__ float act_f(float v, int mode) {
    switch (mode) {
        case 1: return fmaxf(v, 0.f);                      // relu
        case 2: return v >= 0.f ? v : 0.01f * v;           // LeakyReLU(0.01)
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + __expf(-v));
        case 5: return fminf(fmaxf(v, 0.f), 6.f);          // relu6
        default: return v;
    }
}

// y[p, c] = act(y[p, c] * s[c] + t[c] (+ r[p, c]))   (NHWC rows; s / t / r may be null). In place.
extern "C" __global__ void affine_act(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                      const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    GRID_STRIDE(i, P * C) {
        long long p = i / C;
        int c = (int)(i - p * C);
        float v = __half2float(y[p * ystride + c]);
        if (s) v *= s[c];
        if (t) v += t[c];
        if (r) v += __half2float(r[p * rstride + c]);
        y[p * ystride + c] = __float2half(act_f(v, mode));
    }
}

// 8 channels per thread, 16-byte accesses (C, strides multiple of 8, 16-byte aligned base pointers)
extern "C" __global__ void affine_act8(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                       const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 8;
        uint4 raw = *reinterpret_cast<const uint4*>(y + p * ystride + c0);
        __half* hv = reinterpret_cast<__half*>(&raw);
        uint4 rraw;
        __half* rv = reinterpret_cast<__half*>(&rraw);
        if (r) rraw = *reinterpret_cast<const uint4*>(r + p * rstride + c0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(hv[k]);
            if (s) v *= s[c0 + k];
            if (t) v += t[c0 + k];
            if (r) v += __half2float(rv[k]);
            hv[k] = __float2half(act_f(v, mode));
        }
        *reinterpret_cast<uint4*>(y + p * ystride + c0) = raw;
    }
}

// Combined geometry encoding volume lookup (IGEV), 2 pyramid levels, both volumes, = grid_sample bilinear
// (align_corners=True, zero padding) along the disparity / x axis. Only level 0 is stored (fp16); level-1 values
// are upstream's avg_pool(1x2) computed on the fly in fp32: v1[j] = (v0[2j] + v0[2j+1]) / 2, j < D0/2.
// One thread per (pixel, row): row jj = l * (C + 1) + j; j < C: geometry channel j, j == C: correlation.
// out [P, 2*(C+1)*K] fp16, channel order per level: geo (c*K + k), then corr (k)   (== upstream's cat/permute)
__device__ __forceinline__ float lvl_val(const __half* v, int i, int n, int l) {
    if (l == 0) return (i >= 0 && i < n) ? __half2float(v[i]) : 0.f;
    return (i >= 0 && i < n / 2) ? 0.5f * (__half2float(v[2 * i]) + __half2float(v[2 * i + 1])) : 0.f;
}

extern "C" __global__ void geo_lookup(const __half* __restrict__ g0, int C, int D0, const __half* __restrict__ c0,
                                      int W0, int Wimg, const float* __restrict__ disp, __half* __restrict__ out,
                                      long long P, int r) {
    const int K = 2 * r + 1;
    const int rows = 2 * (C + 1);
    GRID_STRIDE(i, P * rows) {
        long long p = i / rows;
        int jj = (int)(i - p * rows);
        int l = jj / (C + 1), j = jj - l * (C + 1);
        float sc = l ? 0.5f : 1.f;
        float d = disp[p] * sc;
        __half* o = out + p * (2 * (C + 1) * K) + l * (C + 1) * K + j * K;
        const __half* v;
        int n;
        float base;
        if (j < C) { v = g0 + (p * C + j) * D0; n = D0; base = d; }
        else { v = c0 + p * W0; n = W0; base = (float)(p % Wimg) * sc - d; }
        for (int k = 0; k < K; ++k) {
            float x = base + (float)(k - r);
            float xf = floorf(x);
            int x0 = (int)xf;
            float w = x - xf;
            o[k] = __float2half(lvl_val(v, x0, n, l) * (1.f - w) + lvl_val(v, x0 + 1, n, l) * w);
        }
    }
}

// ConvGRU gates. zr [P, 2*Hd] = conv([h|x]) without bias; czr [P, 2*Hd] = context + conv biases.
// z = sigmoid(zr[:, :Hd] + czr[:, :Hd]) stored back into zr; r = sigmoid(...); rh[p, c] = r * h  (rh: NHWC slice)
extern "C" __global__ void gru_zr(__half* zr, const __half* __restrict__ czr, const __half* __restrict__ h, int hstride,
                                  __half* rh, int rhstride, long long P, int Hd) {
    GRID_STRIDE(i, P * Hd) {
        long long p = i / Hd;
        int c = (int)(i - p * Hd);
        long long b = p * 2 * Hd;
        float z = 1.f / (1.f + __expf(-(__half2float(zr[b + c]) + __half2float(czr[b + c]))));
        float rr = 1.f / (1.f + __expf(-(__half2float(zr[b + Hd + c]) + __half2float(czr[b + Hd + c]))));
        zr[b + c] = __float2half(z);
        rh[p * rhstride + c] = __float2half(rr * __half2float(h[p * hstride + c]));
    }
}

// h = (1 - z) * h + z * tanh(q + cq), in place on h
extern "C" __global__ void gru_update(const __half* __restrict__ q, const __half* __restrict__ cq,
                                      const __half* __restrict__ zr, __half* h, int hstride, long long P, int Hd) {
    GRID_STRIDE(i, P * Hd) {
        long long p = i / Hd;
        int c = (int)(i - p * Hd);
        float z = __half2float(zr[p * 2 * Hd + c]);
        float qq = tanhf(__half2float(q[i]) + __half2float(cq[i]));
        float hh = __half2float(h[p * hstride + c]);
        h[p * hstride + c] = __float2half((1.f - z) * hh + z * qq);
    }
}

// ---- same formulas as PyTorch, NHWC slices, C multiple of 8
// avg_pool2d(kernel 3, stride 2, padding 1, count_include_pad) of a C-channel slice into another slice.
__device__ __forceinline__ void pool2x_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            int sy = 2 * y - 1 + dy;
            if (sy < 0 || sy >= Hs) continue;
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                int sx = 2 * x - 1 + dx;
                if (sx < 0 || sx >= Ws) continue;
                uint4 v = *reinterpret_cast<const uint4*>(src + ((long long)sy * Ws + sx) * sstride + c);
                const __half* h = reinterpret_cast<const __half*>(&v);
#pragma unroll
                for (int k = 0; k < 8; ++k) s[k] += __half2float(h[k]);
            }
        }
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(s[k] / 9.f);
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void pool2x_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    pool2x_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}

// Bilinear upsampling with align_corners=True (PyTorch formula) of a C-channel slice into another slice.
__device__ __forceinline__ void interp_body(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                            int dstride, __half* dst2, int Hd, int Wd, int C) {
    float rh = Hd > 1 ? (float)(Hs - 1) / (float)(Hd - 1) : 0.f;
    float rw = Wd > 1 ? (float)(Ws - 1) / (float)(Wd - 1) : 0.f;
    const int CV = C / 8;  // 8 channels per thread
    GRID_STRIDE(i, (long long)Hd * Wd * CV) {
        int c = (int)(i % CV) * 8;
        long long pd = i / CV;
        int y = (int)(pd / Wd), x = (int)(pd % Wd);
        float h1r = rh * (float)y, w1r = rw * (float)x;
        int h1 = (int)h1r, w1 = (int)w1r;
        int h1p = h1 < Hs - 1 ? 1 : 0, w1p = w1 < Ws - 1 ? 1 : 0;
        float h1l = h1r - (float)h1, w1l = w1r - (float)w1;
        float h0l = 1.f - h1l, w0l = 1.f - w1l;
        uint4 a = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1) * sstride + c);
        uint4 b = *reinterpret_cast<const uint4*>(src + ((long long)h1 * Ws + w1 + w1p) * sstride + c);
        uint4 cc = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1) * sstride + c);
        uint4 d = *reinterpret_cast<const uint4*>(src + ((long long)(h1 + h1p) * Ws + w1 + w1p) * sstride + c);
        const __half *v00 = reinterpret_cast<const __half*>(&a), *v01 = reinterpret_cast<const __half*>(&b);
        const __half *v10 = reinterpret_cast<const __half*>(&cc), *v11 = reinterpret_cast<const __half*>(&d);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = __float2half(h0l * (w0l * __half2float(v00[k]) + w1l * __half2float(v01[k])) +
                                h1l * (w0l * __half2float(v10[k]) + w1l * __half2float(v11[k])));
        *reinterpret_cast<uint4*>(dst + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
        if (dst2) *reinterpret_cast<uint4*>(dst2 + pd * dstride + c) = *reinterpret_cast<const uint4*>(o);
    }
}
extern "C" __global__ void interp_into(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                       int dstride, int Hd, int Wd, int C) {
    interp_body(src, sstride, Hs, Ws, dst, dstride, nullptr, Hd, Wd, C);
}

// same, written to two buffers with the same row stride (the fused GRU's [h | x] and [r*h | x] conv inputs)
extern "C" __global__ void pool2x_into2(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                        __half* dst2, int dstride, int Hd, int Wd, int C) {
    pool2x_body(src, sstride, Hs, Ws, dst, dstride, dst2, Hd, Wd, C);
}
extern "C" __global__ void interp_into2(const __half* __restrict__ src, int sstride, int Hs, int Ws, __half* dst,
                                        __half* dst2, int dstride, int Hd, int Wd, int C) {
    interp_body(src, sstride, Hs, Ws, dst, dstride, dst2, Hd, Wd, C);
}

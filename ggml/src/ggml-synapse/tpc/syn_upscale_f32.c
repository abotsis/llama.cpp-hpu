// UPSCALE (interpolation of dims 0/1; dims 2/3 nearest, i02 = (int) (i2 / sf2) as the CPU does), F32.
// Modes: NEAREST (-DNEAREST), BILINEAR (-DBILINEAR), BICUBIC (-DBICUBIC); ANTIALIAS is a
// bilinear-only registration (-DANTIALIAS). The mode scalar carries the flags:
// 256 = ALIGN_CORNERS (pixel_offset 0, scale (out-1)/(in-1) when both dims exceed 1),
// 512 = ANTIALIAS. Coordinate formulas and boundary treatment ported exactly from
// ggml_compute_forward_upscale_f32 (ggml/src/ggml-cpu/ops.cpp:8290):
//  - nearest: i00 = (int)(i0 / sf0) without a half-pixel offset (the CPU's float-division
//    truncation toward zero; scales are positive so this is floor).
//  - bilinear / bicubic: x = (i0 + 0.5)/sf0 - 0.5, floor taps clamped to the source bounds;
//    bicubic coefficient -0.75, clamped 4x4 taps.
//  - antialias: triangle filter, support max(1, 1/scale), taps clipped to [0, in), weights
//    normalized; a zero total weight yields exact zero.
// src view (element offset + strides of dims 0..3, dim-0 stride 1); out contiguous
// [ne0, ne1, ne2, ne3].
// tensors: 0 src | 1 out
#include "syn_common.h"

// floorf semantics through a scalar conversion (SW_RD rounds toward -infinity)
static inline int ifl(float x)
{
    return s_convert_f32_to_i32(x, SW_RD);
}

static inline int clampi(int x, int hi) // hi exclusive
{
    return x < 0 ? 0 : (x > hi - 1 ? hi - 1 : x);
}

#if defined(BICUBIC)
// weight1(x) = ((a+2)x - (a+3))x^2 + 1; weight2(x) = ((a x - 5a)x + 8a)x - 4a, a = -0.75
static inline float w1(float x)
{
    const float a = -0.75f;
    return ((a + 2) * x - (a + 3)) * x * x + 1;
}
static inline float w2(float x)
{
    const float a = -0.75f;
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
}
static inline float bic1(float p0, float p1, float p2, float p3, float x)
{
    return p0 * w2(x + 1) + p1 * w1(x + 0) + p2 * w1(1 - x) + p3 * w2(2 - x);
}
#endif
#if defined(ANTIALIAS)
static inline float tri(float x)
{
    const float ax = x < 0 ? -x : x;
    return 1.0f - ax > 0.0f ? 1.0f - ax : 0.0f;
}
#endif

void main(tensor src, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int ne00, int ne01,
          int off0, int s01, int s02, int s03,
          int mode, int ne02, int ne03)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int align_corners = mode & 256;
    float sf0 = (float) ne0 / (float) ne00;
    float sf1 = (float) ne1 / (float) ne01;
    const float sf2 = (float) ne2 / (float) ne02;
    const float sf3 = (float) ne3 / (float) ne03;
    float pixel_offset = 0.5f;
    if (align_corners) {
        pixel_offset = 0.0f;
        if (ne0 > 1 && ne00 > 1) sf0 = (float) (ne0 - 1) / (float) (ne00 - 1);
        if (ne1 > 1 && ne01 > 1) sf1 = (float) (ne1 - 1) / (float) (ne01 - 1);
    }
#if defined(BILINEAR) && defined(ANTIALIAS)
    const float support1 = 1.0f / sf1 < 1.0f ? 1.0f : 1.0f / sf1;
    const float invscale1 = 1.0f / support1;
    const float support0 = 1.0f / sf0 < 1.0f ? 1.0f : 1.0f / sf0;
    const float invscale0 = 1.0f / support0;
#endif
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int row = r0; row < r1e; row++) {
            const int i3 = row / (ne1 * ne2);
            const int rr = row - i3 * ne1 * ne2;
            const int i2 = rr / ne1;
            const int i1_ = rr - i2 * ne1;
            const int i02 = ifl((float) i2 / sf2);
            const int i03 = ifl((float) i3 / sf3);
            const int pm = off0 + i02 * s02 + i03 * s03;
            const int po = row * ne0;
            const float yf = ((float) i1_ + pixel_offset) / sf1;
#if defined(NEAREST)
            const int py = clampi(ifl((float) i1_ / sf1), ne01) * s01;
            for (int i0 = 0; i0 < ne0; i0++) {
                const int xc = clampi(ifl((float) i0 / sf0), ne00);
                int5 c = {pm + py + xc, 0, 0, 0, 0};
                int5 d = {po + i0, 0, 0, 0, 0};
                s_f32_st_g((__global float *) gen_addr(d, out), s_f32_ld_g((__global float *) gen_addr(c, src)));
            }
#else
#if defined(BICUBIC)
            const int y0 = ifl(yf - pixel_offset);
            const float dy = (yf - pixel_offset) - (float) y0;
#elif defined(ANTIALIAS)
            int y_min = ifl(yf - support1 + pixel_offset);
            if (y_min < 0) y_min = 0;
            int y_max = ifl(yf + support1 + pixel_offset);
            if (y_max > ne01) y_max = ne01;
#else
            const float y = yf - pixel_offset;
            const int yfloor = ifl(y);
            const int y0 = clampi(yfloor, ne01), y1 = clampi(yfloor + 1, ne01);
            float dyv = y - (float) yfloor;
            if (dyv < 0.0f) dyv = 0.0f;
            if (dyv > 1.0f) dyv = 1.0f;
#endif
            for (int i0 = 0; i0 < ne0; i0++) {
                const float xf = ((float) i0 + pixel_offset) / sf0;
                float v;
#if defined(BICUBIC)
                {
                    const int x0 = ifl(xf - pixel_offset);
                    const float dx = (xf - pixel_offset) - (float) x0;
                    float pr[4];
                    for (int yy = 0; yy < 4; yy++) {
                        const int yc = clampi(y0 + yy - 1, ne01);
                        float rw[4];
                        for (int xx = 0; xx < 4; xx++) {
                            const int xc = clampi(x0 + xx - 1, ne00);
                            int5 c = {pm + yc * s01 + xc, 0, 0, 0, 0};
                            rw[xx] = s_f32_ld_g((__global float *) gen_addr(c, src));
                        }
                        pr[yy] = bic1(rw[0], rw[1], rw[2], rw[3], dx);
                    }
                    v = bic1(pr[0], pr[1], pr[2], pr[3], dy);
                }
#elif defined(ANTIALIAS)
                {
                    int x_min = ifl(xf - support0 + pixel_offset);
                    if (x_min < 0) x_min = 0;
                    int x_max = ifl(xf + support0 + pixel_offset);
                    if (x_max > ne00) x_max = ne00;
                    float val = 0.0f, tw = 0.0f;
                    for (int yy = y_min; yy < y_max; yy++) {
                        const float wy = tri(((float) yy - yf + pixel_offset) * invscale1);
                        for (int xx = x_min; xx < x_max; xx++) {
                            const float wx = tri(((float) xx - xf + pixel_offset) * invscale0);
                            const float w = wx * wy;
                            if (w <= 0.0f) continue;
                            int5 c = {pm + yy * s01 + xx, 0, 0, 0, 0};
                            val += s_f32_ld_g((__global float *) gen_addr(c, src)) * w;
                            tw += w;
                        }
                    }
                    v = tw > 0.0f ? val / tw : 0.0f;
                }
#else // BILINEAR
                {
                    const float x = xf - pixel_offset;
                    const int xfloor = ifl(x);
                    const int x0 = clampi(xfloor, ne00), x1 = clampi(xfloor + 1, ne00);
                    float dxv = x - (float) xfloor;
                    if (dxv < 0.0f) dxv = 0.0f;
                    if (dxv > 1.0f) dxv = 1.0f;
                    int5 c00 = {pm + y0 * s01 + x0, 0, 0, 0, 0};
                    int5 c10 = c00; c10[0] += x1 - x0;
                    int5 c01 = c00; c01[0] += (y1 - y0) * s01;
                    int5 c11 = c00; c11[0] += (x1 - x0) + (y1 - y0) * s01;
                    const float a = s_f32_ld_g((__global float *) gen_addr(c00, src));
                    const float b = s_f32_ld_g((__global float *) gen_addr(c10, src));
                    const float cc = s_f32_ld_g((__global float *) gen_addr(c01, src));
                    const float d = s_f32_ld_g((__global float *) gen_addr(c11, src));
                    v = a * (1 - dxv) * (1 - dyv) + b * dxv * (1 - dyv)
                      + cc * (1 - dxv) * dyv + d * dxv * dyv;
                }
#endif
                int5 d = {po + i0, 0, 0, 0, 0};
                s_f32_st_g((__global float *) gen_addr(d, out), v);
            }
#endif
        }
    }
}

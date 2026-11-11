// Strided copy with type conversion: out (contiguous [ne0, ne1, ne2, ne3], DST type)
// = a (view, SRC type, element offset + strides of dims 0..3).
// Built per (SRC, DST) pair: -DSRC_F32/-DSRC_F16/-DSRC_BF16, -DDST_F32/-DDST_F16/-DDST_BF16;
// -DSRC_I32 -DDST_I32: a raw 32-bit copy (integer loads / stores). -DSRC_F32 -DDST_I32: f32 -> i32 toward zero (C);
// -DSRC_I32 -DDST_F32: i32 -> f32 (the casts of index math: the backend top-p sampler, sparse-attention selections).
// tensors: 0 a | 1 out
// -DKV: also takes the K / V caches, not read: the dep orders the copy after the cache writes.
// tensors: 0 a  1 k  2 v | out
#include "syn_common.h"

#if defined(SRC_F16)
#define SLD(c, t) v_f16_ld_tnsr_b(c, t)
#define S2F(x) f16_to_f32(x)
#define SSLD(addr) ((float) s_f16_ld_g((__global half *) (addr)))
#elif defined(SRC_BF16)
#define SLD(c, t) v_bf16_ld_tnsr_b(c, t)
#define S2F(x) bf16_to_f32(x)
#define SSLD(addr) ((float) s_bf16_ld_g((__global bf16 *) (addr)))
#elif defined(SRC_I32)
#define SSLD(addr) s_i32_ld_g((__global int *) (addr))
#else
#define SSLD(addr) s_f32_ld_g((__global float *) (addr))
#endif

// elements ps .. ps + 127 of a -> lo, hi (f32 lanes; raw ints for I32)
#if defined(SRC_I32)
#define CH_T int64
#define CH_LD(ps, lo, hi) { int5 c_ = {(ps), 0, 0, 0, 0}; lo = v_i32_ld_tnsr_b(c_, a); c_[0] += 64; hi = v_i32_ld_tnsr_b(c_, a); }
#elif defined(SRC_F16) || defined(SRC_BF16)
#define CH_T float64
#define CH_LD(ps, lo, hi) { int5 c_ = {(ps), 0, 0, 0, 0}; const float128 x_ = S2F(SLD(c_, a)); lo = x_.v1; hi = x_.v2; }
#else
#define CH_T float64
#define CH_LD(ps, lo, hi) { int5 c_ = {(ps), 0, 0, 0, 0}; lo = v_f32_ld_tnsr_b(c_, a); c_[0] += 64; hi = v_f32_ld_tnsr_b(c_, a); }
#endif
// lane conversions of the mixed f32 / i32 copies
#if defined(DST_I32) && !defined(SRC_I32)
#define TO_I(x) v_convert_f32_to_i32_b((x), SW_RZ, 0, 1, 0)
#else
#define TO_I(x) (x)
#endif
#if defined(SRC_I32) && !defined(DST_I32)
#define TO_F(x) v_convert_i32_to_f32_b((x), SW_RHNE, 0, 1, 0)
#else
#define TO_F(x) (x)
#endif
// lo, hi -> out at element po, n (>= 1) elements
#if defined(DST_I32)
#define CH_ST(po, lo, hi, n) \
    { \
        int5 d0 = {(po), 0, 0, 0, 0}; \
        if ((n) >= 64) v_i32_st_tnsr(d0, out, TO_I(lo)); else v_i32_st_tnsr_partial(d0, out, TO_I(lo), (n) - 1, 0); \
        if ((n) > 64) { \
            int5 d1 = {(po) + 64, 0, 0, 0, 0}; \
            if ((n) >= 128) v_i32_st_tnsr(d1, out, TO_I(hi)); else v_i32_st_tnsr_partial(d1, out, TO_I(hi), (n) - 65, 0); \
        } \
    }
#elif defined(DST_F16)
#define CH_ST(po, lo, hi, n) \
    { int5 d_ = {(po), 0, 0, 0, 0}; const half128 y_ = cvt_f16(lo, hi); \
      if ((n) >= 128) v_f16_st_tnsr(d_, out, y_); else v_f16_st_tnsr_partial(d_, out, y_, (n) - 1, 0); }
#elif defined(DST_BF16)
#define CH_ST(po, lo, hi, n) \
    { int5 d_ = {(po), 0, 0, 0, 0}; const bfloat128 y_ = cvt_bf16(lo, hi); \
      if ((n) >= 128) v_bf16_st_tnsr(d_, out, y_); else v_bf16_st_tnsr_partial(d_, out, y_, (n) - 1, 0); }
#else
#define CH_ST(po, lo, hi, n) \
    { ST_F32(out, (po), TO_F(lo), (n)); if ((n) > 64) { ST_F32(out, (po) + 64, TO_F(hi), (n) - 64); } }
#endif

#if defined(KV)
void main(tensor a, tensor k, tensor v, tensor dep, tensor out,
#else
void main(tensor a, tensor dep, tensor out,
#endif
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int off0, int s00, int s01, int s02, int s03)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nrows = ne1 * ne2 * ne3;
    if (s00 != 1) {
        // element-wise path (transposing views): scalar loads
        scalar_cache_reset();
        const int n = nrows * ne0;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(n, nmem, m, i0, i1)
            for (int i = i0; i < i1; i++) {
                const int r = i / ne0;
                const int e = i - r * ne0;
                const int r3 = r / (ne1 * ne2);
                const int rr = r - r3 * ne1 * ne2;
                const int r2 = rr / ne1;
                const int r1 = rr - r2 * ne1;
                int5 c = {off0 + e * s00 + r1 * s01 + r2 * s02 + r3 * s03, 0, 0, 0, 0};
                int5 d = {i, 0, 0, 0, 0};
#if defined(SRC_I32) && defined(DST_I32)
                s_i32_st_g((__global int *) gen_addr(d, out), SSLD(gen_addr(c, a)));
                continue;
#elif defined(SRC_I32)
                s_f32_st_g((__global float *) gen_addr(d, out), (float) SSLD(gen_addr(c, a)));
                continue;
#elif defined(DST_I32)
                s_i32_st_g((__global int *) gen_addr(d, out), s_convert_f32_to_i32(SSLD(gen_addr(c, a)), SW_RZ));
                continue;
#endif
                const float v = SSLD(gen_addr(c, a));
#if defined(DST_F16) && defined(SRC_BF16)
                // (the mirrored tpc-clang 2.7.33 cast gap, bf16 -> f32 -> f16): round to nearest
                // even on the f32 bits, stored as the 16-bit pattern (inf stays inf, NaN stays NaN)
                const int u_ = as_int(v);
                s_u16_st_g((__global unsigned short *) gen_addr(d, out), (unsigned short) ((u_ + 0x7FFF + ((u_ >> 16) & 1)) >> 16));
#elif defined(DST_F16)
                s_f16_st_g((__global half *) gen_addr(d, out), (half) v);
#elif defined(DST_BF16) && defined(SRC_F16)
                // (tpc-clang 2.7.33 cannot select a direct f16 -> f32 -> bf16 scalar cast): round to nearest
                // even on the f32 bits, stored as the 16-bit pattern (inf stays inf, NaN stays NaN)
                const int u_ = as_int(v);
                s_u16_st_g((__global unsigned short *) gen_addr(d, out), (unsigned short) ((u_ + 0x7FFF + ((u_ >> 16) & 1)) >> 16));
#elif defined(DST_BF16)
                s_bf16_st_g((__global bf16 *) gen_addr(d, out), (bf16) v);
#else
                s_f32_st_g((__global float *) gen_addr(d, out), v);
#endif
            }
        }
        return;
    }
    if (s01 == ne0 && s02 == ne0 * ne1 && s03 == ne0 * ne1 * ne2) {
        // whole tensor contiguous
        const int N = nrows * ne0;
        const int nit = (N + 127) >> 7;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(nit, nmem, m, i0, i1)
            int it = i0;
            const int full = i1 < (N >> 7) ? i1 : (N >> 7);   // chunks [.., full) are whole
            for (; it + 4 <= full; it += 4) {
                const int ps = off0 + (it << 7), po = it << 7;
                CH_T l0, h0, l1, h1, l2, h2, l3, h3;
                CH_LD(ps, l0, h0)
                CH_LD(ps + 128, l1, h1)
                CH_LD(ps + 256, l2, h2)
                CH_LD(ps + 384, l3, h3)
                CH_ST(po, l0, h0, 128)
                CH_ST(po + 128, l1, h1, 128)
                CH_ST(po + 256, l2, h2, 128)
                CH_ST(po + 384, l3, h3, 128)
            }
            for (; it < i1; it++) {
                const int po = it << 7, n = N - po;
                CH_T lo, hi;
                CH_LD(off0 + po, lo, hi)
                CH_ST(po, lo, hi, n)
            }
        }
        return;
    }
    // rows with a contiguous dim 0
    const int nch = (ne0 + 127) >> 7;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nch;
        int ch = i0 - r * nch;
        int r3 = r / (ne1 * ne2);
        int rr = r - r3 * ne1 * ne2;
        int r2 = rr / ne1;
        int r1 = rr - r2 * ne1;
        for (int it = i0; it < i1; it++) {
            const int e0 = ch << 7;
            const int ps = off0 + r1 * s01 + r2 * s02 + r3 * s03 + e0;
            int5 c = {ps, 0, 0, 0, 0};
#if defined(SRC_I32)
            const int64 lo = v_i32_ld_tnsr_b(c, a);
            c[0] = ps + 64;
            const int64 hi = v_i32_ld_tnsr_b(c, a);
#elif defined(SRC_F16) || defined(SRC_BF16)
            const float128 x = S2F(SLD(c, a));
            const float64 lo = x.v1, hi = x.v2;
#else
            const float64 lo = v_f32_ld_tnsr_b(c, a);
            c[0] = ps + 64;
            const float64 hi = v_f32_ld_tnsr_b(c, a);
#endif
            const int n = ne0 - e0;
            const int po = r * ne0 + e0;
            int5 d = {po, 0, 0, 0, 0};
#if defined(DST_I32)
            {
                int5 d0 = {po, 0, 0, 0, 0};
                if (n >= 64) v_i32_st_tnsr(d0, out, TO_I(lo)); else v_i32_st_tnsr_partial(d0, out, TO_I(lo), n - 1, 0);
                if (n > 64) {
                    int5 d1 = {po + 64, 0, 0, 0, 0};
                    if (n >= 128) v_i32_st_tnsr(d1, out, TO_I(hi)); else v_i32_st_tnsr_partial(d1, out, TO_I(hi), n - 65, 0);
                }
            }
#elif defined(DST_F16) || defined(DST_BF16)
#if defined(DST_F16)
            const half128 y = cvt_f16(lo, hi);
            if (n >= 128) v_f16_st_tnsr(d, out, y); else v_f16_st_tnsr_partial(d, out, y, n - 1, 0);
#else
            const bfloat128 y = cvt_bf16(lo, hi);
            if (n >= 128) v_bf16_st_tnsr(d, out, y); else v_bf16_st_tnsr_partial(d, out, y, n - 1, 0);
#endif
#else
            ST_F32(out, po, TO_F(lo), n);
            if (n > 64) {
                ST_F32(out, po + 64, TO_F(hi), n - 64);
            }
#endif
            if (++ch == nch) {
                ch = 0;
                r++;
                if (++r1 == ne1) {
                    r1 = 0;
                    if (++r2 == ne2) {
                        r2 = 0;
                        r3++;
                    }
                }
            }
        }
    }
}

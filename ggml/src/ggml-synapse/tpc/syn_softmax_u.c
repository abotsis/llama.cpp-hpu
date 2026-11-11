// Unnormalized softmax of attention scores (flash attention): p = exp(x * scale + mask - M) with
// M the row max, in f16 (-DOUT_F16) or bf16 (-DOUT_BF16), and l = 1 / (sum p (+ exp(sink - M))) per row in f32
// [ne2, ne1, ne3] (heads fastest);
// the P V product is scaled by l afterwards (syn_fa_norm). x contiguous [ne0, ne1 queries, ne2 heads, ne3];
// the mask (f16, optional) is shared by the heads (row (i % nm1, b % nm3)); hg <= HGMAX heads per group.
// Two passes over x (max, then exp + store). Lanes past ne0 get -inf through the mask
// chunk (read past the row end, masked here).
// tensors: 0 x  1 mask  2 sinks | 3 out (p)  4 l
#include "syn_common.h"

// exp(x), x <= 0: degree-5 polynomial on |r| <= ln2 / 2 (~2e-7 relative), as syn_softmax.c
static inline float64 exp_p(float64 x)
{
    x = v_f32_max_b(x, -87.0f);
    const int64 n = v_convert_f32_to_i32_b(x * 1.4426950408889634f, SW_RHNE, 0, 1, 0);
    const float64 r = v_f32_mac_b(v_convert_i32_to_f32_b(n, SW_RHNE, 0, 1, 0), -0.6931471805599453f, x);
    float64 pl = 1.0f / 120.0f;
    pl = v_f32_mac_b(pl, r, 1.0f / 24.0f);
    pl = v_f32_mac_b(pl, r, 1.0f / 6.0f);
    pl = v_f32_mac_b(pl, r, 0.5f);
    pl = v_f32_mac_b(pl, r, 1.0f);
    pl = v_f32_mac_b(pl, r, 1.0f);
    return pl * as_float64(v_i32_shl_b(n + 127, (int64) 23));
}

#define CV 16
#define HGMAX 16
__local__ float64 ML[CV];
__local__ float64 MX[HGMAX + 3];
__local__ float64 AC[HGMAX + 3];

#if defined(OUT_F16)
#define ST_P(c, y) v_f16_st_tnsr(c, out, y)
#define STP_P(c, y, n) v_f16_st_tnsr_partial(c, out, y, (n) - 1, 0)
#define CVT_P(a, b) cvt_f16(a, b)
#define PT half128
#else
#define ST_P(c, y) v_bf16_st_tnsr(c, out, y)
#define STP_P(c, y, n) v_bf16_st_tnsr_partial(c, out, y, (n) - 1, 0)
#define CVT_P(a, b) cvt_bf16(a, b)
#define PT bfloat128
#endif

void main(tensor x, tensor mask, tensor sinks, tensor dep, tensor out, tensor l,
          int nmem, int ne0, int ne1, int ne2, int ne3, int hg,
          int offm, int sm1, int sm3, int nm1, int nm3, int has_mask, float scale, int has_sinks)
{
    if (has_sinks) {
        scalar_cache_reset();
    }
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 ninf = as_float64((int64) 0xFF800000);
    const int ngrp = (ne2 + hg - 1) / hg;
    const int nitems = ne1 * ngrp * ne3;
    const int hstride = ne1 * ne0;    // elements between the rows of consecutive heads
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int i = it % ne1;
            const int t = it / ne1;
            const int g = t % ngrp;
            const int b = t / ngrp;
            const int h0 = g * hg;
            const int nh = ne2 - h0 < hg ? ne2 - h0 : hg;
            const int r0 = i + ne1 * (h0 + ne2 * b);          // row of head h0
            const int pm = offm + (i % nm1) * sm1 + (b % nm3) * sm3;
            for (int h = 0; h < nh; h++) {
                MX[h] = ninf;
                AC[h] = 0.0f;
            }
            // the chunk's mask (or 0) as f32, -inf past ne0
#define LOAD_ML(c0, nvc)                                                                       \
            for (int v = 0; v < (nvc); v += 2) {                                               \
                float64 a_ = 0.0f, b_ = 0.0f;                                                  \
                if (has_mask) {                                                                \
                    int5 cm = {pm + (c0) + 64 * v, 0, 0, 0, 0};                                \
                    const float128 m2 = f16_to_f32(v_f16_ld_tnsr_b(cm, mask));                 \
                    a_ = m2.v1;                                                                \
                    b_ = m2.v2;                                                                \
                }                                                                              \
                ML[v] = v_f32_sel_less_f32_b(lane, (float) (ne0 - (c0) - 64 * v), a_, ninf);   \
                ML[v + 1] = v_f32_sel_less_f32_b(lane, (float) (ne0 - (c0) - 64 * v - 64), b_, ninf); \
            }
            // pass 1: lane max per head
            for (int c0 = 0; c0 < ne0; c0 += CV * 64) {
                const int nvc = (ne0 - c0 + 63) >> 6 < CV ? (ne0 - c0 + 63) >> 6 : CV;
                LOAD_ML(c0, nvc)
                int px = r0 * ne0 + c0;
                for (int h = 0; h < nh; h++, px += hstride) {
                    float64 m0 = MX[h], m1 = ninf, m2 = ninf, m3 = ninf;
                    int5 cx = {px, 0, 0, 0, 0};
                    int v = 0;
                    for (; v + 4 <= nvc; v += 4) {
                        const float64 x0 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x1 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x2 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x3 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        m0 = v_f32_max_b(m0, v_f32_mac_b(x0, scale, ML[v]));
                        m1 = v_f32_max_b(m1, v_f32_mac_b(x1, scale, ML[v + 1]));
                        m2 = v_f32_max_b(m2, v_f32_mac_b(x2, scale, ML[v + 2]));
                        m3 = v_f32_max_b(m3, v_f32_mac_b(x3, scale, ML[v + 3]));
                    }
                    for (; v < nvc; v++) {
                        m0 = v_f32_max_b(m0, v_f32_mac_b(v_f32_ld_tnsr_b(cx, x), scale, ML[v]));
                        cx[0] += 64;
                    }
                    MX[h] = v_f32_max_b(v_f32_max_b(m0, m1), v_f32_max_b(m2, m3));
                }
            }
            // row maxima (a row with every key masked: p = 0, not exp(-inf + inf))
            for (int h = 0; h < nh; h += 4) {
                float64 M0, M1, M2, M3;
                NL_REDUCE4_MAX(MX[h], MX[h + 1], MX[h + 2], MX[h + 3], IDX, M0, M1, M2, M3)
                MX[h] = v_f32_sel_eq_f32_b(M0, ninf, 0.0f, M0);
                MX[h + 1] = v_f32_sel_eq_f32_b(M1, ninf, 0.0f, M1);
                MX[h + 2] = v_f32_sel_eq_f32_b(M2, ninf, 0.0f, M2);
                MX[h + 3] = v_f32_sel_eq_f32_b(M3, ninf, 0.0f, M3);
            }
            // pass 2: p = exp(s - M), stored unnormalized; lane sums
            for (int c0 = 0; c0 < ne0; c0 += CV * 64) {
                const int nvc = (ne0 - c0 + 63) >> 6 < CV ? (ne0 - c0 + 63) >> 6 : CV;
                LOAD_ML(c0, nvc)
                int px = r0 * ne0 + c0;
                for (int h = 0; h < nh; h++, px += hstride) {
                    const float64 M = MX[h];
                    float64 a0 = AC[h], a1 = 0.0f;
                    int5 cx = {px, 0, 0, 0, 0}, co = {px, 0, 0, 0, 0};
                    int v = 0;
                    // whole vectors: plain stores; the rest below
                    const int nfull = (ne0 - c0) >> 6 < CV ? (ne0 - c0) >> 6 : CV;
                    for (; v + 8 <= nfull; v += 8) {
                        float64 pv[8];
                        #pragma unroll
                        for (int u = 0; u < 8; u++) {
                            pv[u] = exp_p(v_f32_mac_b(v_f32_ld_tnsr_b(cx, x), scale, ML[v + u]) - M);
                            cx[0] += 64;
                        }
                        a0 += (pv[0] + pv[2]) + (pv[4] + pv[6]);
                        a1 += (pv[1] + pv[3]) + (pv[5] + pv[7]);
                        #pragma unroll
                        for (int u = 0; u < 8; u += 2) {
                            ST_P(co, CVT_P(pv[u], pv[u + 1]));
                            co[0] += 128;
                        }
                    }
                    for (; v + 4 <= nfull; v += 4) {
                        const float64 x0 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x1 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x2 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 x3 = v_f32_ld_tnsr_b(cx, x);
                        cx[0] += 64;
                        const float64 p0 = exp_p(v_f32_mac_b(x0, scale, ML[v]) - M);
                        const float64 p1 = exp_p(v_f32_mac_b(x1, scale, ML[v + 1]) - M);
                        const float64 p2 = exp_p(v_f32_mac_b(x2, scale, ML[v + 2]) - M);
                        const float64 p3 = exp_p(v_f32_mac_b(x3, scale, ML[v + 3]) - M);
                        a0 += p0 + p2;
                        a1 += p1 + p3;
                        ST_P(co, CVT_P(p0, p1));
                        co[0] += 128;
                        ST_P(co, CVT_P(p2, p3));
                        co[0] += 128;
                    }
                    for (; v < nvc; v += 2) {
                        // a last odd vector pairs with a vector past the row (-inf mask: p = 0, not stored)
                        const float64 p0 = exp_p(v_f32_mac_b(ld_f32(x, px + 64 * v), scale, ML[v]) - M);
                        const float64 p1 = v + 1 < nvc ? exp_p(v_f32_mac_b(ld_f32(x, px + 64 * v + 64), scale, ML[v + 1]) - M) : 0.0f;
                        a0 += p0;
                        a1 += p1;
                        const int n0 = ne0 - c0 - 64 * v;
                        int5 co = {px + 64 * v, 0, 0, 0, 0};
                        const PT y0 = CVT_P(p0, p1);
                        if (n0 >= 128) ST_P(co, y0); else STP_P(co, y0, n0);
                    }
                    AC[h] = a0 + a1;
                }
            }
#undef LOAD_ML
            // l = 1 / row sums
            for (int h = 0; h < nh; h += 4) {
                float64 S[4];
                NL_REDUCE4_ADD(AC[h], AC[h + 1], AC[h + 2], AC[h + 3], IDX, S[0], S[1], S[2], S[3])
                if (has_sinks) {
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        // (scalar loads have no bounds check: heads past the group read the last one)
                        int5 cs = {h0 + (h + u < nh ? h + u : nh - 1), 0, 0, 0, 0};
                        const float sk = s_f32_ld_g(gen_addr(cs, sinks));
                        S[u] = S[u] + exp_p(sk - MX[h + u]);
                    }
                }
                int5 cl = {h0 + h + ne2 * (i + ne1 * b), 0, 0, 0, 0};
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float64 il = v_f32_sel_grt_f32_b(S[u], 0.0f, nl_recip(v_f32_max_b(S[u], 1e-30f)), 0.0f);
                    if (h + u < nh) {
                        v_f32_st_tnsr_partial(cl, l, il, 0, 0);
                    }
                    cl[0] += 1;
                }
            }
        }
    }
}

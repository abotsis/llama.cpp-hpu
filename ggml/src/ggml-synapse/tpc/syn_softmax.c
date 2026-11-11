// SOFT_MAX over dim 0: out = softmax(x * scale + slope * mask), f32 (-DOUT_F16 / -DOUT_BF16: f16 / bf16).
// x is a view; mask (has_mask) is a view broadcast over rows (index i_d % nm_d), f16 (-DMASK_F16) or f32.
// alibi: slope of head i2 from slopes [ne2] (else 1); has_sinks: one more logit sinks[i2] per row
// in the normalization (no output).
// tensors: 0 x  1 mask  2 slopes  3 sinks | 4 out
#include "syn_common.h"

static inline float64 ld_mask(tensor t, int p)
{
#if defined(MASK_F16)
    int5 c = {p, 0, 0, 0, 0};
    return f16_to_f32(v_f16_ld_tnsr_b(c, t)).v1;
#else
    return ld_f32(t, p);
#endif
}

// exp(x), x <= 0, for f16 / bf16 probabilities: degree-5 polynomial on |r| <= ln2 / 2 (~2e-7 relative)
#if defined(OUT_F16) || defined(OUT_BF16)
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
#else
#define exp_p(x) nl_exp(x)
#endif

#define EMAX 256
__local__ float64 E[EMAX + 4];

void main(tensor x, tensor mask, tensor slopes, tensor sinks, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int offx, int sx1, int sx2, int sx3,
          int offm, int sm1, int sm2, int sm3, int nm1, int nm2, int nm3,
          int has_mask, float scale, int alibi, int has_sinks)
{
    if (alibi || has_sinks) {
        scalar_cache_reset();
    }
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int px = offx + r1 * sx1 + r2 * sx2 + r3 * sx3;
            const int pm = offm + (r1 % nm1) * sm1 + (r2 % nm2) * sm2 + (r3 % nm3) * sm3;
            int5 ch = {r2, 0, 0, 0, 0};
            const float slope = alibi ? s_f32_ld_g(gen_addr(ch, slopes)) : 1.0f;
            const float sk = has_sinks ? s_f32_ld_g(gen_addr(ch, sinks)) : -1e30f;
            float64 mx = sk;
            const int nv = (ne0 + 63) >> 6;
            if (nv <= EMAX) {
                for (int c = 0; c < nv; c += 4) {
                    const int e = c << 6;
                    float64 v0 = ld_f32(x, px + e), v1 = ld_f32(x, px + e + 64);
                    float64 v2 = ld_f32(x, px + e + 128), v3 = ld_f32(x, px + e + 192);
                    if (has_mask) {
#if defined(MASK_F16)
                        int5 cm = {pm + e, 0, 0, 0, 0};
                        const float128 mA = f16_to_f32(v_f16_ld_tnsr_b(cm, mask));
                        cm[0] += 128;
                        const float128 mB = f16_to_f32(v_f16_ld_tnsr_b(cm, mask));
                        const float64 m0 = mA.v1, m1 = mA.v2, m2 = mB.v1, m3 = mB.v2;
#else
                        const float64 m0 = ld_mask(mask, pm + e), m1 = ld_mask(mask, pm + e + 64);
                        const float64 m2 = ld_mask(mask, pm + e + 128), m3 = ld_mask(mask, pm + e + 192);
#endif
                        v0 = v_f32_mac_b(m0, slope, v0 * scale); v1 = v_f32_mac_b(m1, slope, v1 * scale);
                        v2 = v_f32_mac_b(m2, slope, v2 * scale); v3 = v_f32_mac_b(m3, slope, v3 * scale);
                    } else {
                        v0 = v0 * scale; v1 = v1 * scale; v2 = v2 * scale; v3 = v3 * scale;
                    }
                    v0 = v_f32_sel_less_f32_b(lane, (float) (ne0 - e), v0, -1e30f);
                    v1 = v_f32_sel_less_f32_b(lane, (float) (ne0 - e - 64), v1, -1e30f);
                    v2 = v_f32_sel_less_f32_b(lane, (float) (ne0 - e - 128), v2, -1e30f);
                    v3 = v_f32_sel_less_f32_b(lane, (float) (ne0 - e - 192), v3, -1e30f);
                    E[c] = v0; E[c + 1] = v1; E[c + 2] = v2; E[c + 3] = v3;
                    mx = v_f32_max_b(mx, v_f32_max_b(v_f32_max_b(v0, v1), v_f32_max_b(v2, v3)));
                }
                float64 M;
                NL_REDUCE_MAX(mx, IDX, M);
                float64 a0 = 0.0f, a1 = 0.0f;
                for (int c = 0; c < nv; c += 2) {
                    const float64 p0 = exp_p(E[c] - M), p1 = exp_p(E[c + 1] - M);
                    E[c] = p0; E[c + 1] = p1;
                    a0 += p0; a1 += p1;
                }
                float64 S;
                const float64 acc2 = a0 + a1;
                NL_REDUCE_ADD(acc2, IDX, S);
                if (has_sinks) {
                    S = S + nl_exp(sk - M);
                }
                const float64 is = nl_recip(S);
#if defined(OUT_F16) || defined(OUT_BF16)
                // E[nv] of an odd nv is read but not stored
                for (int c = 0; c < nv; c += 2) {
                    const int e = c << 6, n = ne0 - e;
                    int5 co = {r * ne0 + e, 0, 0, 0, 0};
#if defined(OUT_F16)
                    const half128 y = cvt_f16(E[c] * is, E[c + 1] * is);
                    if (n >= 128) v_f16_st_tnsr(co, out, y); else v_f16_st_tnsr_partial(co, out, y, n - 1, 0);
#else
                    const bfloat128 y = cvt_bf16(E[c] * is, E[c + 1] * is);
                    if (n >= 128) v_bf16_st_tnsr(co, out, y); else v_bf16_st_tnsr_partial(co, out, y, n - 1, 0);
#endif
                }
#else
                for (int c = 0; c < nv; c++) {
                    ST_OUT(out, r * ne0 + (c << 6), E[c] * is, ne0 - (c << 6));
                }
#endif
                continue;
            }
            // three passes over x (max, sum, output)
#define LDV(e, v)                                                              \
            {                                                                  \
                v = ld_f32(x, px + (e)) * scale;                               \
                if (has_mask) v = v_f32_mac_b(ld_mask(mask, pm + (e)), slope, v); \
                v = v_f32_sel_less_f32_b(lane, (float) (ne0 - (e)), v, -1e30f); \
            }
            float64 mxs[4] = {mx, mx, mx, mx};
            for (int e = 0; e < ne0; e += 256) {
                float64 v0, v1, v2, v3;
                LDV(e, v0) LDV(e + 64, v1) LDV(e + 128, v2) LDV(e + 192, v3)
                mxs[0] = v_f32_max_b(mxs[0], v0); mxs[1] = v_f32_max_b(mxs[1], v1);
                mxs[2] = v_f32_max_b(mxs[2], v2); mxs[3] = v_f32_max_b(mxs[3], v3);
            }
            mx = v_f32_max_b(v_f32_max_b(mxs[0], mxs[1]), v_f32_max_b(mxs[2], mxs[3]));
            float64 M;
            NL_REDUCE_MAX(mx, IDX, M);
            float64 acs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int e = 0; e < ne0; e += 256) {
                float64 v0, v1, v2, v3;
                LDV(e, v0) LDV(e + 64, v1) LDV(e + 128, v2) LDV(e + 192, v3)
                acs[0] += exp_p(v0 - M); acs[1] += exp_p(v1 - M); acs[2] += exp_p(v2 - M); acs[3] += exp_p(v3 - M);
            }
            const float64 acc = (acs[0] + acs[1]) + (acs[2] + acs[3]);
            float64 S;
            NL_REDUCE_ADD(acc, IDX, S);
            if (has_sinks) {
                S = S + nl_exp(sk - M);
            }
            const float64 is = nl_recip(S);
#if defined(OUT_F16) || defined(OUT_BF16)
            for (int e = 0; e < ne0; e += 256) {
                float64 v0, v1, v2, v3;
                LDV(e, v0) LDV(e + 64, v1) LDV(e + 128, v2) LDV(e + 192, v3)
                const float64 o0 = exp_p(v0 - M) * is, o1 = exp_p(v1 - M) * is;
                const float64 o2 = exp_p(v2 - M) * is, o3 = exp_p(v3 - M) * is;
                const int n0 = ne0 - e, n1 = ne0 - e - 128;
                int5 co = {r * ne0 + e, 0, 0, 0, 0};
#if defined(OUT_F16)
                const half128 y0 = cvt_f16(o0, o1), y1 = cvt_f16(o2, o3);
                if (n0 >= 128) v_f16_st_tnsr(co, out, y0); else v_f16_st_tnsr_partial(co, out, y0, n0 - 1, 0);
                co[0] += 128;
                if (n1 >= 128) v_f16_st_tnsr(co, out, y1); else if (n1 > 0) v_f16_st_tnsr_partial(co, out, y1, n1 - 1, 0);
#else
                const bfloat128 y0 = cvt_bf16(o0, o1), y1 = cvt_bf16(o2, o3);
                if (n0 >= 128) v_bf16_st_tnsr(co, out, y0); else v_bf16_st_tnsr_partial(co, out, y0, n0 - 1, 0);
                co[0] += 128;
                if (n1 >= 128) v_bf16_st_tnsr(co, out, y1); else if (n1 > 0) v_bf16_st_tnsr_partial(co, out, y1, n1 - 1, 0);
#endif
            }
#else
            for (int e = 0; e < ne0; e += 64) {
                float64 v;
                LDV(e, v)
                ST_OUT(out, r * ne0 + e, exp_p(v - M) * is, ne0 - e);
            }
#endif
#undef LDV
        }
    }
}

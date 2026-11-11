// Hyper-connection mixing (ggml_dsv4_hc_pre / _gated, ggml_dsv4_hc_post; DeepSeek V4, Qwen3.8-Flash-Next), f32.
// -DPRE : out[i, t] = scale * sum_h x[i, h, t] * w(i, h, t); gated: w = sigmoid(gate[i, h, t]), else w = weights[h, t]
//         x: element offset + strides of h / t (dim 0 contiguous); gate likewise; weights: strides of h / t
// -DPOST: out[i, d, t] = x[i, t] * post[d, t] + sum_s residual[i, s, t] * comb[d, s, t]  (has_comb), else
//         x[i, t] * post[d, t] + residual[i, d, t]; out contiguous [ne0, hc, T].
//         sig: post[d, t] = pb * sigmoid(pa * p[d, t])
// -DNORM (with -DPOST): also the grouped RMSNorm of each output row,
//         xn[:, d, t] = out[:, d, t] / rms(out[:, d, t]) * gamma[:, d]; -DOUT_BF16: a bf16 copy of xn as well
// hc <= HCMAX. Sums in the CPU's order (x * w first, then the terms in h / s order).
// tensors: PRE 0 x  1 w | 2 out;  POST 0 x  1 residual  2 post  3 comb | 4 out;
//          NORM 0 x  1 residual  2 post  3 comb  4 gamma | 5 out  6 xn [7 xn bf16]
#include "syn_common.h"

#define HCMAX 8
#define MAXN 8192

static inline float lds(tensor t, int i)
{
    int5 c = {i, 0, 0, 0, 0};
    return s_f32_ld_g((__global float *) gen_addr(c, t));
}

#if defined(PRE)
void main(tensor x, tensor w, tensor dep, tensor out, int nmem, int ne0, int hc, int T, int offx, int sx1, int sx2,
          int offw, int sw0, int sw1, int sw2, float scale, int gated, int chunk)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nch = (ne0 + chunk - 1) / chunk;
    const int nitems = T * nch;
    float wt[HCMAX];
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t = it / nch, c0 = (it - t * nch) * chunk;
            const int c1 = c0 + chunk < ne0 ? c0 + chunk : ne0;
            const int px = offx + t * sx2, pw = offw + t * sw2;
            if (!gated) {
                for (int h = 0; h < hc; h++) {
                    wt[h] = lds(w, offw + h * sw0 + t * sw1);
                }
            }
            if (gated && hc == 4) {
                for (int e = c0; e < c1; e += 64) {
                    const float64 x0 = ld_f32(x, px + e), x1 = ld_f32(x, px + e + sx1);
                    const float64 x2 = ld_f32(x, px + e + 2 * sx1), x3 = ld_f32(x, px + e + 3 * sx1);
                    const float64 g0 = ld_f32(w, pw + e), g1 = ld_f32(w, pw + e + sw1);
                    const float64 g2 = ld_f32(w, pw + e + 2 * sw1), g3 = ld_f32(w, pw + e + 3 * sw1);
                    float64 acc = x0 * nl_sigm_acc(g0);
                    acc = acc + x1 * nl_sigm_acc(g1);
                    acc = acc + x2 * nl_sigm_acc(g2);
                    acc = acc + x3 * nl_sigm_acc(g3);
                    ST_F32(out, t * ne0 + e, acc * scale, c1 - e);
                }
                continue;
            }
            for (int e = c0; e < c1; e += 64) {
                float64 acc = 0.0f;
                for (int h = 0; h < hc; h++) {
                    const float64 xv = ld_f32(x, px + e + h * sx1);
                    if (gated) {
                        acc = acc + xv * nl_sigm_acc(ld_f32(w, pw + e + h * sw1));
                    } else {
                        acc = acc + xv * wt[h];
                    }
                }
                ST_F32(out, t * ne0 + e, acc * scale, c1 - e);
            }
        }
    }
}
#elif defined(NORM)
__local__ float64 YV[MAXN / 64];

void main(tensor x, tensor res, tensor post, tensor comb, tensor gamma, tensor dep, tensor out, tensor xn,
#if defined(OUT_BF16)
          tensor xb,
#endif
          int nmem, int ne0, int hc, int T, int offx, int sx1, int offr, int sr1, int sr2, int offp, int sp0, int sp1,
          int offc, int sc0, int sc1, int sc2, int has_comb, int sig, float pa, float pb, int offg, int sg1, float eps)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float inv = 1.0f / (float) ne0;
    const int nitems = T * hc;
    float cv[HCMAX];
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t = it / hc, d = it - t * hc;
            const float ps = lds(post, offp + d * sp0 + t * sp1);
            const float64 pv = sig ? pb * nl_sigm_acc((float64) (pa * ps)) : (float64) ps;
            if (has_comb) {
                for (int s = 0; s < hc; s++) {
                    cv[s] = lds(comb, offc + d * sc0 + s * sc1 + t * sc2);
                }
            }
            const int px = offx + t * sx1, pr = offr + t * sr2, po = (t * hc + d) * ne0;
            float64 acc0 = 0.0f, acc1 = 0.0f;
            for (int e = 0; e < ne0; e += 128) {
                const float64 xa = ld_f32(x, px + e), xb2 = ld_f32(x, px + e + 64);
                float64 ya, yb;
                if (has_comb) {
                    ya = xa * pv;
                    yb = xb2 * pv;
                    for (int s = 0; s < hc; s++) {
                        ya = ya + ld_f32(res, pr + s * sr1 + e) * cv[s];
                        yb = yb + ld_f32(res, pr + s * sr1 + e + 64) * cv[s];
                    }
                } else {
                    const float64 ra = ld_f32(res, pr + d * sr1 + e), rb = ld_f32(res, pr + d * sr1 + e + 64);
                    ya = xa * pv + ra;
                    yb = xb2 * pv + rb;
                }
                ST_F32(out, po + e, ya, 64);
                ST_F32(out, po + e + 64, yb, 64);
                YV[e >> 6] = ya;
                YV[(e >> 6) + 1] = yb;
                acc0 = v_f32_mac_b(ya, ya, acc0);
                acc1 = v_f32_mac_b(yb, yb, acc1);
            }
            float64 ss;
            NL_REDUCE_ADD(acc0 + acc1, IDX, ss);
            const float64 r = nl_rsqrt(ss * inv + eps);
            const int pg = offg + d * sg1;
            for (int e = 0; e < ne0; e += 128) {
                const float64 ga = ld_f32(gamma, pg + e), gb = ld_f32(gamma, pg + e + 64);
                const float64 na = YV[e >> 6] * r * ga, nb = YV[(e >> 6) + 1] * r * gb;
                ST_F32(xn, po + e, na, 64);
                ST_F32(xn, po + e + 64, nb, 64);
#if defined(OUT_BF16)
                int5 cb = {po + e, 0, 0, 0, 0};
                v_bf16_st_tnsr(cb, xb, cvt_bf16(na, nb));
#endif
            }
        }
    }
}
#else
void main(tensor x, tensor res, tensor post, tensor comb, tensor dep, tensor out, int nmem, int ne0, int hc, int T,
          int offx, int sx1, int offr, int sr1, int sr2, int offp, int sp0, int sp1, int offc, int sc0, int sc1, int sc2,
          int has_comb, int sig, float pa, float pb, int chunk)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nch = (ne0 + chunk - 1) / chunk;
    const int nitems = T * nch;
    float64 pv[HCMAX];
    float cv[HCMAX * HCMAX];
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t = it / nch, c0 = (it - t * nch) * chunk;
            const int c1 = c0 + chunk < ne0 ? c0 + chunk : ne0;
            for (int d = 0; d < hc; d++) {
                const float ps = lds(post, offp + d * sp0 + t * sp1);
                pv[d] = sig ? pb * nl_sigm_acc((float64) (pa * ps)) : (float64) ps;
                if (has_comb) {
                    for (int s = 0; s < hc; s++) {
                        cv[d * HCMAX + s] = lds(comb, offc + d * sc0 + s * sc1 + t * sc2);
                    }
                }
            }
            if (!has_comb && hc == 4) {
                for (int e = c0; e < c1; e += 64) {
                    const float64 xv = ld_f32(x, offx + e + t * sx1);
                    const int pr = offr + e + t * sr2;
                    const float64 r0 = ld_f32(res, pr), r1 = ld_f32(res, pr + sr1);
                    const float64 r2 = ld_f32(res, pr + 2 * sr1), r3 = ld_f32(res, pr + 3 * sr1);
                    const int po = t * 4 * ne0 + e;
                    ST_F32(out, po, xv * pv[0] + r0, c1 - e);
                    ST_F32(out, po + ne0, xv * pv[1] + r1, c1 - e);
                    ST_F32(out, po + 2 * ne0, xv * pv[2] + r2, c1 - e);
                    ST_F32(out, po + 3 * ne0, xv * pv[3] + r3, c1 - e);
                }
                continue;
            }
            for (int e = c0; e < c1; e += 64) {
                const float64 xv = ld_f32(x, offx + e + t * sx1);
                for (int d = 0; d < hc; d++) {
                    float64 y = xv * pv[d];
                    if (has_comb) {
                        for (int s = 0; s < hc; s++) {
                            y = y + ld_f32(res, offr + e + s * sr1 + t * sr2) * cv[d * HCMAX + s];
                        }
                    } else {
                        y = y + ld_f32(res, offr + e + d * sr1 + t * sr2);
                    }
                    ST_F32(out, (t * hc + d) * ne0 + e, y, c1 - e);
                }
            }
        }
    }
}
#endif

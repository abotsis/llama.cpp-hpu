// Row norms over dim 0, f32 in. out contiguous [ne0, ne1, ne2, ne3]; a is a view.
// op: 0 rms_norm, 1 l2_norm, 2 norm (mean/variance)
// wmode 1: multiply by w[offw + e] (gamma, broadcast over rows)
// -DADD: the normalized value is x = a + b (b a view); x is also written to output xo. b may be nrb
//        slices db elements apart, summed (a tensor-parallel all-gather: the ranks' partial sums)
// -DADD -DMOE: x = a + hc * c + sum_k wv[k] * b_k (a MoE layer's output: the residual, the shared expert,
//        the routed experts' outputs weighted per token; wv[offwv + k sw1 + token sw2])
//        -DIND: b_k for token t is row pr[t nrb + k] of b (rows of ne0: the paged MUL_MAT_ID output; -1: none;
//        offb / sb* / db unused)
// -DOUT_BF16: out is bf16
// -DROPE: NEOX rope of the first n_dims elements after the norm (tab: host table, row = token i2,
//         [cos(0..n_dims/2) | sin(..)]); n_dims/2 == 32 or a multiple of 64
// tensors: 0 a  1 b  2 w  [3 tab]  dep | out [xo]
#include "syn_common.h"

#define MAXV 128
__local__ float64 XV[MAXV];

#if defined(MOE)
__local__ float WK[16];
#if defined(IND)
__local__ int RK[16];
void main(tensor a, tensor b, tensor w, tensor c, tensor wv, tensor pr, tensor dep, tensor out, tensor xo,
#else
void main(tensor a, tensor b, tensor w, tensor c, tensor wv, tensor dep, tensor out, tensor xo,
#endif
#elif defined(ADD)
void main(tensor a, tensor b, tensor w, tensor dep, tensor out, tensor xo,
#elif defined(ROPE)
void main(tensor a, tensor b, tensor w, tensor tab, tensor dep, tensor out,
#else
void main(tensor a, tensor b, tensor w, tensor dep, tensor out,
#endif
          int nmem, int op, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03, float eps, int wmode, int offw,
          int offb, int sb1, int sb2, int sb3
#if defined(ROPE)
          , int n_dims
#endif
#if defined(ADD)
          , int nrb, int db
#endif
#if defined(MOE)
          , int offc, int sc1, float hc, int offwv, int sw1, int sw2
#endif
          )
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int nrows = ne1 * ne2 * ne3;
    const float inv_n = 1.0f / (float) ne0;
#if defined(MOE)
    // the routing weights (and IND rows) are scalar loads: drop lines cached by an earlier launch at the same address
    scalar_cache_reset();
#endif
#if !defined(ADD) && !defined(ROPE)
    if (ne0 == 128 && op != 2) {
        for (int m = ioff[0]; m < iend[0]; m++) {
            int r0, r1e;
            SPLIT_RANGE(nrows, nmem, m, r0, r1e)
            if (r0 >= r1e) {
                continue;
            }
            int r3 = r0 / (ne1 * ne2);
            const int rr = r0 - r3 * ne1 * ne2;
            int r2 = rr / ne1;
            int r1 = rr - r2 * ne1;
            for (int r = r0; r < r1e; r += 4) {
                float64 x[8], q[4], ss[4];
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    int5 c = {off0 + r1 * s01 + r2 * s02 + r3 * s03, 0, 0, 0, 0};
                    x[2 * u] = v_f32_ld_tnsr_b(c, a);
                    c[0] += 64;
                    x[2 * u + 1] = v_f32_ld_tnsr_b(c, a);
                    q[u] = v_f32_mac_b(x[2 * u + 1], x[2 * u + 1], x[2 * u] * x[2 * u]);
                    if (++r1 == ne1) {
                        r1 = 0;
                        if (++r2 == ne2) {
                            r2 = 0;
                            r3++;
                        }
                    }
                }
                NL_REDUCE4_ADD(q[0], q[1], q[2], q[3], IDX, ss[0], ss[1], ss[2], ss[3])
                float64 g0 = 1.0f, g1 = 1.0f;
                if (wmode) {
                    g0 = ld_f32(w, offw);
                    g1 = ld_f32(w, offw + 64);
                }
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float64 sc = op == 1 ? nl_rsqrt(v_f32_max_b(ss[u], eps * eps)) : nl_rsqrt(ss[u] * inv_n + eps);
                    x[2 * u] = x[2 * u] * sc * g0;
                    x[2 * u + 1] = x[2 * u + 1] * sc * g1;
                }
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    if (r + u < r1e) {
                        ST_OUT(out, (r + u) * 128, x[2 * u], 64);
                        ST_OUT(out, (r + u) * 128 + 64, x[2 * u + 1], 64);
                    }
                }
            }
        }
        return;
    }
#endif
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
#if defined(MOE) && defined(IND)
            // the token's expert rows in the pages (slots without a row skipped)
            const int pc = offc + r1 * sc1;
            int nk = 0;
            for (int k_ = 0; k_ < nrb; k_++) {
                int5 cr = {r1 * nrb + k_, 0, 0, 0, 0};
                const int rw = s_i32_ld_g((__global int *) gen_addr(cr, pr));
                int5 cw = {offwv + k_ * sw1 + r1 * sw2, 0, 0, 0, 0};
                const float wk = s_f32_ld_g((__global float *) gen_addr(cw, wv));
                if (rw >= 0) {
                    RK[nk] = rw * ne0;
                    WK[nk] = wk;
                    nk++;
                }
            }
#define LDX(e) ({ float64 x_ = v_f32_mac_b(ld_f32(c, pc + (e)), hc, ld_f32(a, pa + (e))); \
                  for (int k_ = 0; k_ < nk; k_++) { x_ = v_f32_mac_b(ld_f32(b, RK[k_] + (e)), WK[k_], x_); } x_; })
#elif defined(MOE)
            const int pb = offb + r1 * sb1 + r2 * sb2 + r3 * sb3;
            const int pc = offc + r1 * sc1;
            for (int k_ = 0; k_ < nrb; k_++) {
                int5 cw = {offwv + k_ * sw1 + r1 * sw2, 0, 0, 0, 0};
                WK[k_] = s_f32_ld_g((__global float *) gen_addr(cw, wv));
            }
#define LDX(e) ({ float64 x_ = v_f32_mac_b(ld_f32(c, pc + (e)), hc, ld_f32(a, pa + (e))); \
                  for (int k_ = 0; k_ < nrb; k_++) { x_ = v_f32_mac_b(ld_f32(b, pb + k_ * db + (e)), WK[k_], x_); } x_; })
#elif defined(ADD)
            const int pb = offb + r1 * sb1 + r2 * sb2 + r3 * sb3;
#define LDX(e) ({ float64 x_ = ld_f32(a, pa + (e)) + ld_f32(b, pb + (e)); \
                  for (int k_ = 1; k_ < nrb; k_++) { x_ += ld_f32(b, pb + k_ * db + (e)); } x_; })
#else
#define LDX(e) ld_f32(a, pa + (e))
#endif
            if (op != 2 && ne0 % 256 == 0 && ne0 <= 64 * MAXV) {
                float64 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
                for (int e = 0; e < ne0; e += 256) {
                    const float64 x0 = LDX(e), x1 = LDX(e + 64), x2 = LDX(e + 128), x3 = LDX(e + 192);
                    XV[(e >> 6)] = x0; XV[(e >> 6) + 1] = x1; XV[(e >> 6) + 2] = x2; XV[(e >> 6) + 3] = x3;
#if defined(ADD)
                    ST_F32(xo, r * ne0 + e, x0, 64);
                    ST_F32(xo, r * ne0 + e + 64, x1, 64);
                    ST_F32(xo, r * ne0 + e + 128, x2, 64);
                    ST_F32(xo, r * ne0 + e + 192, x3, 64);
#endif
                    c0 = v_f32_mac_b(x0, x0, c0); c1 = v_f32_mac_b(x1, x1, c1);
                    c2 = v_f32_mac_b(x2, x2, c2); c3 = v_f32_mac_b(x3, x3, c3);
                }
                float64 ss;
                NL_REDUCE_ADD((c0 + c1) + (c2 + c3), IDX, ss);
                const float64 sc = op == 1 ? nl_rsqrt(v_f32_max_b(ss, eps * eps)) : nl_rsqrt(ss * inv_n + eps);
#if defined(ROPE)
                {
                    const int r2 = (r / ne1) % ne2;
                    const int pt = r2 * n_dims;
                    const int hd = n_dims >> 1;
                    for (int e = 0; e < ne0; e += 64) {
                        float64 y = XV[e >> 6] * sc;
                        if (wmode) {
                            y = y * ld_f32(w, offw + e);
                        }
                        XV[e >> 6] = y;
                    }
                    if (hd == 32) {
                        const float64 t = ld_f32(tab, pt);
                        const float64 ts = SWAP_HALVES(t);
                        const float64 cv = v_f32_sel_less_f32_b(lane, 32.0f, t, ts);
                        const float64 sv = v_f32_sel_less_f32_b(lane, 32.0f, -ts, t);
                        const float64 y = XV[0];
                        XV[0] = v_f32_mac_b(SWAP_HALVES(y), sv, y * cv);
                    } else {
                        for (int e = 0; e < hd; e += 64) {
                            const float64 c = ld_f32(tab, pt + e), sn = ld_f32(tab, pt + hd + e);
                            const float64 x0 = XV[e >> 6], x1 = XV[(hd + e) >> 6];
                            XV[e >> 6] = x0 * c - x1 * sn;
                            XV[(hd + e) >> 6] = v_f32_mac_b(x1, c, x0 * sn);
                        }
                    }
                    for (int e = 0; e < ne0; e += 64) {
                        ST_OUT(out, r * ne0 + e, XV[e >> 6], 64);
                    }
                    continue;
                }
#endif
                for (int e = 0; e < ne0; e += 128) {
                    float64 y0 = XV[e >> 6] * sc, y1 = XV[(e >> 6) + 1] * sc;
                    if (wmode) {
                        y0 = y0 * ld_f32(w, offw + e);
                        y1 = y1 * ld_f32(w, offw + e + 64);
                    }
#if defined(OUT_BF16)
                    int5 co = {r * ne0 + e, 0, 0, 0, 0};
                    v_bf16_st_tnsr(co, out, cvt_bf16(y0, y1));
#else
                    ST_F32(out, r * ne0 + e, y0, 64);
                    ST_F32(out, r * ne0 + e + 64, y1, 64);
#endif
                }
                continue;
            }
            float64 mean = 0.0f;
            if (op == 2) {
                float64 acc = 0.0f;
                for (int e = 0; e < ne0; e += 64) {
                    acc += v_f32_sel_less_f32_b(lane, (float) (ne0 - e), LDX(e), 0.0f);
                }
                NL_REDUCE_ADD(acc, IDX, mean);
                mean = mean * inv_n;
            }
            float64 acc = 0.0f;
            for (int e = 0; e < ne0; e += 64) {
                const float64 x = LDX(e);
#if defined(ADD)
                ST_F32(xo, r * ne0 + e, x, ne0 - e);
#endif
                const float64 xm = v_f32_sel_less_f32_b(lane, (float) (ne0 - e), x - mean, 0.0f);
                acc = v_f32_mac_b(xm, xm, acc);
            }
            float64 ss;
            NL_REDUCE_ADD(acc, IDX, ss);
            float64 sc;
            if (op == 1) {
                sc = nl_rsqrt(v_f32_max_b(ss, eps * eps));
            } else {
                sc = nl_rsqrt(ss * inv_n + eps);
            }
            for (int e = 0; e < ne0; e += 64) {
                float64 y = (LDX(e) - mean) * sc;
                if (wmode) {
                    y = y * ld_f32(w, offw + e);
                }
                ST_OUT(out, r * ne0 + e, y, ne0 - e);
            }
#undef LDX
        }
    }
}

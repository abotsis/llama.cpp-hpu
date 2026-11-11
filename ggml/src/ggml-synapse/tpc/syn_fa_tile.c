// Tiled flash attention over a window of W keys (one launch of the window recipe; the host launches it once per window).
// Scores s [W, C] f32 (row c = iq + nq h contiguous over the window's keys), running state ml [4 C] f32:
// [0, C) 1 / l at h + H iq, [C, 2C) m, [2C, 3C) l, [3C, 4C) alpha = exp(m_old - m_new) of this window (rows c).
// Table (i32): [0] first window, [2 + 4 iq ..] lo1 hi1 lo2 hi2: keys [lo1, hi1) and [lo2, hi2) of the window are valid for
// query iq (two ranges: a ring-buffer cache that wrapped).
//
// -DSM: online softmax step. tensors: 0 s  1 tab  2 ml_in  [3 sinks (-DSINKS)]  dep | p (f16 -DOUT_F16 / bf16
//   -DOUT_BF16 [W, C])  ml_out. p = exp(s scale - m_new) on valid keys, 0 elsewhere (every key of the window written).
//   Sinks: the first window starts from m = sink[h], l = 1 (the sink's logit, no value).
// -DMRG: accumulator merge. tensors: 0 o [Rv, C]  1 ml (this window's ml_out)  2 acc_in  3 tab  4 dep | 5 acc_out
//   acc_out = first ? o : acc_in alpha + o, on the Dv rows of row c's KV head only (the rest is never read)
// -DMASK: the window's additive bf16 mask for the native SDPA. tensors: 0 tab  1 dep | 2 mask [W, nq]:
//   0 on the valid keys of query iq, -30000 elsewhere (finite: a row without keys in this window gets ~0 weight)
// -DPMRG: merge of a normalized SDPA window result. tensors: 0 o [D, rows] f32  1 mx [rows]  2 linv [rows]  3 acc_in
//   4 ml_in [2 rows] (m, l)  5 tab  6 dep | 7 acc_out  8 ml_out. rows = nq H (row iq + nq h).
//   n = max(m, mx), wa = exp(m - n) l, ws = exp(mx - n) / linv: acc = (wa acc + ws o) / (wa + ws), m = n, l = wa + ws
#include "syn_common.h"

#if defined(SM)
// exp(x), x <= 0: degree-5 polynomial on |r| <= ln2 / 2 (~2e-7 relative), as syn_softmax_u.c
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

#if defined(OUT_F16)
#define ST_P(c, y) v_f16_st_tnsr(c, p, y)
#define CVT_P(a, b) cvt_f16(a, b)
#else
#define ST_P(c, y) v_bf16_st_tnsr(c, p, y)
#define CVT_P(a, b) cvt_bf16(a, b)
#endif

#if defined(SINKS)
void main(tensor s, tensor tab, tensor ml_in, tensor sinks, tensor dep, tensor p, tensor ml_out,
          int nmem, int W, int C, int nq, int H, float scale)
#else
void main(tensor s, tensor tab, tensor ml_in, tensor dep, tensor p, tensor ml_out,
          int nmem, int W, int C, int nq, int H, float scale)
#endif
{
    scalar_cache_reset();
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 ninf = as_float64((int64) 0xFF800000);
    int5 ct = {0, 0, 0, 0, 0};
    const int first = s_i32_ld_g(gen_addr(ct, tab));
    for (int m = ioff[0]; m < iend[0]; m++) {
        int c0, c1;
        SPLIT_RANGE(C, nmem, m, c0, c1)
        for (int c = c0; c < c1; c++) {
            const int h = c / nq;
            const int iq = c - h * nq;
            ct[0] = 2 + 4 * iq;
            int lo = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            int hi = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            int lo2 = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            int hi2 = s_i32_ld_g(gen_addr(ct, tab));
            hi = hi > W ? W : hi;
            lo = lo > hi ? hi : lo;
            hi2 = hi2 > W ? W : hi2;
            lo2 = lo2 > hi2 ? hi2 : lo2;
            float64 Mo = ninf, Lo = 0.0f;
            if (!first) {
                int5 cm = {C + c, 0, 0, 0, 0};
                const float m_old = s_f32_ld_g(gen_addr(cm, ml_in));
                cm[0] = 2 * C + c;
                const float l_old = s_f32_ld_g(gen_addr(cm, ml_in));
                Mo = m_old;
                Lo = l_old;
            }
#if defined(SINKS)
            else {
                int5 cs = {h, 0, 0, 0, 0};
                const float sk = s_f32_ld_g(gen_addr(cs, sinks));
                Mo = sk;
                Lo = 1.0f;
            }
#endif
            const int px = c * W;
            // pass 1: max of the valid scores (vectors spanning both ranges)
            float64 m0 = ninf, m1 = ninf;
            const int e1 = lo < hi, e2 = lo2 < hi2;
            const int klo = e1 && e2 ? (lo < lo2 ? lo : lo2) : (e1 ? lo : (e2 ? lo2 : 0));
            const int khi = e1 && e2 ? (hi > hi2 ? hi : hi2) : (e1 ? hi : (e2 ? hi2 : 0));
            const int v0 = klo >> 6, v1 = (khi + 63) >> 6;
            for (int v = v0; v < v1; v++) {
                const float64 x = ld_f32(s, px + 64 * v) * scale;
                const float64 xv = v_f32_sel_less_f32_b(lane, (float) (hi - 64 * v), x, ninf);
                const float64 xa = v_f32_sel_less_f32_b(lane, (float) (lo - 64 * v), ninf, xv);
                const float64 yv = v_f32_sel_less_f32_b(lane, (float) (hi2 - 64 * v), x, ninf);
                const float64 yb = v_f32_sel_less_f32_b(lane, (float) (lo2 - 64 * v), ninf, yv);
                const float64 xw = v_f32_max_b(xa, yb);
                if (v & 1) {
                    m1 = v_f32_max_b(m1, xw);
                } else {
                    m0 = v_f32_max_b(m0, xw);
                }
            }
            float64 MT;
            NL_REDUCE_MAX(v_f32_max_b(m0, m1), IDX, MT)
            const float64 Mn = v_f32_max_b(MT, Mo);
            const float64 M = v_f32_sel_eq_f32_b(Mn, ninf, 0.0f, Mn);
            // pass 2: p over the whole window (zeros outside [lo, hi))
            float64 s0 = 0.0f, s1 = 0.0f;
            int5 co = {px, 0, 0, 0, 0};
            for (int v = 0; v < (W >> 6); v += 2) {
                float64 p0 = 0.0f, p1 = 0.0f;
                if (v + 1 >= v0 && v < v1) {
                    const float64 x0 = ld_f32(s, px + 64 * v) * scale - M;
                    const float64 x1 = ld_f32(s, px + 64 * v + 64) * scale - M;
                    const float64 e0 = exp_p(x0), e1v = exp_p(x1);
                    const float64 k0 = lane + (float) (64 * v);
                    const float64 k1 = k0 + 64.0f;
                    // p = e on keys in either range, else 0 (both selections are e or 0: their max)
                    float64 a0 = v_f32_sel_less_f32_b(k0, (float) hi, e0, 0.0f);
                    a0 = v_f32_sel_less_f32_b(k0, (float) lo, 0.0f, a0);
                    float64 b0 = v_f32_sel_less_f32_b(k0, (float) hi2, e0, 0.0f);
                    b0 = v_f32_sel_less_f32_b(k0, (float) lo2, 0.0f, b0);
                    float64 a1 = v_f32_sel_less_f32_b(k1, (float) hi, e1v, 0.0f);
                    a1 = v_f32_sel_less_f32_b(k1, (float) lo, 0.0f, a1);
                    float64 b1 = v_f32_sel_less_f32_b(k1, (float) hi2, e1v, 0.0f);
                    b1 = v_f32_sel_less_f32_b(k1, (float) lo2, 0.0f, b1);
                    p0 = v_f32_max_b(a0, b0);
                    p1 = v_f32_max_b(a1, b1);
                    s0 += p0;
                    s1 += p1;
                }
                ST_P(co, CVT_P(p0, p1));
                co[0] += 128;
            }
            float64 S;
            NL_REDUCE_ADD(s0 + s1, IDX, S)
            // alpha scales the earlier state; exp(-inf - M) underflows to ~0 (no earlier keys)
            const float64 alpha = exp_p(Mo - M);
            const float64 L = v_f32_mac_b(Lo, alpha, S);
            const float64 il = v_f32_sel_grt_f32_b(L, 0.0f, nl_recip(v_f32_max_b(L, 1e-30f)), 0.0f);
            int5 cw = {h + H * iq, 0, 0, 0, 0};
            v_f32_st_tnsr_partial(cw, ml_out, il, 0, 0);
            cw[0] = C + c;
            v_f32_st_tnsr_partial(cw, ml_out, Mn, 0, 0);
            cw[0] = 2 * C + c;
            v_f32_st_tnsr_partial(cw, ml_out, L, 0, 0);
            cw[0] = 3 * C + c;
            v_f32_st_tnsr_partial(cw, ml_out, alpha, 0, 0);
        }
    }
}
#elif defined(MRG)
void main(tensor o, tensor ml, tensor acc_in, tensor tab, tensor dep, tensor acc_out,
          int nmem, int Rv, int C, int nq, int hpg, int Dv)
{
    scalar_cache_reset();
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    int5 ct = {0, 0, 0, 0, 0};
    const int first = s_i32_ld_g(gen_addr(ct, tab));
    for (int m = ioff[0]; m < iend[0]; m++) {
        int c0, c1;
        SPLIT_RANGE(C, nmem, m, c0, c1)
        for (int c = c0; c < c1; c++) {
            const int hk = (c / nq) / hpg;
            const int r0 = c * Rv + hk * Dv;
            int5 ca = {3 * C + c, 0, 0, 0, 0};
            const float alpha = s_f32_ld_g(gen_addr(ca, ml));
            for (int d = 0; d < Dv; d += 64) {
                const float64 x = ld_f32(o, r0 + d);
                float64 y = x;
                if (!first) {
                    y = v_f32_mac_b(ld_f32(acc_in, r0 + d), alpha, x);
                }
                int5 cw = {r0 + d, 0, 0, 0, 0};
                v_f32_st_tnsr(cw, acc_out, y);
            }
        }
    }
}
#elif defined(MASK)
void main(tensor tab, tensor dep, tensor mask, int nmem, int W, int nq)
{
    scalar_cache_reset();
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lane = LANE_F32();
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_RANGE(nq, nmem, m, r0, r1)
        for (int iq = r0; iq < r1; iq++) {
            int5 ct = {2 + 4 * iq, 0, 0, 0, 0};
            const int lo = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            const int hi = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            const int lo2 = s_i32_ld_g(gen_addr(ct, tab));
            ct[0]++;
            const int hi2 = s_i32_ld_g(gen_addr(ct, tab));
            int5 cw = {iq * W, 0, 0, 0, 0};
            for (int k = 0; k < W; k += 128) {
                const float64 k0 = lane + (float) k, k1 = k0 + 64.0f;
                float64 a = v_f32_sel_less_f32_b(k0, (float) hi, 0.0f, -30000.0f);
                a = v_f32_sel_less_f32_b(k0, (float) lo, -30000.0f, a);
                float64 a2 = v_f32_sel_less_f32_b(k0, (float) hi2, 0.0f, -30000.0f);
                a = v_f32_max_b(a, v_f32_sel_less_f32_b(k0, (float) lo2, -30000.0f, a2));
                float64 b = v_f32_sel_less_f32_b(k1, (float) hi, 0.0f, -30000.0f);
                b = v_f32_sel_less_f32_b(k1, (float) lo, -30000.0f, b);
                float64 b2 = v_f32_sel_less_f32_b(k1, (float) hi2, 0.0f, -30000.0f);
                b = v_f32_max_b(b, v_f32_sel_less_f32_b(k1, (float) lo2, -30000.0f, b2));
                v_bf16_st_tnsr(cw, mask, cvt_bf16(a, b));
                cw[0] += 128;
            }
        }
    }
}
#elif defined(PMRG)
void main(tensor o, tensor mx, tensor linv, tensor acc_in, tensor ml_in, tensor tab, tensor dep, tensor acc_out,
          tensor ml_out, int nmem, int D, int rows)
{
    scalar_cache_reset();
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    int5 ct = {0, 0, 0, 0, 0};
    const int first = s_i32_ld_g(gen_addr(ct, tab));
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_RANGE(rows, nmem, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            int5 cr = {r, 0, 0, 0, 0};
            const float64 ms = s_f32_ld_g(gen_addr(cr, mx));
            const float64 li = s_f32_ld_g(gen_addr(cr, linv));
            const float64 ws1 = nl_recip(v_f32_max_b(li, 1.0e-30f));
            float64 a = 0.0f, b = 1.0f, n = ms, den = ws1;
            if (!first) {
                cr[0] = r;
                const float64 am = s_f32_ld_g(gen_addr(cr, ml_in));
                cr[0] = rows + r;
                const float64 al = s_f32_ld_g(gen_addr(cr, ml_in));
                n = v_f32_max_b(am, ms);
                const float64 wa = nl_exp(am - n) * al;
                const float64 ws = nl_exp(ms - n) * ws1;
                den = wa + ws;
                const float64 inv = nl_recip(v_f32_max_b(den, 1.0e-30f));
                a = wa * inv;
                b = ws * inv;
            }
            for (int d = 0; d < D; d += 64) {
                const int pd = r * D + d;
                float64 y = ld_f32(o, pd) * b;
                if (!first) {
                    y = v_f32_mac_b(ld_f32(acc_in, pd), a, y);
                }
                int5 cw = {pd, 0, 0, 0, 0};
                v_f32_st_tnsr(cw, acc_out, y);
            }
            int5 cm = {r, 0, 0, 0, 0};
            v_f32_st_tnsr_partial(cm, ml_out, n, 0, 0);
            cm[0] = rows + r;
            v_f32_st_tnsr_partial(cm, ml_out, den, 0, 0);
        }
    }
}
#endif

// TOP_K of long rows (any n, k <= n) by radix select; indices in ascending element order.
// Keys: the f32 bits as an order-preserving i32 (desc: larger first).
//  -DTHR : per part of a row (np parts: the row's 64-element chunks split evenly), the part's k-th
//          largest key T (32 bit-steps of count(key >= t) >= k), then thr[r * np + part] =
//          {T, k - count(key > T)} and per chunk the counts of keys >= T and == T (cnt [2 nch, rows]:
//          the >= counts, then the == counts). np = 1: the row's threshold.
//          tensors: 0 a  1 dep | thr (i32 [2, np, rows])  cnt (f32 [2 nch, rows])
//  else  : keys > T, and keys == T while the equal budget lasts, written in order to out[r * k + o];
//          a row's chunks are split into np parts (each finds its output offset and remaining budget
//          from the counts of the chunks before its own).
//          indep: the parts' own thresholds (THR with the same np), part p's top k to
//          out[(r * np + p) * k + o] (each part >= k elements), with -DVALS also their values to vals:
//          the candidates of a two-stage top-k (the union of the parts' top k holds the row's, ties
//          included: a part keeps its earliest ones).
//          -DIND: the output is ind[r * n + e] (the second stage, over the candidates' values).
//          tensors: 0 a  1 thr  2 cnt  (3 ind)  dep | out  (vals)
// a is a view (element offset + strides of dims 1..3, dim 0 contiguous).
#include "syn_common.h"

#define MAXV 128

#if defined(THR)
__local__ int64 KV[MAXV];

static inline int64 key_of(float64 x, float64 lane, int left)
{
    const int64 i = as_int64(x);
    const int64 k = v_i32_sel_less_i32_b(i, (int64) 0, i ^ 0x7FFFFFFF, i);
    return v_i32_sel_less_f32_b(lane, (float) left, k, (int64) 0x80000000);
}

void main(tensor a, tensor dep, tensor thr, tensor cnt,
          int nmem, int n, int k, int ne1, int ne2, int ne3, int off0, int s01, int s02, int s03, int desc, int np)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int nv = (n + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nrows * np, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / np, part = it - r * np;
            const int c0 = part * nv / np, c1 = (part + 1) * nv / np;
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
            const float sg = desc ? 1.0f : -1.0f;
            const bool local = c1 - c0 <= MAXV;
            if (local) {
                for (int c = c0; c < c1; c++) {
                    KV[c - c0] = key_of(ld_f32(a, pa + 64 * c) * sg, lane, n - 64 * c);
                }
            }
            int64 tu = 0;
            for (int b = 31; b >= 0; b--) {
                const int64 cu = tu | (int64) (1u << b);
                const int64 t = cu ^ (int64) 0x80000000;
                float64 acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                if (local) {
                    for (int c = 0; c < c1 - c0; c++) {
                        acc[c & 3] += v_f32_sel_geq_i32_b(KV[c], t, (float64) 1.0f, (float64) 0.0f);
                    }
                } else {
                    for (int c = c0; c < c1; c++) {
                        const int64 kk = key_of(ld_f32(a, pa + 64 * c) * sg, lane, n - 64 * c);
                        acc[c & 3] += v_f32_sel_geq_i32_b(kk, t, (float64) 1.0f, (float64) 0.0f);
                    }
                }
                float64 tot;
                NL_REDUCE_ADD((acc[0] + acc[1]) + (acc[2] + acc[3]), IDX, tot)
                tu = v_i32_sel_geq_f32_b(tot, (float) k, cu, tu);
            }
            const int64 T = tu ^ (int64) 0x80000000;
            float64 gt = 0.0f;
            for (int c = c0; c < c1; c++) {
                const int64 kk = local ? KV[c - c0] : key_of(ld_f32(a, pa + 64 * c) * sg, lane, n - 64 * c);
                const float64 g = v_f32_sel_grt_i32_b(kk, T, (float64) 1.0f, (float64) 0.0f);
                gt += g;
                float64 ch, ce;
                const float64 ge = v_f32_sel_geq_i32_b(kk, T, (float64) 1.0f, (float64) 0.0f);
                NL_REDUCE_ADD(ge, IDX, ch)
                NL_REDUCE_ADD(ge - g, IDX, ce)
                int5 cc = {2 * r * nv + c, 0, 0, 0, 0};
                v_f32_st_tnsr_partial(cc, cnt, ch, 0, 0);
                cc[0] += nv;
                v_f32_st_tnsr_partial(cc, cnt, ce, 0, 0);
            }
            float64 cgt;
            NL_REDUCE_ADD(gt, IDX, cgt)
            const int64 req = (int64) k - v_convert_f32_to_i32_b(cgt, SW_RHNE, 0, 1, 0);
            int5 ct = {2 * it, 0, 0, 0, 0};
            v_i32_st_tnsr_partial(ct, thr, v_i32_sel_less_f32_b(lane, 0.5f, T, req), 1, 0);
        }
    }
}
#else
#if defined(IND)
#define IND_T , tensor ind
#else
#define IND_T
#endif
#if defined(VALS)
#define VALS_T , tensor vals
#else
#define VALS_T
#endif
void main(tensor a, tensor thr, tensor cnt IND_T, tensor dep, tensor out VALS_T,
          int nmem, int n, int k, int ne1, int ne2, int ne3, int off0, int s01, int s02, int s03, int desc, int np,
          int indep)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int64 lane = (int64) read_lane_id_4b_b();
    const int nv = (n + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nrows * np, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / np, part = it - r * np;
            const int c0 = part * nv / np, c1 = (part + 1) * nv / np;
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
            int5 ct = {indep ? 2 * it : 2 * r, 0, 0, 0, 0};
            const int T = s_i32_ld_g((__global int *) gen_addr(ct, thr));
            ct[0] += 1;
            int req = s_i32_ld_g((__global int *) gen_addr(ct, thr));
            // the chunks before this part: their keys > T and the equal budget they used
            int o = 0, eqb = 0;
            for (int c = 0; c < (indep ? 0 : c0); c++) {
                int5 cc = {2 * r * nv + c, 0, 0, 0, 0};
                const int ge = (int) s_f32_ld_g((__global float *) gen_addr(cc, cnt));
                cc[0] += nv;
                const int eq = (int) s_f32_ld_g((__global float *) gen_addr(cc, cnt));
                o += ge - eq;
                eqb += eq;
            }
            const int used = eqb < req ? eqb : req;
            o += used;
            req -= used;
            int g = (indep ? it : r) * k + o;
            int vb = g & ~63, va = g;
            int64 v = 0;
#if defined(VALS)
            float64 fv = 0.0f;
#endif
            for (int c = c0; c < c1 && o < k; c++) {
                int5 cc = {2 * r * nv + c, 0, 0, 0, 0};
                if (s_f32_ld_g((__global float *) gen_addr(cc, cnt)) == 0.0f) {
                    continue;
                }
                const int e1 = 64 * c + 64 < n ? 64 * c + 64 : n;
                for (int e = 64 * c; e < e1 && o < k; e++) {
                    int5 ca = {pa + e, 0, 0, 0, 0};
                    const float x0 = s_f32_ld_g((__global float *) gen_addr(ca, a));
                    float x = x0;
                    if (!desc) {
                        x = -x;
                    }
                    const int i = as_int(x);
                    const int kk = i < 0 ? (i ^ 0x7FFFFFFF) : i;
                    if (kk > T || (kk == T && req > 0)) {
                        if (kk == T) {
                            req--;
                        }
#if defined(IND)
                        int5 cn = {r * n + e, 0, 0, 0, 0};
                        const int ev = s_i32_ld_g((__global int *) gen_addr(cn, ind));
#else
                        const int ev = e;
#endif
                        v = v_i32_sel_eq_i32_b(lane, (int64) (g - vb), (int64) ev, v);
#if defined(VALS)
                        fv = v_f32_sel_eq_i32_b(lane, (int64) (g - vb), (float64) x0, fv);
#endif
                        g++;
                        o++;
                        if (g - vb == 64) {
                            int5 co = {va, 0, 0, 0, 0};
                            v_i32_st_tnsr_partial(co, out, v, g - va - 1, va - vb);
#if defined(VALS)
                            v_f32_st_tnsr_partial(co, vals, fv, g - va - 1, va - vb);
#endif
                            vb = g;
                            va = g;
                        }
                    }
                }
            }
            if (va < g) {
                int5 co = {va, 0, 0, 0, 0};
                v_i32_st_tnsr_partial(co, out, v, g - va - 1, va - vb);
#if defined(VALS)
                v_f32_st_tnsr_partial(co, vals, fv, g - va - 1, va - vb);
#endif
            }
        }
    }
}
#endif

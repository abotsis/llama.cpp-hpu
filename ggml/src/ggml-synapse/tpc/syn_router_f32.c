// MoE routing with sigmoid scores (DeepSeek-V3 / GLM style), per token row:
//   p = sigmoid(l), s = p + bias; ids = the k largest s (ties: lowest index, as syn_argsort),
//   w = p[ids] (/ max(sum, cmin) when norm) * scale
// smax: softmax scores (Qwen style), p = softmax(l), s = p (bias not read)
// ids are the first k of the token's argsort row (ids [E, T] i32, row stride E); w is [k, T].
// l is a view (element offset + token stride, dim 0 contiguous); bias [E] at offb. E <= 64 * MAXV.
// tensors: 0 l  1 bias | 2 ids  3 w
#include "syn_common.h"

#define MAXV 8

__local__ float64 S[MAXV];
__local__ float64 P[MAXV];

void main(tensor l, tensor bias, tensor dep, tensor ids, tensor w,
          int nmem, int E, int T, int k, int offl, int sl1, int offb, float cmin, float scale, int norm, int smax)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 ninf = as_float64((int64) 0xFF800000);
    const int nv = (E + 63) >> 6;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int t0, t1;
        SPLIT_RANGE(T, nmem, m, t0, t1)
        for (int t = t0; t < t1; t++) {
            if (smax) {
                // max, exp(l - max) and its sum, then p = e / sum (the CPU's order); lanes past E never win
                float64 bm = ninf;
                for (int c = 0; c < nv; c++) {
                    const float64 x = v_f32_sel_less_f32_b(lane, (float) (E - 64 * c), ld_f32(l, offl + t * sl1 + 64 * c), ninf);
                    S[c] = x;
                    bm = v_f32_max_b(bm, x);
                }
                float64 M;
                NL_REDUCE_MAX(bm, IDX, M)
                float64 acc = 0.0f;
                for (int c = 0; c < nv; c++) {
                    const float64 e = v_f32_sel_less_f32_b(lane, (float) (E - 64 * c), nl_exp(S[c] - M), 0.0f);
                    P[c] = e;
                    acc = acc + e;
                }
                float64 sum;
                NL_REDUCE_ADD(acc, IDX, sum);
                float64 inv = nl_recip(sum);
                inv = v_f32_mac_b(inv, v_f32_mac_b(sum, -inv, (float64) 1.0f), inv);
                for (int c = 0; c < nv; c++) {
                    const float64 p = P[c] * inv;
                    P[c] = p;
                    S[c] = v_f32_sel_less_f32_b(lane, (float) (E - 64 * c), p, ninf);
                }
            } else {
                for (int c = 0; c < nv; c++) {
                    const float64 p = nl_sigm_acc(ld_f32(l, offl + t * sl1 + 64 * c));
                    P[c] = p;
                    // lanes past E never win
                    S[c] = v_f32_sel_less_f32_b(lane, (float) (E - 64 * c), p + ld_f32(bias, offb + 64 * c), ninf);
                }
            }
            float64 iv = 0.0f, wv = 0.0f;
            for (int j = 0; j < k; j++) {
                float64 bm = ninf;
                for (int c = 0; c < nv; c++) {
                    bm = v_f32_max_b(bm, S[c]);
                }
                float64 M;
                NL_REDUCE_MAX(bm, IDX, M)
                float64 bi = 1e30f;
                for (int c = 0; c < nv; c++) {
                    bi = v_f32_min_b(bi, v_f32_sel_eq_f32_b(S[c], M, lane + (float) (64 * c), 1e30f));
                }
                float64 ni;
                NL_REDUCE_MAX(-bi, IDX, ni)
                const float64 idx = -ni;
                float64 pm = ninf;
                for (int c = 0; c < nv; c++) {
                    const float64 hit = v_f32_sel_eq_f32_b(lane + (float) (64 * c), idx, 1.0f, 0.0f);
                    pm = v_f32_max_b(pm, v_f32_sel_eq_f32_b(hit, 1.0f, P[c], ninf));
                    S[c] = v_f32_sel_eq_f32_b(hit, 1.0f, ninf, S[c]);   // taken
                }
                float64 pj;
                NL_REDUCE_MAX(pm, IDX, pj)
                iv = v_f32_sel_eq_f32_b(lane, (float) j, idx, iv);
                wv = v_f32_sel_eq_f32_b(lane, (float) j, pj, wv);
            }
            if (norm) {
                float64 sum;
                NL_REDUCE_ADD(v_f32_sel_less_f32_b(lane, (float) k, wv, 0.0f), IDX, sum);
                wv = wv * nl_recip(v_f32_max_b(sum, cmin));   // (as syn_binary div)
            }
            wv = wv * scale;
            int5 co = {t * E, 0, 0, 0, 0};
            v_i32_st_tnsr_partial(co, ids, v_convert_f32_to_i32_b(iv, SW_RHNE, 0, 1, 0), k - 1, 0);
            ST_F32(w, t * k, wv, k);
        }
    }
}

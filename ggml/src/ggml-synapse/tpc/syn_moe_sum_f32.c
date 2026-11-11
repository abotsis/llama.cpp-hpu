// A MoE layer's output without a norm after it (a tensor-parallel partial sum):
//   out[:, t] = hc * c[:, t] + sum_k wv[k, t] * b_k[:, t]   (hc == 0: c is not read; unw: unit weights, wv not read)
// b_k the expert outputs (slice k at offb + k db, token t at + t sb1), c the shared expert (offc + t sc1),
// wv the routing weights (offwv + k sw1 + t sw2). out [ne0, T] contiguous.
// nrb <= NRB_MAX (checked by the lowering).
// tensors: 0 b  1 c  2 wv | 3 out
// -DIND: b_k[:, t] is row pr[t nrb + k] of b (rows of ne0, paged MUL_MAT_ID output; -1: none), offb/db/sb1 unused.
// tensors: 0 b  1 c  2 wv  3 pr | 4 out
#include "syn_common.h"

#define NRB_MAX 32

#if defined(IND)
void main(tensor b, tensor c, tensor wv, tensor pr, tensor dep, tensor out, int nmem, int ne0, int T, int nrb, int db,
          int offb, int sb1, int offc, int sc1, float hc, int offwv, int sw1, int sw2, int unw)
#else
void main(tensor b, tensor c, tensor wv, tensor dep, tensor out, int nmem, int ne0, int T, int nrb, int db,
          int offb, int sb1, int offc, int sc1, float hc, int offwv, int sw1, int sw2, int unw)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nv = (ne0 + 63) >> 6;
    // the weights (and IND rows) are scalar loads: drop lines cached by an earlier launch at the same address
    scalar_cache_reset();
    int ro[NRB_MAX];
    float w[NRB_MAX];
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(T * nv, nmem, m, i0, i1)
        int t = i0 / nv;
        int e = (i0 - t * nv) * 64;
        for (int it = i0; it < i1;) {
            // the token's rows (element offsets) and weights; rows of -1 dropped
            int n = 0;
            for (int k = 0; k < nrb; k++) {
                int5 cw = {offwv + k * sw1 + t * sw2, 0, 0, 0, 0};
                w[n] = unw ? 1.0f : s_f32_ld_g((__global float *) gen_addr(cw, wv));
#if defined(IND)
                int5 cr = {t * nrb + k, 0, 0, 0, 0};
                const int r = s_i32_ld_g((__global int *) gen_addr(cr, pr));
                ro[n] = r * ne0;
                n += r >= 0 ? 1 : 0;
#else
                ro[n] = offb + k * db + t * sb1;
                n++;
#endif
            }
            const int ce = offc + t * sc1, oe = t * ne0;
            for (; it < i1 && e < ne0; it++, e += 64) {
                float64 x = hc != 0.0f ? ld_f32(c, ce + e) * hc : (float64) 0.0f;
                int k = 0;
                for (; k + 4 <= n; k += 4) {
                    const float64 b0 = ld_f32(b, ro[k] + e);
                    const float64 b1 = ld_f32(b, ro[k + 1] + e);
                    const float64 b2 = ld_f32(b, ro[k + 2] + e);
                    const float64 b3 = ld_f32(b, ro[k + 3] + e);
                    x = v_f32_mac_b(b0, w[k], x);
                    x = v_f32_mac_b(b1, w[k + 1], x);
                    x = v_f32_mac_b(b2, w[k + 2], x);
                    x = v_f32_mac_b(b3, w[k + 3], x);
                }
                for (; k < n; k++) {
                    x = v_f32_mac_b(ld_f32(b, ro[k] + e), w[k], x);
                }
                ST_F32(out, oe + e, x, ne0 - e);
            }
            t++;
            e = 0;
        }
    }
}

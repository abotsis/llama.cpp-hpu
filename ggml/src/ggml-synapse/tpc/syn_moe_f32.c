// MUL_MAT_ID helpers around a dense product y over all experts and all tokens:
//  -DGATHER: xe[e][t] (bf16 [K]) = x[:, s, t] for the slot s with ids[s, t] == e, else zeros.
//            tensors: 0 x  1 ids | 2 xe
//  else    : out[:, s, t] = y[ids[s, t] * se + t * st + (0..M)] (f32).
//            tensors: 0 y  1 ids | 2 out
#include "syn_common.h"

#if defined(GATHER)
void main(tensor x, tensor ids, tensor dep, tensor xe, int nmem, int K, int nu, int T, int E,
          int offx, int sx1, int sx2, int offi, int si0, int si1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nitems = T * E;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t = it / E, e = it - t * E;
            int s = -1;
            for (int j = 0; j < nu; j++) {
                int5 ci = {offi + j * si0 + t * si1, 0, 0, 0, 0};
                if (s_i32_ld_g((__global int *) gen_addr(ci, ids)) == e) {
                    s = j;
                }
            }
            const int px = offx + s * sx1 + t * sx2;
            const int po = (e * T + t) * K;
            for (int k = 0; k < K; k += 128) {
                bfloat128 v = (bf16) 0.0f;
                if (s >= 0) {
                    v = cvt_bf16(ld_f32(x, px + k), ld_f32(x, px + k + 64));
                }
                int5 co = {po + k, 0, 0, 0, 0};
                const int left = K - k;
                if (left >= 128) v_bf16_st_tnsr(co, xe, v); else v_bf16_st_tnsr_partial(co, xe, v, left - 1, 0);
            }
        }
    }
}
#else
void main(tensor y, tensor ids, tensor dep, tensor out, int nmem, int M, int nu, int T, int se, int st,
          int offi, int si0, int si1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nitems = T * nu;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t = it / nu, s = it - t * nu;
            int5 ci = {offi + s * si0 + t * si1, 0, 0, 0, 0};
            const int e = s_i32_ld_g((__global int *) gen_addr(ci, ids));
            const int py = e * se + t * st;
            for (int k = 0; k < M; k += 64) {
                // a negative id skips the slot: its row is zero
                float64 v = 0.0f;
                if (e >= 0) {
                    v = ld_f32(y, py + k);
                }
                ST_F32(out, it * M + k, v, M - k);
            }
        }
    }
}
#endif

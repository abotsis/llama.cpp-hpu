// COL2IM_1D: columns [K*OC, T_in] -> signal [T_out, OC], T_out = (T_in-1)*s + K - 2p.
// out[t, c] = sum_k col[c*K + k, t_in] where t_in*s + k == t + p, 0 <= k < K, 0 <= t_in < T_in.
// Cropped taps contribute nothing; overlapping windows
// accumulate; outputs with no contributing tap write exact zero. Input and output the same
// declared type: F32 (default), F16 (-DSRC_F16) or BF16 (-DSRC_BF16) storage both sides.
// F32 accumulation.
// Input contiguous [K*OC, T_in] (element (c*K + k, t_in) at (c*K + k) + t_in*K*OC); output
// contiguous [T_out, OC] (element (t, c) at t + c*T_out).
// tensors: 0 col | 1 out
#include "syn_common.h"

#if defined(SRC_F16)
#define LD_COL(c) f16_to_f32(v_f16_ld_tnsr_b((c), col)).v1
#define ST_C2I(c0, v) { int5 c_ = {(c0), 0, 0, 0, 0}; v_f16_st_tnsr_partial(c_, out, cvt_f16((v), (v)), 0, 0); }
#elif defined(SRC_BF16)
#define LD_COL(c) bf16_to_f32(v_bf16_ld_tnsr_b((c), col)).v1
#define ST_C2I(c0, v) { int5 c_ = {(c0), 0, 0, 0, 0}; v_bf16_st_tnsr_partial(c_, out, cvt_bf16((v), (v)), 0, 0); }
#else
#define LD_COL(c) ({ int5 c_ = (c); ld_f32(col, c_[0]); })
#define ST_C2I(c0, v) ST_F32(out, (c0), (v), 1)
#endif

void main(tensor col, tensor dep, tensor out,
          int nmem, int s, int OC, int p, int K, int T_in, int T_out)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int K_OC = K * OC;
    const int items = T_out * OC;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int c = it % OC;
            const int t = it / OC;
            const int t_abs = t + p;
            // t_in*s + k == t_abs with 0 <= k < K: t_in >= (t_abs - K + 1)/s, t_in <= t_abs/s
            int t_in = t_abs - K + 1;
            if (t_in < 0) t_in = 0;
            t_in = (t_in + s - 1) / s;
            const int t_in_max = t_abs / s;
            float64 acc = 0.0f;
            for (; t_in <= t_in_max; t_in++) {
                const int k = t_abs - t_in * s;
                if (k >= 0 && k < K) {
                    int5 cc = {c * K + k + t_in * K_OC, 0, 0, 0, 0};
                    acc += LD_COL(cc);
                }
            }
            ST_C2I(t + c * T_out, acc);
        }
    }
}

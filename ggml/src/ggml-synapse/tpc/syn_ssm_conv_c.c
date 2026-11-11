// ggml_ssm_conv over channel-fastest operands: out[c, t, s] = sum_k xt[c + C s + sxt (t + k)] * wt[c + C k]
// (xt: the input transposed to [C ns, nc - 1 + nt]; wt: the weights transposed to [C, nc]).
// tensors: 0 xt  1 wt | 2 out
#include "syn_common.h"

void main(tensor xt, tensor wt, tensor dep, tensor out,
          int nmem, int nc, int C, int nt, int ns, int sxt)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int ncb = (C + 63) >> 6, ntb = (nt + 63) >> 6;
    const int nitems = ns * ncb * ntb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int tb = it % ntb, rest = it / ntb;
            const int cb = rest % ncb, s = rest / ncb;
            const int c0 = cb << 6;
            const int t0 = tb << 6;
            const int t1 = t0 + 64 < nt ? t0 + 64 : nt;
            for (int t = t0; t < t1; t++) {
                float64 acc = 0.0f;
                for (int k = 0; k < nc; k++) {
                    acc = v_f32_mac_b(ld_f32(xt, c0 + C * s + sxt * (t + k)), ld_f32(wt, c0 + C * k), acc);
                }
                ST_F32(out, c0 + C * t + C * nt * s, acc, C - c0);
            }
        }
    }
}

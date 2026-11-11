// out = bf16(attn * sigmoid(gate)), attn from the syn_fattn partials (D = 256, B = 1). See syn_fattn_comb_f32.c for the partial format.
// tensors: 0 part  1 gate  2 dep | 3 out (bf16)
#include "syn_common.h"

void main(tensor part, tensor gate, tensor dep, tensor out, int nmem, int chunk, int nq, int H, int nch,
          int offg, int sgh, int sgi, float inv_H)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nitems = H * nq;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int i = FDIV(it, inv_H);
            const int h = it - i * H;
            const int pb = (h * nq + i) * nch * 384;
            float64 M = -1e30f;
            #pragma unroll 1
            for (int c = 0; c < nch; c++) {
                M = v_f32_max_b(M, ld_f32(part, pb + c * 384 + 256));
            }
            float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, l = 0.0f;
            #pragma unroll 1
            for (int c = 0; c < nch; c++) {
                const int pc = pb + c * 384;
                const float64 w = v_exp_f32(ld_f32(part, pc + 256) - M);
                l = v_f32_mac_b(ld_f32(part, pc + 320), w, l);
                a0 = v_f32_mac_b(ld_f32(part, pc), w, a0);
                a1 = v_f32_mac_b(ld_f32(part, pc + 64), w, a1);
                a2 = v_f32_mac_b(ld_f32(part, pc + 128), w, a2);
                a3 = v_f32_mac_b(ld_f32(part, pc + 192), w, a3);
            }
            const float64 il = v_f32_sel_grt_f32_b(l, 0.0f, v_reciprocal_f32(v_f32_max_b(l, 1e-30f)), 0.0f);
            const int pg = offg + h * sgh + i * sgi;
            const int po = (i * H + h) * 256;
            int5 co = {po, 0, 0, 0, 0};
            v_bf16_st_tnsr(co, out, cvt_bf16(a0 * il * v_sigmoid_f32(ld_f32(gate, pg)),
                                             a1 * il * v_sigmoid_f32(ld_f32(gate, pg + 64))));
            co[0] = po + 128;
            v_bf16_st_tnsr(co, out, cvt_bf16(a2 * il * v_sigmoid_f32(ld_f32(gate, pg + 128)),
                                             a3 * il * v_sigmoid_f32(ld_f32(gate, pg + 192))));
        }
    }
}

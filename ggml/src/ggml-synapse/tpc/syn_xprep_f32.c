// Activation prep for syn_qmm: x [K, N] f32 (column view: element offset + column stride, dim 0
// contiguous) -> xb bf16 [N][K] and xs f32 [N][K / 16] (sums of 16 consecutive bf16-rounded x).
// K is a multiple of 16.
// tensors: 0 x | 1 xb  2 xs
#include "syn_common.h"

void main(tensor x, tensor dep, tensor xb, tensor xs, int nmem, int K, int N, int offx, int sx1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nb = (K + 127) >> 7;
    const int nitems = N * nb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int n = it / nb, b = it - n * nb;
            const int k0 = b << 7;
            const int left = K - k0;
            const bfloat128 v = cvt_bf16(ld_f32(x, offx + n * sx1 + k0), ld_f32(x, offx + n * sx1 + k0 + 64));
            int5 c = {n * K + k0, 0, 0, 0, 0};
            if (left >= 128) v_bf16_st_tnsr(c, xb, v); else v_bf16_st_tnsr_partial(c, xb, v, left - 1, 0);
            const float128 r = bf16_to_f32(v);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const float64 f = h ? r.v2 : r.v1;
                float64 s = SHUF(f, IDX[0]);
                #pragma unroll
                for (int q = 1; q < 16; q++) {
                    s += SHUF(f, IDX[q]);
                }
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    if (64 * h + 16 * g < left) {
                        int5 cs = {n * (K >> 4) + (k0 >> 4) + 4 * h + g, 0, 0, 0, 0};
                        v_f32_st_tnsr_partial(cs, xs, s, 0, 16 * g);
                    }
                }
            }
        }
    }
}

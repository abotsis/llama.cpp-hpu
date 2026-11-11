// f32 GEMV: y[n, m] = sum_k W[m, k] x[n, k]. W row m at offw + m sw1 and
// x column n at offx + n sx1 (dim 0 contiguous, K % 64 == 0); y [M, N] contiguous.
// -DW_BF16: bf16 W (K % 128 == 0)
// tensors: 0 W  1 x | 2 y
#include "syn_common.h"

void main(tensor W, tensor x, tensor dep, tensor y, int nmem, int K, int M, int N, int offw, int sw1, int offx, int sx1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int ng = (M + 3) >> 2;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(N * ng, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int n = it / ng, m0 = (it - n * ng) * 4;
            const int px = offx + n * sx1;
            // rows past M read row M - 1 (discarded)
            const int r1 = m0 + 1 < M ? m0 + 1 : M - 1, r2 = m0 + 2 < M ? m0 + 2 : M - 1, r3 = m0 + 3 < M ? m0 + 3 : M - 1;
            const int p0 = offw + m0 * sw1, p1 = offw + r1 * sw1, p2 = offw + r2 * sw1, p3 = offw + r3 * sw1;
            float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#if defined(W_BF16)
            for (int k = 0; k < K; k += 128) {
                const float64 xa = ld_f32(x, px + k), xb = ld_f32(x, px + k + 64);
                int5 c0 = {p0 + k, 0, 0, 0, 0}, c1 = {p1 + k, 0, 0, 0, 0}, c2 = {p2 + k, 0, 0, 0, 0}, c3 = {p3 + k, 0, 0, 0, 0};
                const float128 w0 = bf16_to_f32(v_bf16_ld_tnsr_b(c0, W)), w1 = bf16_to_f32(v_bf16_ld_tnsr_b(c1, W));
                const float128 w2 = bf16_to_f32(v_bf16_ld_tnsr_b(c2, W)), w3 = bf16_to_f32(v_bf16_ld_tnsr_b(c3, W));
                a0 = v_f32_mac_b(w0.v2, xb, v_f32_mac_b(w0.v1, xa, a0));
                a1 = v_f32_mac_b(w1.v2, xb, v_f32_mac_b(w1.v1, xa, a1));
                a2 = v_f32_mac_b(w2.v2, xb, v_f32_mac_b(w2.v1, xa, a2));
                a3 = v_f32_mac_b(w3.v2, xb, v_f32_mac_b(w3.v1, xa, a3));
            }
#else
            for (int k = 0; k < K; k += 64) {
                const float64 xv = ld_f32(x, px + k);
                a0 = v_f32_mac_b(ld_f32(W, p0 + k), xv, a0);
                a1 = v_f32_mac_b(ld_f32(W, p1 + k), xv, a1);
                a2 = v_f32_mac_b(ld_f32(W, p2 + k), xv, a2);
                a3 = v_f32_mac_b(ld_f32(W, p3 + k), xv, a3);
            }
#endif
            float64 o0, o1, o2, o3;
            NL_REDUCE4_ADD(a0, a1, a2, a3, IDX, o0, o1, o2, o3)
            const float64 v = v_f32_sel_less_f32_b(lane, 1.0f, o0,
                              v_f32_sel_less_f32_b(lane, 2.0f, o1, v_f32_sel_less_f32_b(lane, 3.0f, o2, o3)));
            const int left = M - m0;
            ST_F32(y, n * M + m0, v, left < 4 ? left : 4);
        }
    }
}

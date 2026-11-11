// OUT_PROD, f32, general broadcast / strided operands.
// out[m, n, b2, b3] = sum_p a[m, p, b2/r2, b3/r3] * b[n, p, b2, b3]
//   r2 = out.ne2 / a.ne2, r3 = out.ne3 / a.ne3 (division-based batch broadcast, host computed).
// out is contiguous [M, N, B2, B3]; a, b are views: element offset + element strides of dims 0..3
// (arbitrary strides, including the dflash attention transpose of src1; a's dim-0 stride must be 1 -
// the lowering rejects transposed src0).
// Every output element is written exactly once.
// tensors: 0 a  1 b | 2 out
// params (dwords): nmem, M, N, K, B2, B3, r2, r3, offa, sa1, sa2, sa3, offb, sb0, sb1, sb2, sb3,
//                  inv_r2, inv_r3  - 19 dwords total (under the 32-dword ABI cap)
// inv_r2 = 1 / r2, inv_r3 = 1 / r3 from the host (exact for small integers) for FDIV.
#include "syn_common.h"

void main(tensor a, tensor b, tensor dep, tensor out,
          int nmem, int M, int N, int K, int B2, int B3, int r2, int r3,
          int offa, int sa1, int sa2, int sa3,
          int offb, int sb0, int sb1, int sb2, int sb3, float inv_r2, float inv_r3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    DECL_IDX(IDX)
    const int nrow = N * B2 * B3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nrow, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int n = i0 % N;
        int rest = i0 / N;
        int b2 = rest % B2;
        int b3 = rest / B2;
        for (int it = i0; it < i1; it++) {
            // the source batch of a for this output batch (division-based broadcast)
            const int a2 = FDIV(b2, inv_r2);
            const int a3 = FDIV(b3, inv_r3);
            const int pb = offb + n * sb0 + b2 * sb2 + b3 * sb3;
            const int pab = offa + a2 * sa2 + a3 * sa3;
            for (int m0 = 0; m0 < M; m0 += 64) {
                const int nm = M - m0 < 64 ? M - m0 : 64;
                float64 acc = 0.0f;
                for (int p = 0; p < K; p++) {
                    const float64 bv = BCAST0(ld_f32(b, pb + p * sb1), IDX);
                    const float64 av = ld_f32(a, pab + m0 + p * sa1);
                    acc = v_f32_mac_b(av, bv, acc);
                }
                // outputs: scratch[row = it, column = m] at it * M + m
                // (the lowering transposes the scratch to the [m, row] ggml layout)
                int5 co = {it * M + m0, 0, 0, 0, 0};
                if (nm >= 64) {
                    v_f32_st_tnsr(co, out, acc);
                } else {
                    v_f32_st_tnsr_partial(co, out, acc, nm - 1, 0);
                }
            }
            if (++n == N) {
                n = 0;
                if (++b2 == B2) {
                    b2 = 0;
                    b3++;
                }
            }
        }
    }
}

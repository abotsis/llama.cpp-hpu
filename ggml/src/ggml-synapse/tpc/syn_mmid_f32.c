// MUL_MAT_ID: out[:, s, t] = W[ids[s, t]] x[:, s or 0, t].
// W: bf16, per expert e and block of 128 rows rb: [128 rows (lanes), K] at ((e * nrb + rb) * K) * 128
// (dequant_weight layout 1; rows past M are zero). x: f32 view [K, nx1, T]; ids: i32 view
// [n_used, T]; out contiguous [M, n_used, T] f32.
// tensors: 0 W  1 x  2 ids | 3 out
#include "syn_common.h"

void main(tensor W, tensor x, tensor ids, tensor dep, tensor out,
          int nmem, int K, int M, int nrb, int nu, int T, int bcast,
          int offx, int sx1, int sx2, int offi, int si0, int si1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    DECL_IDX(IDX)
    const int nitems = T * nu * nrb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int rb = it % nrb;
            const int st = it / nrb;        // t * nu + s
            const int s = st % nu;
            const int t = st / nu;
            int5 ci = {offi + s * si0 + t * si1, 0, 0, 0, 0};
            const int e = s_i32_ld_g((__global int *) gen_addr(ci, ids));
            const int px = offx + (bcast ? 0 : s * sx1) + t * sx2;
            int5 cw = {(e * nrb + rb) * K * 128, 0, 0, 0, 0};
            float64 a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
            for (int kb = 0; kb < K; kb += 64) {
                const float64 xv = ld_f32(x, px + kb);
                const int kn = K - kb;
                for (int dg = 0; dg < 4 && dg * 16 < kn; dg++) {
                    float64 bd;
                    if (dg == 0) bd = BCAST_DG(xv, 0);
                    else if (dg == 1) bd = BCAST_DG(xv, 1);
                    else if (dg == 2) bd = BCAST_DG(xv, 2);
                    else bd = BCAST_DG(xv, 3);
                    const int ln = kn - dg * 16;
                    if (ln >= 16) {
                        #pragma unroll
                        for (int l = 0; l < 16; l += 2) {
                            const float64 x0 = SHUF(bd, IDX[l]);
                            const float64 x1 = SHUF(bd, IDX[l + 1]);
                            const float128 w0 = bf16_to_f32(v_bf16_ld_tnsr_b(cw, W));
                            cw[0] += 128;
                            const float128 w1 = bf16_to_f32(v_bf16_ld_tnsr_b(cw, W));
                            cw[0] += 128;
                            a0 = v_f32_mac_b(w0.v1, x0, a0);
                            a1 = v_f32_mac_b(w0.v2, x0, a1);
                            b0 = v_f32_mac_b(w1.v1, x1, b0);
                            b1 = v_f32_mac_b(w1.v2, x1, b1);
                        }
                    } else {
                        // tail (K % 16)
                        for (int l = 0; l < ln; l++) {
                            const float64 x0 = SHUF(bd, IDX[l]);
                            const float128 w0 = bf16_to_f32(v_bf16_ld_tnsr_b(cw, W));
                            cw[0] += 128;
                            a0 = v_f32_mac_b(w0.v1, x0, a0);
                            a1 = v_f32_mac_b(w0.v2, x0, a1);
                        }
                    }
                }
            }
            const int po = st * M + rb * 128;
            const int rem = M - rb * 128;
            ST_F32(out, po, a0 + b0, rem);
            if (rem > 64) {
                ST_F32(out, po + 64, a1 + b1, rem - 64);
            }
        }
    }
}

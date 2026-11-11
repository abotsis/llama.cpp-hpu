// CONV_TRANSPOSE_1D / 2D (p = 0, d = 1), F32 accumulation, one output element per item.
// Gather form of the CPU scatter: output x takes tap k from input i where i*s + k == x.
// Kernel f32 or f16 (-DW_F16); kernel and source are views (element offset + strides of dims 1..3,
// dim 0 contiguous); out contiguous f32.
// ND=1 (default): knl [K, OC, IC], src [T, IC], out [OT, OC], OT = (T-1)*s + K.
// ND=2: knl [KW, KH, OC, IC], src [IW, IH, IC, B], out [OW, OH, OC, B], OW = (IW-1)*s + KW, OH = (IH-1)*s + KH.
// tensors: 0 knl  1 src | 2 out
#include "syn_common.h"

static inline float64 ld_knl(tensor knl, int c0)
{
#if defined(W_F16)
    int5 c = {c0, 0, 0, 0, 0};
    return f16_to_f32(v_f16_ld_tnsr_b(c, knl)).v1;
#else
    return ld_f32(knl, c0);
#endif
}

#if defined(ND2)
void main(tensor knl, tensor src, tensor dep, tensor out,
          int nmem, int s,
          int KW, int KH, int OC, int IC, int IW, int IH, int B, int OW, int OH,
          int offk, int kb1, int kb2, int kb3,
          int offs, int sb1, int sb2, int sb3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int items = OW * OH * OC * B;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int r = it;
            const int ox = r % OW; r /= OW;
            const int oy = r % OH; r /= OH;
            const int oc = r % OC;
            const int b  = r / OC;
            float64 acc = 0.0f;
            // taps with (oy - ky) % s == 0 and 0 <= (oy - ky) / s < IH
            int ky0 = oy - (IH - 1) * s;
            ky0 = ky0 < 0 ? oy % s : ky0;
            for (int ky = ky0; ky < KH && ky <= oy; ky += s) {
                const int iy = (oy - ky) / s;
                int kx0 = ox - (IW - 1) * s;
                kx0 = kx0 < 0 ? ox % s : kx0;
                for (int kx = kx0; kx < KW && kx <= ox; kx += s) {
                    const int ix = (ox - kx) / s;
                    const int pk = offk + kx + ky * kb1 + oc * kb2;
                    const int ps = offs + ix + iy * sb1 + b * sb3;
                    for (int ic = 0; ic < IC; ic++) {
                        acc = v_f32_mac_b(ld_knl(knl, pk + ic * kb3), ld_f32(src, ps + ic * sb2), acc);
                    }
                }
            }
            ST_F32(out, it, acc, 1);
        }
    }
}
#else // ND=1
void main(tensor knl, tensor src, tensor dep, tensor out,
          int nmem, int s, int K, int OC, int IC, int T, int OT,
          int offk, int kb1, int kb2,
          int offs, int sb1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int items = OT * OC;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int t  = it % OT;
            const int oc = it / OT;
            float64 acc = 0.0f;
            // taps with (t - k) % s == 0 and 0 <= (t - k) / s < T: k from max(t % s, t - (T-1) s) in steps of s
            int k0 = t - (T - 1) * s;
            k0 = k0 < 0 ? t % s : k0;
            for (int k = k0; k < K && k <= t; k += s) {
                const int tin = (t - k) / s;
                const int pk = offk + k + oc * kb1;
                const int ps = offs + tin;
                for (int ic = 0; ic < IC; ic++) {
                    acc = v_f32_mac_b(ld_knl(knl, pk + ic * kb2), ld_f32(src, ps + ic * sb1), acc);
                }
            }
            ST_F32(out, it, acc, 1);
        }
    }
}
#endif

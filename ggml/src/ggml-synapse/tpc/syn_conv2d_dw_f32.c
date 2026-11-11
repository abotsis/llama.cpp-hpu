// CONV_2D_DW: depthwise 2D convolution, F32 output, F16 (-DW_F16) or F32 weights, F32 accumulation
// (CPU ggml_compute_forward_conv_2d_dw). Out-of-range taps contribute nothing.
// src [SW, SH, C, N] view: (x, y, c, n) at offs + x sb0 + y sb1 + c sb2 + n sb3.
// knl [KW, KH, 1, C] view: (kx, ky, c) at offk + kx kb0 + ky kb1 + c kb3.
// WHCN (default): one output element per item, out dense [DW, DH, C, N].
// CWHN (-DCWHN): 64 channels per vector (needs sb2 == 1 and kb3 == 1), out in the constructor's
//   CWHN order: (x, y, c, n) at ((n DH + y) DW + x) C + c.
// params {sx, sy, px, py, dx, dy}.
// tensors: 0 knl  1 src | 2 out
#include "syn_common.h"

#if defined(W_F16)
#define LD_K(c0) ({ int5 c_ = {(c0), 0, 0, 0, 0}; f16_to_f32(v_f16_ld_tnsr_b(c_, knl)).v1; })
#else
#define LD_K(c0) ld_f32(knl, (c0))
#endif

void main(tensor knl, tensor src, tensor dep, tensor out,
          int nmem, int C, int N, int SW, int SH, int DW, int DH, int KW, int KH,
          int sx, int sy, int px, int py, int dx, int dy,
          int offs, int sb0, int sb1, int sb2, int sb3,
          int offk, int kb0, int kb1, int kb3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
#if defined(CWHN)
    (void) sb2;
    const int nch = (C + 63) >> 6;
    const int items = DW * DH * N * nch;
#else
    const int items = DW * DH * C * N;
#endif
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int r = it;
#if defined(CWHN)
            const int c = (r % nch) << 6; r /= nch;
#endif
            const int ox = r % DW; r /= DW;
            const int oy = r % DH; r /= DH;
#if defined(CWHN)
            const int n = r;
#else
            const int c = r % C;
            const int n = r / C;
#endif
            const int bx = ox * sx - px;
            const int by = oy * sy - py;
            const int ps = offs + c * sb2 + n * sb3;
            const int pk = offk + c * kb3;
            float64 acc = 0.0f;
            for (int ky = 0; ky < KH; ky++) {
                const int y = by + ky * dy;
                if (y < 0 || y >= SH) continue;
                for (int kx = 0; kx < KW; kx++) {
                    const int x = bx + kx * dx;
                    if (x < 0 || x >= SW) continue;
                    acc = v_f32_mac_b(LD_K(pk + kx * kb0 + ky * kb1), ld_f32(src, ps + x * sb0 + y * sb1), acc);
                }
            }
#if defined(CWHN)
            const int nc = C - c < 64 ? C - c : 64;
            ST_F32(out, ((n * DH + oy) * DW + ox) * C + c, acc, nc);
#else
            ST_F32(out, it, acc, 1);
#endif
        }
    }
}

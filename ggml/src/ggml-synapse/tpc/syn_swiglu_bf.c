// SwiGLU: out = bf16(silu(a) * b). Rows with one stride per
// operand (a/b may be slices of one tensor); ne0 % 256 == 0.
// -DOUT_FP8: y = silu(a) * b (syn_binary's math), then fp8_143 codes
//            y / s[r] with s[r] = max |y[r]| / 240 (the [1, N] activation scale; p2: a power of two, see
//            FP8_SCALE); ne0 <= 16384
// tensors: 0 a  1 b  2 dep | 3 out (bf16);  OUT_FP8: 3 out (fp8)  4 sc
#include "syn_common.h"

#if defined(OUT_FP8)
__local__ float64 XV[256];

void main(tensor a, tensor b, tensor dep, tensor out, tensor sc,
          int nmem, int chunk, int ne0, int nrows, int offa, int sa, int offb, int sb, int p2)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_CHUNK(nrows, chunk, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            const int pa = offa + r * sa, pb = offb + r * sb;
            float64 mx = 0.0f;
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 512) {
                float64 y[8];
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    y[u] = nl_silu_acc(ld_f32(a, pa + e + 64 * u)) * ld_f32(b, pb + e + 64 * u);
                }
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    XV[(e >> 6) + u] = y[u];
                }
                mx = v_f32_max_b(mx, v_f32_max_b(v_f32_max_b(v_f32_abs_b(y[0]), v_f32_abs_b(y[1])),
                                                 v_f32_max_b(v_f32_abs_b(y[2]), v_f32_abs_b(y[3]))));
                if (e + 256 < ne0) {
                    mx = v_f32_max_b(mx, v_f32_max_b(v_f32_max_b(v_f32_abs_b(y[4]), v_f32_abs_b(y[5])),
                                                     v_f32_max_b(v_f32_abs_b(y[6]), v_f32_abs_b(y[7]))));
                }
            }
            float64 M, s, inv;
            NL_REDUCE_MAX(mx, IDX, M);
            FP8_SCALE(M, p2, s, inv)
            FP8_ROW(XV, ne0, inv, out, r * ne0)
            ST_F32(sc, r, s, 1);
        }
    }
}
#else
void main(tensor a, tensor b, tensor dep, tensor out,
          int nmem, int chunk, int ne0, int nrows, int offa, int sa, int offb, int sb, float inv_nq)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nq = ne0 >> 8;
    const int nitems = nrows * nq;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        #pragma unroll 1
        for (int it = i0; it < i1; it++) {
            const int r = FDIV(it, inv_nq);
            const int e = (it - r * nq) << 8;
            const int pa = offa + r * sa + e, pb = offb + r * sb + e, po = r * ne0 + e;
            const float64 g0 = ld_f32(a, pa), g1 = ld_f32(a, pa + 64), g2 = ld_f32(a, pa + 128), g3 = ld_f32(a, pa + 192);
            const float64 u0 = ld_f32(b, pb), u1 = ld_f32(b, pb + 64), u2 = ld_f32(b, pb + 128), u3 = ld_f32(b, pb + 192);
            int5 co = {po, 0, 0, 0, 0};
            v_bf16_st_tnsr(co, out, cvt_bf16(g0 * v_sigmoid_f32(g0) * u0, g1 * v_sigmoid_f32(g1) * u1));
            co[0] = po + 128;
            v_bf16_st_tnsr(co, out, cvt_bf16(g2 * v_sigmoid_f32(g2) * u2, g3 * v_sigmoid_f32(g3) * u3));
        }
    }
}
#endif

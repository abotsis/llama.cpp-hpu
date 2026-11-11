// Residual add + RMSNorm * gamma: xo = a + b (f32), out = bf16(rms(xo) * w).
// Rows: element offset + one row stride per operand; ne0 % 256 == 0, ne0 <= 4096.
// flags: 1 = no add (x = a), 2 = xo receives the f32 normalized rows instead of x, 4 = no xo.
// -DOUT_FP8: out = fp8_143 rows y / s[r] with a scale per row s[r] = max |y[r]| / 240 (an fp8 GEMM's
//            [1, N] activation scale; flags 8: a power of two, see FP8_SCALE); ne0 <= 8192.
// tensors: 0 a  1 b  2 w  3 dep | 4 out (bf16)  5 xo;  OUT_FP8: 4 out (fp8)  5 sc  6 xo
#include "syn_common.h"

#if defined(OUT_FP8)
__local__ float64 XV[128];

void main(tensor a, tensor b, tensor w, tensor dep, tensor out, tensor sc, tensor xo,
#else
__local__ float64 XV[64];

void main(tensor a, tensor b, tensor w, tensor dep, tensor out, tensor xo,
#endif
          int nmem, int chunk, int ne0, int nrows, int offa, int sa, int offb, int sb, int offw, float eps,
          float inv_n, int flags)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float bmul = (flags & 1) ? 0.0f : 1.0f;
    const bool st_x = (flags & 6) == 0;
    const bool st_y = (flags & 2) != 0;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_CHUNK(nrows, chunk, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            const int pa = offa + r * sa, pb = offb + r * sb, po = r * ne0;
            float64 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 256) {
                const float64 x0 = v_f32_mac_b(ld_f32(b, pb + e), bmul, ld_f32(a, pa + e));
                const float64 x1 = v_f32_mac_b(ld_f32(b, pb + e + 64), bmul, ld_f32(a, pa + e + 64));
                const float64 x2 = v_f32_mac_b(ld_f32(b, pb + e + 128), bmul, ld_f32(a, pa + e + 128));
                const float64 x3 = v_f32_mac_b(ld_f32(b, pb + e + 192), bmul, ld_f32(a, pa + e + 192));
                const int v = e >> 6;
                XV[v] = x0; XV[v + 1] = x1; XV[v + 2] = x2; XV[v + 3] = x3;
                if (st_x) {
                    ST_F32(xo, po + e, x0, 64);
                    ST_F32(xo, po + e + 64, x1, 64);
                    ST_F32(xo, po + e + 128, x2, 64);
                    ST_F32(xo, po + e + 192, x3, 64);
                }
                c0 = v_f32_mac_b(x0, x0, c0); c1 = v_f32_mac_b(x1, x1, c1);
                c2 = v_f32_mac_b(x2, x2, c2); c3 = v_f32_mac_b(x3, x3, c3);
            }
            float64 ss;
            NL_REDUCE_ADD((c0 + c1) + (c2 + c3), IDX, ss);
#if defined(OUT_FP8)
            // (1 / ne0 and rsqrt as in syn_norm)
            const float64 rs = nl_rsqrt(ss * (1.0f / (float) ne0) + eps);
            float64 mx = 0.0f;
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 128) {
                const float64 y0 = XV[e >> 6] * rs * ld_f32(w, offw + e);
                const float64 y1 = XV[(e >> 6) + 1] * rs * ld_f32(w, offw + e + 64);
                XV[e >> 6] = y0;
                XV[(e >> 6) + 1] = y1;
                mx = v_f32_max_b(mx, v_f32_max_b(v_f32_abs_b(y0), v_f32_abs_b(y1)));
                if (st_y) {
                    ST_F32(xo, po + e, y0, 64);
                    ST_F32(xo, po + e + 64, y1, 64);
                }
            }
            float64 M, s, inv;
            NL_REDUCE_MAX(mx, IDX, M);
            FP8_SCALE(M, flags & 8, s, inv)
            FP8_ROW(XV, ne0, inv, out, po)
            ST_F32(sc, r, s, 1);
#else
            const float64 sc = nl_rsqrt2(ss * inv_n + eps);
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 128) {
                const float64 y0 = XV[e >> 6] * sc * ld_f32(w, offw + e);
                const float64 y1 = XV[(e >> 6) + 1] * sc * ld_f32(w, offw + e + 64);
                int5 co = {po + e, 0, 0, 0, 0};
                v_bf16_st_tnsr(co, out, cvt_bf16(y0, y1));
                if (st_y) {
                    ST_F32(xo, po + e, y0, 64);
                    ST_F32(xo, po + e + 64, y1, 64);
                }
            }
#endif
        }
    }
}

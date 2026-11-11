// FP8 (Gaudi2 fp8_143, +-240) quantization of an f32 activation x [K, N] (column view: element offset +
// column stride, dim 0 contiguous) for an fp8 GEMM, with one scale for the tensor (fp8 is a float
// format: tokens of a smaller range keep their precision down to 2^-14 of the max).
//  -DAMAX: am[n] = max |x[:, n]|
//  else  : s = max(am) / 240, q[:, n] = fp8(x[:, n] / s) (2-D [K, N]), sc[0] = s
//  -DONE : as the default, with max |x| computed here (no am input)
//  -DTOK : a scale per token (column): s[n] = am[n] / 240, q[:, n] = fp8(x[:, n] / s[n]), sc[0] = 1 (the
//          GEMM's activation scale) and st[n] = s[n], applied to the GEMM output columns (-DCOLS)
//  -DCOLS: y[:, n] = g[:, n] * st[n], g [M, N] contiguous
//  -DCOLS -DROWS: also the weight's per-row scales of batch b = n / ncol: y[m, n] = g[m, n] * st[n] * rs[b M + m]
//          (a batched fp8 GEMM [M, ncol, nb] without scales)
//  -DROWQ: a scale per token: s[n] = max |x[:, n]| / 240, q[:, n] = fp8(x[:, n] / s[n]), st[n] = s[n]
//          (the GEMM's [1, N] activation scale; p2: a power of two, see FP8_SCALE), q flat [K N]
// tensors: AMAX 0 x | 1 am;  ONE 0 x | 1 q  2 sc;  TOK 0 x  1 am | 2 q  3 sc  4 st;  COLS 0 g  1 st | 2 y;
//          COLS ROWS 0 g  1 st  2 rs | 3 y;  ROWQ 0 x | 1 q  2 st;
//          else 0 x  1 am | 2 q  3 sc
#include "syn_common.h"

#if defined(ROWQ)
#define MAXV 256
__local__ float64 XV[MAXV];

void main(tensor x, tensor dep, tensor q, tensor st, int nmem, int K, int N, int offx, int sx1, int p2)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const bool vlm = (K & 255) == 0 && K <= 64 * MAXV;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int n0, n1;
        SPLIT_RANGE(N, nmem, m, n0, n1)
        for (int n = n0; n < n1; n++) {
            const int px = offx + n * sx1;
            float64 a = 0.0f;
            if (vlm) {
                #pragma unroll 1
                for (int k = 0; k < K; k += 256) {
                    const float64 x0 = ld_f32(x, px + k), x1 = ld_f32(x, px + k + 64);
                    const float64 x2 = ld_f32(x, px + k + 128), x3 = ld_f32(x, px + k + 192);
                    XV[k >> 6] = x0; XV[(k >> 6) + 1] = x1; XV[(k >> 6) + 2] = x2; XV[(k >> 6) + 3] = x3;
                    a = v_f32_max_b(a, v_f32_max_b(v_f32_max_b(v_f32_abs_b(x0), v_f32_abs_b(x1)),
                                                   v_f32_max_b(v_f32_abs_b(x2), v_f32_abs_b(x3))));
                }
            } else {
                for (int k = 0; k < K; k += 64) {
                    a = v_f32_max_b(a, v_f32_sel_less_f32_b(lane, (float) (K - k), v_f32_abs_b(ld_f32(x, px + k)), 0.0f));
                }
            }
            float64 M, s, inv;
            NL_REDUCE_MAX(a, IDX, M)
            FP8_SCALE(M, p2, s, inv)
            ST_F32(st, n, s, 1);
            if (vlm) {
                FP8_ROW(XV, K, inv, q, n * K)
                continue;
            }
            for (int k = 0; k < K; k += 256) {
                float256 t;
                t.v1 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k) * inv, -240.0f), 240.0f);
                t.v2 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 64) * inv, -240.0f), 240.0f);
                t.v3 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 128) * inv, -240.0f), 240.0f);
                t.v4 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 192) * inv, -240.0f), 240.0f);
                const minifloat256 y = convert_float256_to_minifloat256(t, SW_RHNE | SW_LINEAR);
                int5 c = {n * K + k, 0, 0, 0, 0};   // q is flat [K N]
                const int left = K - k;
                if (left >= 256) v_f8_st_tnsr(c, q, y); else v_f8_st_tnsr_partial(c, q, y, left - 1, 0);
            }
        }
    }
}
#elif defined(ONE)
void main(tensor x, tensor dep, tensor q, tensor sc, int nmem, int K, int N, int offx, int sx1)
{
    if (get_index_space_offset()[0] != 0) {
        return;
    }
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    float64 a = 0.0f;
    for (int n = 0; n < N; n++) {
        const int px = offx + n * sx1;
        for (int k = 0; k < K; k += 64) {
            a = v_f32_max_b(a, v_f32_sel_less_f32_b(lane, (float) (K - k), v_f32_abs_b(ld_f32(x, px + k)), 0.0f));
        }
    }
    float64 M;
    NL_REDUCE_MAX(a, IDX, M)
    const float64 s = v_f32_max_b(M, 1e-30f) * (1.0f / 240.0f);
    const float64 inv = nl_recip(s);
    ST_F32(sc, 0, s, 1);
    for (int n = 0; n < N; n++) {
        const int px = offx + n * sx1;
        for (int k = 0; k < K; k += 256) {
            float256 t;
            t.v1 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k) * inv, -240.0f), 240.0f);
            t.v2 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 64) * inv, -240.0f), 240.0f);
            t.v3 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 128) * inv, -240.0f), 240.0f);
            t.v4 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 192) * inv, -240.0f), 240.0f);
            const minifloat256 y = convert_float256_to_minifloat256(t, SW_RHNE | SW_LINEAR);
            int5 c = {k, n, 0, 0, 0};
            const int left = K - k;
            if (left >= 256) v_f8_st_tnsr(c, q, y); else v_f8_st_tnsr_partial(c, q, y, left - 1, 0);
        }
    }
}
#elif defined(COLS)
#if defined(ROWS)
void main(tensor g, tensor st, tensor rs, tensor dep, tensor y, int nmem, int M, int N, int ncol)
#else
void main(tensor g, tensor st, tensor dep, tensor y, int nmem, int M, int N)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nch = (M + 1023) >> 10;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(N * nch, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int n = it / nch, r0 = (it - n * nch) * 1024;
            const int r1 = r0 + 1024 < M ? r0 + 1024 : M;
            const float64 s = BCAST0(ld_f32(st, n), IDX);
            const int po = n * M;
#if defined(ROWS)
            const int pr = (n / ncol) * M;
#endif
            for (int r = r0; r < r1; r += 256) {
                float64 v[4];
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    v[u] = ld_f32(g, po + r + 64 * u);
#if defined(ROWS)
                    v[u] = v[u] * ld_f32(rs, pr + r + 64 * u);
#endif
                }
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    if (r + 64 * u < r1) {
                        ST_F32(y, po + r + 64 * u, v[u] * s, r1 - r - 64 * u);
                    }
                }
            }
        }
    }
}
#else

#if defined(AMAX)
void main(tensor x, tensor dep, tensor am, int nmem, int K, int N, int offx, int sx1)
#elif defined(TOK)
void main(tensor x, tensor am, tensor dep, tensor q, tensor sc, tensor st, int nmem, int K, int N, int offx, int sx1)
#else
void main(tensor x, tensor am, tensor dep, tensor q, tensor sc, int nmem, int K, int N, int offx, int sx1)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
#if defined(TOK)
    if (ioff[0] == 0) {
        ST_F32(sc, 0, (float64) 1.0f, 1);
    }
#elif !defined(AMAX)
    // the tensor's max |x| from the column maxima
    float64 mx = 0.0f;
    for (int n = 0; n < N; n += 64) {
        mx = v_f32_max_b(mx, v_f32_sel_less_f32_b(lane, (float) (N - n), ld_f32(am, n), 0.0f));
    }
    float64 M;
    NL_REDUCE_MAX(mx, IDX, M)
    const float64 s = v_f32_max_b(M, 1e-30f) * (1.0f / 240.0f);
    const float64 inv = nl_recip(s);
    if (ioff[0] == 0) {
        ST_F32(sc, 0, s, 1);
    }
#endif
    for (int m = ioff[0]; m < iend[0]; m++) {
        int n0, n1;
        SPLIT_RANGE(N, nmem, m, n0, n1)
        for (int n = n0; n < n1; n++) {
            const int px = offx + n * sx1;
#if defined(AMAX)
            float64 a = 0.0f;
            for (int k = 0; k < K; k += 64) {
                a = v_f32_max_b(a, v_f32_sel_less_f32_b(lane, (float) (K - k), v_f32_abs_b(ld_f32(x, px + k)), 0.0f));
            }
            float64 r;
            NL_REDUCE_MAX(a, IDX, r)
            ST_F32(am, n, r, 1);
#else
#if defined(TOK)
            const float64 s = v_f32_max_b(BCAST0(ld_f32(am, n), IDX), 1e-30f) * (1.0f / 240.0f);
            const float64 inv = nl_recip(s);
            ST_F32(st, n, s, 1);
#endif
            for (int k = 0; k < K; k += 256) {
                float256 t;
                t.v1 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k) * inv, -240.0f), 240.0f);
                t.v2 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 64) * inv, -240.0f), 240.0f);
                t.v3 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 128) * inv, -240.0f), 240.0f);
                t.v4 = v_f32_min_b(v_f32_max_b(ld_f32(x, px + k + 192) * inv, -240.0f), 240.0f);
                const minifloat256 y = convert_float256_to_minifloat256(t, SW_RHNE | SW_LINEAR);
                int5 c = {k, n, 0, 0, 0};   // q is [K, N]
                const int left = K - k;
                if (left >= 256) v_f8_st_tnsr(c, q, y); else v_f8_st_tnsr_partial(c, q, y, left - 1, 0);
            }
#endif
        }
    }
}
#endif // ONE

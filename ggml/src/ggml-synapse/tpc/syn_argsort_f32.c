// ARGSORT / TOP_K of rows (n <= 4096), i32 indices: pass p writes the index of the best element
// not yet taken (desc: largest, else smallest; ties: lowest index) to out[r * k + p], p < np.
// a is a view (element offset + strides of dims 1..3, dim 0 contiguous); out [k, rows] contiguous.
// tensors: 0 a | 1 out
#include "syn_common.h"

#define MAXV 64

__local__ float64 X[MAXV];
__local__ float64 TK[MAXV];

void main(tensor a, tensor dep, tensor out,
          int nmem, int n, int k, int np, int ne1, int ne2, int ne3, int off0, int s01, int s02, int s03, int desc)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 ninf = as_float64((int64) 0xFF800000);
    const int nv = (n + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
            for (int c = 0; c < nv; c++) {
                float64 x = ld_f32(a, pa + 64 * c);
                if (!desc) {
                    x = -x;
                }
                X[c] = x;
                // lanes past n are taken from the start
                TK[c] = v_f32_sel_less_f32_b(lane, (float) (n - 64 * c), 0.0f, 1.0f);
            }
            for (int p = 0; p < np; p++) {
                float64 bm = ninf;
                for (int c = 0; c < nv; c++) {
                    bm = v_f32_max_b(bm, v_f32_sel_eq_f32_b(TK[c], 0.0f, X[c], ninf));
                }
                float64 M;
                NL_REDUCE_MAX(bm, IDX, M)
                float64 bi = 1e30f;
                for (int c = 0; c < nv; c++) {
                    const float64 hit = v_f32_sel_eq_f32_b(X[c], M, lane + (float) (64 * c), 1e30f);
                    bi = v_f32_min_b(bi, v_f32_sel_eq_f32_b(TK[c], 0.0f, hit, 1e30f));
                }
                float64 ni;
                NL_REDUCE_MAX(-bi, IDX, ni)
                const float64 idx = -ni;
                for (int c = 0; c < nv; c++) {
                    TK[c] = v_f32_sel_eq_f32_b(lane + (float) (64 * c), idx, 1.0f, TK[c]);
                }
                int5 co = {r * k + p, 0, 0, 0, 0};
                v_i32_st_tnsr_partial(co, out, v_convert_f32_to_i32_b(idx, SW_RHNE, 0, 1, 0), 0, 0);
            }
        }
    }
}

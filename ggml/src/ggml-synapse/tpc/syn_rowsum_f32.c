// out[r] = scale * sum of row r of a (f32). -DARGMAX: out[r] = index of the first maximum (i32). -DROWMAX: out[r] =
// the row's maximum (f32; POOL_1D max over whole rows).
// a is a view (element offset + strides of dims 1..3, dim 0 contiguous); out [ne1, ne2, ne3].
// tensors: 0 a | 1 out
#include "syn_common.h"

void main(tensor a, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3, int off0, int s01, int s02, int s03, float scale)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nrows = ne1 * ne2 * ne3;
    const float64 lane = LANE_F32();
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
#if defined(ARGMAX)
            float64 best = as_float64((int64) 0xFF800000);
            float64 bi = 0.0f;
            for (int e = 0; e < ne0; e += 64) {
                const float64 x = v_f32_sel_less_f32_b(lane, (float) (ne0 - e), ld_f32(a, pa + e), as_float64((int64) 0xFF800000));
                const float64 idx = lane + (float) e;
                bi = v_f32_sel_grt_f32_b(x, best, idx, bi);
                best = v_f32_max_b(x, best);
            }
            float64 mx;
            NL_REDUCE_MAX(best, IDX, mx)
            // lowest index among the lanes holding the maximum
            const float64 cand = -v_f32_sel_eq_f32_b(best, mx, bi, 1e30f);
            float64 ni;
            NL_REDUCE_MAX(cand, IDX, ni)
            int5 c = {r, 0, 0, 0, 0};
            v_i32_st_tnsr_partial(c, out, v_convert_f32_to_i32_b(-ni, SW_RHNE, 0, 1, 0), 0, 0);
#elif defined(ROWMAX)
            float64 best = as_float64((int64) 0xFF800000);
            for (int e = 0; e < ne0; e += 64) {
                best = v_f32_max_b(best, v_f32_sel_less_f32_b(lane, (float) (ne0 - e), ld_f32(a, pa + e), as_float64((int64) 0xFF800000)));
            }
            float64 mx;
            NL_REDUCE_MAX(best, IDX, mx)
            ST_F32(out, r, mx, 1);
#else
            float64 acc = 0.0f;
            for (int e = 0; e < ne0; e += 64) {
                acc += v_f32_sel_less_f32_b(lane, (float) (ne0 - e), ld_f32(a, pa + e), 0.0f);
            }
            float64 s;
            NL_REDUCE_ADD(acc, IDX, s)
            ST_F32(out, r, s * scale, 1);
#endif
        }
    }
}

// CUMSUM along dim 0: out[:, r] = inclusive prefix sums of a[:, r] (f32). out contiguous [ne0, ne1, ne2, ne3];
// a a view (element offset + strides of dims 1..3, dim 0 contiguous).
// -DARANGE: out[i] = start + i * step, i < ne0 (no source; tensors: 0 dep | 1 out).
// tensors: 0 a  1 dep | 2 out
#include "syn_common.h"

#if defined(ARANGE)
void main(tensor dep, tensor out, int nmem, int n, float start, float step)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lane = LANE_F32();
    const int nch = (n + 63) >> 6;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nch, nmem, m, i0, i1)
        for (int c = i0; c < i1; c++) {
            const float64 x = v_f32_mac_b(lane + (float) (c << 6), step, (float64) start);
            ST_F32(out, c << 6, x, n - (c << 6));
        }
    }
}
#else
void main(tensor a, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3, int offa, int sa1, int sa2, int sa3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    // the lane's index in its dual group (16 lanes)
    const float64 lg = lane - v_f32_sel_less_f32_b(lane, 16.0f, (float64) 0.0f,
                              v_f32_sel_less_f32_b(lane, 32.0f, (float64) 16.0f,
                                                   v_f32_sel_less_f32_b(lane, 48.0f, (float64) 32.0f, (float64) 48.0f)));
    const int nrows = ne1 * ne2 * ne3;
    const int nch = (ne0 + 63) >> 6;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_RANGE(nrows, nmem, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int i2 = rr / ne1;
            const int i1 = rr - i2 * ne1;
            const int pa = offa + i1 * sa1 + i2 * sa2 + r3 * sa3;
            float64 carry = 0.0f;
            for (int c = 0; c < nch; c++) {
                const int left = ne0 - (c << 6);
                float64 x = ld_f32(a, pa + (c << 6));
                x = v_f32_sel_less_f32_b(lane, (float) left, x, (float64) 0.0f);
                float64 acc = 0.0f;
                #pragma unroll
                for (int e = 0; e < 16; e++) {
                    const float64 b = SHUF(x, IDX[e]);
                    acc += v_f32_sel_less_f32_b(lg, (float) e, (float64) 0.0f, b);
                }
                // each dual group's total (its lane 15), carried to the groups above it
                const float64 tot = SHUF(acc, IDX[15]);
                acc += v_f32_sel_less_f32_b(lane, 16.0f, (float64) 0.0f, BCAST_DG(tot, 0));
                acc += v_f32_sel_less_f32_b(lane, 32.0f, (float64) 0.0f, BCAST_DG(tot, 1));
                acc += v_f32_sel_less_f32_b(lane, 48.0f, (float64) 0.0f, BCAST_DG(tot, 2));
                acc += carry;
                ST_F32(out, r * ne0 + (c << 6), acc, left);
                carry = SHUF(BCAST_DG(acc, 3), IDX[15]);
            }
        }
    }
}
#endif

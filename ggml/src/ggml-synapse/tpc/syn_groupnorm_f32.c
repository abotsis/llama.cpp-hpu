// GROUP_NORM, f32 in / f32 out. out = (x - mean) / sqrt(var + eps) per (batch, group),
// channels = ne2, groups from op param 0 (n_groups), eps from f32 op param 1.
// Group g of batch b covers channels [g * ceil(C/G), min(C, (g+1) * ceil(C/G))); mean and variance
// cover ne0 * ne1 * (the group's actual channel count) - no channel-divisibility restriction.
// Empty groups are skipped without division or writes. Epsilon placement is the CPU reference's:
// (x - mean) / sqrt(var + eps).
// tensors: 0 x | 1 out. out contiguous [ne0, ne1, C, B3]; x a view (offset + strides 1..3).
// params: nmem, ne0, ne1, C, B3, n_groups, offx, sx1, sx2, sx3, eps  - 11 dwords
#include "syn_common.h"

void main(tensor x, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int C, int B3, int n_groups,
          int offx, int sx1, int sx2, int sx3, float eps)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int64 lane = LANE_I32();
    const int nitems = B3 * n_groups;
    const int cpg = (C + n_groups - 1) / n_groups;   // ceil(C/G), host may pass any C, G > 0
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int b3 = it / n_groups;
            const int g = it - b3 * n_groups;
            const int c0 = g * cpg;
            const int c1 = c0 + cpg < C ? c0 + cpg : C;
            const int nch = c1 - c0;
            if (nch <= 0 || ne0 <= 0 || ne1 <= 0) {
                continue;   // empty group: no division, no writes
            }
            const float64 lf = LANE_F32();
            const int np = (ne0 + 63) >> 6;
            // pass 1: sum of x over the group's channels * ne0 * ne1
            float64 acc = 0.0f;
            for (int c = c0; c < c1; c++) {
                const int px = offx + c * sx2 + b3 * sx3;
                for (int r1 = 0; r1 < ne1; r1++) {
                    const int prow = px + r1 * sx1;
                    for (int e = 0; e < ne0; e += 64) {
                        acc += v_f32_sel_less_f32_b(lf, (float) (ne0 - e), ld_f32(x, prow + e), 0.0f);
                    }
                }
            }
            float64 s;
            NL_REDUCE_ADD(acc, IDX, s)
            const float64 mean = s * (1.0f / (float) ((int) ne0 * ne1 * nch));
            // pass 2: sum of (x - mean)^2
            float64 acc2 = 0.0f;
            for (int c = c0; c < c1; c++) {
                const int px = offx + c * sx2 + b3 * sx3;
                for (int r1 = 0; r1 < ne1; r1++) {
                    const int prow = px + r1 * sx1;
                    for (int e = 0; e < ne0; e += 64) {
                        // mask after centering: masked lanes must not contribute (0 - mean)^2
                        const float64 v = v_f32_sel_less_f32_b(lf, (float) (ne0 - e), ld_f32(x, prow + e) - mean, 0.0f);
                        acc2 = v_f32_mac_b(v, v, acc2);
                    }
                }
            }
            float64 s2;
            NL_REDUCE_ADD(acc2, IDX, s2)
            const float64 inv = nl_rsqrt(s2 * (1.0f / (float) ((int) ne0 * ne1 * nch)) + eps);
            // output pass over the immutable input
            for (int c = c0; c < c1; c++) {
                const int px = offx + c * sx2 + b3 * sx3;
                const int po = ((b3 * C + c) * ne1) * ne0;
                for (int r1 = 0; r1 < ne1; r1++) {
                    const int prow = px + r1 * sx1;
                    for (int e = 0; e < ne0; e += 64) {
                        const float64 y = (ld_f32(x, prow + e) - mean) * inv;
                        ST_F32(out, po + r1 * ne0 + e, y, ne0 - e < 64 ? ne0 - e : 64);
                    }
                }
            }
            (void) lane;
        }
    }
}

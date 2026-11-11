// rms_norm * gamma + NEOX rope (attention q/k heads).
// Rows of ne0 (% 64 == 0, <= 1024) from a view (row r -> head r % nh, token r / nh);
// rope over the first n_dims (64, or a multiple of 128) with a host table row per token.
// tensors: 0 a  1 w  2 tab  3 dep | 4 out (f32)
#include "syn_common.h"

__local__ float64 XV[16];

void main(tensor a, tensor w, tensor tab, tensor dep, tensor out,
          int nmem, int chunk, int ne0, int nh, int nrows, int offa, int sh, int st, int offw, float eps, int n_dims,
          float inv_n, float inv_nh)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int hd = n_dims >> 1;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_CHUNK(nrows, chunk, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            const int t = FDIV(r, inv_nh);
            const int h = r - t * nh;
            const int pa = offa + h * sh + t * st;
            float64 acc = 0.0f;
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 64) {
                const float64 x = ld_f32(a, pa + e);
                XV[e >> 6] = x;
                acc = v_f32_mac_b(x, x, acc);
            }
            float64 ss;
            NL_REDUCE_ADD(acc, IDX, ss);
            const float64 sc = nl_rsqrt2(ss * inv_n + eps);
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 64) {
                XV[e >> 6] = XV[e >> 6] * sc * ld_f32(w, offw + e);
            }
            const int pt = t * n_dims;
            if (hd == 32) {
                const float64 tv = ld_f32(tab, pt);
                const float64 ts = SWAP_HALVES(tv);
                const float64 cv = v_f32_sel_less_f32_b(lane, 32.0f, tv, ts);
                const float64 sv = v_f32_sel_less_f32_b(lane, 32.0f, -ts, tv);
                const float64 y = XV[0];
                XV[0] = v_f32_mac_b(SWAP_HALVES(y), sv, y * cv);
            } else {
                #pragma unroll 1
                for (int e = 0; e < hd; e += 64) {
                    const float64 c = ld_f32(tab, pt + e), sn = ld_f32(tab, pt + hd + e);
                    const float64 x0 = XV[e >> 6], x1 = XV[(hd + e) >> 6];
                    XV[e >> 6] = x0 * c - x1 * sn;
                    XV[(hd + e) >> 6] = v_f32_mac_b(x1, c, x0 * sn);
                }
            }
            #pragma unroll 1
            for (int e = 0; e < ne0; e += 64) {
                ST_F32(out, r * ne0 + e, XV[e >> 6], 64);
            }
        }
    }
}

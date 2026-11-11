// DSV4_HC_COMB (hc = 4): per token the 4x4 combination matrix from mixes[8..23] (element idst + 4 isrc):
//   v = mixes * scale[2] + base;  rows (isrc) softmaxed over idst, + eps;  columns normalized
//   (sum + eps), then n_iter - 1 times rows and columns again (Sinkhorn).
// mixes: view [24, nt] (dim 0 contiguous); scale: view [>= 3] (stride ss0); base: [24] (contiguous).
// out: contiguous [4, 4, nt].
// tensors: 0 mixes  1 scale  2 base | 3 out
#include "syn_common.h"

// the 16 broadcasts of v's dual group elements
#define BC16(v, b)                                                       \
    _Pragma("unroll")                                                    \
    for (int e_ = 0; e_ < 16; e_++) b[e_] = SHUF((v), IDX[e_]);

// per lane: OP over its row (lanes 4r..4r+3) / its column (lanes d, d+4, d+8, d+12)
#define ROWS(b, OP, lane, out)                                                                      \
    {                                                                                               \
        const float64 r0_ = OP(OP(b[0], b[1]), OP(b[2], b[3])), r1_ = OP(OP(b[4], b[5]), OP(b[6], b[7])); \
        const float64 r2_ = OP(OP(b[8], b[9]), OP(b[10], b[11])), r3_ = OP(OP(b[12], b[13]), OP(b[14], b[15])); \
        out = v_f32_sel_less_f32_b(lane, 4.0f, r0_, v_f32_sel_less_f32_b(lane, 8.0f, r1_,           \
              v_f32_sel_less_f32_b(lane, 12.0f, r2_, r3_)));                                        \
    }
#define COLS(b, lm, out)                                                                            \
    {                                                                                               \
        const float64 c0_ = (b[0] + b[4]) + (b[8] + b[12]), c1_ = (b[1] + b[5]) + (b[9] + b[13]);   \
        const float64 c2_ = (b[2] + b[6]) + (b[10] + b[14]), c3_ = (b[3] + b[7]) + (b[11] + b[15]); \
        out = v_f32_sel_less_f32_b(lm, 1.0f, c0_, v_f32_sel_less_f32_b(lm, 2.0f, c1_,               \
              v_f32_sel_less_f32_b(lm, 3.0f, c2_, c3_)));                                           \
    }
#define ADD_(a, b) ((a) + (b))

void main(tensor mixes, tensor scale, tensor base, tensor dep, tensor out,
          int nmem, int nt, int offm, int sm1, int offs, int ss0, int offb, float eps, int n_iter)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    // the column of a lane (lane mod 4, lanes 0..15)
    const float64 lm = lane - v_f32_sel_less_f32_b(lane, 4.0f, (float64) 0.0f, v_f32_sel_less_f32_b(lane, 8.0f, (float64) 4.0f,
                              v_f32_sel_less_f32_b(lane, 12.0f, (float64) 8.0f, (float64) 12.0f)));
    int5 cs = {offs + 2 * ss0, 0, 0, 0, 0};
    const float sc = s_f32_ld_g((__global float *) gen_addr(cs, scale));
    const float64 bv = ld_f32(base, offb + 8);
    for (int m = ioff[0]; m < iend[0]; m++) {
        int t0, t1;
        SPLIT_RANGE(nt, nmem, m, t0, t1)
        for (int t = t0; t < t1; t++) {
            float64 b[16];
            float64 v = v_f32_mac_b(ld_f32(mixes, offm + t * sm1 + 8), sc, bv);
            // rows: softmax + eps
            float64 mx, sm;
            BC16(v, b)
            ROWS(b, v_f32_max_b, lane, mx)
            v = nl_exp(v - mx);
            BC16(v, b)
            ROWS(b, ADD_, lane, sm)
            v = v_f32_mac_b(v, nl_recip(sm), (float64) eps);
            // columns, then (rows, columns) n_iter - 1 times
            BC16(v, b)
            COLS(b, lm, sm)
            v = v * nl_recip(sm + eps);
            for (int i = 1; i < n_iter; i++) {
                BC16(v, b)
                ROWS(b, ADD_, lane, sm)
                v = v * nl_recip(sm + eps);
                BC16(v, b)
                COLS(b, lm, sm)
                v = v * nl_recip(sm + eps);
            }
            ST_F32(out, t * 16, v, 16);
        }
    }
}

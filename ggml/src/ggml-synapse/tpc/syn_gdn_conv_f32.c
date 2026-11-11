// Causal conv1d (d_conv = 4) + SiLU of a gated-delta-net layer, one token per sequence,
// with the conv state read from and written back to the recurrent cache (side-effect write,
// input 0). The cache row keeps ggml's layout: h[3c + k] = tap k of channel c.
//   out[c] = silu(sum_k<3 w[c,k] h[3c+k] + w[c,3] x[c]);   new h[3c + k] = (h[3c+k+1] | x[c])
// W[3c+k] = w[c,k] (host pre-arranged).
// src row = idx[s], dst row = drow + s. C % 64 == 0.
// tensors: 0 R  1 idx  2 x  3 W (interleaved taps 0..2, 3C)  4 W3 (tap 3, C)  5 dep | 6 out
#include "syn_common.h"

// gather lane i of the output dual group from element (3i) % 16 of A[(3i) / 16]
#define GATHER16(A0, A1, A2, IDX, lane16, out)                                       \
    {                                                                                \
        out = SHUF(A0, IDX[0]);                                    \
        out = v_f32_sel_eq_f32_b(lane16, 1.0f, SHUF(A0, IDX[3]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 2.0f, SHUF(A0, IDX[6]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 3.0f, SHUF(A0, IDX[9]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 4.0f, SHUF(A0, IDX[12]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 5.0f, SHUF(A0, IDX[15]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 6.0f, SHUF(A1, IDX[2]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 7.0f, SHUF(A1, IDX[5]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 8.0f, SHUF(A1, IDX[8]), out);   \
        out = v_f32_sel_eq_f32_b(lane16, 9.0f, SHUF(A1, IDX[11]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 10.0f, SHUF(A1, IDX[14]), out); \
        out = v_f32_sel_eq_f32_b(lane16, 11.0f, SHUF(A2, IDX[1]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 12.0f, SHUF(A2, IDX[4]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 13.0f, SHUF(A2, IDX[7]), out);  \
        out = v_f32_sel_eq_f32_b(lane16, 14.0f, SHUF(A2, IDX[10]), out); \
        out = v_f32_sel_eq_f32_b(lane16, 15.0f, SHUF(A2, IDX[13]), out); \
    }

// interleaved [3c + k] (three vectors from element offset p + k of t) -> lane c: element k of 64 channels
#define GATHER_TAP(t, p, IDX, lane16, out)                                                        \
    {                                                                                             \
        const float64 s0_ = ld_f32(t, (p)), s1_ = ld_f32(t, (p) + 64), s2_ = ld_f32(t, (p) + 128); \
        float64 a0_ = v_f32_mov_dual_group_all_b(s0_, 0xFFFFFFFF, 0, 3, 0, 0, MkWrA(0b11, 0b11, 0b00, 0b00), s0_); \
        a0_ = v_f32_mov_dual_group_all_b(s1_, 0xFFFFFFFF, 0, 0, 2, 0, MkWrA(0b00, 0b00, 0b11, 0b00), a0_);        \
        a0_ = v_f32_mov_dual_group_all_b(s2_, 0xFFFFFFFF, 0, 0, 0, 1, MkWrA(0b00, 0b00, 0b00, 0b11), a0_);        \
        float64 a1_ = v_f32_mov_dual_group_all_b(s0_, 0xFFFFFFFF, 1, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), s0_); \
        a1_ = v_f32_mov_dual_group_all_b(s1_, 0xFFFFFFFF, 0, 0, 3, 0, MkWrA(0b00, 0b11, 0b11, 0b00), a1_);        \
        a1_ = v_f32_mov_dual_group_all_b(s2_, 0xFFFFFFFF, 0, 0, 0, 2, MkWrA(0b00, 0b00, 0b00, 0b11), a1_);        \
        float64 a2_ = v_f32_mov_dual_group_all_b(s0_, 0xFFFFFFFF, 2, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), s0_); \
        a2_ = v_f32_mov_dual_group_all_b(s1_, 0xFFFFFFFF, 0, 1, 0, 0, MkWrA(0b00, 0b11, 0b00, 0b00), a2_);        \
        a2_ = v_f32_mov_dual_group_all_b(s2_, 0xFFFFFFFF, 0, 0, 0, 3, MkWrA(0b00, 0b00, 0b11, 0b11), a2_);        \
        GATHER16(a0_, a1_, a2_, IDX, lane16, out)                                                 \
    }

#if defined(MT)
// -DMT: T tokens per sequence (verify batches), K rollback slots. Per channel the sequence
// u = [h0 h1 h2 x0 .. x(T-1)] (h = the old taps); out[t] = silu(sum_k<4 w[c,k] u[t+k]); slot r (0..K-1) gets
// taps u[j..j+2], j = max(0, T - r) (the state after token T-1-r; the old state when r >= T), at row
// drow + s + r * srs. x token t at offx + s * sxs + t * sxt; out token t of sequence s at (s * T + t) * C.
#define MAXT 8
__local__ float64 UV[MAXT + 3];

void main(tensor R, tensor idx, tensor x, tensor W, tensor W3, tensor dep, tensor out,
          int nmem, int C, int ns, int srow, int offi, int drow, int offx, int sxs, int offw, int offw3,
          int T, int K, int sxt, int srs)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane16 = v_convert_i32_to_f32_b(LANE_I32() & 15, SW_RHNE, 0, 1, 0);
    const int nb = C >> 6;
    const int n = nb * ns;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(n, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int s = it / nb;
            const int b = it - s * nb;
            int5 ci = {offi + s, 0, 0, 0, 0};
            const int row = s_i32_ld_g((__global int *) gen_addr(ci, idx));
            const int c0 = b << 6;
            const int ph = row * srow + 3 * c0;
            const int pw = offw + 3 * c0;
            float64 w0, w1, w2;
            #pragma unroll 1
            for (int k = 0; k < 3; k++) {
                float64 g;
                GATHER_TAP(R, ph + k, IDX, lane16, g)
                UV[k] = g;
            }
            GATHER_TAP(W, pw, IDX, lane16, w0)
            GATHER_TAP(W, pw + 1, IDX, lane16, w1)
            GATHER_TAP(W, pw + 2, IDX, lane16, w2)
            const float64 w3 = ld_f32(W3, offw3 + c0);
            const int px = offx + s * sxs + c0;
            #pragma unroll 1
            for (int t = 0; t < T; t++) {
                const float64 xv = ld_f32(x, px + t * sxt);
                UV[t + 3] = xv;
                const float64 conv = v_f32_mac_b(w3, xv, v_f32_mac_b(w2, UV[t + 2], v_f32_mac_b(w1, UV[t + 1], w0 * UV[t])));
                ST_F32(out, (s * T + t) * C + c0, nl_silu(conv), 64);
            }
            #pragma unroll 1
            for (int r = 0; r < K; r++) {
                const int j = T - r > 0 ? T - r : 0;
                const int pd = (drow + s + r * srs) * srow + 3 * c0;
                const float64 u0 = UV[j], u1 = UV[j + 1], u2 = UV[j + 2];
                #pragma unroll 1
                for (int c = 0; c < 64; c++) {
                    int5 c_a = {pd + 3 * c, 0, 0, 0, 0};
                    int5 c_b = {pd + 3 * c + 1, 0, 0, 0, 0};
                    int5 c_c = {pd + 3 * c + 2, 0, 0, 0, 0};
                    v_f32_st_tnsr_partial(c_a, R, u0, 0, c);
                    v_f32_st_tnsr_partial(c_b, R, u1, 0, c);
                    v_f32_st_tnsr_partial(c_c, R, u2, 0, c);
                }
            }
        }
    }
}
#else
void main(tensor R, tensor idx, tensor x, tensor W, tensor W3, tensor dep, tensor out,
          int nmem, int C, int ns, int srow, int offi, int drow, int offx, int sxs, int offw, int offw3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane16 = v_convert_i32_to_f32_b(LANE_I32() & 15, SW_RHNE, 0, 1, 0);
    const int nb = C >> 6;
    const int n = nb * ns;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(n, nmem, m, i0, i1)
        int scur = -1;
        int psrc = 0, pdst = 0, px = 0;
        for (int it = i0; it < i1; it++) {
            const int s = it / nb;
            const int b = it - s * nb;
            if (s != scur) {
                scur = s;
                int5 ci = {offi + s, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, idx));
                psrc = row * srow;
                pdst = (drow + s) * srow;
                px = offx + s * sxs;
            }
            const int c0 = b << 6;
            const int ph = psrc + 3 * c0;
            const int pw = offw + 3 * c0;
            // interleaved partial sums
            float64 S0, S1, S2, H1a, H1b, H1c;
            {
                const float64 h00 = ld_f32(R, ph), h01 = ld_f32(R, ph + 1), h02 = ld_f32(R, ph + 2);
                const float64 h10 = ld_f32(R, ph + 64), h11 = ld_f32(R, ph + 65), h12 = ld_f32(R, ph + 66);
                const float64 h20 = ld_f32(R, ph + 128), h21 = ld_f32(R, ph + 129), h22 = ld_f32(R, ph + 130);
                S0 = v_f32_mac_b(h02, ld_f32(W, pw + 2), v_f32_mac_b(h01, ld_f32(W, pw + 1), h00 * ld_f32(W, pw)));
                S1 = v_f32_mac_b(h12, ld_f32(W, pw + 66), v_f32_mac_b(h11, ld_f32(W, pw + 65), h10 * ld_f32(W, pw + 64)));
                S2 = v_f32_mac_b(h22, ld_f32(W, pw + 130), v_f32_mac_b(h21, ld_f32(W, pw + 129), h20 * ld_f32(W, pw + 128)));
                H1a = h01; H1b = h11; H1c = h21;
            }
            // A_r: dual group d of A_r = global dual group 3d + r of [S0 | S1 | S2]
            float64 A0 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 0, 3, 0, 0, MkWrA(0b11, 0b11, 0b00, 0b00), S0);
            A0 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 0, 2, 0, MkWrA(0b00, 0b00, 0b11, 0b00), A0);
            A0 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 1, MkWrA(0b00, 0b00, 0b00, 0b11), A0);
            float64 A1 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 1, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), S0);
            A1 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 0, 3, 0, MkWrA(0b00, 0b11, 0b11, 0b00), A1);
            A1 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 2, MkWrA(0b00, 0b00, 0b00, 0b11), A1);
            float64 A2 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 2, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), S0);
            A2 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 1, 0, 0, MkWrA(0b00, 0b11, 0b00, 0b00), A2);
            A2 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 3, MkWrA(0b00, 0b00, 0b11, 0b11), A2);
            float64 G;
            GATHER16(A0, A1, A2, IDX, lane16, G)
            const float64 xv = ld_f32(x, px + c0);
            const float64 conv = v_f32_mac_b(ld_f32(W3, offw3 + c0), xv, G);
            ST_F32(out, s * C + c0, nl_silu(conv), 64);
            // new state: shifted history, then x[c] into the tap-2 slots
            const int pd = pdst + 3 * c0;
            ST_F32(R, pd, H1a, 64);
            ST_F32(R, pd + 64, H1b, 64);
            ST_F32(R, pd + 128, H1c, 64);
            for (int c = 0; c < 64; c++) {
                int5 cd = {pd + 3 * c + 2, 0, 0, 0, 0};
                v_f32_st_tnsr_partial(cd, R, xv, 0, c);
            }
        }
    }
}
#endif

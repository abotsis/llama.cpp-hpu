// Gated-delta-net layer core, one token per sequence: causal conv1d (d_conv 4) + SiLU of the head's q/k/v channels (conv state read from
// and shifted back into the conv cache Rr), l2 norms, gates, delta rule on the recurrent cache Rs,
// gated RMSNorm. Both caches are side-effect writes (inputs 0 and 1). Requires H == Hk (each head
// owns its q/k channels), S = 128, C = 3 * 2048-style layout [q (Hk*128) | k (Hk*128) | v (H*128)].
//   tensors: 0 Rr  1 Rs  2 idx  3 x (projection: conv input channels)  4 W (conv weights, host
//            arranged: [taps 0..2 interleaved (3C) | tap 3 (C) | EL lane table (192)])  5 a  6 b
//            7 dt  8 A  9 z  10 w  11 dep | 12 out (bf16 with -DOUT_BF16)
#include "syn_common.h"

#define DS 128

__local__ float64 CV[6];
__local__ float64 DV[2];
__local__ float64 OV[2];
__local__ float64 ZV[2];
__local__ float64 SV[2 * DS];

#define ROT_DG(a) v_f32_mov_dual_group_all_b(a, 0xFFFFFFFF, 1, 2, 3, 0, WR_ALL, undef_f64())

void main(tensor Rr, tensor Rs, tensor idx, tensor x, tensor W, tensor ta, tensor tb, tensor dt, tensor A,
          tensor z, tensor w, tensor dep, tensor out,
          int nmem, int chunk, int H, int ns, int C, int srow_r, int drow_r, int srow_s, int drow_s, int offi,
          int offx, int sxs, int offw,
          int offa, int sa, int offb, int sb, int offdt, int offA,
          int offz, int sz, int offwn,
          float eps_qk, float qk_scale, float eps_o, float inv_H)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const float64 lane16 = v_convert_i32_to_f32_b(LANE_I32() & 15, SW_RHNE, 0, 1, 0);
    // EL_m lane L: element index (c % 16) of x for a tap-2 slot e = 64m + L = 3c + 2, else -1
    const float64 EL0 = ld_f32(W, offw + 4 * C), EL1 = ld_f32(W, offw + 4 * C + 64), EL2 = ld_f32(W, offw + 4 * C + 128);
    const float inv = 1.0f / (float) DS;
    const float oscale = 0.08838834764831845f;   // 1/sqrt(128)
    const int nitems = ns * H;
    const int koff = H * DS, voff = 2 * H * DS;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int s = FDIV(it, inv_H);
            const int h = it - s * H;
            int5 ci = {offi + s, 0, 0, 0, 0};
            const int srcrow = s_i32_ld_g((__global int *) gen_addr(ci, idx));
            const int prs = srcrow * srow_r, prd = (drow_r + s) * srow_r;
            const int px = offx + s * sxs;
            // conv + silu of the head's q, k, v channels, conv state shifted in place
            #pragma unroll 1
            for (int blk = 0; blk < 6; blk++) {
                const int c0 = (blk < 2 ? 0 : (blk < 4 ? koff : voff)) + h * DS + (blk & 1) * 64;
                const int ph = prs + 3 * c0;
                const int pw = offw + 3 * c0;
                const float64 h00 = ld_f32(Rr, ph), h01 = ld_f32(Rr, ph + 1), h02 = ld_f32(Rr, ph + 2);
                const float64 h10 = ld_f32(Rr, ph + 64), h11 = ld_f32(Rr, ph + 65), h12 = ld_f32(Rr, ph + 66);
                const float64 h20 = ld_f32(Rr, ph + 128), h21 = ld_f32(Rr, ph + 129), h22 = ld_f32(Rr, ph + 130);
                const float64 S0 = v_f32_mac_b(h02, ld_f32(W, pw + 2), v_f32_mac_b(h01, ld_f32(W, pw + 1), h00 * ld_f32(W, pw)));
                const float64 S1 = v_f32_mac_b(h12, ld_f32(W, pw + 66), v_f32_mac_b(h11, ld_f32(W, pw + 65), h10 * ld_f32(W, pw + 64)));
                const float64 S2 = v_f32_mac_b(h22, ld_f32(W, pw + 130), v_f32_mac_b(h21, ld_f32(W, pw + 129), h20 * ld_f32(W, pw + 128)));
                float64 A0 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 0, 3, 0, 0, MkWrA(0b11, 0b11, 0b00, 0b00), S0);
                A0 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 0, 2, 0, MkWrA(0b00, 0b00, 0b11, 0b00), A0);
                A0 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 1, MkWrA(0b00, 0b00, 0b00, 0b11), A0);
                float64 A1 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 1, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), S0);
                A1 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 0, 3, 0, MkWrA(0b00, 0b11, 0b11, 0b00), A1);
                A1 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 2, MkWrA(0b00, 0b00, 0b00, 0b11), A1);
                float64 A2 = v_f32_mov_dual_group_all_b(S0, 0xFFFFFFFF, 2, 0, 0, 0, MkWrA(0b11, 0b00, 0b00, 0b00), S0);
                A2 = v_f32_mov_dual_group_all_b(S1, 0xFFFFFFFF, 0, 1, 0, 0, MkWrA(0b00, 0b11, 0b00, 0b00), A2);
                A2 = v_f32_mov_dual_group_all_b(S2, 0xFFFFFFFF, 0, 0, 0, 3, MkWrA(0b00, 0b00, 0b11, 0b11), A2);
                float64 G = SHUF(A0, IDX[0]);
                G = v_f32_sel_eq_f32_b(lane16, 1.0f, SHUF(A0, IDX[3]), G);
                G = v_f32_sel_eq_f32_b(lane16, 2.0f, SHUF(A0, IDX[6]), G);
                G = v_f32_sel_eq_f32_b(lane16, 3.0f, SHUF(A0, IDX[9]), G);
                G = v_f32_sel_eq_f32_b(lane16, 4.0f, SHUF(A0, IDX[12]), G);
                G = v_f32_sel_eq_f32_b(lane16, 5.0f, SHUF(A0, IDX[15]), G);
                G = v_f32_sel_eq_f32_b(lane16, 6.0f, SHUF(A1, IDX[2]), G);
                G = v_f32_sel_eq_f32_b(lane16, 7.0f, SHUF(A1, IDX[5]), G);
                G = v_f32_sel_eq_f32_b(lane16, 8.0f, SHUF(A1, IDX[8]), G);
                G = v_f32_sel_eq_f32_b(lane16, 9.0f, SHUF(A1, IDX[11]), G);
                G = v_f32_sel_eq_f32_b(lane16, 10.0f, SHUF(A1, IDX[14]), G);
                G = v_f32_sel_eq_f32_b(lane16, 11.0f, SHUF(A2, IDX[1]), G);
                G = v_f32_sel_eq_f32_b(lane16, 12.0f, SHUF(A2, IDX[4]), G);
                G = v_f32_sel_eq_f32_b(lane16, 13.0f, SHUF(A2, IDX[7]), G);
                G = v_f32_sel_eq_f32_b(lane16, 14.0f, SHUF(A2, IDX[10]), G);
                G = v_f32_sel_eq_f32_b(lane16, 15.0f, SHUF(A2, IDX[13]), G);
                const float64 xv = ld_f32(x, px + c0);
                const float64 conv = v_f32_mac_b(ld_f32(W, offw + 3 * C + c0), xv, G);
                // new state: tap slots 3c+2 take x[c], the others the shifted history
                const int pd = prd + 3 * c0;
                const float64 B0 = v_f32_mov_dual_group_all_b(xv, 0xFFFFFFFF, 0, 0, 0, 1, WR_ALL, undef_f64());
                const float64 B1 = v_f32_mov_dual_group_all_b(xv, 0xFFFFFFFF, 1, 1, 2, 2, WR_ALL, undef_f64());
                const float64 B2 = v_f32_mov_dual_group_all_b(xv, 0xFFFFFFFF, 2, 3, 3, 3, WR_ALL, undef_f64());
                float64 N0 = h01, N1 = h11, N2 = h21;
                #pragma unroll 1
                for (int el = 0; el < 16; el++) {
                    const float elf = (float) el;
                    N0 = v_f32_sel_eq_f32_b(EL0, elf, SHUF(B0, IDX[el]), N0);
                    N1 = v_f32_sel_eq_f32_b(EL1, elf, SHUF(B1, IDX[el]), N1);
                    N2 = v_f32_sel_eq_f32_b(EL2, elf, SHUF(B2, IDX[el]), N2);
                }
                ST_F32(Rr, pd, N0, 64);
                ST_F32(Rr, pd + 64, N1, 64);
                ST_F32(Rr, pd + 128, N2, 64);
                CV[blk] = nl_silu(conv);
            }
            // l2 norms of q and k; gates softplus(a + dt) and sigmoid(b)
            float64 rq, rk, eg, beta;
            {
                const float64 q0 = CV[0], q1 = CV[1], k0 = CV[2], k1 = CV[3];
                float64 fq = v_f32_mac_b(q0, q0, q1 * q1), fk = v_f32_mac_b(k0, k0, k1 * k1);
                fq = fq + v_f32_mov_dual_group_all_b(fq, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64());
                fk = fk + v_f32_mov_dual_group_all_b(fk, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64());
                fq = fq + v_f32_mov_dual_group_all_b(fq, 0xFFFFFFFF, 2, 3, 0, 1, WR_ALL, undef_f64());
                fk = fk + v_f32_mov_dual_group_all_b(fk, 0xFFFFFFFF, 2, 3, 0, 1, WR_ALL, undef_f64());
                const float64 p = v_f32_mov_dual_group_all_b(fk, 0xFFFFFFFF, 0, 1, 1, 1, MkWrA(0b00, 0b11, 0b00, 0b00), fq);
                float64 r0 = SHUF(p, IDX[0]), r1 = SHUF(p, IDX[1]);
                #pragma unroll
                for (int e = 2; e < 16; e += 2) {
                    r0 = r0 + SHUF(p, IDX[e]);
                    r1 = r1 + SHUF(p, IDX[e + 1]);
                }
                const float64 r = nl_rsqrt2((r0 + r1) * inv + eps_qk);
                rq = BCAST_DG(r, 0) * (qk_scale * oscale);
                rk = BCAST_DG(r, 1) * qk_scale;

                const float64 g0 = BCAST0(ld_f32(ta, s * sa + offa + h), IDX) + BCAST0(ld_f32(dt, offdt + h), IDX);
                const float64 b0 = BCAST0(ld_f32(tb, s * sb + offb + h), IDX);
                const float64 g = v_f32_mov_dual_group_all_b(b0, 0xFFFFFFFF, 0, 1, 1, 1, MkWrA(0b00, 0b11, 0b00, 0b00), g0);
                // exp(g) for softplus, exp(-|b|) for an overflow-free sigmoid
                const float64 ag = v_f32_mov_dual_group_all_b(-v_f32_abs_b(g), 0xFFFFFFFF, 0, 1, 1, 1, MkWrA(0b00, 0b11, 0b00, 0b00), g);
                const float64 e1 = 1.0f + nl_exp(ag);
                const float64 sp = v_f32_sel_grt_f32_b(g, 20.0f, g, nl_log(e1));
                const float64 sg = nl_recip(e1);
                beta = BCAST_DG(v_f32_sel_less_f32_b(g, 0.0f, 1.0f - sg, sg), 1);
                eg = nl_exp(BCAST_DG(sp, 0) * BCAST0(ld_f32(A, offA + h), IDX));
                // output gate
                const int pz = s * sz + offz + h * DS;
                ZV[0] = nl_silu(ld_f32(z, pz));
                ZV[1] = nl_silu(ld_f32(z, pz + 64));
            }
            const int psrc = srcrow * srow_s + h * DS * DS;
            const int pdst = (drow_s + s) * srow_s + h * DS * DS;
            // d_j = (v_j - M_j.k) * beta for every decayed row
            #pragma unroll 1
            for (int hf = 0; hf < 2; hf++) {
                const float64 k0 = CV[2] * rk, k1 = CV[3] * rk;
                float64 dv = 0.0f;
                #pragma unroll 1
                for (int jj = 0; jj < 64; jj += 4) {
                    const int j = 64 * hf + jj;
                    const int pr = psrc + j * DS;
                    const float64 a00 = ld_f32(Rs, pr) * eg,           a01 = ld_f32(Rs, pr + 64) * eg;
                    const float64 a10 = ld_f32(Rs, pr + DS) * eg,      a11 = ld_f32(Rs, pr + DS + 64) * eg;
                    const float64 a20 = ld_f32(Rs, pr + 2 * DS) * eg,  a21 = ld_f32(Rs, pr + 2 * DS + 64) * eg;
                    const float64 a30 = ld_f32(Rs, pr + 3 * DS) * eg,  a31 = ld_f32(Rs, pr + 3 * DS + 64) * eg;
                    SV[2 * j] = a00;     SV[2 * j + 1] = a01;
                    SV[2 * j + 2] = a10; SV[2 * j + 3] = a11;
                    SV[2 * j + 4] = a20; SV[2 * j + 5] = a21;
                    SV[2 * j + 6] = a30; SV[2 * j + 7] = a31;
                    float64 d0, d1, d2, d3;
                    NL_REDUCE4_ADD(v_f32_mac_b(a00, k0, a01 * k1), v_f32_mac_b(a10, k0, a11 * k1),
                                   v_f32_mac_b(a20, k0, a21 * k1), v_f32_mac_b(a30, k0, a31 * k1), IDX, d0, d1, d2, d3)
                    const float jf = (float) jj;
                    dv = v_f32_sel_eq_f32_b(lane, jf, d0, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 1.0f, d1, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 2.0f, d2, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 3.0f, d3, dv);
                }
                DV[hf] = (CV[4 + hf] - dv) * beta;
            }
            // M_j += d_j k, o_j = M_j.q, rows stored back
            #pragma unroll 1
            for (int hf = 0; hf < 2; hf++) {
                const float64 k0 = CV[2] * rk, k1 = CV[3] * rk;
                const float64 q0 = CV[0] * rq, q1 = CV[1] * rq;
                float64 dd = DV[hf];
                float64 o = 0.0f;
                #pragma unroll 1
                for (int g16 = 0; g16 < 64; g16 += 16) {
                    const float64 db = BCAST_DG(dd, 0);
                    #pragma unroll 1
                    for (int e = 0; e < 16; e += 4) {
                        const int j = 64 * hf + g16 + e;
                        const float64 d0 = SHUF(db, IDX[e]);
                        const float64 d1 = SHUF(db, IDX[e + 1]);
                        const float64 d2 = SHUF(db, IDX[e + 2]);
                        const float64 d3 = SHUF(db, IDX[e + 3]);
                        const float64 a00 = v_f32_mac_b(k0, d0, SV[2 * j]),     a01 = v_f32_mac_b(k1, d0, SV[2 * j + 1]);
                        const float64 a10 = v_f32_mac_b(k0, d1, SV[2 * j + 2]), a11 = v_f32_mac_b(k1, d1, SV[2 * j + 3]);
                        const float64 a20 = v_f32_mac_b(k0, d2, SV[2 * j + 4]), a21 = v_f32_mac_b(k1, d2, SV[2 * j + 5]);
                        const float64 a30 = v_f32_mac_b(k0, d3, SV[2 * j + 6]), a31 = v_f32_mac_b(k1, d3, SV[2 * j + 7]);
                        const int pw = pdst + j * DS;
                        ST_F32(Rs, pw, a00, 64);           ST_F32(Rs, pw + 64, a01, 64);
                        ST_F32(Rs, pw + DS, a10, 64);      ST_F32(Rs, pw + DS + 64, a11, 64);
                        ST_F32(Rs, pw + 2 * DS, a20, 64);  ST_F32(Rs, pw + 2 * DS + 64, a21, 64);
                        ST_F32(Rs, pw + 3 * DS, a30, 64);  ST_F32(Rs, pw + 3 * DS + 64, a31, 64);
                        float64 r0, r1, r2, r3;
                        NL_REDUCE4_ADD(v_f32_mac_b(a00, q0, a01 * q1), v_f32_mac_b(a10, q0, a11 * q1),
                                       v_f32_mac_b(a20, q0, a21 * q1), v_f32_mac_b(a30, q0, a31 * q1), IDX, r0, r1, r2, r3)
                        const float jf = (float) (g16 + e);
                        o = v_f32_sel_eq_f32_b(lane, jf, r0, o);
                        o = v_f32_sel_eq_f32_b(lane, jf + 1.0f, r1, o);
                        o = v_f32_sel_eq_f32_b(lane, jf + 2.0f, r2, o);
                        o = v_f32_sel_eq_f32_b(lane, jf + 3.0f, r3, o);
                    }
                    dd = ROT_DG(dd);
                }
                OV[hf] = o;
            }
            float64 so;
            {
                const float64 o0 = OV[0], o1 = OV[1];
                NL_REDUCE_ADD(v_f32_mac_b(o0, o0, o1 * o1), IDX, so);
            }
            const float64 ro = nl_rsqrt2(so * inv + eps_o);
            const int po = (s * H + h) * DS;
            const float64 y0 = OV[0] * ro * ld_f32(w, offwn) * ZV[0];
            const float64 y1 = OV[1] * ro * ld_f32(w, offwn + 64) * ZV[1];
#if defined(OUT_BF16)
            int5 co = {po, 0, 0, 0, 0};
            v_bf16_st_tnsr(co, out, cvt_bf16(y0, y1));
#else
            ST_F32(out, po, y0, 64);
            ST_F32(out, po + 64, y1, 64);
#endif
        }
    }
}

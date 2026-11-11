// FLASH_ATTN_EXT variant: D = 64 ND (-DND=1/2/4, default 4), f16 K/V,
// f16 mask, no softcap / alibi. Same partial format as syn_fattn.
// kvl[i]: number of KV rows of query i to attend (host table, from the mask).
// tensors: 0 q  1 k  2 v  3 mask  4 kvl  5 dep | 6 part
// -DGATE_OUT (nch = 1): out = bf16(softmax(qk) v * sigmoid(gate))
// tensors: 0 q  1 k  2 v  3 mask  4 kvl  5 gate  6 dep | 7 out (bf16)
#include "syn_common.h"

#ifndef ND
#define ND 4
#endif
#define D (64 * ND)
#if defined(GATE_OUT) && ND != 4
#error "GATE_OUT: D = 256"
#endif

// a row of D f16 values as ND f32 vectors (the others zero)
#define LD4(t, p, X0, X1, X2, X3)                                                  \
    {                                                                              \
        int5 c_ = {(p), 0, 0, 0, 0};                                               \
        float128 a_ = f16_to_f32(v_f16_ld_tnsr_b(c_, t));                          \
        X0 = a_.v1; X1 = ND > 1 ? a_.v2 : (float64) 0.0f;                          \
        if (ND > 2) {                                                              \
            c_[0] = (p) + 128;                                                     \
            a_ = f16_to_f32(v_f16_ld_tnsr_b(c_, t));                               \
            X2 = a_.v1; X3 = a_.v2;                                                \
        } else {                                                                   \
            X2 = 0.0f; X3 = 0.0f;                                                  \
        }                                                                          \
    }

#if defined(GATE_OUT)
__local__ float64 AV[4];

void main(tensor q, tensor k, tensor v, tensor mask, tensor kvl, tensor gate, tensor dep, tensor out,
#else
void main(tensor q, tensor k, tensor v, tensor mask, tensor kvl, tensor dep, tensor part,
#endif
          int nmem, int mchunk, int nq, int H, int hpg, int nkv, int chunk, int nch,
          float inv_nch, float inv_nq, float inv_hpg,
          int offq, int sq1, int sq2, int offk, int sk1, int sk2, int offv, int sv1, int sv2,
          int offm, int sm1, int has_mask, float scale, int offg, int sgh, int sgi)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const int nitems = H * nq * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, mchunk, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int hi = FDIV(it, inv_nch);
            const int c = it - hi * nch;
            const int h = FDIV(hi, inv_nq);
            const int i = hi - h * nq;
            const int hk = FDIV(h, inv_hpg);
            int5 cl = {i, 0, 0, 0, 0};
            const int nv = s_i32_ld_g((__global int *) gen_addr(cl, kvl));
            const int j0 = c * chunk;
            int j1 = j0 + chunk;
            if (j1 > nv) j1 = nv;
            const int pq = offq + i * sq1 + h * sq2;
            const float64 q0 = ld_f32(q, pq) * scale;
            const float64 q1 = ND > 1 ? ld_f32(q, pq + 64) * scale : (float64) 0.0f;
            const float64 q2 = ND > 2 ? ld_f32(q, pq + 128) * scale : (float64) 0.0f;
            const float64 q3 = ND > 2 ? ld_f32(q, pq + 192) * scale : (float64) 0.0f;
            const int pk0 = offk + hk * sk2, pv0 = offv + hk * sv2;
            float64 mr = -1e30f, lr = 0.0f;
            float64 A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f;
            for (int jb = j0; jb < j1; jb += 64) {
                int je = jb + 64;
                if (je > j1) je = j1;
                float64 S = -1e30f;
                #pragma unroll 1
                for (int j = jb; j < je; j += 4) {
                    const int r1 = j + 1 < je ? j + 1 : j, r2 = j + 2 < je ? j + 2 : j, r3 = j + 3 < je ? j + 3 : j;
                    float64 x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3, w0, w1, w2, w3;
                    LD4(k, pk0 + j * sk1, x0, x1, x2, x3)
                    LD4(k, pk0 + r1 * sk1, y0, y1, y2, y3)
                    LD4(k, pk0 + r2 * sk1, z0, z1, z2, z3)
                    LD4(k, pk0 + r3 * sk1, w0, w1, w2, w3)
                    const float64 a0 = v_f32_mac_b(q3, x3, v_f32_mac_b(q2, x2, v_f32_mac_b(q1, x1, q0 * x0)));
                    const float64 a1 = v_f32_mac_b(q3, y3, v_f32_mac_b(q2, y2, v_f32_mac_b(q1, y1, q0 * y0)));
                    const float64 a2 = v_f32_mac_b(q3, z3, v_f32_mac_b(q2, z2, v_f32_mac_b(q1, z1, q0 * z0)));
                    const float64 a3 = v_f32_mac_b(q3, w3, v_f32_mac_b(q2, w2, v_f32_mac_b(q1, w1, q0 * w0)));
                    float64 s0, s1, s2, s3;
                    NL_REDUCE4_ADD(a0, a1, a2, a3, IDX, s0, s1, s2, s3)
                    const float jl = (float) (j - jb);
                    S = v_f32_sel_eq_f32_b(lane, jl, s0, S);
                    S = v_f32_sel_eq_f32_b(lane, jl + 1.0f, s1, S);
                    S = v_f32_sel_eq_f32_b(lane, jl + 2.0f, s2, S);
                    S = v_f32_sel_eq_f32_b(lane, jl + 3.0f, s3, S);
                }
                if (has_mask) {
                    int5 cm = {offm + i * sm1 + jb, 0, 0, 0, 0};
                    S = S + f16_to_f32(v_f16_ld_tnsr_b(cm, mask)).v1;
                }
                S = v_f32_sel_less_f32_b(lane, (float) (je - jb), S, -1e30f);
                float64 mb;
                NL_REDUCE_MAX(S, IDX, mb);
                const float64 mn = v_f32_max_b(mr, mb);
                const float64 corr = nl_exp(mr - mn);
                const float64 P = v_f32_sel_less_f32_b(lane, (float) (je - jb), nl_exp(S - mn), 0.0f);
                float64 ps;
                NL_REDUCE_ADD(P, IDX, ps);
                lr = v_f32_mac_b(lr, corr, ps);
                mr = mn;
                A0 = A0 * corr; A1 = A1 * corr; A2 = A2 * corr; A3 = A3 * corr;
                #pragma unroll 1
                for (int j = jb; j < je; j++) {
                    const int jj = j - jb;
                    float64 pb;
                    if (jj < 16) pb = BCAST_DG(P, 0);
                    else if (jj < 32) pb = BCAST_DG(P, 1);
                    else if (jj < 48) pb = BCAST_DG(P, 2);
                    else pb = BCAST_DG(P, 3);
                    const float64 p = SHUF(pb, IDX[jj & 15]);
                    float64 x0, x1, x2, x3;
                    LD4(v, pv0 + j * sv1, x0, x1, x2, x3)
                    A0 = v_f32_mac_b(p, x0, A0); A1 = v_f32_mac_b(p, x1, A1);
                    A2 = v_f32_mac_b(p, x2, A2); A3 = v_f32_mac_b(p, x3, A3);
                }
            }
#if defined(GATE_OUT)
            const float64 il = v_f32_sel_grt_f32_b(lr, 0.0f, nl_recip(v_f32_max_b(lr, 1e-30f)), 0.0f);
            AV[0] = A0 * il; AV[1] = A1 * il; AV[2] = A2 * il; AV[3] = A3 * il;
            const int pg = offg + h * sgh + i * sgi;
            const int po = (i * H + h) * 256;
            #pragma unroll 1
            for (int e = 0; e < 4; e += 2) {
                const float64 y0 = AV[e] * nl_sigm(ld_f32(gate, pg + 64 * e));
                const float64 y1 = AV[e + 1] * nl_sigm(ld_f32(gate, pg + 64 * e + 64));
                int5 co = {po + 64 * e, 0, 0, 0, 0};
                v_bf16_st_tnsr(co, out, cvt_bf16(y0, y1));
            }
#else
            const int po = ((h * nq + i) * nch + c) * (D + 128);
            ST_F32(part, po, A0, 64);
            if (ND > 1) {
                ST_F32(part, po + 64, A1, 64);
            }
            if (ND > 2) {
                ST_F32(part, po + 128, A2, 64);
                ST_F32(part, po + 192, A3, 64);
            }
            ST_F32(part, po + D, mr, 64);
            ST_F32(part, po + D + 64, lr, 64);
#endif
        }
    }
}

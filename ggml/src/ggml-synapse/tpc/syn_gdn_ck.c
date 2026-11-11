// Chunked (WY) gated delta rule. Head size 128, chunks of CH = 64 tokens; per (head, chunk) blocks
// are row-major 64 x 128 (dims (128, 64, heads, NC)); 64 x 64 blocks (64, 64, nv, NC).
// Value head h reads q/k head h % nk. Tokens past T are zero (no state change).
//  -DPREP : ggml q, k, v [128, heads, T] views (q scaled by 1/sqrt(128)), g / beta as [T, nv] (token index contiguous) ->
//           Q, K (128, CH, nk, NC), V (128, CH, nv, NC), GB (CH, 2, nv, NC) [chunk cumsum of g | beta]
//           tensors: 0 q 1 k 2 v 3 gT 4 bT dep | Q K V GB
//  -DSOLVE: from KK = K K^T, QK = Q K^T: P = decay-masked QK (s <= t), T = (I + A)^-1 by forward
//           substitution (A = beta decay-masked KK, s < t), Tb = T diag(beta), Tbg = T diag(beta gamma),
//           Tkb / Tkbg: rows scaled by exp(G_63 - G_t).
//           tensors: 0 KK 1 QK 2 GB dep | P Tb Tbg Tkb Tkbg
//  -DSTEP : S' = gamma_C S - X + B (one chunk).  tensors: 0 S 1 X 2 B 3 GB dep | S'
//  -DOUT  : out[j + 128 h + 128 nv t] = gamma_t QS[t][j] + PD[t][j] (t < T), then the final state
//           (transposed for ggml, St [128, 128, nv]) after the T 128 nv outputs.
//           tensors: 0 QS 1 PD 2 GB 3 St dep | out
#include "syn_common.h"

#define CH 64
#define DK 128

#if defined(PREP)
void main(tensor q, tensor k, tensor v, tensor gT, tensor bT, tensor dep, tensor Q, tensor K, tensor V, tensor GB,
          int nmem, int T, int NC, int nk, int nv, int offq, int sq1, int sq2, int offk, int sk1, int sk2,
          int offv, int sv1, int sv2, float scale)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int nitems = nv * NC;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int c = it / nv, h = it - c * nv;
            const int t0 = c * CH;
            // gates (gT / bT [T, nv]: a row of tokens per head): chunk cumsum of g, beta; 0 past T
            const float lim = (float) (T - t0);
            const float64 gv = v_f32_sel_less_f32_b(lane, lim, ld_f32(gT, h * T + t0), 0.0f);
            const float64 Bt = v_f32_sel_less_f32_b(lane, lim, ld_f32(bT, h * T + t0), 0.0f);
            float64 G = 0.0f;
            #pragma unroll
            for (int dg = 0; dg < 4; dg++) {
                float64 bd;
                if (dg == 0) bd = BCAST_DG(gv, 0);
                else if (dg == 1) bd = BCAST_DG(gv, 1);
                else if (dg == 2) bd = BCAST_DG(gv, 2);
                else bd = BCAST_DG(gv, 3);
                #pragma unroll
                for (int e = 0; e < 16; e++) {
                    G = v_f32_sel_geq_f32_b(lane, (float) (dg * 16 + e), G + SHUF(bd, IDX[e]), G);
                }
            }
            int5 cgb = {0, 0, h, c, 0};
            v_f32_st_tnsr(cgb, GB, G);
            cgb[1] = 1;
            v_f32_st_tnsr(cgb, GB, Bt);
            for (int i = 0; i < CH; i++) {
                const int t = t0 + i;
                const int ok = t < T;
                int5 co = {0, i, h, c, 0};
                float64 a0 = 0.0f, a1 = 0.0f;
                if (ok) {
                    a0 = ld_f32(v, offv + h * sv1 + t * sv2);
                    a1 = ld_f32(v, offv + h * sv1 + t * sv2 + 64);
                }
                v_f32_st_tnsr(co, V, a0);
                co[0] = 64;
                v_f32_st_tnsr(co, V, a1);
                if (h < nk) {
                    float64 b0 = 0.0f, b1 = 0.0f, k0 = 0.0f, k1 = 0.0f;
                    if (ok) {
                        b0 = ld_f32(q, offq + h * sq1 + t * sq2) * scale;
                        b1 = ld_f32(q, offq + h * sq1 + t * sq2 + 64) * scale;
                        k0 = ld_f32(k, offk + h * sk1 + t * sk2);
                        k1 = ld_f32(k, offk + h * sk1 + t * sk2 + 64);
                    }
                    co[0] = 0;
                    v_f32_st_tnsr(co, Q, b0);
                    v_f32_st_tnsr(co, K, k0);
                    co[0] = 64;
                    v_f32_st_tnsr(co, Q, b1);
                    v_f32_st_tnsr(co, K, k1);
                }
            }
        }
    }
}
#elif defined(SOLVE)
__local__ float64 TV[CH];
__local__ float64 AV[CH];

void main(tensor KK, tensor QK, tensor GB, tensor dep, tensor P, tensor Tb, tensor Tbg, tensor Tkb, tensor Tkbg,
          int nmem, int nk, int nv, int NC)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lanef = LANE_F32();
    DECL_IDX(IDX)
    const int nitems = nv * NC;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int c = it / nv, h = it - c * nv;
            const int kh = h % nk;
            int5 gc = {0, 0, h, c, 0};
            const float64 G = v_f32_ld_tnsr_b(gc, GB);
            gc[1] = 1;
            const float64 Bt = v_f32_ld_tnsr_b(gc, GB);
            const float64 BG = Bt * nl_exp(G);
            const float64 G63 = SHUF(BCAST_DG(G, 3), IDX[15]);
            const float64 KT = nl_exp(G63 - G);
            for (int s2 = 0; s2 < CH; s2++) TV[s2] = 0.0f;
// acc -= sum over s in dual group DS of A[t][s] T_s, blocks up to the diagonal (A[t][s] = 0 for
// s >= t, rows of T not computed yet are 0)
#define COLS(DS) if ((DS) * 16 <= t) {                                                      \
                const float64 Ad = BCAST_DG(arow, DS);                                       \
                float64 c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;                          \
                _Pragma("unroll")                                                            \
                for (int es = 0; es < 16; es += 4) {                                         \
                    c0 = v_f32_mac_b(SHUF(Ad, IDX[es]), TV[(DS) * 16 + es], c0);             \
                    c1 = v_f32_mac_b(SHUF(Ad, IDX[es + 1]), TV[(DS) * 16 + es + 1], c1);     \
                    c2 = v_f32_mac_b(SHUF(Ad, IDX[es + 2]), TV[(DS) * 16 + es + 2], c2);     \
                    c3 = v_f32_mac_b(SHUF(Ad, IDX[es + 3]), TV[(DS) * 16 + es + 3], c3);     \
                }                                                                            \
                acc = acc - ((c0 + c1) + (c2 + c3));                                         \
            }
// A rows, P rows out
#define AROWS(DT) {                                                                         \
            const float64 Gd = BCAST_DG(G, DT), Bd = BCAST_DG(Bt, DT);                       \
            for (int et = 0; et < 16; et++) {                                                \
                const int t = (DT) * 16 + et;                                                \
                const float lo = (float) t - 0.5f, hi = (float) t + 0.5f;                    \
                const float64 Gt = SHUF(Gd, IDX[et]);                                        \
                const float64 bt = SHUF(Bd, IDX[et]);                                        \
                const float64 dec = nl_exp(Gt - G);                                          \
                int5 rc = {0, t, h, c, 0}, kc = {0, t, kh, c, 0};                            \
                AV[t] = v_f32_sel_less_f32_b(lanef, lo, bt * dec * v_f32_ld_tnsr_b(kc, KK), 0.0f); \
                v_f32_st_tnsr(rc, P, v_f32_sel_less_f32_b(lanef, hi, dec * v_f32_ld_tnsr_b(kc, QK), 0.0f)); \
            } }
            AROWS(0) AROWS(1) AROWS(2) AROWS(3)
#undef AROWS
// forward substitution and the scaled outputs
#define ROWS(DT) {                                                                          \
            const float64 Kd = BCAST_DG(KT, DT);                                             \
            for (int et = 0; et < 16; et++) {                                                \
                const int t = (DT) * 16 + et;                                                \
                const float lo = (float) t - 0.5f, hi = (float) t + 0.5f;                    \
                const float64 ktt = SHUF(Kd, IDX[et]);                                       \
                const float64 arow = AV[t];                                                  \
                float64 acc = v_f32_sel_less_f32_b(lanef, lo, 0.0f, v_f32_sel_less_f32_b(lanef, hi, 1.0f, 0.0f)); \
                COLS(0) COLS(1) COLS(2) COLS(3)                                              \
                TV[t] = acc;                                                                 \
                int5 rc = {0, t, h, c, 0};                                                   \
                const float64 rb = acc * Bt, rbg = acc * BG;                                 \
                v_f32_st_tnsr(rc, Tb, rb);                                                   \
                v_f32_st_tnsr(rc, Tbg, rbg);                                                 \
                v_f32_st_tnsr(rc, Tkb, rb * ktt);                                            \
                v_f32_st_tnsr(rc, Tkbg, rbg * ktt);                                          \
            } }
            ROWS(0) ROWS(1) ROWS(2) ROWS(3)
#undef ROWS
#undef COLS
        }
    }
}
#elif defined(STEP)
void main(tensor S, tensor X, tensor B, tensor GB, tensor dep, tensor Sn, int nmem, int nv)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lanef = LANE_F32();
    DECL_IDX(IDX)
    const int nitems = nv * 16;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int h = it >> 4, rb = it & 15;
            int5 gc = {0, 0, h, 0, 0};
            const float64 GCv = v_f32_sel_less_f32_b(lanef, 62.5f, 0.0f, v_f32_ld_tnsr_b(gc, GB));
            float64 GC;
            NL_REDUCE_ADD(GCv, IDX, GC)
            const float64 gam = nl_exp(GC);
            for (int i = rb * 8; i < rb * 8 + 8; i++) {
                int5 c = {0, i, h, 0, 0};
                v_f32_st_tnsr(c, Sn, v_f32_mac_b(gam, v_f32_ld_tnsr_b(c, S), v_f32_ld_tnsr_b(c, B) - v_f32_ld_tnsr_b(c, X)));
                c[0] = 64;
                v_f32_st_tnsr(c, Sn, v_f32_mac_b(gam, v_f32_ld_tnsr_b(c, S), v_f32_ld_tnsr_b(c, B) - v_f32_ld_tnsr_b(c, X)));
            }
        }
    }
}
#else
void main(tensor QS, tensor PD, tensor GB, tensor St, tensor dep, tensor out, int nmem, int T, int nv, int NC)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nblk = nv * NC;
    const int nst = nv * DK;                 // state rows of 128
    const int nitems = nblk + nst / 16;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            if (it >= nblk) {
                // final state rows
                const int r0 = (it - nblk) * 16;
                for (int r = r0; r < r0 + 16; r++) {
                    const int po = T * DK * nv + r * DK;
                    ST_F32(out, po, ld_f32(St, r * DK), 64);
                    ST_F32(out, po + 64, ld_f32(St, r * DK + 64), 64);
                }
                continue;
            }
            const int c = it / nv, h = it - c * nv;
            int5 gc = {0, 0, h, c, 0};
            const float64 GAM = nl_exp(v_f32_ld_tnsr_b(gc, GB));
            for (int i = 0; i < CH; i++) {
                const int t = c * CH + i;
                if (t >= T) {
                    break;
                }
                float64 gd;
                if (i < 16) gd = BCAST_DG(GAM, 0);
                else if (i < 32) gd = BCAST_DG(GAM, 1);
                else if (i < 48) gd = BCAST_DG(GAM, 2);
                else gd = BCAST_DG(GAM, 3);
                const float64 gt = SHUF(gd, IDX[i & 15]);
                int5 cq = {0, i, h, c, 0};
                const float64 o0 = v_f32_mac_b(gt, v_f32_ld_tnsr_b(cq, QS), v_f32_ld_tnsr_b(cq, PD));
                cq[0] = 64;
                const float64 o1 = v_f32_mac_b(gt, v_f32_ld_tnsr_b(cq, QS), v_f32_ld_tnsr_b(cq, PD));
                const int po = (t * nv + h) * DK;
                ST_F32(out, po, o0, 64);
                ST_F32(out, po + 64, o1, 64);
            }
        }
    }
}
#endif

// SSM_SCAN (Mamba-1 / Mamba-2 selective scan), f32, K rollback snapshots.
// State rows s[ii][:] (ii = i1 + h nr, nr = head_dim) are contiguous over the state index c < nc. Per token
//   dts = softplus(dt[h]);  dA = exp(dts * A[h]) (Mamba-2: one per head; Mamba-1 (a1): exp(dts * A[h][c]))
//   s[ii] = s[ii] * dA + B[g] * (x[ii] * dts);  y[ii] = s[ii] . C[g]      (g = h / (nh / ng))
// Requires nr % 4 == 0, or nr == 1 (a1) with (nh / ng) % 4 == 0. Lanes past nc are zero.
// s: contiguous [nc, nr, nh, slots], sequence i3 starts from slot ids[i3]; x: view [nr, nh, nt, ns] with
// contiguous heads; dt: contiguous [nh, nt, ns]; A: contiguous [1|nc, nh]; B, C: views [nc, ng, nt, ns].
// out: [nr*nh*nt*ns outputs | K x nc*nr*nh*ns states], slot k = the state after token nt-1-k (k < min(nt, K)).
// nc <= 256 (a1: nc <= 64).
// tensors: 0 s  1 x  2 dt  3 A  4 B  5 C  6 ids | 7 out
#include "syn_common.h"

#define MAXV 4

void main(tensor st, tensor x, tensor dt, tensor A, tensor B, tensor C, tensor ids, tensor dep, tensor out,
          int nmem, int nc, int nr, int nh, int ng, int nt, int ns, int K, int a1,
          int offs, int offx, int sx2, int sx3, int offdt, int offa,
          int offb, int sb2, int sb3, int offc, int sc2, int sc3, int offi)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const int nv = (nc + 63) >> 6;
    const int R = nr * nh;
    const int nq = R >> 2;
    const int nitems = ns * nq;
    const int sst = nc * R;            // one sequence's state
    const int hpg = nh / ng;
    const int mono = nr == 1;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int i3 = it / nq;
            const int j0 = (it - i3 * nq) << 2;
            const int h0 = j0 / nr;
            const int g = h0 / hpg;
            int5 ci = {offi + i3, 0, 0, 0, 0};
            const int slot0 = s_i32_ld_g((__global int *) gen_addr(ci, ids));
            // state rows, lanes past nc zeroed
            float64 M0[MAXV], M1[MAXV], M2[MAXV], M3[MAXV];
            const int ps = offs + slot0 * sst + j0 * nc;
            #pragma unroll
            for (int u = 0; u < MAXV; u++) {
                if (u < nv) {
                    const float lim = (float) (nc - 64 * u);
                    M0[u] = v_f32_sel_less_f32_b(lane, lim, ld_f32(st, ps + u * 64), 0.0f);
                    M1[u] = v_f32_sel_less_f32_b(lane, lim, ld_f32(st, ps + nc + u * 64), 0.0f);
                    M2[u] = v_f32_sel_less_f32_b(lane, lim, ld_f32(st, ps + 2 * nc + u * 64), 0.0f);
                    M3[u] = v_f32_sel_less_f32_b(lane, lim, ld_f32(st, ps + 3 * nc + u * 64), 0.0f);
                }
            }
            // decay rates: per head or per (row, c) (a1)
            const float64 ab = BCAST_DG(ld_f32(A, offa + (a1 ? 0 : h0)), 0);
            float64 A0 = 0.0f, A1 = 0.0f, A2 = 0.0f, A3 = 0.0f;
            if (a1) {
                const float lim = (float) nc;
                A0 = v_f32_sel_less_f32_b(lane, lim, ld_f32(A, offa + h0 * nc), 0.0f);
                A1 = v_f32_sel_less_f32_b(lane, lim, ld_f32(A, offa + (h0 + mono) * nc), 0.0f);
                A2 = v_f32_sel_less_f32_b(lane, lim, ld_f32(A, offa + (h0 + 2 * mono) * nc), 0.0f);
                A3 = v_f32_sel_less_f32_b(lane, lim, ld_f32(A, offa + (h0 + 3 * mono) * nc), 0.0f);
            } else {
                A0 = SHUF(ab, IDX[0]);
                A1 = mono ? SHUF(ab, IDX[1]) : A0;
                A2 = mono ? SHUF(ab, IDX[2]) : A0;
                A3 = mono ? SHUF(ab, IDX[3]) : A0;
            }
            for (int t = 0; t < nt; t++) {
                // x of the rows, softplus(dt) of their heads
                const float64 xb = BCAST_DG(ld_f32(x, offx + t * sx2 + i3 * sx3 + j0), 0);
                const float64 db = BCAST_DG(nl_softplus(ld_f32(dt, offdt + (i3 * nt + t) * nh + h0)), 0);
                const float64 d0 = SHUF(db, IDX[0]);
                const float64 d1 = mono ? SHUF(db, IDX[1]) : d0;
                const float64 d2 = mono ? SHUF(db, IDX[2]) : d0;
                const float64 d3 = mono ? SHUF(db, IDX[3]) : d0;
                const float64 x0 = SHUF(xb, IDX[0]) * d0, x1 = SHUF(xb, IDX[1]) * d1;
                const float64 x2 = SHUF(xb, IDX[2]) * d2, x3 = SHUF(xb, IDX[3]) * d3;
                float64 e0, e1, e2, e3;
                if (a1) {
                    e0 = nl_exp(d0 * A0); e1 = nl_exp(d1 * A1); e2 = nl_exp(d2 * A2); e3 = nl_exp(d3 * A3);
                } else {
                    e0 = nl_exp(d0 * A0);
                    e1 = mono ? nl_exp(d1 * A1) : e0;
                    e2 = mono ? nl_exp(d2 * A2) : e0;
                    e3 = mono ? nl_exp(d3 * A3) : e0;
                }
                const int pb = offb + g * nc + t * sb2 + i3 * sb3;
                const int pc = offc + g * nc + t * sc2 + i3 * sc3;
                float64 a0 = 0.0f, a1v = 0.0f, a2 = 0.0f, a3 = 0.0f;
                #pragma unroll
                for (int u = 0; u < MAXV; u++) {
                    if (u < nv) {
                        const float lim = (float) (nc - 64 * u);
                        const float64 bv = v_f32_sel_less_f32_b(lane, lim, ld_f32(B, pb + u * 64), 0.0f);
                        const float64 cv = v_f32_sel_less_f32_b(lane, lim, ld_f32(C, pc + u * 64), 0.0f);
                        M0[u] = v_f32_mac_b(bv, x0, M0[u] * e0);
                        M1[u] = v_f32_mac_b(bv, x1, M1[u] * e1);
                        M2[u] = v_f32_mac_b(bv, x2, M2[u] * e2);
                        M3[u] = v_f32_mac_b(bv, x3, M3[u] * e3);
                        a0 = v_f32_mac_b(M0[u], cv, a0);
                        a1v = v_f32_mac_b(M1[u], cv, a1v);
                        a2 = v_f32_mac_b(M2[u], cv, a2);
                        a3 = v_f32_mac_b(M3[u], cv, a3);
                    }
                }
                float64 o0, o1, o2, o3;
                NL_REDUCE4_ADD(a0, a1v, a2, a3, IDX, o0, o1, o2, o3)
                float64 o = v_f32_sel_less_f32_b(lane, 1.0f, o0, o1);
                o = v_f32_sel_less_f32_b(lane, 2.0f, o, o2);
                o = v_f32_sel_less_f32_b(lane, 3.0f, o, o3);
                ST_F32(out, (i3 * nt + t) * R + j0, o, 4);
                const int slot = nt - 1 - t;
                if (slot < K) {
                    const int po = ns * nt * R + (slot * ns + i3) * sst + j0 * nc;
                    #pragma unroll
                    for (int u = 0; u < MAXV; u++) {
                        if (u < nv) {
                            const int n = nc - 64 * u;
                            ST_F32(out, po + u * 64, M0[u], n);
                            ST_F32(out, po + nc + u * 64, M1[u], n);
                            ST_F32(out, po + 2 * nc + u * 64, M2[u], n);
                            ST_F32(out, po + 3 * nc + u * 64, M3[u], n);
                        }
                    }
                }
            }
        }
    }
}

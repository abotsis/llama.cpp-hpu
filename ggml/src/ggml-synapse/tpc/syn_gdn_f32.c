// ggml_gated_delta_net, f32, K rollback snapshots.
// State rows: M[j][:] (j = value index) is contiguous over the key index i (ggml layout).
// Every row evolves independently: per token
//   M[j] *= exp(g);  d = (v[j] - M[j].k) * beta;  M[j] += d * k;  o[j] = (M[j].q) * scale
// q, k: views [S, Hq|Hk, T, ns]; v: view [S, H, T, ns]; g: contiguous [1|S, H, T, ns];
// beta: contiguous [1, H, T, ns]; state: contiguous [S, S, H, ns] (seq stride ss3).
// out: [S*H*T*ns attention outputs | K x S*S*H*ns states], slot s = the state after token T-1-s (s < min(T, K)).
// Tokens t0..t1-1 are run from the state given;
// ostore: write the attention outputs; obase: element offset of slot 0 in out. S in {64, 128, 192, 256}.
// tensors: 0 q  1 k  2 v  3 g  4 beta  5 state | 6 out
#include "syn_common.h"

#define MAXV 4

void main(tensor q, tensor k, tensor v, tensor g, tensor beta, tensor st, tensor dep, tensor out,
          int nmem, int S, int H, int T, int ns, int Hq, int Hk,
          int offq, int sq1, int sq2, int sq3,
          int offk, int sk1, int sk2, int sk3,
          int offv, int sv1, int sv2, int sv3,
          int offg, int offb, int offs, int ss3, int kda, float scale, int K, int t0, int t1, int ostore, int obase)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nv = S >> 6;
    const int ng = S >> 2;
    const int nitems = ns * H * ng;
    const int snap = S * S * H * ns;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int sh = it / ng;
            const int j0 = (it - sh * ng) << 2;
            const int s = sh / H;
            const int h = sh - s * H;
            const int hq = h % Hq;
            const int hk = h % Hk;
            // load the state rows
            float64 M0[MAXV], M1[MAXV], M2[MAXV], M3[MAXV];
            const int ps = offs + s * ss3 + (h * S + j0) * S;
            #pragma unroll
            for (int u = 0; u < MAXV; u++) {
                if (u < nv) {
                    M0[u] = ld_f32(st, ps + u * 64);
                    M1[u] = ld_f32(st, ps + S + u * 64);
                    M2[u] = ld_f32(st, ps + 2 * S + u * 64);
                    M3[u] = ld_f32(st, ps + 3 * S + u * 64);
                }
            }
            for (int t = t0; t < t1; t++) {
                const int pq = offq + hq * sq1 + t * sq2 + s * sq3;
                const int pk = offk + hk * sk1 + t * sk2 + s * sk3;
                const int pv = offv + h * sv1 + t * sv2 + s * sv3 + j0;
                const int gi = (s * T + t) * H + h;
                const float64 bt = BCAST0(ld_f32(beta, offb + gi), IDX);
                float64 kv[MAXV], qv[MAXV], eg[MAXV];
                if (kda) {
                    #pragma unroll
                    for (int u = 0; u < MAXV; u++) {
                        if (u < nv) eg[u] = nl_exp(ld_f32(g, offg + gi * S + u * 64));
                    }
                } else {
                    const float64 e = nl_exp(BCAST0(ld_f32(g, offg + gi), IDX));
                    #pragma unroll
                    for (int u = 0; u < MAXV; u++) eg[u] = e;
                }
                float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                #pragma unroll
                for (int u = 0; u < MAXV; u++) {
                    if (u < nv) {
                        kv[u] = ld_f32(k, pk + u * 64);
                        qv[u] = ld_f32(q, pq + u * 64);
                        M0[u] = M0[u] * eg[u];
                        M1[u] = M1[u] * eg[u];
                        M2[u] = M2[u] * eg[u];
                        M3[u] = M3[u] * eg[u];
                        a0 = v_f32_mac_b(M0[u], kv[u], a0);
                        a1 = v_f32_mac_b(M1[u], kv[u], a1);
                        a2 = v_f32_mac_b(M2[u], kv[u], a2);
                        a3 = v_f32_mac_b(M3[u], kv[u], a3);
                    }
                }
                float64 d0, d1, d2, d3;
                NL_REDUCE4_ADD(a0, a1, a2, a3, IDX, d0, d1, d2, d3)
                // v[j0..j0+3]
                const float64 vv = BCAST_DG(ld_f32(v, pv), 0);
                d0 = (SHUF(vv, IDX[0]) - d0) * bt;
                d1 = (SHUF(vv, IDX[1]) - d1) * bt;
                d2 = (SHUF(vv, IDX[2]) - d2) * bt;
                d3 = (SHUF(vv, IDX[3]) - d3) * bt;
                float64 b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
                #pragma unroll
                for (int u = 0; u < MAXV; u++) {
                    if (u < nv) {
                        M0[u] = v_f32_mac_b(kv[u], d0, M0[u]);
                        M1[u] = v_f32_mac_b(kv[u], d1, M1[u]);
                        M2[u] = v_f32_mac_b(kv[u], d2, M2[u]);
                        M3[u] = v_f32_mac_b(kv[u], d3, M3[u]);
                        b0 = v_f32_mac_b(M0[u], qv[u], b0);
                        b1 = v_f32_mac_b(M1[u], qv[u], b1);
                        b2 = v_f32_mac_b(M2[u], qv[u], b2);
                        b3 = v_f32_mac_b(M3[u], qv[u], b3);
                    }
                }
                float64 o0, o1, o2, o3;
                NL_REDUCE4_ADD(b0, b1, b2, b3, IDX, o0, o1, o2, o3)
                // o[j0..j0+3]
                const float64 lane = LANE_F32();
                float64 o = v_f32_sel_less_f32_b(lane, 1.0f, o0, o1);
                o = v_f32_sel_less_f32_b(lane, 2.0f, o, o2);
                o = v_f32_sel_less_f32_b(lane, 3.0f, o, o3);
                if (ostore) {
                    ST_F32(out, ((s * T + t) * H + h) * S + j0, o * scale, 4);
                }
                const int slot = T - 1 - t;
                if (slot < K) {
                    const int po = obase + slot * snap + ((s * H + h) * S + j0) * S;
                    #pragma unroll
                    for (int u = 0; u < MAXV; u++) {
                        if (u < nv) {
                            ST_F32(out, po + u * 64, M0[u], 64);
                            ST_F32(out, po + S + u * 64, M1[u], 64);
                            ST_F32(out, po + 2 * S + u * 64, M2[u], 64);
                            ST_F32(out, po + 3 * S + u * 64, M3[u], 64);
                        }
                    }
                }
            }
        }
    }
}

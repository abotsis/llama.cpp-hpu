// RWKV_WKV6 / GATED_LINEAR_ATTN / RWKV_WKV7, f32. Inputs contiguous [S, H, T] (T = ns sequences of T/ns
// tokens), state [S, S, H, ns]; out [S*H*T outputs | S*S*H*ns states]. Index of (t, h, i): t C + h S + i.
// -DWKV6 / -DGLA (state rows i = key index, contiguous over the value index j): the output sums all rows.
//   Per token, per row i (prev = the state before the token):
//   WKV6: o += (prev + u_i k_i v) r_i;  S_i = prev w_i + k_i v
//   GLA:  S_i = prev g_i + k_i v;  o += S_i q_i scale
//   tensors (WKV6): 0 k  1 v  2 r  3 u  4 w  5 state | 6 out    (GLA): 0 k  1 v  2 q  3 g  4 state | 5 out
// -DWKV7 (state rows i = value index, contiguous over the key index j): rows are independent.
//   S_i = S_i w + k v_i + (a . S_i) b;  o_i = S_i . r
//   tensors: 0 r  1 w  2 k  3 v  4 a  5 b  6 state | 7 out
// S % 64 == 0; WKV6 / GLA: S <= 128, WKV7: S <= 256.
#include "syn_common.h"

#if defined(WKV6) || defined(GLA)
__local__ float64 SL[128];

static inline float sld(tensor t, int i)
{
    int5 c = {i, 0, 0, 0, 0};
    return s_f32_ld_g((__global float *) gen_addr(c, t));
}

#if defined(WKV6)
void main(tensor k, tensor v, tensor r, tensor u, tensor w, tensor st, tensor dep, tensor out,
          int nmem, int S, int H, int T, int ns)
#else
void main(tensor k, tensor v, tensor r, tensor w, tensor st, tensor dep, tensor out,
          int nmem, int S, int H, int T, int ns, float scale)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int C = S * H;
    const int nb = S >> 6;
    const int nitems = ns * H * nb;
    const int tps = T / ns;
    const int sbase = C * T;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int sh = it / nb;
            const int j0 = (it - sh * nb) << 6;
            const int s = sh / H;
            const int h = sh - s * H;
            const int ps = (s * H + h) * S * S + j0;
            for (int i = 0; i < S; i++) {
                SL[i] = ld_f32(st, ps + i * S);
            }
            for (int tt = 0; tt < tps; tt++) {
                const int t = s * tps + tt;
                const int pt = t * C + h * S;
                const float64 vv = ld_f32(v, pt + j0);
                float64 o = 0.0f;
                for (int i = 0; i < S; i++) {
                    const float ki = sld(k, pt + i), ri = sld(r, pt + i), wi = sld(w, pt + i);
                    const float64 pv = SL[i];
#if defined(WKV6)
                    const float uk = sld(u, h * S + i) * ki;
                    o = v_f32_mac_b(v_f32_mac_b(vv, uk, pv), ri, o);
                    SL[i] = v_f32_mac_b(vv, ki, pv * wi);
#else
                    const float64 ns_ = v_f32_mac_b(vv, ki, pv * wi);
                    o = v_f32_mac_b(ns_, ri * scale, o);
                    SL[i] = ns_;
#endif
                }
                ST_F32(out, pt + j0, o, 64);
            }
            for (int i = 0; i < S; i++) {
                ST_F32(out, sbase + ps + i * S, SL[i], 64);
            }
        }
    }
}
#else
#define MAXV 4

void main(tensor r, tensor w, tensor k, tensor v, tensor a, tensor b, tensor st, tensor dep, tensor out,
          int nmem, int S, int H, int T, int ns)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int C = S * H;
    const int nv = S >> 6;
    const int ng = S >> 2;
    const int nitems = ns * H * ng;
    const int tps = T / ns;
    const int sbase = C * T;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int sh = it / ng;
            const int j0 = (it - sh * ng) << 2;
            const int s = sh / H;
            const int h = sh - s * H;
            float64 M0[MAXV], M1[MAXV], M2[MAXV], M3[MAXV];
            const int ps = ((s * H + h) * S + j0) * S;
            #pragma unroll
            for (int u = 0; u < MAXV; u++) {
                if (u < nv) {
                    M0[u] = ld_f32(st, ps + u * 64);
                    M1[u] = ld_f32(st, ps + S + u * 64);
                    M2[u] = ld_f32(st, ps + 2 * S + u * 64);
                    M3[u] = ld_f32(st, ps + 3 * S + u * 64);
                }
            }
            for (int tt = 0; tt < tps; tt++) {
                const int pt = (s * tps + tt) * C + h * S;
                float64 av[MAXV];
                float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                #pragma unroll
                for (int u = 0; u < MAXV; u++) {
                    if (u < nv) {
                        av[u] = ld_f32(a, pt + u * 64);
                        a0 = v_f32_mac_b(M0[u], av[u], a0);
                        a1 = v_f32_mac_b(M1[u], av[u], a1);
                        a2 = v_f32_mac_b(M2[u], av[u], a2);
                        a3 = v_f32_mac_b(M3[u], av[u], a3);
                    }
                }
                float64 sa0, sa1, sa2, sa3;
                NL_REDUCE4_ADD(a0, a1, a2, a3, IDX, sa0, sa1, sa2, sa3)
                const float64 vb = BCAST_DG(ld_f32(v, pt + j0), 0);
                const float64 v0 = SHUF(vb, IDX[0]), v1 = SHUF(vb, IDX[1]), v2 = SHUF(vb, IDX[2]), v3 = SHUF(vb, IDX[3]);
                float64 b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
                #pragma unroll
                for (int u = 0; u < MAXV; u++) {
                    if (u < nv) {
                        const float64 wv = ld_f32(w, pt + u * 64), kv = ld_f32(k, pt + u * 64);
                        const float64 bv = ld_f32(b, pt + u * 64), rv = ld_f32(r, pt + u * 64);
                        M0[u] = v_f32_mac_b(sa0, bv, v_f32_mac_b(kv, v0, M0[u] * wv));
                        M1[u] = v_f32_mac_b(sa1, bv, v_f32_mac_b(kv, v1, M1[u] * wv));
                        M2[u] = v_f32_mac_b(sa2, bv, v_f32_mac_b(kv, v2, M2[u] * wv));
                        M3[u] = v_f32_mac_b(sa3, bv, v_f32_mac_b(kv, v3, M3[u] * wv));
                        b0 = v_f32_mac_b(M0[u], rv, b0);
                        b1 = v_f32_mac_b(M1[u], rv, b1);
                        b2 = v_f32_mac_b(M2[u], rv, b2);
                        b3 = v_f32_mac_b(M3[u], rv, b3);
                    }
                }
                float64 o0, o1, o2, o3;
                NL_REDUCE4_ADD(b0, b1, b2, b3, IDX, o0, o1, o2, o3)
                float64 o = v_f32_sel_less_f32_b(lane, 1.0f, o0, o1);
                o = v_f32_sel_less_f32_b(lane, 2.0f, o, o2);
                o = v_f32_sel_less_f32_b(lane, 3.0f, o, o3);
                ST_F32(out, pt + j0, o, 4);
            }
            #pragma unroll
            for (int u = 0; u < MAXV; u++) {
                if (u < nv) {
                    ST_F32(out, sbase + ps + u * 64, M0[u], 64);
                    ST_F32(out, sbase + ps + S + u * 64, M1[u], 64);
                    ST_F32(out, sbase + ps + 2 * S + u * 64, M2[u], 64);
                    ST_F32(out, sbase + ps + 3 * S + u * 64, M3[u], 64);
                }
            }
        }
    }
}
#endif

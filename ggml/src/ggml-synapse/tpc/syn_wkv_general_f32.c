// RWKV_WKV6 / GATED_LINEAR_ATTN / RWKV_WKV7 general device paths, f32: any positive S (syn_wkv_f32.c needs
// S % 64 == 0 and S <= 128 (WKV6 / GLA) or 256 (WKV7)).
// Inputs contiguous [S, H, T] with T = ns sequences of tps = T/ns tokens (index of (t, h, i): t*C + h*S + i,
// C = S*H). State [S*S*H, ns]: sequence s, head h at (s*H + h)*S*S. out: [C*T outputs | S*S*H*ns final states].
// Every item runs all the tokens of its sequence with its state tile in local memory; every tensor has one writer.
//
// WKV6 / GLA (state rows i = key index, contiguous over the value index j). Per token, per row i (prev = the
// state before the token):
//   WKV6: o += (prev + u_i k_i v) r_i;   S_i = prev * w_i + k_i * v
//   GLA:  S_i = prev * g_i + k_i * v;    o += S_i * q_i * scale
// STEP: one item per (sequence, head, 32-row tile rt, 64-column chunk vc); per token the tile's partial of the
// output chunk goes to part[rt][t*C + h*S + j] (part: [nr32][C*T]), the final state tile to sts [S*S*H*ns].
// REDUCE sums the row-tile partials in increasing tile order into the outputs and copies sts into the state part
// of out.
// WKV6  STEP tensors: 0 k  1 v  2 r  3 u  4 w  5 st  6 dep | 7 part  8 sts
// GLA   STEP tensors: 0 k  1 v  2 q  3 g  4 st  5 dep | 6 part  7 sts
//   (scalars: nmem, S, H, T, ns; GLA adds scale)
// REDUCE tensors: 0 part  1 sts  2 dep | 3 out   (scalars: nmem, S, H, T, ns)
//
// WKV7 (state rows i = value index, contiguous over the key index j; rows are independent):
//   S_i = S_i * w + k * v_i + dot(a, S_i) * b;  o_i = S_i . r
// one node, one item per (sequence, head, row) holding the whole row in local memory (S <= 64 * WKV7_MAXV).
// WKV7 tensors: 0 r  1 w  2 k  3 v  4 a  5 b  6 st  7 dep | 8 out   (scalars: nmem, S, H, T, ns)
#include "syn_common.h"

#if defined(WKV7)
#define WKV7_MAXV 128
__local__ float64 SL[WKV7_MAXV];

void main(tensor r, tensor w, tensor k, tensor v, tensor a, tensor b, tensor st, tensor dep, tensor out,
          int nmem, int S, int H, int T, int ns)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int C = S * H;
    const int nkc = (S + 63) >> 6;
    const int tps = T / ns;
    const int CT = C * T;
    const int nitems = ns * H * S;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int i = it % S;
            const int sh = it / S;                     // s * H + h
            const int s = sh / H;
            const int h = sh - s * H;
            const int ps = sh * S * S + i * S;
            for (int u = 0; u < nkc; u++) {
                const int cl = S - u * 64 < 64 ? S - u * 64 : 64;
                SL[u] = v_f32_sel_less_f32_b(lane, (float) cl, ld_f32(st, ps + u * 64), 0.0f);
            }
            for (int tt = 0; tt < tps; tt++) {
                const int pt = (s * tps + tt) * C + h * S;
                // dot(a, old_row) before any update of the row (lanes past S are 0 in SL)
                float64 dacc = 0.0f;
                for (int u = 0; u < nkc; u++) {
                    dacc = v_f32_mac_b(ld_f32(a, pt + u * 64), SL[u], dacc);
                }
                float64 sa;
                NL_REDUCE_ADD(dacc, IDX, sa)
                const float64 vi = BCAST0(ld_f32(v, pt + i), IDX);
                float64 oacc = 0.0f;
                for (int u = 0; u < nkc; u++) {
                    const int cl = S - u * 64 < 64 ? S - u * 64 : 64;
                    const float64 wv = ld_f32(w, pt + u * 64);
                    const float64 kv = ld_f32(k, pt + u * 64);
                    const float64 bv = ld_f32(b, pt + u * 64);
                    const float64 rv = ld_f32(r, pt + u * 64);
                    // new_row = old_row * w + k * v_i + dot * b; lanes past S stay 0
                    float64 nu = v_f32_mac_b(bv, sa, v_f32_mac_b(kv, vi, SL[u] * wv));
                    nu = v_f32_sel_less_f32_b(lane, (float) cl, nu, 0.0f);
                    SL[u] = nu;
                    oacc = v_f32_mac_b(nu, rv, oacc);
                }
                float64 o;
                NL_REDUCE_ADD(oacc, IDX, o)
                ST_F32(out, pt + i, o, 1);
            }
            for (int u = 0; u < nkc; u++) {
                const int cl = S - u * 64 < 64 ? S - u * 64 : 64;
                ST_F32(out, CT + ps + u * 64, SL[u], cl);
            }
        }
    }
}
#elif defined(REDUCE)
void main(tensor part, tensor sts, tensor dep, tensor out, int nmem, int S, int H, int T, int ns)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int CT = S * H * T;
    const int nr32 = (S + 31) >> 5;
    const int nout = (CT + 63) >> 6;
    const int nst = S * S * H * ns;
    const int ncopy = (nst + 63) >> 6;
    const int nitems = nout + ncopy;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            if (it >= nout) {
                const int e0 = (it - nout) << 6;
                const int left = nst - e0;
                ST_F32(out, CT + e0, ld_f32(sts, e0), left < 64 ? left : 64);
                continue;
            }
            const int e0 = it << 6;
            const int left = CT - e0;
            float64 o = 0.0f;
            for (int u = 0; u < nr32; u++) {           // increasing tile order
                o = o + ld_f32(part, u * CT + e0);
            }
            ST_F32(out, e0, o, left < 64 ? left : 64);
        }
    }
}
#else
// WKV6 / GLA STEP
__local__ float64 SL[32];

#if defined(WKV6)
void main(tensor k, tensor v, tensor r, tensor u, tensor w, tensor st, tensor dep, tensor part, tensor sts,
          int nmem, int S, int H, int T, int ns)
#else
void main(tensor k, tensor v, tensor q, tensor g, tensor st, tensor dep, tensor part, tensor sts,
          int nmem, int S, int H, int T, int ns, float scale)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const int C = S * H;
    const int nr32 = (S + 31) >> 5;
    const int nvc = (S + 63) >> 6;
    const int CT = C * T;
    const int tps = T / ns;
    const int nitems = ns * H * nr32 * nvc;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int x = it;
            const int vc = x % nvc;   x /= nvc;
            const int rt = x % nr32;  x /= nr32;
            const int sh = x;                          // s * H + h
            const int s = sh / H;
            const int h = sh - s * H;
            const int ps = sh * S * S + rt * 32 * S + vc * 64;
            const int rows = S - rt * 32 < 32 ? S - rt * 32 : 32;
            const int vlim = S - vc * 64 < 64 ? S - vc * 64 : 64;
            for (int i = 0; i < rows; i++) {
                SL[i] = v_f32_sel_less_f32_b(lane, (float) vlim, ld_f32(st, ps + i * S), 0.0f);
            }
            for (int tt = 0; tt < tps; tt++) {
                const int pt = (s * tps + tt) * C + h * S;
                const float64 vj = v_f32_sel_less_f32_b(lane, (float) vlim, ld_f32(v, pt + vc * 64), 0.0f);
                float64 o = 0.0f;
                for (int i = 0; i < rows; i++) {
                    const int row = rt * 32 + i;
                    const float64 ki = BCAST0(ld_f32(k, pt + row), IDX);
                    const float64 pv = SL[i];
#if defined(WKV6)
                    // o += (prev + u_i k_i v) r_i;  S_i = prev w_i + k_i v
                    const float64 ri = BCAST0(ld_f32(r, pt + row), IDX);
                    const float64 uk = BCAST0(ld_f32(u, h * S + row), IDX) * ki;
                    o = v_f32_mac_b(v_f32_mac_b(uk, vj, pv), ri, o);
                    SL[i] = v_f32_mac_b(vj, ki, pv * BCAST0(ld_f32(w, pt + row), IDX));
#else
                    // S_i = prev g_i + k_i v;  o += S_i q_i scale
                    const float64 gi = BCAST0(ld_f32(g, pt + row), IDX);
                    const float64 ns_ = v_f32_mac_b(vj, ki, pv * gi);
                    o = v_f32_mac_b(ns_, BCAST0(ld_f32(q, pt + row), IDX) * scale, o);
                    SL[i] = ns_;
#endif
                }
                ST_F32(part, rt * CT + pt + vc * 64, o, vlim);
            }
            for (int i = 0; i < rows; i++) {
                ST_F32(sts, ps + i * S, SL[i], vlim);
            }
        }
    }
}
#endif

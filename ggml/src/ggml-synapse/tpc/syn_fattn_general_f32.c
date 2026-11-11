// FLASH_ATTN_EXT, f32.
// Inputs are dense F32. q [Dk, nq, H, B], k [Dk, nkv, Hkv, Bkv],
// v [Dv, nkv, Hkv, Bkv] dense flat; mask F16 [nkv, nqp, M2, M3] scalar-read with the KV dim
// contiguous (sm1/sm2/sm3 are the query/head/batch strides, the same mask view contract as
// syn_fattn; head/batch broadcast via %M2 / %M3). Sinks [H] f32.
// Score per key: d = q.k; s = d * sc (sc = softcap != 0 ?
// scale / softcap : scale, computed by the lowerer); capped: s = softcap * tanh(s);
// s += slope_h * mask (alibi: slope table [H], slope 1.0 without alibi). Keys whose mask is
// -INFINITY are skipped before any maximum / sum update (no -1e30 substitute).
// Online softmax: max m, sum l (the sink is added to l once, no numerator) and the
// accumulator, both rescaled when m increases.
// l == 0 (empty / all-masked row) yields exact zero output.
// out contiguous: element (d, h, i, b) at ((b*nq+i)*H+h)*Dv + d.
// tensors: 0 q  1 k  2 v  3 mask  4 slopes  5 sinks | 6 out
// scalars: nmem, Dk, Dv, nq, H, nkv, B, hpg, offq, offk, offv, offm, sm1, sm2, sm3, M2, M3,
//   sc, softcap, flags  (20 <= 32 dwords; sc folds scale and softcap)
#include "syn_common.h"

void main(tensor q, tensor k, tensor v, tensor mask, tensor slopes, tensor sinks, tensor dep, tensor out,
          int nmem, int Dk, int Dv, int nq, int H, int nkv, int B, int hpg,
          int offq, int offk, int offv, int offm, int sm1, int sm2, int sm3, int M2, int M3,
          float sc, float softcap, int flags)
{
    // flags: has_mask | alibi << 1 | has_sinks << 2
    const int has_mask = flags & 1, alibi = (flags >> 1) & 1, has_sinks = (flags >> 2) & 1;
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const float neginf = -1.0f / 0.0f;               // exact -INFINITY scalar operand
    const int Hkv = H / hpg;
    const int nvt = (Dv + 63) >> 6;
    const int nkc = (Dk + 63) >> 6;
    const int nitems = B * nq * H * nvt;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int x = it;
            const int vt = x % nvt;  x /= nvt;
            const int hh = x % H;    x /= H;
            const int i  = x % nq;   x /= nq;
            const int b  = x;
            const int hk = hh / hpg;
            const int pq = offq + ((b * H + hh) * nq + i) * Dk;
            const int pk = offk + (b * Hkv + hk) * nkv * Dk;
            const int pv = offv + (b * Hkv + hk) * nkv * Dv;
            const int pm = offm + i * sm1 + (hh % M2) * sm2 + (b % M3) * sm3;
            float slope = 1.0f;
            if (alibi) {
                int5 cs = {hh, 0, 0, 0, 0};
                slope = s_f32_ld_g(gen_addr(cs, slopes));
            }
            // online softmax state; the sink (if any) is its initial maximum with weight 1
            float64 M = -1e30f, l = 0.0f;
            if (has_sinks) {
                int5 cs = {hh, 0, 0, 0, 0};
                M = s_f32_ld_g(gen_addr(cs, sinks));
                l = 1.0f;
            }
            float64 acc = 0.0f;
            const int vlim = Dv - vt * 64 < 64 ? Dv - vt * 64 : 64;
            for (int j = 0; j < nkv; j++) {
                // dot product q.k
                float64 d = 0.0f;
                for (int u = 0; u < nkc; u++) {
                    const int lim = Dk - u * 64 < 64 ? Dk - u * 64 : 64;
                    const float64 qc = v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(q, pq + u * 64), 0.0f);
                    const float64 kc = v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(k, pk + j * Dk + u * 64), 0.0f);
                    d = v_f32_mac_b(qc, kc, d);
                }
                NL_REDUCE_ADD(d, IDX, d)
                float64 s = d * sc;
                if (softcap != 0.0f) {
                    s = softcap * nl_tanh(s);
                }
                if (has_mask) {
                    int5 cm = {pm + j, 0, 0, 0, 0};
                    const float64 mv = (float64) s_f16_ld_g((__global half *) gen_addr(cm, mask));
                    const float64 ms = mv * slope;
                    s = s + ms;
                    // slope * mask == -INFINITY: skip before any maximum / sum update
                    s = v_f32_sel_eq_f32_b(ms, neginf, M, s);
                    const float64 mx = v_f32_max_b(M, s);
                    const float64 corr = nl_exp(M - mx);
                    const float64 p = v_f32_sel_eq_f32_b(ms, neginf, 0.0f, nl_exp(s - mx));
                    acc = acc * corr;
                    l = l * corr + p;
                    M = mx;
                    acc = v_f32_mac_b(p, v_f32_sel_less_f32_b(lane, (float) vlim, ld_f32(v, pv + j * Dv + vt * 64), 0.0f), acc);
                } else {
                    const float64 mx = v_f32_max_b(M, s);
                    const float64 corr = nl_exp(M - mx);
                    const float64 p = nl_exp(s - mx);
                    acc = acc * corr;
                    l = l * corr + p;
                    M = mx;
                    acc = v_f32_mac_b(p, v_f32_sel_less_f32_b(lane, (float) vlim, ld_f32(v, pv + j * Dv + vt * 64), 0.0f), acc);
                }
            }
            const float64 il = v_f32_sel_grt_f32_b(l, 0.0f, nl_recip(v_f32_max_b(l, 1e-30f)), 0.0f);
            const int po = ((b * nq + i) * H + hh) * Dv + vt * 64;
            ST_F32(out, po, acc * il, vlim);
        }
    }
}

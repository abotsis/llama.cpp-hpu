// SSM_SCAN general device path, f32, K rollback snapshots: any valid nc and nr (syn_ssm_scan_f32.c needs
// (nr % 4 == 0) || (nr == 1 && (nh/ng) % 4 == 0) and nc bounds).
// State rows s[ii][:] are contiguous over the state index c < nc; state [nc, nr, nh, slots], sequence i3 starts
// from slot ids[i3]. Per token t of sequence i3
//   dts = softplus(dt[h]);  dA = exp(dts * A[h]) (Mamba-2: per head; Mamba-1 (a1): A[h][c])
//   s[c] = s[c] * dA + B[g][c] * (x[ii] * dts);  y = sum_c s[c] * C[g][c]   (g = h / (nh/ng))
// STEP: one item per (sequence, row, 64-state tile) runs every token with its tile in a register: per token the
// tile's partial C-dot goes to part[t][ii][tile], the state after token nt-1-k to snapshot k < K (sts).
// REDUCE sums a row's tile partials in increasing tile order into the token output, and copies the snapshots
// into the state part of out. Every tensor has one writer.
// STEP tensors: 0 st  1 x  2 dt  3 A  4 B  5 C  6 ids  7 dep | 8 part  9 sts
// STEP scalars: nmem, nc, nr, nh, ng, nt, ns, K, a1, offs, offx, sx2, sx3, offdt, offa, offb, sb2, sb3, offc, sc2,
//   sc3, offi
// REDUCE tensors: 0 part  1 sts  2 dep | 3 out [nr nh nt ns outputs | K x nc nr nh ns states]
#include "syn_common.h"

#if defined(REDUCE)
void main(tensor part, tensor sts, tensor dep, tensor out, int nmem, int nc, int nr, int nh, int nt, int ns, int R, int K)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int ntile = (nc + 63) >> 6;
    const int nout = ns * R * nt;
    const int nst = K * ns * R * nc;                 // snapshot floats
    const int ncopy = (nst + 63) >> 6;
    const int nitems = nout + ncopy;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            if (it >= nout) {
                const int e0 = (it - nout) << 6;
                const int left = nst - e0;
                ST_F32(out, nout + e0, ld_f32(sts, e0), left < 64 ? left : 64);
                continue;
            }
            int x = it;
            const int t = x % nt;  x /= nt;
            const int ii = x % R;  x /= R;
            const int i3 = x;
            const int pp = ((i3 * nt + t) * R + ii) * ntile;
            float64 o = 0.0f;
            for (int u = 0; u < ntile; u += 64) {
                const int lim = ntile - u < 64 ? ntile - u : 64;
                o = o + v_f32_sel_less_f32_b(LANE_F32(), (float) lim, ld_f32(part, pp + u), 0.0f);
            }
            float64 oo;
            NL_REDUCE_ADD(o, IDX, oo)
            ST_F32(out, (i3 * nt + t) * R + ii, oo, 1);
        }
    }
}
#else
// x is a view [nr, nh, nt, ns] with contiguous heads (element offset offx, strides per dim 2/3); dt contiguous
// [nh, nt, ns]; A contiguous [1|nc, nh]; B, C views [nc, ng, nt, ns] with element offsets and dim 2/3 strides.
void main(tensor st, tensor x, tensor dt, tensor A, tensor B, tensor C, tensor ids, tensor dep, tensor part,
          tensor sts, int nmem, int nc, int nr, int nh, int ng, int nt, int ns, int K, int a1,
          int offs, int offx, int sx2, int sx3, int offdt, int offa, int offb, int sb2, int sb3,
          int offc, int sc2, int sc3, int offi)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const int R = nr * nh;
    const int ntile = (nc + 63) >> 6;
    const int nitems = ns * R * ntile;
    const int hpg = nh / ng;
    const int sst = nc * R;                           // one sequence's state
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int xi = it;
            const int u  = xi % ntile;  xi /= ntile;
            const int ii = xi % R;      xi /= R;
            const int i3 = xi;
            const int h = ii / nr;
            const int g = h / hpg;
            int5 ci = {offi + i3, 0, 0, 0, 0};
            const int slot0 = s_i32_ld_g((__global int *) gen_addr(ci, ids));
            const int lim = nc - u * 64 < 64 ? nc - u * 64 : 64;
            // the sequence's state tile, lanes past nc zeroed
            float64 s = v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(st, offs + slot0 * sst + ii * nc + u * 64), 0.0f);
            const float64 av = a1 ? v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(A, offa + h * nc + u * 64), 0.0f) :
                                    BCAST0(ld_f32(A, offa + h), IDX);
            for (int t = 0; t < nt; t++) {
                const float64 dtv = BCAST0(nl_softplus(ld_f32(dt, offdt + (i3 * nt + t) * nh + h)), IDX);
                const float64 e = nl_exp(dtv * av);
                // x rows are contiguous over (nr, nh): row ii's value is at flat offset ii
                const float64 xv = BCAST0(ld_f32(x, offx + t * sx2 + i3 * sx3 + ii), IDX) * dtv;
                const float64 bv = v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(B, offb + g * nc + t * sb2 + i3 * sb3 + u * 64), 0.0f);
                const float64 cv = v_f32_sel_less_f32_b(lane, (float) lim, ld_f32(C, offc + g * nc + t * sc2 + i3 * sc3 + u * 64), 0.0f);
                s = v_f32_mac_b(bv, xv, s * e);
                float64 d;
                NL_REDUCE_ADD(s * cv, IDX, d)
                ST_F32(part, ((i3 * nt + t) * R + ii) * ntile + u, d, 1);
                // snapshots: slot k = the state after token nt-1-k
                const int slot = nt - 1 - t;
                if (slot < K) {
                    ST_F32(sts, (slot * ns + i3) * sst + ii * nc + u * 64, s, lim);
                }
            }
        }
    }
}
#endif

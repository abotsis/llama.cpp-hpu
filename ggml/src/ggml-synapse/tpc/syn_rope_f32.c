// rope with a host-computed table, f32 (or f16 in and out with -DIO_F16).
// NEOX style (default): pairs (i, i + n_dims/2) from n_offs; tab row t (token i2) =
//   [cos(0..hd) | sin(0..hd)], hd = n_dims / 2.
// -DNORM (GGML_ROPE_TYPE_NORMAL): adjacent pairs (2i, 2i + 1) from n_offs; tab row t =
//   [C(n_dims) | S(n_dims)], C[2i] = C[2i+1] = cos(i), S[2i] = -sin(i), S[2i+1] = sin(i).
// out contiguous [ne0, ne1, ne2, ne3]; a is a view (dim-0 contiguous). Channels outside
// [n_offs, n_offs + n_dims) are copied.
// out row (i1, i2 i3) at i1 ost1 + (i2 + ne2 i3) ost2 (contiguous: ost1 = ne0, ost2 = ne0 ne1).
// -DOUT_BF16: bf16 out (f32 in), e.g. the attention's Q as [D, tokens, heads].
// tensors: 0 a  1 tab | 2 out
#include "syn_common.h"

#if defined(IO_F16)
#define LD_X(c, t) f16_to_f32(v_f16_ld_tnsr_b((c), (t))).v1
#define STP_X(c, t, v, lp) v_f16_st_tnsr_partial((c), (t), cvt_f16((v), (v)), (lp), 0)
#elif defined(OUT_BF16)
#define LD_X(c, t) v_f32_ld_tnsr_b((c), (t))
#define STP_X(c, t, v, lp) v_bf16_st_tnsr_partial((c), (t), cvt_bf16((v), (v)), (lp), 0)
#else
#define LD_X(c, t) v_f32_ld_tnsr_b((c), (t))
#define STP_X(c, t, v, lp) v_f32_st_tnsr_partial((c), (t), (v), (lp), 0)
#endif

void main(tensor a, tensor tab, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3, int n_dims, int n_offs,
          int off0, int s01, int s02, int s03, int hb, int nhb, int chunk, int ost1, int ost2)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nitems = ne2 * ne3 * nhb;
#if defined(NORM)
    // 0 on even lanes, 0.5 on odd lanes
    const float64 lh = LANE_F32() * 0.5f;
    const float64 par = lh - v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(lh, SW_RD, 0, 1, 0), SW_RHNE, 0, 1, 0);
#endif
    const int hd = n_dims >> 1;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r23 = i0 / nhb;
        int hblk = i0 - r23 * nhb;
        int r3 = r23 / ne2;
        int r2 = r23 - r3 * ne2;
        for (int it = i0; it < i1; it++) {
            const int h0 = hblk * hb;
            const int h1 = h0 + hb < ne1 ? h0 + hb : ne1;
            const int pa0 = off0 + r2 * s02 + r3 * s03;
            const int po0 = r23 * ost2;
#if defined(NORM)
            const int pt = r2 * 2 * n_dims;
            for (int e = 0; e < n_dims; e += 64) {
                const float64 c = ld_f32(tab, pt + e);
                const float64 s = ld_f32(tab, pt + n_dims + e);
                const int lp = n_dims - e >= 64 ? 63 : n_dims - e - 1;
                int5 ca = {pa0 + h0 * s01 + n_offs + e, 0, 0, 0, 0};
                int5 cn = ca, cp = ca, co = {po0 + h0 * ost1 + n_offs + e, 0, 0, 0, 0};
                cn[0] += 1;
                cp[0] -= 1;
                int h = h0;
                for (; h + 3 < h1; h += 4) {
                    float64 x[4], xn[4], xp[4];
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        x[u] = LD_X(ca, a);
                        xn[u] = LD_X(cn, a);
                        xp[u] = LD_X(cp, a);
                        ca[0] += s01;
                        cn[0] += s01;
                        cp[0] += s01;
                    }
                    #pragma unroll
                    for (int u = 0; u < 4; u++) {
                        const float64 y = v_f32_sel_eq_f32_b(par, 0.0f, xn[u], xp[u]);
                        STP_X(co, out, v_f32_mac_b(y, s, x[u] * c), lp);
                        co[0] += ost1;
                    }
                }
                for (; h < h1; h++) {
                    const float64 x = LD_X(ca, a), xn = LD_X(cn, a), xp = LD_X(cp, a);
                    const float64 y = v_f32_sel_eq_f32_b(par, 0.0f, xn, xp);
                    STP_X(co, out, v_f32_mac_b(y, s, x * c), lp);
                    ca[0] += s01;
                    cn[0] += s01;
                    cp[0] += s01;
                    co[0] += ost1;
                }
            }
#else
            const int pt = r2 * n_dims;
            for (int e = 0; e < hd; e += 64) {
                const float64 c = ld_f32(tab, pt + e);
                const float64 s = ld_f32(tab, pt + hd + e);
                const int lp = hd - e >= 64 ? 63 : hd - e - 1;
                int5 c0 = {pa0 + h0 * s01 + n_offs + e, 0, 0, 0, 0};
                int5 c1 = c0, o0 = {po0 + h0 * ost1 + n_offs + e, 0, 0, 0, 0};
                c1[0] += hd;
                int5 o1 = o0;
                o1[0] += hd;
                int h = h0;
                for (; h + 1 < h1; h += 2) {
                    float64 x0[2], x1[2];
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        x0[u] = LD_X(c0, a);
                        x1[u] = LD_X(c1, a);
                        c0[0] += s01;
                        c1[0] += s01;
                    }
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        STP_X(o0, out, x0[u] * c - x1[u] * s, lp);
                        STP_X(o1, out, v_f32_mac_b(x1[u], c, x0[u] * s), lp);
                        o0[0] += ost1;
                        o1[0] += ost1;
                    }
                }
                if (h < h1) {
                    const float64 x0 = LD_X(c0, a), x1 = LD_X(c1, a);
                    STP_X(o0, out, x0 * c - x1 * s, lp);
                    STP_X(o1, out, v_f32_mac_b(x1, c, x0 * s), lp);
                }
            }
#endif
            if (n_offs > 0 || n_offs + n_dims < ne0) {
                for (int h = h0; h < h1; h++) {
                    const int pa = pa0 + h * s01, po = po0 + h * ost1;
                    for (int e = 0; e < n_offs; e += 64) {
                        int5 ci = {pa + e, 0, 0, 0, 0}, cq = {po + e, 0, 0, 0, 0};
                        STP_X(cq, out, LD_X(ci, a), n_offs - e >= 64 ? 63 : n_offs - e - 1);
                    }
                    for (int e = n_offs + n_dims; e < ne0; e += 64) {
                        int5 ci = {pa + e, 0, 0, 0, 0}, cq = {po + e, 0, 0, 0, 0};
                        STP_X(cq, out, LD_X(ci, a), ne0 - e >= 64 ? 63 : ne0 - e - 1);
                    }
                }
            }
            if (++hblk == nhb) {
                hblk = 0;
                r23++;
                if (++r2 == ne2) {
                    r2 = 0;
                    r3++;
                }
            }
        }
    }
}

// out = f(a), f32: the less common unary ops.
// out contiguous [ne0, ne1, ne2, ne3]; a is a view (dim-0 contiguous).
// op: 0 gelu_erf, 1 gelu_quick, 2 elu, 3 hardsigmoid, 4 hardswish, 5 step, 6 sgn,
//     7 leaky_relu (slope f0), 8 expm1, 9 floor, 10 ceil, 11 round, 12 trunc, 13 log,
//     14 sin, 15 cos, 16 clamp (f0, f1), 17 xielu (alpha_n f0, alpha_p f1, beta f2, eps f3)
// tensors: 0 a | 1 out
#include "syn_common.h"

// erf, Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
static inline float64 nl_erf(float64 x)
{
    const float64 ax = v_f32_abs_b(x);
    const float64 t = nl_recip(v_f32_mac_b(ax, 0.3275911f, (float64) 1.0f));
    float64 pl = 1.061405429f;
    pl = v_f32_mac_b(pl, t, -1.453152027f);
    pl = v_f32_mac_b(pl, t, 1.421413741f);
    pl = v_f32_mac_b(pl, t, -0.284496736f);
    pl = v_f32_mac_b(pl, t, 0.254829592f);
    const float64 y = 1.0f - pl * t * nl_exp(-ax * ax);
    return v_f32_sel_less_f32_b(x, 0.0f, -y, y);
}

// exp(x) - 1 without cancellation near 0
static inline float64 nl_expm1(float64 x)
{
    const float64 s = x * (1.0f + x * (0.5f + x * (1.0f / 6.0f + x * (1.0f / 24.0f + x * (1.0f / 120.0f)))));
    const float64 e = v_f32_sel_grt_f32_b(x, 88.72283f, as_float64((int64) 0x7F800000), nl_exp(x) - 1.0f);
    return v_f32_sel_less_f32_b(v_f32_abs_b(x), 0.03f, s, e);
}

// x rounded with mode md (exact for |x| >= 2^23)
#define NL_RINT(x, md) v_f32_sel_less_f32_b(v_f32_abs_b(x), 8388608.0f,                            \
    v_convert_i32_to_f32_b(v_convert_f32_to_i32_b((x), (md), 0, 1, 0), SW_RHNE, 0, 1, 0), (x))

// sin (cs = 0) or cos (cs = 1): quadrant reduction by pi/2 (Cody-Waite), minimax polynomials
static inline float64 nl_sincos(float64 x, int cs)
{
    const float64 kq = NL_RINT(x * 0.63661977236758134f, SW_RHNE);
    float64 r = v_f32_mac_b(kq, -1.5703125f, x);
    r = v_f32_mac_b(kq, -4.8375129699707031e-4f, r);
    r = v_f32_mac_b(kq, -7.5497899548918821e-8f, r);
    const float64 r2 = r * r;
    float64 sp = -1.9515295891e-4f;
    sp = v_f32_mac_b(sp, r2, 8.3321608736e-3f);
    sp = v_f32_mac_b(sp, r2, -1.6666654611e-1f);
    const float64 sn = v_f32_mac_b(sp * r2, r, r);
    float64 cp = 2.443315711809948e-5f;
    cp = v_f32_mac_b(cp, r2, -1.388731625493765e-3f);
    cp = v_f32_mac_b(cp, r2, 4.166664568298827e-2f);
    const float64 cn = v_f32_mac_b(cp * r2, r2, 1.0f - 0.5f * r2);
    // quadrant q = (k + cs) mod 4: 0 sin, 1 cos, 2 -sin, 3 -cos
    const float64 q = kq + (float) cs;
    const float64 q4 = q - 4.0f * NL_RINT(q * 0.25f, SW_RD);
    const float64 odd = q4 - 2.0f * NL_RINT(q4 * 0.5f, SW_RD);
    const float64 v = v_f32_sel_eq_f32_b(odd, 0.0f, sn, cn);
    return v_f32_sel_geq_f32_b(q4, 2.0f, -v, v);
}

void main(tensor a, tensor dep, tensor out,
          int nmem, int op, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03, float f0, float f1, float f2, float f3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nch = (ne0 + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nch;
        int ch = i0 - r * nch;
        int r3 = r / (ne1 * ne2);
        int rr = r - r3 * ne1 * ne2;
        int r2 = rr / ne1;
        int r1 = rr - r2 * ne1;
        for (int it = i0; it < i1; it++) {
            const int e0 = ch << 6;
            const float64 x = ld_f32(a, off0 + r1 * s01 + r2 * s02 + r3 * s03 + e0);
            float64 v;
            if (op == 0) {
                v = 0.5f * x * (1.0f + nl_erf(x * 0.70710678118654752f));
            } else if (op == 1) {
                v = x * nl_sigm_acc(1.702f * x);
            } else if (op == 2) {
                v = v_f32_sel_grt_f32_b(x, 0.0f, x, nl_expm1(x));
            } else if (op == 3 || op == 4) {
                v = v_f32_min_b(v_f32_max_b((x + 3.0f) * (1.0f / 6.0f), 0.0f), 1.0f);
                if (op == 4) {
                    v = v * x;
                }
            } else if (op == 5) {
                v = v_f32_sel_grt_f32_b(x, 0.0f, 1.0f, 0.0f);
            } else if (op == 6) {
                v = v_f32_sel_grt_f32_b(x, 0.0f, 1.0f, v_f32_sel_less_f32_b(x, 0.0f, -1.0f, 0.0f));
            } else if (op == 7) {
                v = v_f32_sel_grt_f32_b(x, 0.0f, x, x * f0);
            } else if (op == 8) {
                v = nl_expm1(x);
            } else if (op == 9) {
                v = NL_RINT(x, SW_RD);
            } else if (op == 10) {
                v = NL_RINT(x, SW_RU);
            } else if (op == 11) {
                // half away from zero
                const float64 ax = v_f32_abs_b(x);
                const float64 rv = NL_RINT(ax + 0.5f, SW_RD);
                const float64 ra = v_f32_sel_less_f32_b(ax, 8388608.0f, rv, ax);
                v = v_f32_sel_less_f32_b(x, 0.0f, -ra, ra);
            } else if (op == 12) {
                v = NL_RINT(x, SW_RZ);
            } else if (op == 13) {
                const float64 l = nl_log(x);
                v = v_f32_sel_eq_f32_b(x, 0.0f, as_float64((int64) 0xFF800000), l);
                v = v_f32_sel_less_f32_b(x, 0.0f, as_float64((int64) 0x7FC00000), v);
                v = v_f32_sel_eq_f32_b(x, as_float64((int64) 0x7F800000), x, v);
            } else if (op == 14) {
                v = nl_sincos(x, 0);
            } else if (op == 15) {
                v = nl_sincos(x, 1);
            } else if (op == 16) {
                v = v_f32_min_b(v_f32_max_b(x, f0), f1);
            } else {
                const float64 neg = (nl_expm1(v_f32_min_b(x, f3)) - x) * f0 + f2 * x;
                v = v_f32_sel_grt_f32_b(x, 0.0f, v_f32_mac_b(x * x, f1, f2 * x), neg);
            }
            ST_F32(out, r * ne0 + e0, v, ne0 - e0);
            if (++ch == nch) {
                ch = 0;
                r++;
                if (++r1 == ne1) {
                    r1 = 0;
                    if (++r2 == ne2) {
                        r2 = 0;
                        r3++;
                    }
                }
            }
        }
    }
}

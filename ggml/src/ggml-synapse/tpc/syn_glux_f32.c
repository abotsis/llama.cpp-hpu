// out = f(a) * g(b), f32: the less common gated activations.
// out contiguous [ne0, ne1, ne2, ne3]; a, b are views (element offset + strides of dims 1..3).
// op: 0 geglu_erf, 1 geglu_quick, 2 swiglu_oai (alpha f0, limit f1), 3 swiglu_clamp (limit f1)
// tensors: 0 a  1 b | 2 out
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

void main(tensor a, tensor b, tensor dep, tensor out,
          int nmem, int op, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03,
          int off1, int s11, int s12, int s13, float f0, float f1)
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
            const float64 y = ld_f32(b, off1 + r1 * s11 + r2 * s12 + r3 * s13 + e0);
            float64 v;
            if (op == 0) {
                v = 0.5f * x * (1.0f + nl_erf(x * 0.70710678118654752f)) * y;
            } else if (op == 1) {
                v = x * nl_sigm_acc(1.702f * x) * y;
            } else {
                const float64 xc = v_f32_min_b(x, f1);
                const float64 yc = v_f32_min_b(v_f32_max_b(y, -f1), f1);
                v = op == 2 ? xc * nl_sigm_acc(f0 * xc) * (yc + 1.0f) : xc * nl_sigm_acc(xc) * yc;
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

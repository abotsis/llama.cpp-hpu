// TIMESTEP_EMBEDDING, f32: src [T] (or [T, 1, 1, 1]), out [T, dim] contiguous.
// half = dim / 2; for j < half: out[j] = cos(t * exp(-log(max_period) * j / half)),
// out[j + half] = sin(same argument); odd dim: out[2*half] = 0. half == 0: no division, only
// the odd trailing element zeroed (dim == 1).
// LUT-free device math (nl_exp / nl_log from syn_common.h, nl_sincos ported from
// syn_unaryx_f32.c); no host table staging.
// tensors: 0 src | 1 out
#include "syn_common.h"

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

void main(tensor src, tensor dep, tensor out,
          int nmem, int dim, int max_period, int T)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int hf = dim / 2;
    const int odd = dim - 2 * hf;
    const float64 lane = LANE_F32();
    if (hf > 0) {
        // freq[j] = exp(-log(max_period) * j / half)
        const float64 lm = nl_log((float64) max_period);
        const float64 k = -lm / (float64) hf;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(T, nmem, m, i0, i1)
            for (int i = i0; i < i1; i++) {
                int5 cs = {i, 0, 0, 0, 0};
                const float64 t = (float64) s_f32_ld_g((__global float *) gen_addr(cs, src));
                const int po = i * dim;
                for (int j = 0; j < hf; j += 64) {
                    const int n_ = hf - j < 64 ? hf - j : 64;
                    const float64 f = nl_exp((lane + (float) j) * k);
                    const float64 a = t * f;
                    ST_F32(out, po + j, nl_sincos(a, 1), n_);
                    ST_F32(out, po + j + hf, nl_sincos(a, 0), n_);
                }
                if (odd) {
                    int5 d = {po + 2 * hf, 0, 0, 0, 0};
                    s_f32_st_g((__global float *) gen_addr(d, out), 0.0f);
                }
            }
        }
    } else {
        // half == 0: only the odd trailing element of each row (dim must be 1)
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(T, nmem, m, i0, i1)
            for (int i = i0; i < i1; i++) {
                if (odd) {
                    int5 d = {i * dim + 2 * hf, 0, 0, 0, 0};
                    s_f32_st_g((__global float *) gen_addr(d, out), 0.0f);
                }
            }
        }
    }
}

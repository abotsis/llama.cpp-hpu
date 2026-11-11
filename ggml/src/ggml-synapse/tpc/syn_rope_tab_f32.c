// rope cos/sin table on the device (the layout the rotation kernels of syn_rope_f32.c consume):
// row t (token i2) =
//   NEOX / MROPE / IMROPE: [cos(0..hd) | sin(0..hd)], hd = n_dims / 2 (row n_dims)
//   VISION:                [cos(0..nd) | sin(0..nd)], nd = n_dims (row 2 * n_dims = ne0; the
//                          sections restart their theta)
//   NORMAL:                [C | S], C[2i] = C[2i + 1] = cos(i), S[2i] = -sin(i), S[2i + 1] = sin(i)
//                          (row 2 * n_dims)
// Bit-identical to the host rope_table reference (theta per pair from the position planes,
// sequential multiplication by theta_scale, YaRN ramp / amplitude, freq-factor division).
// The positions are a device i32 tensor ([ntok] or [4, ntok] for MROPE / IMROPE).
// Host-provided scalars: n_dims, mode, sections, corr dims (computed once from shape / params),
// theta_scale, freq_scale, ext_factor, attn_factor, and the f32 frequency-factor tensor's stride
// (no factor tensor: 0), the positions' element offset poff. out contiguous [ntok, rl] f32, rl = the row length above.
// tensors: 0 pos  1 ffac (optional; a 1-element dummy when absent) | 2 out
#include "syn_common.h"

// x rounded with mode md (exact for |x| >= 2^23)
#define NL_RINT(x, md) v_f32_sel_less_f32_b(v_f32_abs_b(x), 8388608.0f,                            \
    v_convert_i32_to_f32_b(v_convert_f32_to_i32_b((x), (md), 0, 1, 0), SW_RHNE, 0, 1, 0), (x))

// sin (cs = 0) or cos (cs = 1): quadrant reduction by pi/2 (Cody-Waite), minimax polynomials
// (the syn_unaryx_f32.c construction)
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

// i / 2 in [low, high] -> 0, outside -> 1 (the host rope_yarn_ramp called with i0 = 2 i, so the
// argument here is the pair index i)
static inline float64 yarn_ramp(float64 low, float64 high, int64 i)
{
    const float64 hf = v_convert_i32_to_f32_b(i, SW_RHNE, 0, 1, 0);
    const float64 y = (hf - low) / v_f32_max_b(v_f32_sub_b(high, low), 0.001f);
    const float64 c = v_f32_max_b(v_f32_min_b(y, 1.0f), 0.0f);
    return 1.0f - c;
}

void main(tensor pos, tensor ffac, tensor dep, tensor out,
          int nmem, int ntok, int n_dims, int mode, int rl, int poff,
          int sect0, int sect1, int sect2, int sect3,
          int has_ff, int ff_stride,
          float theta_scale, float freq_scale, float ext_factor,
          float attn_factor, float corr_lo, float corr_hi)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int vision = mode == 24;
    const int normal = mode == 0;
    const int mrope = (mode & 8) != 0;
    const int imrope = mode == 40;
    const int np = vision ? n_dims : n_dims / 2;   // rotated pairs
    const int sd = sect0 + sect1 + sect2 + sect3;
    const int sw = sect1 + sect0;
    const int se = sect2 + sw;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(ntok, nmem, m, i0, i1)
        for (int t = i0; t < i1; t++) {
            int5 cp = {poff + t, 0, 0, 0, 0};
            const int64 b_t = s_i32_ld_g((__global int *) gen_addr(cp, pos));
            int64 b_h = 0, b_w = 0, b_e = 0;
            if (mrope) {
                cp[0] = poff + t + ntok;
                b_h = s_i32_ld_g((__global int *) gen_addr(cp, pos));
                cp[0] = poff + t + 2 * ntok;
                b_w = s_i32_ld_g((__global int *) gen_addr(cp, pos));
                cp[0] = poff + t + 3 * ntok;
                b_e = s_i32_ld_g((__global int *) gen_addr(cp, pos));
            }
            // the th values as f32 bit patterns
            float64 th_t = v_convert_i32_to_f32_b(b_t, SW_RHNE, 0, 1, 0);
            float64 th_h = v_convert_i32_to_f32_b(b_h, SW_RHNE, 0, 1, 0);
            float64 th_w = v_convert_i32_to_f32_b(b_w, SW_RHNE, 0, 1, 0);
            float64 th_e = v_convert_i32_to_f32_b(b_e, SW_RHNE, 0, 1, 0);
            for (int i = 0; i < np; i++) {
                const int i0 = 2 * i;
                float ff = 1.0f;
                if (has_ff) {
                    int5 cf = {i * ff_stride, 0, 0, 0, 0};
                    ff = s_f32_ld_g((__global float *) gen_addr(cf, ffac));
                }
                // theta selection (the host reference's sector logic)
                float64 theta = th_t;
                if (mrope) {
                    const int sector = i % sd;
                    if (vision) {
                        // sections restart their theta (indep_sects)
                        if (sector == 0) {
                            th_t = v_convert_i32_to_f32_b(b_t, SW_RHNE, 0, 1, 0);
                        } else if (sector == sect0) {
                            th_h = v_convert_i32_to_f32_b(b_h, SW_RHNE, 0, 1, 0);
                        } else if (sector == sw) {
                            th_w = v_convert_i32_to_f32_b(b_w, SW_RHNE, 0, 1, 0);
                        } else if (sector == se) {
                            th_e = v_convert_i32_to_f32_b(b_e, SW_RHNE, 0, 1, 0);
                        }
                        theta = th_t;
                    }
                    if (imrope) {
                        if (sector % 3 == 1 && sector < 3 * sect1) {
                            theta = th_h;
                        } else if (sector % 3 == 2 && sector < 3 * sect2) {
                            theta = th_w;
                        } else if (sector % 3 == 0 && sector < 3 * sect0) {
                            theta = th_t;
                        } else {
                            theta = th_e;
                        }
                    } else {
                        if (sector >= sect0 && sector < sw) {
                            theta = th_h;
                        } else if (sector >= sw && sector < sw + sect2) {
                            theta = th_w;
                        } else if (sector >= sw + sect2) {
                            theta = th_e;
                        }
                    }
                }
                // theta / ff as an exact f32 division (the CPU's theta / ff; the frequency
                // factors are arbitrary model floats, so a reciprocal multiply would drift)
                float64 th = theta / (float64) ff;
                float64 theta_interp = v_f32_mul_b(th, freq_scale);
                float64 mix = theta_interp;
                float64 ms = attn_factor;
                if (ext_factor != 0.0f) {
                    const float64 rm = v_f32_mul_b(yarn_ramp(corr_lo, corr_hi, i), ext_factor);
                    mix = v_f32_mac_b(th, rm, theta_interp * (1.0f - rm));
                    ms = v_f32_mul_b(ms, v_f32_mac_b(nl_log(1.0f / freq_scale), 0.1f, (float64) 1.0f));
                }
                const float64 cs = v_f32_mul_b(nl_sincos(mix, 1), ms);
                const float64 sn = v_f32_mul_b(nl_sincos(mix, 0), ms);
                // the consumed layout
                if (normal) {
                    ST_F32(out, t * rl + i0, cs, 2);
                    ST_F32(out, t * rl + n_dims + i0, -sn, 1);
                    ST_F32(out, t * rl + n_dims + i0 + 1, sn, 1);
                } else {
                    ST_F32(out, t * rl + i, cs, 1);
                    ST_F32(out, t * rl + np + i, sn, 1);
                }
                th_t = v_f32_mul_b(th_t, theta_scale);
                th_h = v_f32_mul_b(th_h, theta_scale);
                th_w = v_f32_mul_b(th_w, theta_scale);
                th_e = v_f32_mul_b(th_e, theta_scale);
            }
        }
    }
}

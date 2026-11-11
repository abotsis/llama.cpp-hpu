// PAD: out [ne0, ne1, ne2, ne3] contiguous, out[i] = a[i - lp] where 0 <= i - lp <
// (sn0, sn1, sn2, sn3), else 0. a: view (element offset + strides of dims 1..3, dim 0
// contiguous). (The backend sampler's padding row of the logits.)
// -DCIRC: circular padding (op_params[8] == 1 at the host): the source coordinate is the positive
// modulo of (i - lp) in each dimension, wrapping repeatedly when the pad exceeds the width. A zero-
// sized source dimension stays out of range (no wrap), per the zero-fill contract.
// tensors: 0 a  1 dep | 2 out
#include "syn_common.h"

#if defined(CIRC)
// positive modulo of i in [0, n), n > 0 (wraps repeatedly for out-of-range input)
static inline int pmod0(int i, int n)
{
    const int r = i % n;
    return r < 0 ? r + n : r;
}

static inline float64 i2f(int64 v)
{
    return v_convert_i32_to_f32_b(v, SW_RHNE, 0, 1, 0);
}
#endif

void main(tensor a, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int lp0, int lp1, int lp2, int lp3, int sn0, int sn1, int sn2, int sn3,
          int offa, int sa1, int sa2, int sa3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const int nrows = ne1 * ne2 * ne3;
    const int nch = (ne0 + 63) >> 6;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nch;
            const int e0 = (it - r * nch) << 6;
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            int j1 = r1 - lp1, j2 = r2 - lp2, j3 = r3 - lp3;
            float64 x = 0.0f;
#if defined(CIRC)
            // each dim wraps independently and repeatedly (positive modulo of output - left pad);
            // a zero-sized source dim stays out of range (zero fill, no modulo)
            if (sn1 > 0) {
                j1 = pmod0(j1, sn1);
            }
            if (sn2 > 0) {
                j2 = pmod0(j2, sn2);
            }
            if (sn3 > 0) {
                j3 = pmod0(j3, sn3);
            }
            const int base = offa + j1 * sa1 + j2 * sa2 + j3 * sa3;
            if (sn0 > 0 && j1 < sn1 && j2 < sn2 && j3 < sn3) {
                const int i0 = e0 - lp0;                    // unwrapped source coordinate
                if (i0 >= 0 && i0 + 64 <= sn0) {
                    x = ld_f32(a, base + i0);
                } else {
                    for (int e = 0; e < 64 && e0 + e < ne0; e++) {
                        int5 ca = {base + pmod0(i0 + e, sn0), 0, 0, 0, 0};
                        const float64 v = s_f32_ld_g((__global float *) gen_addr(ca, a));
                        x = v_f32_sel_eq_f32_b(LANE_F32(), i2f(e), v, x);
                    }
                }
            }
            ST_F32(out, r * ne0 + e0, x, ne0 - e0);
            continue;
#endif
            if (j1 >= 0 && j1 < sn1 && j2 >= 0 && j2 < sn2 && j3 >= 0 && j3 < sn3 && e0 + 64 > lp0 && e0 < lp0 + sn0) {
                x = ld_f32(a, offa + j1 * sa1 + j2 * sa2 + j3 * sa3 + e0 - lp0);
                x = v_f32_sel_less_f32_b(lane, (float) (lp0 - e0), (float64) 0.0f, x);
                x = v_f32_sel_less_f32_b(lane, (float) (lp0 + sn0 - e0), x, (float64) 0.0f);
            }
            ST_F32(out, r * ne0 + e0, x, ne0 - e0);
        }
    }
}

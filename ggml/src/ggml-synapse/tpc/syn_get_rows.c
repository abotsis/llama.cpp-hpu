// out[:, i1, i2, i3] = f32(a[:, idx[i1, i2, i3], i2, i3])
// a: view (SRC type, dim-0 contiguous); idx: i32 view (is64: i64 read as i32 pairs).
// out contiguous f32 [ne0, n1, n2, n3]. Built per SRC type (-DSRC_F32/F16/BF16; -DSRC_I32: a raw
// 32-bit gather into an i32 out, the backend top-k sampler's candidate ids).
// tensors: 0 a  1 idx | 2 out
#include "syn_common.h"

void main(tensor a, tensor idx, tensor dep, tensor out,
          int nmem, int ne0, int n1, int n2, int n3,
          int offa, int sa1, int sa2, int sa3,
          int offi, int si0, int si1, int si2, int is64)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nrows = n1 * n2 * n3;
    const int nch = (ne0 + 127) >> 7;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        int rcur = -1;
        int base = 0;
        for (int it = i0; it < i1; it++) {
            const int r = it / nch;
            const int ch = it - r * nch;
            if (r != rcur) {
                rcur = r;
                const int r3 = r / (n1 * n2);
                const int rr = r - r3 * n1 * n2;
                const int r2 = rr / n1;
                const int r1 = rr - r2 * n1;
                const int pi = offi + r1 * si0 + r2 * si1 + r3 * si2;
                int5 ci = {is64 ? 2 * pi : pi, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, idx));
                base = offa + row * sa1 + r2 * sa2 + r3 * sa3;
            }
            const int e0 = ch << 7;
            int5 c = {base + e0, 0, 0, 0, 0};
#if defined(SRC_F16)
            const float128 x = f16_to_f32(v_f16_ld_tnsr_b(c, a));
            const float64 lo = x.v1, hi = x.v2;
#elif defined(SRC_BF16)
            const float128 x = bf16_to_f32(v_bf16_ld_tnsr_b(c, a));
            const float64 lo = x.v1, hi = x.v2;
#elif defined(SRC_I32)
            const int64 lo = v_i32_ld_tnsr_b(c, a);
            c[0] += 64;
            const int64 hi = v_i32_ld_tnsr_b(c, a);
#else
            const float64 lo = v_f32_ld_tnsr_b(c, a);
            c[0] += 64;
            const float64 hi = v_f32_ld_tnsr_b(c, a);
#endif
            const int n = ne0 - e0;
            const int po = r * ne0 + e0;
#if defined(SRC_I32)
            int5 d0 = {po, 0, 0, 0, 0};
            if (n >= 64) v_i32_st_tnsr(d0, out, lo); else v_i32_st_tnsr_partial(d0, out, lo, n - 1, 0);
            if (n > 64) {
                int5 d1 = {po + 64, 0, 0, 0, 0};
                if (n >= 128) v_i32_st_tnsr(d1, out, hi); else v_i32_st_tnsr_partial(d1, out, hi, n - 65, 0);
            }
#else
            ST_F32(out, po, lo, n);
            if (n > 64) {
                ST_F32(out, po + 64, hi, n - 64);
            }
#endif
        }
    }
}

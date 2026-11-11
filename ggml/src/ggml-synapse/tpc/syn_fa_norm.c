// Strided f32 copy scaled per row: out (contiguous [ne0, ne1, ne2, ne3]) = a (element offset + strides of dims
// 1..3, dim 0 contiguous) * l[r1 * l1 + r2 * l2 + r3 * l3], l = 1 / row sum.
// tensors: 0 a  1 l | 2 out
#include "syn_common.h"

void main(tensor a, tensor l, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03, int l1, int l2, int l3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nch = (ne0 + 127) >> 7;
    const int nitems = ne1 * ne2 * ne3 * nch;
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
            const int e0 = ch << 7;
            const int ps = off0 + r1 * s01 + r2 * s02 + r3 * s03 + e0;
            int5 c = {ps, 0, 0, 0, 0};
            const float64 lo = v_f32_ld_tnsr_b(c, a);
            c[0] = ps + 64;
            const float64 hi = v_f32_ld_tnsr_b(c, a);
            int5 cl = {r1 * l1 + r2 * l2 + r3 * l3, 0, 0, 0, 0};
            const float s = s_f32_ld_g(gen_addr(cl, l));
            const int n = ne0 - e0;
            const int po = r * ne0 + e0;
            ST_F32(out, po, lo * s, n);
            if (n > 64) {
                ST_F32(out, po + 64, hi * s, n - 64);
            }
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

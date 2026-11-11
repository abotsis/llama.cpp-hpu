// out = concat(a, b) along dim, f32. out contiguous [ne0, ne1, ne2, ne3]; na = a's extent
// along dim. a, b: views (element offset + strides of dims 0..3; the dim-0 stride must be 1
// unless that view has a single element along dim 0).
// tensors: 0 a  1 b | 2 out
#include "syn_common.h"

void main(tensor a, tensor b, tensor dep, tensor out,
          int nmem, int dim, int ne0, int ne1, int ne2, int ne3, int na,
          int offa, int sa1, int sa2, int sa3,
          int offb, int sb1, int sb2, int sb3, int sa0, int sb0)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lane = LANE_F32();
    if (sa0 != 1 || sb0 != 1) {
        // a source with a strided dim 0 (a transposed view): element-wise, scalar loads
        scalar_cache_reset();
        const int n = ne0 * ne1 * ne2 * ne3;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(n, nmem, m, i0, i1)
            for (int i = i0; i < i1; i++) {
                int c0 = i % ne0;
                int r = i / ne0;
                int c1 = r % ne1;
                r /= ne1;
                int c2 = r % ne2;
                int c3 = r / ne2;
                int useb = 0;
                if (dim == 0) { useb = c0 >= na; if (useb) c0 -= na; }
                else if (dim == 1) { useb = c1 >= na; if (useb) c1 -= na; }
                else if (dim == 2) { useb = c2 >= na; if (useb) c2 -= na; }
                else { useb = c3 >= na; if (useb) c3 -= na; }
                float v;
                if (useb) {
                    int5 c = {offb + c0 * sb0 + c1 * sb1 + c2 * sb2 + c3 * sb3, 0, 0, 0, 0};
                    v = s_f32_ld_g((__global float *) gen_addr(c, b));
                } else {
                    int5 c = {offa + c0 * sa0 + c1 * sa1 + c2 * sa2 + c3 * sa3, 0, 0, 0, 0};
                    v = s_f32_ld_g((__global float *) gen_addr(c, a));
                }
                int5 d = {i, 0, 0, 0, 0};
                s_f32_st_g((__global float *) gen_addr(d, out), v);
            }
        }
        return;
    }
    const int nch = (ne0 + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    const int nitems = nrows * nch;
    if (dim == 0) {
        // each lane from a (column < na) or b
        const int nq = (ne0 + 255) >> 8;
        const int nit = nrows * nq;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(nit, nmem, m, i0, i1)
            if (i0 >= i1) {
                continue;
            }
            int r = i0 / nq;
            int e = (i0 - r * nq) << 8;
            int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            int r2 = rr / ne1;
            int r1 = rr - r2 * ne1;
            for (int it = i0; it < i1; it++) {
                int5 ca = {offa + r1 * sa1 + r2 * sa2 + r3 * sa3 + e, 0, 0, 0, 0};
                int5 cb = {offb + r1 * sb1 + r2 * sb2 + r3 * sb3 + e - na, 0, 0, 0, 0};
                int5 co = {r * ne0 + e, 0, 0, 0, 0};
                const int left = ne0 - e;
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float64 x = v_f32_ld_tnsr_b(ca, a), y = v_f32_ld_tnsr_b(cb, b);
                    const float64 v = v_f32_sel_less_f32_b(lane, (float) (na - e - 64 * u), x, y);
                    if (left >= 64 * u + 64) {
                        v_f32_st_tnsr(co, out, v);
                    } else if (left > 64 * u) {
                        v_f32_st_tnsr_partial(co, out, v, left - 64 * u - 1, 0);
                    }
                    ca[0] += 64;
                    cb[0] += 64;
                    co[0] += 64;
                }
                e += 256;
                if (e >= ne0) {
                    e = 0;
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
        return;
    }
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
            float64 v;
            if (dim == 0) {
                const float64 x = ld_f32(a, offa + r1 * sa1 + r2 * sa2 + r3 * sa3 + e0);
                const float64 y = ld_f32(b, offb + r1 * sb1 + r2 * sb2 + r3 * sb3 + e0 - na);
                v = v_f32_sel_less_f32_b(lane, (float) (na - e0), x, y);
            } else {
                int c1 = r1, c2 = r2, c3 = r3;
                int useb;
                if (dim == 1) {
                    useb = c1 >= na;
                    if (useb) c1 -= na;
                } else if (dim == 2) {
                    useb = c2 >= na;
                    if (useb) c2 -= na;
                } else {
                    useb = c3 >= na;
                    if (useb) c3 -= na;
                }
                if (useb) {
                    v = ld_f32(b, offb + c1 * sb1 + c2 * sb2 + c3 * sb3 + e0);
                } else {
                    v = ld_f32(a, offa + c1 * sa1 + c2 * sa2 + c3 * sa3 + e0);
                }
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

// General POOL_1D / POOL_2D (-DPOOL2D): MAX and AVG over windows with independent kernel size,
// stride and padding (CPU ggml_compute_forward_pool_1d / pool_2d). One output element per item.
//  1D AVG divides by the number of in-range taps (0 for an empty window);
//  2D AVG divides by the full k0 * k1 area including padding.
// MAX starts at -FLT_MAX. src: f32 view (dim 0 contiguous); out contiguous f32 [ne0, ne1, ne2, ne3].
// 1D: rows (dims 1..3) map 1:1 to source rows. 2D: dims 2..3 map 1:1, dim 1 is pooled.
// tensors: 0 src | 1 out
#include "syn_common.h"

void main(tensor a, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int sn0, int sn1, int sn2, int sn3,
          int offa, int sa1, int sa2, int sa3,
          int k0, int s0, int p0, int k1, int s1, int p1, int avg)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    (void) sn2; (void) sn3;
    const int nitems = ne0 * ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int r = it;
            const int ox = r % ne0; r /= ne0;
            const int o1 = r % ne1; r /= ne1;
            const int o2 = r % ne2;
            const int o3 = r / ne2;
#if defined(POOL2D)
            const int by = o1 * s1 - p1;
            const int pb = offa + o2 * sa2 + o3 * sa3;
            const int kh = k1;
#else
            const int by = 0;
            const int pb = offa + o1 * sa1 + o2 * sa2 + o3 * sa3;
            const int kh = 1;
            (void) s1; (void) p1;
#endif
            const int bx = ox * s0 - p0;
            float acc = avg ? 0.0f : -3.40282347e+38f;
            int cnt = 0;
            for (int ky = 0; ky < kh; ky++) {
                const int y = by + ky;
                if (y < 0 || y >= sn1) {
                    continue;
                }
                for (int kx = 0; kx < k0; kx++) {
                    const int x = bx + kx;
                    if (x < 0 || x >= sn0) {
                        continue;
                    }
                    int5 c = {pb + y * sa1 + x, 0, 0, 0, 0};
                    const float v = s_f32_ld_g((__global float *) gen_addr(c, a));
                    if (avg) {
                        acc = acc + v;
                    } else {
                        acc = v > acc ? v : acc;
                    }
                    cnt++;
                }
            }
            if (avg) {
#if defined(POOL2D)
                acc = acc / (float) (k0 * k1);
#else
                acc = cnt > 0 ? acc / (float) cnt : 0.0f;
#endif
            }
            int5 d = {it, 0, 0, 0, 0};
            s_f32_st_g((__global float *) gen_addr(d, out), acc);
        }
    }
}

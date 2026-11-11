// PAD_REFLECT_1D and ROLL, f32, source view (element offset + strides of dims 0..3).
// REFLECT (-DREFLECT): pads p0 (left) p1 (right); each pad < width (constructor). The edge
//   element is NOT repeated (even reflection, CPU ggml_compute_forward_pad_reflect_1d): per
//   output coordinate i0 with c = i0 - p0 the source coordinate is
//     c < 0        -> -c            (p0 - i0)
//     0 <= c < w   -> c             (i0 - p0)
//     c >= w       -> 2*w - 2 - c   (reflected right of the last element)
//   out contiguous [w + p0 + p1, ne1, ne2, ne3]; dims 1..3 of the source map 1:1 (dim0 of the
//   source is contiguous).
// ROLL (-DROLL): out[i] = src[(i - s_d) mod ne_d] per dim d (positive modulo, negative shifts
//   accepted; CPU ggml_compute_forward_roll wraps each dim); shifts s0..s3 params; the source
//   may be a permuted view (strides of dims 0..3 explicit; s00 may differ from 1). Each output
//   element written exactly once.
// tensors: 0 src | 1 out
#include "syn_common.h"

#if defined(ROLL)
void main(tensor src, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int off0, int s00, int s01, int s02, int s03,
          int sh0, int sh1, int sh2, int sh3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
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
            // source coordinate (c - shift) wrapped into [0, ne); |shift| < ne (the lowering checks)
            int j0 = c0 - sh0;
            j0 = j0 < 0 ? j0 + ne0 : (j0 >= ne0 ? j0 - ne0 : j0);
            int j1 = c1 - sh1;
            j1 = j1 < 0 ? j1 + ne1 : (j1 >= ne1 ? j1 - ne1 : j1);
            int j2 = c2 - sh2;
            j2 = j2 < 0 ? j2 + ne2 : (j2 >= ne2 ? j2 - ne2 : j2);
            int j3 = c3 - sh3;
            j3 = j3 < 0 ? j3 + ne3 : (j3 >= ne3 ? j3 - ne3 : j3);
            int5 c = {off0 + j0 * s00 + j1 * s01 + j2 * s02 + j3 * s03, 0, 0, 0, 0};
            int5 d = {i, 0, 0, 0, 0};
            s_f32_st_g((__global float *) gen_addr(d, out),
                       s_f32_ld_g((__global float *) gen_addr(c, src)));
        }
    }
}
#else // REFLECT
void main(tensor src, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03,
          int p0, int p1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int w = ne0 - p0 - p1; // source width along dim0 (each pad < w)
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int psrc = off0 + r1 * s01 + r2 * s02 + r3 * s03;
            const int po = r * ne0;
            for (int i0 = 0; i0 < ne0; i0++) {
                const int c = i0 - p0;
                int j0;
                if (c < 0) {
                    j0 = -c;
                } else if (c >= w) {
                    j0 = 2 * w - 2 - c;
                } else {
                    j0 = c;
                }
                int5 cs = {psrc + j0, 0, 0, 0, 0};
                int5 d = {po + i0, 0, 0, 0, 0};
                s_f32_st_g((__global float *) gen_addr(d, out),
                           s_f32_ld_g((__global float *) gen_addr(cs, src)));
            }
        }
    }
}
#endif

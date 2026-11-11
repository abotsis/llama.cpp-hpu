// Side-effect writes into an existing tensor (a persistent input or an internal root):
// the destination is INPUT 0 and is stored through; output 0 is a small sink that keeps the
// node alive. The graph orders these nodes with control dependencies.
//  -DSET_ROWS: dst[:, idx[i, i2 % ni1, i3 % ni2], i2, i3] = src[:, i, i2, i3]
//  otherwise : dst view [ne0, ne1, ne2, ne3] = src view (strided copy, dim-0 contiguous)
// Built per destination type (-DDST_F32/F16/BF16); the source is f32 (-DSRC_SAME: same type
// as the destination, copied without conversion for 16-bit types). -DDST_I32: an i32 source (a raw copy).
// tensors: 0 dst  1 src  [2 idx]  dep | sink
#include "syn_common.h"

#if defined(SET_ROWS)
void main(tensor dst, tensor src, tensor idx, tensor dep, tensor sink,
          int nmem, int ne0, int n, int ne2, int ne3,
          int offd, int sd1, int sd2, int sd3,
          int offs, int ss1, int ss2, int ss3,
          int offi, int si0, int si1, int si2, int ni1, int ni2, int is64)
#else
void main(tensor dst, tensor src, tensor dep, tensor sink,
          int nmem, int ne0, int n, int ne2, int ne3,
          int offd, int sd1, int sd2, int sd3,
          int offs, int ss1, int ss2, int ss3)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
#if defined(SET_ROWS)
    scalar_cache_reset();
#endif
    const int nrows = n * ne2 * ne3;
    const int nch = (ne0 + 127) >> 7;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nch;
        int ch = i0 - r * nch;
        int r3 = r / (n * ne2);
        int r2 = (r - r3 * n * ne2) / n;
        int r1 = r - r3 * n * ne2 - r2 * n;
        int pd = 0, ps = 0;
        bool newrow = true;
        for (int it = i0; it < i1; it++) {
            if (newrow) {
                newrow = false;
                ps = offs + r1 * ss1 + r2 * ss2 + r3 * ss3;
#if defined(SET_ROWS)
                const int pi = offi + r1 * si0 + (r2 % ni1) * si1 + (r3 % ni2) * si2;
                int5 ci = {is64 ? 2 * pi : pi, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, idx));
                pd = offd + row * sd1 + r2 * sd2 + r3 * sd3;
#else
                pd = offd + r1 * sd1 + r2 * sd2 + r3 * sd3;
#endif
            }
            const int e0 = ch << 7;
            const int nn = ne0 - e0;
            if (++ch == nch) {
                ch = 0;
                newrow = true;
                if (++r1 == n) {
                    r1 = 0;
                    if (++r2 == ne2) {
                        r2 = 0;
                        r3++;
                    }
                }
            }
            int5 cs = {ps + e0, 0, 0, 0, 0};
            int5 cd = {pd + e0, 0, 0, 0, 0};
#if defined(SRC_SAME) && defined(DST_F16)
            const half128 y = v_f16_ld_tnsr_b(cs, src);
            if (nn >= 128) v_f16_st_tnsr(cd, dst, y); else v_f16_st_tnsr_partial(cd, dst, y, nn - 1, 0);
#elif defined(SRC_SAME) && defined(DST_BF16)
            const bfloat128 y = v_bf16_ld_tnsr_b(cs, src);
            if (nn >= 128) v_bf16_st_tnsr(cd, dst, y); else v_bf16_st_tnsr_partial(cd, dst, y, nn - 1, 0);
#elif defined(DST_I32)
            const int64 lo = v_i32_ld_tnsr_b(cs, src);
            cs[0] += 64;
            const int64 hi = v_i32_ld_tnsr_b(cs, src);
            if (nn >= 64) v_i32_st_tnsr(cd, dst, lo); else v_i32_st_tnsr_partial(cd, dst, lo, nn - 1, 0);
            if (nn > 64) {
                cd[0] += 64;
                if (nn >= 128) v_i32_st_tnsr(cd, dst, hi); else v_i32_st_tnsr_partial(cd, dst, hi, nn - 65, 0);
            }
#else
            const float64 lo = v_f32_ld_tnsr_b(cs, src);
            cs[0] += 64;
            const float64 hi = v_f32_ld_tnsr_b(cs, src);
#if defined(DST_F16)
            const half128 y = cvt_f16(lo, hi);
            if (nn >= 128) v_f16_st_tnsr(cd, dst, y); else v_f16_st_tnsr_partial(cd, dst, y, nn - 1, 0);
#elif defined(DST_BF16)
            const bfloat128 y = cvt_bf16(lo, hi);
            if (nn >= 128) v_bf16_st_tnsr(cd, dst, y); else v_bf16_st_tnsr_partial(cd, dst, y, nn - 1, 0);
#else
            ST_F32(dst, pd + e0, lo, nn);
            if (nn > 64) {
                ST_F32(dst, pd + e0 + 64, hi, nn - 64);
            }
#endif
#endif
        }
    }
    if (ioff[0] == 0) {
        int5 z = {0, 0, 0, 0, 0};
        v_f32_st_tnsr(z, sink, 0.0f);
    }
}

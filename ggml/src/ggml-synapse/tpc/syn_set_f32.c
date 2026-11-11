// SET / ACC: dst[:, i1, i2, i3] (+)= src[:, i1, i2, i3] rows of src1 written at dst's offsets.
// ggml_set / ggml_acc: dst (src0) is a contiguous tensor of the result shape; the op params
// {nb1, nb2, nb3, offset, inplace} (bytes) place src1's logical rows in it. Out-of-place: the
// lowering copies src0 to dst first and a DEPENDENT scatter node writes src1's rows (this kernel).
// In-place (inplace = 1): the base copy is skipped, only the scatter runs (same kernel).
// ACC (-DACC): the scatter adds src1's row to dst instead of overwriting (f32 only).
// src1 rows are non-overlapping (the host rejects overlapping destination rows before this runs).
// src1 is a view (dim-0 contiguous element offset
// + element strides of dims 1..3); dst addressed by the byte strides nb1/nb2/nb3 and byte offset
// (offset is in bytes, as in the ggml constructor; all must be multiples of the element size).
// -DSRC_I32: src1 is i32, dst i32 (SET of i32 only; ACC is not built for i32).
// tensors: 0 src0  1 src1 | 2 sink (unused; the stores go through src0)
// params: nmem, ne10, ne11, ne12, ne13 (src1 shape = the scatter extent),
//         off1 (src1 element offset; src1 is a view of its base),
//         sb11, sb12, sb13 (src1 element strides of dims 1..3; dim 0 contiguous),
//         nb1, nb2, nb3, offset (dst element strides / offset; the host pre-divides the op params' bytes),
//         inplace (1: skip nothing here - the scatter is the only mutation)  - 15 dwords
#include "syn_common.h"

void main(tensor src0, tensor src1, tensor dep, tensor dst,
          int nmem, int ne10, int ne11, int ne12, int ne13,
          int off1, int sb11, int sb12, int sb13, int nb1, int nb2, int nb3, int offset, int inplace)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nrows = ne11 * ne12 * ne13;
    const int nch = (ne10 + 63) >> 6;
    const int nitems = nrows * nch;
    const int esz = 4;   // f32 / i32
    (void) inplace;      // ABI parity with the op params; the base copy is a separate node
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nch;
        int ch = i0 - r * nch;
        int r3 = r / (ne11 * ne12);
        int rr = r - r3 * ne11 * ne12;
        int r2 = rr / ne11;
        int r1 = rr - r2 * ne11;
        for (int it = i0; it < i1; it++) {
            const int e0 = ch << 6;
            const int n = ne10 - e0 < 64 ? ne10 - e0 : 64;
            // src1 row element offset (dim 0 contiguous)
            int5 cs = {off1 + r1 * sb11 + r2 * sb12 + r3 * sb13 + e0, 0, 0, 0, 0};
            // dst element offset of the same row start (host pre-divided the byte params)
            int5 cd = {offset + r1 * nb1 + r2 * nb2 + r3 * nb3 + e0, 0, 0, 0, 0};
#if defined(SRC_I32)
            const int64 v = v_i32_ld_tnsr_b(cs, src1);
#else
            const float64 v = v_f32_ld_tnsr_b(cs, src1);
#endif
#if defined(ACC)
            // add to the destination (f32 only)
            const float64 dv = v_f32_ld_tnsr_b(cd, src0);
            v_f32_st_tnsr_partial(cd, src0, dv + v, n - 1, 0);
#elif defined(SRC_I32)
            if (n >= 64) {
                v_i32_st_tnsr(cd, src0, v);
            } else {
                v_i32_st_tnsr_partial(cd, src0, v, n - 1, 0);
            }
#else
            if (n >= 64) {
                v_f32_st_tnsr(cd, src0, v);
            } else {
                v_f32_st_tnsr_partial(cd, src0, v, n - 1, 0);
            }
#endif
            if (++ch == nch) {
                ch = 0;
                if (++r1 == ne11) {
                    r1 = 0;
                    if (++r2 == ne12) {
                        r2 = 0;
                        r3++;
                    }
                }
            }
        }
    }
}

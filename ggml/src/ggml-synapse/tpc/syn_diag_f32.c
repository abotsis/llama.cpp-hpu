// DIAG / TRI / DIAG_MASK_INF, f32 (one source, three registrations via -DDIAG/-DTRI/-DMASK).
// DIAG: out {N, N, B2, B3} = diag of src {N, 1, B2, B3}: out[i, j] = (j == i) ? src[i] : 0,
//   per (i2, i3) matrix. Every output cell written once.
// TRI: out {N, N, B2, B3}: element (i, j) of each matrix (rows restart per matrix: i2/i3 fix the
//   matrix, the predicate uses the row within it) is src[i, j] where the ggml_tri_type predicate
//   on (column j, row i) holds, else 0.
//   ttype: 0 UPPER_DIAG j >= i, 1 UPPER j > i, 2 LOWER_DIAG j <= i, 3 LOWER j < i
//   (the exact ggml/src/ggml-cpu/ops.cpp bipred table).
// MASK (DIAG_MASK_INF): out[i, j] = j > n_past + i ? -INFINITY : src[i, j], per (i2, i3) matrix;
//   n_past from op param 0 (>= 0). Every cell written once.
// tensors: 0 src | 1 out. src a view (element offset + element strides of dims 1..3, dim 0
//   contiguous); out contiguous [N, N, B2, B3].
// params: nmem, N, B2, B3, off, s1, s2, s3, ttype (TRI) / n_past (MASK; unused for DIAG) - 9 dwords
#include "syn_common.h"

void main(tensor src, tensor dep, tensor out,
          int nmem, int N, int B2, int B3, int off, int s1, int s2, int s3, int ttype)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 ninf = as_float64((int64) 0xFF800000);
    const int nb = (N + 63) >> 6;
    const int nitems = B2 * B3 * nb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nb;
        int blk = i0 - r * nb;
        int r3 = r / B2;
        int r2 = r - r3 * B2;
        for (int it = i0; it < i1; it++) {
            const int i0r = blk << 6;
            const int i1r = i0r + 64 < N ? i0r + 64 : N;
            const int psrc = off + r2 * s2 + r3 * s3;
            // out contiguous [N, N, B2, B3]: matrix base (r2 * B3 + r3) * N * N, row i at i * N
            const int pout = (r2 + r3 * B2) * N * N;
#if defined(DIAG)
            for (int j = 0; j < N; j++) {
                // lanes = rows i of this row block; the diagonal value is src[j]
                const float64 d = BCAST0(ld_f32(src, psrc + j), IDX);
                const float64 v = v_f32_sel_eq_f32_b(lane + (float) i0r, (float) j, d, 0.0f);
                ST_F32(out, pout + i0r + j * N, v, i1r - i0r);
            }
#elif defined(TRI)
            for (int j = 0; j < N; j++) {
                // lanes = rows i: src[i, j] is contiguous along i (src dim 0 stride 1)
                const float64 sv = ld_f32(src, psrc + j * s1 + i0r);
                float64 v;
                // the CPU bipred is (i0, i01) = (lane, j): keep when the predicate holds
                const float64 row = lane + (float) i0r;
                if (ttype == 0) {          // UPPER_DIAG: i0 >= i01
                    v = v_f32_sel_geq_f32_b(row, (float) j, sv, 0.0f);
                } else if (ttype == 1) {   // UPPER: i0 > i01
                    v = v_f32_sel_grt_f32_b(row, (float) j, sv, 0.0f);
                } else if (ttype == 2) {   // LOWER_DIAG: i0 <= i01
                    v = v_f32_sel_leq_f32_b(row, (float) j, sv, 0.0f);
                } else {                   // LOWER: i0 < i01
                    v = v_f32_sel_less_f32_b(row, (float) j, sv, 0.0f);
                }
                ST_F32(out, pout + i0r + j * N, v, i1r - i0r);
            }
#else   // MASK
            const int np = ttype;   // op param 0 arrives in the ttype slot for MASK
            for (int j = 0; j < N; j++) {
                // the CPU masks dim0 index i where i > n_past + j (j = dim1); lanes are dim0
                const float64 sv = ld_f32(src, psrc + j * s1 + i0r);
                const float64 masked = v_f32_sel_leq_f32_b(lane + (float) i0r, (float) (np + j), sv, ninf);
                ST_F32(out, pout + i0r + j * N, masked, i1r - i0r);
            }
#endif
            if (++blk == nb) {
                blk = 0;
                if (++r2 == B2) {
                    r2 = 0;
                    r3++;
                }
            }
        }
    }
}

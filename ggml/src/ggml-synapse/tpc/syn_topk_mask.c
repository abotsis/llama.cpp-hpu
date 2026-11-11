// TOP_K_MASK without scores, blocks or extra cells (DSA): out[:, i1, i2, i3] = -inf except out[p] = mask[p] for
// the row's top-k cells p = top_k[j, i1, i2, i3], 0 <= p < n_kv (others skipped). The mask row is
// (i1 % nm1, i2 % nm2, i3 % nm3). The row's -inf fill is stored before its cells (stores of one TPC into the
// same row, issued in program order).
// -DF16 / -DF32: the mask / out type. out contiguous [n_kv, ne1, ne2, ne3].
// tensors: 0 mask  1 top_k | 2 out
#include "syn_common.h"

#if defined(F16)
#define VT half128
#define VLD(c, t) v_f16_ld_tnsr_b(c, t)
#define VST(c, t, v) v_f16_st_tnsr(c, t, v)
#define VSTP(c, t, v, sz, pred) v_f16_st_tnsr_partial(c, t, v, sz, 0, 0, pred, 0)
#define VW 128
#else
#define VT float64
#define VLD(c, t) v_f32_ld_tnsr_b(c, t)
#define VST(c, t, v) v_f32_st_tnsr(c, t, v)
#define VSTP(c, t, v, sz, pred) v_f32_st_tnsr_partial(c, t, v, sz, 0, 0, pred, 0)
#define VW 64
#endif

static inline int ldk(tensor t, int i)
{
    int5 c = {i, 0, 0, 0, 0};
    return s_i32_ld_g((__global int *) gen_addr(c, t));
}

// cell p of the row: mask[pm + p] -> out[po + p] when 0 <= p < n_kv
#define CELL(p, v)                                       \
    {                                                    \
        int5 co_ = {po + (p), 0, 0, 0, 0};               \
        VSTP(co_, out, v, 0, (p) >= 0 && (p) < n_kv);    \
    }

#if defined(GEN)
// -DGEN: the general form (ggml_top_k_mask): picks p = top_k[j, row] with 0 <= p < n_pick; a pick is live when its
// score[p, row] > -inf (has_score) and keeps its cells blocks[0 .. n_blk - 1, p] (has_blocks) or cell p; then the
// row's extra cells extra[0 .. n_extra - 1, row] (has_extra). top_k / score / extra rows are flat (one row stride).
// tensors: 0 mask  1 top_k  2 score  3 blocks  4 extra | 5 out (absent inputs: any tensor, flags off)
static inline float ldf(tensor t, int i)
{
    int5 c = {i, 0, 0, 0, 0};
    return s_f32_ld_g((__global float *) gen_addr(c, t));
}

void main(tensor mask, tensor tk, tensor score, tensor blocks, tensor extra, tensor dep, tensor out, int nmem, int n_kv,
          int n_top, int ne1, int ne2, int ne3, int offm, int sm1, int sm2, int sm3, int nm1, int nm2, int nm3, int offk,
          int sk, int offs, int ss, int n_pick, int offb, int sb, int n_blk, int offe, int se, int n_extra, int flags)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
#if defined(F16)
    const VT ninf = as_half128((ushort128) 0xFC00);
#else
    const VT ninf = as_float64((int64) 0xFF800000);
#endif
    const int has_score = flags & 1, has_blocks = (flags >> 1) & 1, has_extra = (flags >> 2) & 1;
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_RANGE(nrows, nmem, m, r0, r1)
        for (int r = r0; r < r1; r++) {
            const int i3 = r / (ne1 * ne2);
            const int i2 = (r - i3 * ne1 * ne2) / ne1;
            const int i1 = r - i3 * ne1 * ne2 - i2 * ne1;
            const int po = r * n_kv;
            const int pm = offm + (i1 % nm1) * sm1 + (i2 % nm2) * sm2 + (i3 % nm3) * sm3;
            for (int c = 0; c < n_kv; c += VW) {
                int5 co = {po + c, 0, 0, 0, 0};
                const int left = n_kv - c;
                if (left >= VW) VST(co, out, ninf); else VSTP(co, out, ninf, left - 1, 1);
            }
#define KEEP(cell)                                           \
            {                                                \
                const int c_ = (cell);                       \
                int5 cm_ = {pm + c_, 0, 0, 0, 0};            \
                const VT v_ = VLD(cm_, mask);                \
                CELL(c_, v_)                                 \
            }
            for (int j = 0; j < n_top; j++) {
                const int p = ldk(tk, offk + r * sk + j);
                if (p < 0 || p >= n_pick) {
                    continue;
                }
                if (has_score && !(ldf(score, offs + r * ss + p) >= -3.40282347e38f)) {   // (dead: -inf or NaN)
                    continue;
                }
                if (!has_blocks) {
                    KEEP(p)
                    continue;
                }
                for (int k = 0; k < n_blk; k++) {
                    KEEP(ldk(blocks, offb + p * sb + k))
                }
            }
            if (has_extra) {
                for (int j = 0; j < n_extra; j++) {
                    KEEP(ldk(extra, offe + r * se + j))
                }
            }
#undef KEEP
        }
    }
}
#else
void main(tensor mask, tensor tk, tensor dep, tensor out, int nmem, int n_kv, int n_top, int ne1, int ne2, int ne3,
          int offm, int sm1, int sm2, int sm3, int nm1, int nm2, int nm3, int offk, int sk1, int sk2, int sk3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
#if defined(F16)
    const VT ninf = as_half128((ushort128) 0xFC00);
#else
    const VT ninf = as_float64((int64) 0xFF800000);
#endif
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1;
        SPLIT_RANGE(nrows, nmem, m, r0, r1)
        if (r0 >= r1) {
            continue;
        }
        int i3 = r0 / (ne1 * ne2);
        int i2 = (r0 - i3 * ne1 * ne2) / ne1;
        int i1 = r0 - i3 * ne1 * ne2 - i2 * ne1;
        int j1 = i1 % nm1, j2 = i2 % nm2, j3 = i3 % nm3;   // the mask row, advanced with the row
        for (int r = r0; r < r1; r++) {
            const int po = r * n_kv;
            const int pm = offm + j1 * sm1 + j2 * sm2 + j3 * sm3;
            const int pk = offk + i1 * sk1 + i2 * sk2 + i3 * sk3;
            for (int c = 0; c < n_kv; c += VW) {
                int5 co = {po + c, 0, 0, 0, 0};
                const int left = n_kv - c;
                if (left >= VW) VST(co, out, ninf); else VSTP(co, out, ninf, left - 1, 1);
            }
            int j = 0;
            for (; j + 4 <= n_top; j += 4) {
                const int p0 = ldk(tk, pk + j), p1 = ldk(tk, pk + j + 1), p2 = ldk(tk, pk + j + 2), p3 = ldk(tk, pk + j + 3);
                int5 c0 = {pm + p0, 0, 0, 0, 0}, c1 = {pm + p1, 0, 0, 0, 0}, c2 = {pm + p2, 0, 0, 0, 0}, c3 = {pm + p3, 0, 0, 0, 0};
                const VT v0 = VLD(c0, mask), v1 = VLD(c1, mask), v2 = VLD(c2, mask), v3 = VLD(c3, mask);
                CELL(p0, v0)
                CELL(p1, v1)
                CELL(p2, v2)
                CELL(p3, v3)
            }
            for (; j < n_top; j++) {
                const int p = ldk(tk, pk + j);
                int5 c = {pm + p, 0, 0, 0, 0};
                const VT v = VLD(c, mask);
                CELL(p, v)
            }
            if (++i1 == ne1) {
                i1 = 0;
                if (++i2 == ne2) {
                    i2 = 0;
                    i3++;
                    j3 = j3 + 1 == nm3 ? 0 : j3 + 1;
                }
                j2 = i2 % nm2;
            }
            j1 = i1 == 0 ? 0 : (j1 + 1 == nm1 ? 0 : j1 + 1);
        }
    }
}
#endif // GEN

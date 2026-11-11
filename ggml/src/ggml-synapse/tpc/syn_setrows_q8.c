// SET_ROWS into a q8_0 tensor: dst[:, idx[i, i2 % ni1, i3 % ni2], i2, i3] = q8_0(src[:, i, i2, i3]).
// As the x86 CPU quantizer: d = amax / 127 (stored f16), q = round_half_even(x * 127 / amax).
// dst offsets and strides are in bytes (rows of ne0 / 32 blocks of 34 bytes); src is an f32 view.
// tensors: 0 dst  1 src  2 idx  dep | sink
// -DCPY: F32 -> Q8_0, written in place into a contiguous destination: tensors (dst, src, dep | sink),
// scalars nmem, ne0, ne1, ne2, ne3 (the source shape), dst byte offset, source_offset, source_stride1..3;
// row r at dst_offset + r * row_bytes, no index tensor. ne0 must be block-aligned.
#include "syn_common.h"

// f16 bit pattern of x >= 0 (round to nearest even; subnormals included), as f32
static inline float64 f16_bits(float64 x)
{
    const int64 b = as_int64(x);
    const int64 rb = b + 0x0FFF + (v_i32_shr_b(b, (int64) 13) & 1);
    const int64 hn = v_i32_shr_b(rb, (int64) 13) - ((127 - 15) << 10);
    const float64 fn = v_convert_i32_to_f32_b(hn, SW_RHNE, 0, 1, 0);
    const float64 fs = v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(x * 16777216.0f, SW_RHNE, 0, 1, 0), SW_RHNE, 0, 1, 0);
    return v_f32_sel_less_f32_b(x, 6.103515625e-05f, fs, fn);
}

#if defined(CPY)
// F32 -> Q8_0 dense copy. src: f32 view (source_offset + strides of dims 1..3, dim 0
// contiguous), out: the destination, dense quantized rows at byte offd.
// tensors: 0 out  1 src  2 dep | 3 sink
void main(tensor out, tensor src, tensor dep, tensor sink,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int offd, int offs, int ss1, int ss2, int ss3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 lh = lane * 0.5f;
    const float64 par = lh - v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(lh, SW_RD, 0, 1, 0), SW_RHNE, 0, 1, 0);
    const int nrows = ne1 * ne2 * ne3;
    const int nch = (ne0 + 63) >> 6;   // rows are multiples of 32 (block-aligned), the tail chunk masked
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        int rcur = -1;
        int ps = 0;
        for (int it = i0; it < i1; it++) {
            const int r = it / nch;
            const int ch = it - r * nch;
            if (r != rcur) {
                rcur = r;
                const int r3 = r / (ne1 * ne2);
                const int rr = r - r3 * ne1 * ne2;
                const int r2 = rr / ne1;
                const int r1 = rr - r2 * ne1;
                ps = offs + r1 * ss1 + r2 * ss2 + r3 * ss3;
            }
            const int e0 = ch << 6;
            const int pb = offd + r * (ne0 / 32) * 34 + 68 * ch;
            const float64 x = ld_f32(src, ps + e0);
            const float64 ax = v_f32_abs_b(x);
            const float64 t = v_f32_max_b(ax, v_f32_mov_dual_group_all_b(ax, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64()));
            float64 r0 = SHUF(t, IDX[0]), r1 = SHUF(t, IDX[1]), r2 = SHUF(t, IDX[2]), r3 = SHUF(t, IDX[3]);
            #pragma unroll
            for (int e = 4; e < 16; e += 4) {
                r0 = v_f32_max_b(r0, SHUF(t, IDX[e]));
                r1 = v_f32_max_b(r1, SHUF(t, IDX[e + 1]));
                r2 = v_f32_max_b(r2, SHUF(t, IDX[e + 2]));
                r3 = v_f32_max_b(r3, SHUF(t, IDX[e + 3]));
            }
            const float64 amax = v_f32_max_b(v_f32_max_b(r0, r1), v_f32_max_b(r2, r3));
            const float64 d = amax * (1.0f / 127.0f);
            float64 ia = nl_recip(v_f32_max_b(amax, 1e-30f));
            ia = v_f32_mac_b(ia, v_f32_mac_b(amax, -ia, (float64) 1.0f), ia);
            const float64 id = v_f32_sel_grt_f32_b(amax, 0.0f, 127.0f * ia, 0.0f);
            float256 g;
            g.v1 = x * id;
            g.v2 = 0.0f;
            g.v3 = 0.0f;
            g.v4 = 0.0f;
            const char256 qc = convert_float256_to_char256(g, SW_RHNE | SW_LINEAR);
            const int64 hb = v_convert_f32_to_i32_b(f16_bits(d), SW_RHNE, 0, 1, 0);
            const float64 lo = v_convert_i32_to_f32_b(hb & 0xFF, SW_RHNE, 0, 1, 0);
            const float64 hi = v_convert_i32_to_f32_b(v_i32_shr_b(hb, (int64) 8) & 0xFF, SW_RHNE, 0, 1, 0);
            float64 sb = v_f32_sel_eq_f32_b(par, 0.0f, lo, hi);
            sb = v_f32_sel_geq_f32_b(sb, 128.0f, sb - 256.0f, sb);
            float256 gs;
            gs.v1 = sb;
            gs.v2 = 0.0f;
            gs.v3 = 0.0f;
            gs.v4 = 0.0f;
            const char256 sc = convert_float256_to_char256(gs, SW_RHNE | SW_LINEAR);
            int5 c = {pb, 0, 0, 0, 0};
            v_i8_st_tnsr_partial(c, out, sc, 1, 0);
            c[0] = pb + 2;
            v_i8_st_tnsr_partial(c, out, qc, 31, 0);
            if (ne0 - e0 > 32) {
                c[0] = pb + 34;
                v_i8_st_tnsr_partial(c, out, sc, 1, 32);
                c[0] = pb + 36;
                v_i8_st_tnsr_partial(c, out, qc, 31, 32);
            }
        }
    }
}
#else

void main(tensor dst, tensor src, tensor idx, tensor dep, tensor sink,
          int nmem, int ne0, int n, int ne2, int ne3,
          int offd, int sd1, int sd2, int sd3,
          int offs, int ss1, int ss2, int ss3,
          int offi, int si0, int si1, int si2, int ni1, int ni2, int is64)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    DECL_IDX(IDX)
    const float64 lane = LANE_F32();
    const float64 lh = lane * 0.5f;
    const float64 par = lh - v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(lh, SW_RD, 0, 1, 0), SW_RHNE, 0, 1, 0);
    const int nrows = n * ne2 * ne3;
    const int nch = (ne0 + 63) >> 6;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        int rcur = -1;
        int pd = 0, ps = 0;
        for (int it = i0; it < i1; it++) {
            const int r = it / nch;
            const int ch = it - r * nch;
            if (r != rcur) {
                rcur = r;
                const int r3 = r / (n * ne2);
                const int rr = r - r3 * n * ne2;
                const int r2 = rr / n;
                const int r1 = rr - r2 * n;
                ps = offs + r1 * ss1 + r2 * ss2 + r3 * ss3;
                const int pi = offi + r1 * si0 + (r2 % ni1) * si1 + (r3 % ni2) * si2;
                int5 ci = {is64 ? 2 * pi : pi, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, idx));
                pd = offd + row * sd1 + r2 * sd2 + r3 * sd3;
            }
            const int e0 = ch << 6;
            const int pb = pd + 68 * ch;
            const float64 x = ld_f32(src, ps + e0);
            // max |x| of each 32-lane half (block)
            const float64 ax = v_f32_abs_b(x);
            const float64 t = v_f32_max_b(ax, v_f32_mov_dual_group_all_b(ax, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64()));
            float64 r0 = SHUF(t, IDX[0]), r1 = SHUF(t, IDX[1]), r2 = SHUF(t, IDX[2]), r3 = SHUF(t, IDX[3]);
            #pragma unroll
            for (int e = 4; e < 16; e += 4) {
                r0 = v_f32_max_b(r0, SHUF(t, IDX[e]));
                r1 = v_f32_max_b(r1, SHUF(t, IDX[e + 1]));
                r2 = v_f32_max_b(r2, SHUF(t, IDX[e + 2]));
                r3 = v_f32_max_b(r3, SHUF(t, IDX[e + 3]));
            }
            const float64 amax = v_f32_max_b(v_f32_max_b(r0, r1), v_f32_max_b(r2, r3));
            const float64 d = amax * (1.0f / 127.0f);
            float64 ia = nl_recip(v_f32_max_b(amax, 1e-30f));
            ia = v_f32_mac_b(ia, v_f32_mac_b(amax, -ia, (float64) 1.0f), ia);
            const float64 id = v_f32_sel_grt_f32_b(amax, 0.0f, 127.0f * ia, 0.0f);
            float256 g;
            g.v1 = x * id;
            g.v2 = 0.0f;
            g.v3 = 0.0f;
            g.v4 = 0.0f;
            const char256 qc = convert_float256_to_char256(g, SW_RHNE | SW_LINEAR);
            // scale bytes: lane 0 / 32 low byte, lane 1 / 33 high byte of the block's f16 d
            const int64 hb = v_convert_f32_to_i32_b(f16_bits(d), SW_RHNE, 0, 1, 0);
            const float64 lo = v_convert_i32_to_f32_b(hb & 0xFF, SW_RHNE, 0, 1, 0);
            const float64 hi = v_convert_i32_to_f32_b(v_i32_shr_b(hb, (int64) 8) & 0xFF, SW_RHNE, 0, 1, 0);
            float64 sb = v_f32_sel_eq_f32_b(par, 0.0f, lo, hi);
            sb = v_f32_sel_geq_f32_b(sb, 128.0f, sb - 256.0f, sb);
            float256 gs;
            gs.v1 = sb;
            gs.v2 = 0.0f;
            gs.v3 = 0.0f;
            gs.v4 = 0.0f;
            const char256 sc = convert_float256_to_char256(gs, SW_RHNE | SW_LINEAR);
            int5 c = {pb, 0, 0, 0, 0};
            v_i8_st_tnsr_partial(c, dst, sc, 1, 0);
            c[0] = pb + 2;
            v_i8_st_tnsr_partial(c, dst, qc, 31, 0);
            if (ne0 - e0 > 32) {
                c[0] = pb + 34;
                v_i8_st_tnsr_partial(c, dst, sc, 1, 32);
                c[0] = pb + 36;
                v_i8_st_tnsr_partial(c, dst, qc, 31, 32);
            }
        }
    }
    if (ioff[0] == 0) {
        int5 z = {0, 0, 0, 0, 0};
        v_f32_st_tnsr(z, sink, 0.0f);
    }
}
#endif

// Rows of a q4_0 / q4_1 / q5_0 / q5_1 / q8_0 view (-DQ4_0 ...) to contiguous f16 [ne0, ne1, ne2, ne3]
// (-DOUT_F32: f32)
// Source offsets and strides in bytes.
// tensors: 0 src  dep | out
#include "syn_common.h"

#if defined(Q4_0)
#define BB 18
#elif defined(Q4_1)
#define BB 20
#elif defined(Q5_0)
#define BB 22
#elif defined(Q5_1)
#define BB 24
#elif defined(Q8_0)
#define BB 34
#else
#error "no type"
#endif
#if defined(Q4_1) || defined(Q5_1)
#define HAS_M 1
#else
#define HAS_M 0
#endif
#if defined(Q5_0) || defined(Q5_1)
#define HAS_H 1
#else
#define HAS_H 0
#endif
#define OFF_H (2 + 2 * HAS_M)
#define OFF_Q (OFF_H + 4 * HAS_H)

static inline float64 i2f(int64 v)
{
    return v_convert_i32_to_f32_b(v, SW_RHNE, 0, 1, 0);
}

static inline float64 ld_u8(tensor t, int p)
{
    int5 c = {p, 0, 0, 0, 0};
    const float64 f = convert_char256_to_float256(v_i8_ld_tnsr_b(c, t), SW_LINEAR).v1;
    return v_f32_sel_less_f32_b(f, 0.0f, f + 256.0f, f);
}

void main(tensor src, tensor dep, tensor out, int nmem, int ne0, int ne1, int ne2, int ne3,
          int offs, int ss1, int ss2, int ss3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int64 li = (int64) read_lane_id_4b_b();
    const float64 lane = i2f(li);
    const float64 p16 = i2f(li & 16);
    const int nrows = ne1 * ne2 * ne3;
    const int nch = (ne0 + 63) >> 6;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nch;
            const int ch = it - r * nch;
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int p = offs + r1 * ss1 + r2 * ss2 + r3 * ss3 + 2 * BB * ch;
            int5 cs = {p, 0, 0, 0, 0};
            // f16 params: block A at f16 lane 0 (1), block B at lane BB / 2 (+ 1)
            const float64 sv = f16_to_f32(v_f16_ld_tnsr_b(cs, src)).v1;
            const float64 sc = BCAST_DG(sv, 0), sc1 = BCAST_DG(sv, (BB / 2) >> 4);
            const float64 d = v_f32_sel_less_f32_b(lane, 32.0f, SHUF(sc, IDX[0]), SHUF(sc1, IDX[(BB / 2) & 15]));
#if defined(Q8_0)
            int5 ca = {p + 2, 0, 0, 0, 0};
            int5 cb = {p + 4, 0, 0, 0, 0};
            const float64 qa = convert_char256_to_float256(v_i8_ld_tnsr_b(ca, src), SW_LINEAR).v1;
            const float64 qb = convert_char256_to_float256(v_i8_ld_tnsr_b(cb, src), SW_LINEAR).v1;
            const float64 y = v_f32_sel_less_f32_b(lane, 32.0f, qa, qb) * d;
#else
            const float64 f = v_f32_sel_less_f32_b(lane, 32.0f, BCAST_DG(ld_u8(src, p + OFF_Q), 0),
                                                   BCAST_DG(ld_u8(src, p + BB + OFF_Q), 0));
            const float64 hi = i2f(v_i32_shr_b(v_convert_f32_to_i32_b(f, SW_RHNE, 0, 1, 0), (int64) 4));
            float64 q = v_f32_sel_eq_f32_b(p16, 0.0f, v_f32_mac_b(hi, -16.0f, f), hi);
#if HAS_H
            int5 ha = {p + OFF_H, 0, 0, 0, 0};
            int5 hb = {p + BB + OFF_H, 0, 0, 0, 0};
            const float64 qha = BCAST_DG(v_f32_ld_tnsr_b(ha, src), 0);
            const float64 qhb = BCAST_DG(v_f32_ld_tnsr_b(hb, src), 0);
            const int64 qh = as_int64(v_f32_sel_less_f32_b(lane, 32.0f, SHUF(qha, IDX[0]), SHUF(qhb, IDX[0])));
            q += 16.0f * i2f(v_i32_shr_b(qh, li & 31) & 1);
#endif
#if HAS_M
            const float64 mn = v_f32_sel_less_f32_b(lane, 32.0f, SHUF(sc, IDX[1]), SHUF(sc, IDX[BB / 2 + 1]));
            const float64 y = v_f32_mac_b(q, d, mn);
#else
            const float64 y = (q - (HAS_H ? 16.0f : 8.0f)) * d;
#endif
#endif
            const int left = ne0 - (ch << 6);
#if defined(OUT_F32)
            ST_F32(out, r * ne0 + (ch << 6), y, left);
#else
            int5 co = {r * ne0 + (ch << 6), 0, 0, 0, 0};
            const half128 h = cvt_f16(y, y);
            if (left >= 64) v_f16_st_tnsr_partial(co, out, h, 63, 0); else v_f16_st_tnsr_partial(co, out, h, left - 1, 0);
#endif
        }
    }
}

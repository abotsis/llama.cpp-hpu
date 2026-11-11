// SET_ROWS into a q4_0 / q4_1 / q5_0 / q5_1 tensor (-DQ4_0 ...): as quantize_row_q*_ref.
//  q4_0 / q5_0: d = max / -8 (-16) with max the value of largest |x|, q = min(15 (31), (int) (x / d + 8.5 (16.5)))
//  q4_1 / q5_1: d = (max - min) / 15 (31), m = min, q = min(15 (31), (int) ((x - min) / d + 0.5))
// Block (32 elements): f16 d [f16 m] [u32 qh: bit 4 of q[j] at bit j] [16 bytes: q[j] & 15 | (q[j + 16] & 15) << 4].
// dst offsets and strides are in bytes; src is an f32 view.
// tensors: 0 dst  1 src  2 idx  dep | sink
// -DIQ4_NL: SET_ROWS into an IQ4_NL tensor, as quantize_row_iq4_nl_ref (ntry = -1, no quant
// weights): blocks of 32 values, f16 scale + 16 packed nibble bytes (low code j, high code j + 16),
// table T[16]. Midpoint ties choose the higher index; the fitted scale is the weighted least-squares
// d = sum(w q x) / sum(w q q) accumulated in source order (deterministic).
// -DCPY: F32 -> the format, written in place into a contiguous destination: tensors (dst, src, dep | sink),
// scalars nmem, ne0, ne1, ne2, ne3 (the source shape), dst byte offset, source_offset, source_stride1..3;
// row r at dst_offset + r * row_bytes, no index tensor. ne0 and the source row offsets must be block-aligned.
#include "syn_common.h"

#if defined(Q4_0)
#define BB 18
#define NQ 15.0f
#define HAS_M 0
#define HAS_H 0
#elif defined(Q4_1)
#define BB 20
#define NQ 15.0f
#define HAS_M 1
#define HAS_H 0
#elif defined(Q5_0)
#define BB 22
#define NQ 31.0f
#define HAS_M 0
#define HAS_H 1
#elif defined(Q5_1)
#define BB 24
#define NQ 31.0f
#define HAS_M 1
#define HAS_H 1
#elif defined(IQ4_NL)
#define BB 18
#define HAS_M 0
#define HAS_H 0
#else
#error "no type"
#endif
#define OFF_H (2 + 2 * HAS_M)
#define OFF_Q (OFF_H + 4 * HAS_H)

#if defined(IQ4_NL)
// the IQ4_NL code table
static const int T_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// the code of al (best_index_int8 with the >= midpoint tie choosing the HIGHER index: the
// reference's x - val[mu-1] < val[mu] - x comparison becomes >= against each code midpoint)
static inline float64 iq4_code(float64 al)
{
    // the code midpoints (al below the first code's -127 start falls to index 0)
    float64 l = (float64) (char) T_IQ4NL[0];
    for (int j = 0; j < 15; j++) {
        const float64 mid = 0.5f * (T_IQ4NL[j] + T_IQ4NL[j + 1]);
        l = v_f32_sel_geq_f32_b(al, mid, (float64) (char) T_IQ4NL[j + 1], l);
    }
    return l;
}
#endif

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

// f16 bits of x (any sign) as an int
static inline int64 f16_sbits(float64 x)
{
    const int64 h = v_convert_f32_to_i32_b(f16_bits(v_f32_abs_b(x)), SW_RHNE, 0, 1, 0);
    return v_i32_sel_less_f32_b(x, 0.0f, h | 0x8000, h);
}

static inline float64 i2f(int64 v)
{
    return v_convert_i32_to_f32_b(v, SW_RHNE, 0, 1, 0);
}

// floor of x >= 0 (as f32)
static inline float64 floorp(float64 x)
{
    return i2f(v_convert_f32_to_i32_b(x, SW_RD, 0, 1, 0));
}

// byte value 0..255 as the signed value the char conversion stores
static inline float64 sbyte(float64 v)
{
    return v_f32_sel_geq_f32_b(v, 128.0f, v - 256.0f, v);
}

#define SWAP_DG(a) v_f32_mov_dual_group_all_b((a), 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64())

// max and min over each 32-lane half (all lanes of the half get the result)
#define HALF_REDUCE(x, OP, out)                                                    \
    {                                                                              \
        const float64 t_ = OP((x), SWAP_DG(x));                                    \
        float64 r0_ = SHUF(t_, IDX[0]), r1_ = SHUF(t_, IDX[1]);                    \
        float64 r2_ = SHUF(t_, IDX[2]), r3_ = SHUF(t_, IDX[3]);                    \
        _Pragma("unroll") for (int e_ = 4; e_ < 16; e_ += 4) {                     \
            r0_ = OP(r0_, SHUF(t_, IDX[e_]));                                      \
            r1_ = OP(r1_, SHUF(t_, IDX[e_ + 1]));                                  \
            r2_ = OP(r2_, SHUF(t_, IDX[e_ + 2]));                                  \
            r3_ = OP(r3_, SHUF(t_, IDX[e_ + 3]));                                  \
        }                                                                          \
        out = OP(OP(r0_, r1_), OP(r2_, r3_));                                      \
    }

static inline char256 to_char(float64 v)
{
    float256 g;
    g.v1 = v;
    g.v2 = 0.0f;
    g.v3 = 0.0f;
    g.v4 = 0.0f;
    return convert_float256_to_char256(g, SW_RHNE | SW_LINEAR);
}

// quantize the two 32-value blocks of x (64 lanes) and store them at out + pb
// (pb = byte offset of the first block): the Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL quantizers.
// ne0e = ne0 - e0 (the masked second block). IDX / p32 / p16 / w8: the shared lane tables.
static inline void quant2(tensor out, int pb, float64 x, int ne0e, uchar256 IDX[16],
                          float64 p32, float64 p16, float64 w8)
{
#if defined(IQ4_NL)
    // quantize_row_iq4_nl_impl, ntry = -1, no quant weights: weights x * x, signed first
    // maximum, weighted fitted scale, midpoint ties -> the higher index. The reference
    // divides by values[0] = -127: d0 = max / -127 with max the signed first maximum.
    // x holds two 32-value blocks (one per 32-lane half); every reduction below is per half.
    const float64 w = x * x;   // per-lane weights
    const float64 ax = v_f32_abs_b(x);
    float64 amax;
    HALF_REDUCE(ax, v_f32_max_b, amax)
    // the signed first maximum: the earliest carrier of amax in lane order (the reference's
    // strict > keeps the FIRST); the select chain walks the 16 lane slots of each dual group in
    // order, then the block's lower dual group wins when it carries amax (m0 == 0 only for an
    // all-zero block, where either choice is 0)
    float64 mxg = (float64) (char) 0;
    for (int l = 0; l < 16; l++) {
        const float64 al = SHUF(ax, IDX[l]);
        mxg = v_f32_sel_eq_f32_b(al, amax, v_f32_sel_eq_f32_b(mxg, (float64) (char) 0, SHUF(x, IDX[l]), mxg), mxg);
    }
    const float64 m0 = v_f32_mov_dual_group_all_b(mxg, 0xFFFFFFFF, 0, 0, 2, 2, WR_ALL, undef_f64());
    const float64 m1 = v_f32_mov_dual_group_all_b(mxg, 0xFFFFFFFF, 1, 1, 3, 3, WR_ALL, undef_f64());
    const float64 mx = v_f32_sel_eq_f32_b(m0, (float64) (char) 0, m1, m0);
    // first estimate d0 = max / values[0] = mx / -127
    const float64 d0 = v_f32_mul_b(mx, -1.0f / 127.0f);
    const float64 q0 = iq4_code(v_f32_mul_b(x, nl_recip(d0)));
    float64 pqx = v_f32_mac_b(w, v_f32_mul_b(q0, x), 0.0f);
    float64 pq2 = v_f32_mac_b(w, v_f32_mul_b(q0, q0), 0.0f);
    // source-order accumulation: slot l of the unswapped vector is term l (block lanes 0..15),
    // slot l of the swapped vector is term 16 + l (block lanes 16..31); the sequential adds
    // reproduce the reference's j = 0..31 association exactly
    float64 sqx = (float64) 0.0f, sq2 = (float64) 0.0f;
    for (int l = 0; l < 16; l++) {
        sqx = sqx + SHUF(pqx, IDX[l]);
        sqx = sqx + SHUF(SWAP_DG(pqx), IDX[l]);
        sq2 = sq2 + SHUF(pq2, IDX[l]);
        sq2 = sq2 + SHUF(SWAP_DG(pq2), IDX[l]);
    }
    const float64 d = v_f32_sel_grt_f32_b(sq2, 0.0f, v_f32_mul_b(sqx, nl_recip(sq2)), 0.0f);
    // zero block (amax < GROUP_MAX_EPS): scale 0, deterministically zero packed bytes
    const float64 zero = v_f32_sel_less_f32_b(amax, 1e-15f, (float64) 1.0f, 0.0f);
    const float64 df = v_f32_sel_eq_f32_b(zero, 1.0f, 0.0f, d);
    // the packed codes stay the FIRST-MAX-scale codes (the reference's ntry < 0 path keeps L
    // from the single pass and stores only the fitted scale)
    const float64 l = v_f32_sel_eq_f32_b(zero, 1.0f, (float64) (char) 0, q0);
    // the code's index j (0..15) as the nibble (low code j, high code j + 16); zero blocks take
    // index 0 -> deterministic zero packed bytes (the harness compares dequantized values, where
    // the zero scale makes every code read 0)
    float64 jn = (float64) (char) 0;
    for (int jj = 15; jj >= 1; jj--) {
        jn = v_f32_sel_eq_f32_b(l, (float) T_IQ4NL[jj], (float64) jj, jn);
    }
    const char256 qc = to_char(sbyte(v_f32_mac_b(SWAP_DG(jn), 16.0f, jn)));
    // header: f16 d bytes in lanes 0..1 of each half (zero for a zero block)
    const int64 db = f16_sbits(df);
    float64 hv = v_f32_sel_eq_f32_b(p32, 0.0f, i2f(db & 0xFF), i2f(v_i32_shr_b(db, (int64) 8) & 0xFF));
    const char256 hc = to_char(sbyte(hv));
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        if (h == 1 && ne0e <= 32) {
            break;
        }
        const int b = pb + h * BB;
        int5 c = {b, 0, 0, 0, 0};
        v_i8_st_tnsr_partial(c, out, hc, 1, 32 * h);
        c[0] = b + 2;
        v_i8_st_tnsr_partial(c, out, qc, 15, 32 * h);
    }
#else
    float64 mx, mn;
    HALF_REDUCE(x, v_f32_max_b, mx)
    HALF_REDUCE(x, v_f32_min_b, mn)
    const float64 vmax = v_f32_sel_grt_f32_b(-mn, mx, mn, mx);
    float64 d, base, bias;
#if HAS_M
    const float64 a = v_f32_sub_b(mx, mn);
    const float64 q0 = v_f32_mul_b(a, 1.0f / NQ);
    d = v_f32_mac_b(v_f32_mac_b(q0, -NQ, a), 1.0f / NQ, q0);
    base = mn;
    bias = 0.5f;
#else
    d = vmax * (HAS_H ? -1.0f / 16.0f : -1.0f / 8.0f);
    base = 0.0f;
    bias = HAS_H ? 16.5f : 8.5f;
#endif
    const float64 ad = v_f32_abs_b(d);
    float64 ia = nl_recip(v_f32_max_b(ad, 1e-30f));
    ia = v_f32_mac_b(ia, v_f32_mac_b(ad, -ia, (float64) 1.0f), ia);
    ia = v_f32_sel_less_f32_b(d, 0.0f, -ia, ia);
    const float64 id = v_f32_sel_eq_f32_b(d, 0.0f, 0.0f, ia);
    // q = min(NQ, (int) ((x - base) * id + bias)), each step rounded as on the CPU
    const float64 xs = HAS_M ? v_f32_sub_b(x, base) : x;
    const float64 q = v_f32_min_b(floorp(v_f32_add_b(v_f32_mul_b(xs, id), bias)), NQ);
#if HAS_H
    const float64 hb = v_f32_sel_geq_f32_b(q, 16.0f, 1.0f, 0.0f);
    const float64 lo = v_f32_mac_b(hb, -16.0f, q);
#else
    const float64 lo = q;
#endif
    const char256 qc = to_char(sbyte(v_f32_mac_b(SWAP_DG(lo), 16.0f, lo)));
    // header: f16 d (and f16 m) bytes in lanes 0..3 of each half
    const int64 db = f16_sbits(d);
    float64 hv = v_f32_sel_eq_f32_b(p32, 0.0f, i2f(db & 0xFF), i2f(v_i32_shr_b(db, (int64) 8) & 0xFF));
#if HAS_M
    const int64 mb = f16_sbits(mn);
    hv = v_f32_sel_eq_f32_b(p32, 2.0f, i2f(mb & 0xFF), hv);
    hv = v_f32_sel_eq_f32_b(p32, 3.0f, i2f(v_i32_shr_b(mb, (int64) 8) & 0xFF), hv);
#endif
    const char256 hc = to_char(sbyte(hv));
#if HAS_H
    // qh bytes: bit 4 of the codes of 8 lanes; per 16-lane group: lane 0 low byte, lane 1 high byte
    const float64 wq = hb * w8;
    float64 s0 = SHUF(wq, IDX[0]), s1 = SHUF(wq, IDX[8]);
    #pragma unroll
    for (int e = 1; e < 8; e++) {
        s0 += SHUF(wq, IDX[e]);
        s1 += SHUF(wq, IDX[e + 8]);
    }
    const char256 bc = to_char(sbyte(v_f32_sel_eq_f32_b(p16, 0.0f, s0, s1)));
#endif
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        if (h == 1 && ne0e <= 32) {
            break;
        }
        const int b = pb + h * BB;
        int5 c = {b, 0, 0, 0, 0};
        v_i8_st_tnsr_partial(c, out, hc, 1 + 2 * HAS_M, 32 * h);
#if HAS_H
        c[0] = b + OFF_H;
        v_i8_st_tnsr_partial(c, out, bc, 1, 32 * h);
        c[0] = b + OFF_H + 2;
        v_i8_st_tnsr_partial(c, out, bc, 1, 32 * h + 16);
#endif
        c[0] = b + OFF_Q;
        v_i8_st_tnsr_partial(c, out, qc, 15, 32 * h);
    }
#endif
}

#if defined(CPY)
// F32 -> Q4_0 / Q4_1 / Q5_0 / Q5_1 / IQ4_NL dense copy. src: f32 view
// (source_offset + strides of dims 1..3, dim 0 contiguous), out: the destination, dense quantized rows at byte offd.
// tensors: 0 out  1 src  2 dep | 3 sink
void main(tensor out, tensor src, tensor dep, tensor sink,
          int nmem, int ne0, int ne1, int ne2, int ne3,
          int offd, int offs, int ss1, int ss2, int ss3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    DECL_IDX(IDX)
    const int64 li = (int64) read_lane_id_4b_b();
    const float64 p32 = i2f(li & 31), p16 = i2f(li & 15);
#if HAS_H
    const float64 w8 = i2f(v_i32_shl_b((int64) 1, li & 7));
#else
    const float64 w8 = 0.0f;
#endif
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
            const int pb = offd + r * (ne0 / 32) * BB + 2 * BB * ch;
            quant2(out, pb, ld_f32(src, ps + e0), ne0 - e0, IDX, p32, p16, w8);
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
    const int64 li = (int64) read_lane_id_4b_b();
    const float64 p32 = i2f(li & 31), p16 = i2f(li & 15);
#if HAS_H
    const float64 w8 = i2f(v_i32_shl_b((int64) 1, li & 7));
#else
    const float64 w8 = 0.0f;
#endif
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
            const int pb = pd + 2 * BB * ch;
            quant2(dst, pb, ld_f32(src, ps + e0), ne0 - e0, IDX, p32, p16, w8);
        }
    }
    if (ioff[0] == 0) {
        int5 z = {0, 0, 0, 0, 0};
        v_f32_st_tnsr(z, sink, 0.0f);
    }
}
#endif

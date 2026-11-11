// out = f(a), f32. out contiguous [ne0, ne1, ne2, ne3]; a is a view (dim-0 contiguous).
// op: 0 silu, 1 sigmoid, 2 softplus, 3 neg, 4 exp, 5 relu, 6 gelu (tanh form), 7 tanh,
//     8 scale (a * f0 + f1), 9 sqr, 10 sqrt, 11 abs, 12 fill (f0), 13 silu(a * f0)
// -DFLAT: rows of contiguous elements at a uniform stride.
// -DFLAT -DOPC=<op>: the op fixed at build time.
// tensors: 0 a | 1 out
#include "syn_common.h"

static inline float64 unary_op(int op, float64 x, float f0, float f1)
{
    float64 v;
    if (op == 0) {
#if defined(OPC)
        v = x * nl_sigm_f(x);
#else
        v = nl_silu_acc(x);
#endif
    } else if (op == 1) {
#if defined(OPC)
        v = nl_sigm_f(x);
#else
        v = nl_sigm_acc(x);
#endif
    } else if (op == 2) {
        v = nl_softplus(x);
    } else if (op == 3) {
        v = -x;
    } else if (op == 4) {
        v = nl_exp(x);
        v = v_f32_sel_grt_f32_b(x, 88.72283f, as_float64((int64) 0x7F800000), v);
        v = v_f32_sel_less_f32_b(x, -103.0f, 0.0f, v);
    } else if (op == 5) {
        v = v_f32_max_b(x, 0.0f);
    } else if (op == 6) {
#if defined(OPC)
        v = nl_gelu_f(x);
#else
        const float64 u = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
        v = 0.5f * x * (1.0f + nl_tanh(u));
#endif
    } else if (op == 7) {
        v = nl_tanh(x);
    } else if (op == 8) {
        v = v_f32_mac_b(x, f0, (float64) f1);
    } else if (op == 9) {
        v = x * x;
    } else if (op == 10) {
        const float64 y = v_f32_max_b(x, 1e-30f);
        v = v_f32_sel_leq_f32_b(x, 0.0f, 0.0f, y * nl_rsqrt(y));
    } else if (op == 11) {
        v = v_f32_abs_b(x);
    } else if (op == 13) {
        v = nl_silu_acc(x * f0);
    } else {
        v = f0;
    }
    return v;
}

#if defined(FLAT) && defined(OPC)
#if defined(OUT_BF16)
#define STP_OUT(c, t, v, lp) v_bf16_st_tnsr_partial((c), (t), cvt_bf16((v), (v)), (lp), 0)
#else
#define STP_OUT(c, t, v, lp) v_f32_st_tnsr_partial((c), (t), (v), (lp), 0)
#endif
// rows of L contiguous elements at stride rs (out: rows packed)
void main(tensor a, tensor dep, tensor out, int nmem, int op, int L, int nrows, int rs, int off0, float f0, float f1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nq = (L + 511) >> 9;
    const int nitems = nrows * nq;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int r = i0 / nq;
        int e = (i0 - r * nq) << 9;
        for (int it = i0; it < i1; it++) {
            int5 ca = {off0 + r * rs + e, 0, 0, 0, 0};
            int5 co = {r * L + e, 0, 0, 0, 0};
            float64 x[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                x[u] = v_f32_ld_tnsr_b(ca, a);
                ca[0] += 64;
            }
            const int left = L - e;
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                const float64 v = unary_op(OPC, x[u], f0, f1);
                if (left >= 64 * u + 64) {
                    STP_OUT(co, out, v, 63);
                } else if (left > 64 * u) {
                    STP_OUT(co, out, v, left - 64 * u - 1);
                }
                co[0] += 64;
            }
            e += 512;
            if (e >= L) {
                e = 0;
                r++;
            }
        }
    }
}
#elif defined(FLAT)
// rows of L contiguous elements at stride rs (out: rows packed)
void main(tensor a, tensor dep, tensor out, int nmem, int op, int L, int nrows, int rs, int off0, float f0, float f1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nch = (L + 255) >> 8;
    const int nitems = nrows * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nch, e = (it - r * nch) << 8;
            const int pa = off0 + r * rs + e, po = r * L + e, n = L - e;
            const float64 x0 = ld_f32(a, pa), x1 = ld_f32(a, pa + 64);
            const float64 x2 = ld_f32(a, pa + 128), x3 = ld_f32(a, pa + 192);
            const float64 y0 = unary_op(op, x0, f0, f1), y1 = unary_op(op, x1, f0, f1);
            const float64 y2 = unary_op(op, x2, f0, f1), y3 = unary_op(op, x3, f0, f1);
            ST_OUT(out, po, y0, n);
            if (n > 64) ST_OUT(out, po + 64, y1, n - 64);
            if (n > 128) ST_OUT(out, po + 128, y2, n - 128);
            if (n > 192) ST_OUT(out, po + 192, y3, n - 192);
        }
    }
}
#else
void main(tensor a, tensor dep, tensor out,
          int nmem, int op, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03, float f0, float f1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nch = (ne0 + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    const int nitems = nrows * nch;
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
            const float64 x = ld_f32(a, off0 + r1 * s01 + r2 * s02 + r3 * s03 + e0);
            const float64 v = unary_op(op, x, f0, f1);
            ST_OUT(out, r * ne0 + e0, v, ne0 - e0);
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
#endif

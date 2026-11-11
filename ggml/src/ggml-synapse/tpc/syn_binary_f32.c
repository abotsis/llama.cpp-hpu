// out = a (op) b, f32. out is contiguous [ne0, ne1, ne2, ne3]; a, b are views
// (element offset + strides of dims 1..3; a stride of 0 broadcasts that dim).
// bc0 = 1: b has one element along dim 0 (broadcast).
// op: 0 add, 1 sub, 2 mul, 3 div, 4 silu(a)*b, 5 gelu(a)*b, 6 relu(a)*b
// flat = 1: a, b and out contiguous over the whole tensor. -DOPC=<op>: the op
// fixed at build time; flat = 1 then means rows of ne0 elements (rows = ne1 ne2 ne3), a row r at
// off0 + r * s01, b row at off1 + r * s11 (0: one row for all; bc0: one element per row).
// tensors: 0 a  1 b | 2 out
#include "syn_common.h"

#if defined(OPC)
#define OP OPC
#else
#define OP op
#endif

static inline float64 bin_op(int op, float64 x, float64 y)
{
    float64 v;
    if (op == 0) {
        v = x + y;
    } else if (op == 1) {
        v = x - y;
    } else if (op == 2) {
        v = x * y;
    } else if (op == 3) {
        v = x * nl_recip(y);
    } else if (op == 4) {
#if defined(OPC)
        v = x * nl_sigm_f(x) * y;
#else
        v = nl_silu_acc(x) * y;
#endif
    } else if (op == 5) {
#if defined(OPC)
        v = nl_gelu_f(x) * y;
#else
        const float64 u = 0.7978845608028654f * x * (1.0f + 0.044715f * x * x);
        v = 0.5f * x * (1.0f + nl_tanh(u)) * y;
#endif
    } else {
        v = v_f32_max_b(x, 0.0f) * y;
    }
    return v;
}

#if defined(OUT_BF16)
#define STP_OUT(c, t, v, lp) v_bf16_st_tnsr_partial((c), (t), cvt_bf16((v), (v)), (lp), 0)
#else
#define STP_OUT(c, t, v, lp) v_f32_st_tnsr_partial((c), (t), (v), (lp), 0)
#endif

void main(tensor a, tensor b, tensor dep, tensor out,
          int nmem, int op, int ne0, int ne1, int ne2, int ne3,
          int off0, int s01, int s02, int s03,
          int off1, int s11, int s12, int s13, int bc0, int flat)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nch = (ne0 + 63) >> 6;
    const int nrows = ne1 * ne2 * ne3;
    const int nitems = nrows * nch;
#if defined(OPC)
    if (flat) {
        const int nq = (ne0 + 511) >> 9;
        const int nit = nrows * nq;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int i0, i1;
            SPLIT_RANGE(nit, nmem, m, i0, i1)
            if (i0 >= i1) {
                continue;
            }
            int r = i0 / nq;
            int e = (i0 - r * nq) << 9;
            for (int it = i0; it < i1; it++) {
                int5 ca = {off0 + r * s01 + e, 0, 0, 0, 0};
                int5 cb = {off1 + r * s11 + (bc0 ? 0 : e), 0, 0, 0, 0};
                int5 co = {r * ne0 + e, 0, 0, 0, 0};
                float64 x[8], y[8];
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    x[u] = v_f32_ld_tnsr_b(ca, a);
                    ca[0] += 64;
                }
                if (bc0) {
                    const float64 yb = BCAST0(v_f32_ld_tnsr_b(cb, b), IDX);
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        y[u] = yb;
                    }
                } else {
                    #pragma unroll
                    for (int u = 0; u < 8; u++) {
                        y[u] = v_f32_ld_tnsr_b(cb, b);
                        cb[0] += 64;
                    }
                }
                const int left = ne0 - e;
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    const float64 v = bin_op(OP, x[u], y[u]);
                    if (left >= 64 * u + 64) {
                        STP_OUT(co, out, v, 63);
                    } else if (left > 64 * u) {
                        STP_OUT(co, out, v, left - 64 * u - 1);
                    }
                    co[0] += 64;
                }
                e += 512;
                if (e >= ne0) {
                    e = 0;
                    r++;
                }
            }
        }
        return;
    }
#else
    if (flat) {
        // a, b and out are contiguous over the whole tensor
        const int n = ne0 * nrows;
        const int nq = (n + 255) >> 8;
        for (int m = ioff[0]; m < iend[0]; m++) {
            int q0, q1;
            SPLIT_RANGE(nq, nmem, m, q0, q1)
            for (int q = q0; q < q1; q++) {
                const int e = q << 8;
                float64 v[4];
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float64 x = ld_f32(a, off0 + e + 64 * u);
                    const float64 y = ld_f32(b, off1 + e + 64 * u);
                    v[u] = bin_op(op, x, y);
                }
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int left = n - e - 64 * u;
                    if (left > 0) {
                        ST_OUT(out, e + 64 * u, v[u], left);
                    }
                }
            }
        }
        return;
    }
#endif
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
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03 + e0;
            const int pb = off1 + r1 * s11 + r2 * s12 + r3 * s13 + (bc0 ? 0 : e0);
            const float64 x = ld_f32(a, pa);
            float64 y = ld_f32(b, pb);
            if (bc0) {
                y = BCAST0(y, IDX);
            }
            const float64 v = bin_op(OP, x, y);
            const int n = ne0 - e0;
            ST_OUT(out, r * ne0 + e0, v, n);
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

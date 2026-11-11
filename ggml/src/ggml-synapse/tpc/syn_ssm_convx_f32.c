// silu(ssm_conv(concat(state, transpose(x)))): per channel the
// sequence u = [h0 h1 h2 x0 .. x(T-1)] (h = the old conv state); out[c, t] = silu(sum_k<4 w[k, c] u[t + k]),
// tail = the last three u of each channel (the new conv state), in the state layout [3, C, ns].
// x: x[c, t, s] at offx + c + t sxt + s sxs; state h[k + 3 c] at offs + s sss; w tap-major [4][C].
// C % 64 == 0.
// tensors: 0 x  1 st  2 w | 3 out [C, T, ns]  4 tail [3, C, ns]
#include "syn_common.h"

// u[j] (j = t - 3 .. ; j < 0: state column 3 + j) for channels c0 .. c0 + 63
static inline float64 ld_u(tensor x, tensor st, int j, int px, int sxt, int ps)
{
    if (j >= 0) {
        int5 c = {px + j * sxt, 0, 0, 0, 0};
        return v_f32_ld_tnsr_b(c, x);
    }
    // stride-3 gather from the state: lane i <- h[3 (c0 + i) + 3 + j]
    float64 v = 0.0f;
    int5 c = {ps + 3 + j, 0, 0, 0, 0};
    for (int i = 0; i < 64; i++) {
        v = v_f32_ld_tnsr_partial_b(c, st, 0, i, 0, v);
        c[0] += 3;
    }
    return v;
}

static inline float64 conv_silu(float64 w0, float64 w1, float64 w2, float64 w3, float64 u0, float64 u1, float64 u2, float64 u3)
{
    const float64 o = v_f32_mac_b(w3, u3, v_f32_mac_b(w2, u2, v_f32_mac_b(w1, u1, w0 * u0)));
    return o * nl_sigm_f(o);
}

void main(tensor x, tensor st, tensor w, tensor dep, tensor out, tensor tail,
          int nmem, int C, int T, int ns, int offx, int sxt, int sxs, int offs, int sss, int tb, int ntb, int chunk)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int ncb = C >> 6;
    const int nitems = ns * ncb * ntb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        if (i0 >= i1) {
            continue;
        }
        int rest = i0 / ntb;
        int tbk = i0 - rest * ntb;
        int s = rest / ncb;
        int cb = rest - s * ncb;
        for (int it = i0; it < i1; it++) {
            const int c0 = cb << 6;
            const int px = offx + s * sxs + c0, ps = offs + s * sss + 3 * c0;
            int5 cw = {c0, 0, 0, 0, 0};
            const float64 w0 = v_f32_ld_tnsr_b(cw, w);
            cw[0] += C;
            const float64 w1 = v_f32_ld_tnsr_b(cw, w);
            cw[0] += C;
            const float64 w2 = v_f32_ld_tnsr_b(cw, w);
            cw[0] += C;
            const float64 w3 = v_f32_ld_tnsr_b(cw, w);
            const int t0 = tbk * tb;
            const int t1 = t0 + tb < T ? t0 + tb : T;
            float64 u0 = ld_u(x, st, t0 - 3, px, sxt, ps);
            float64 u1 = ld_u(x, st, t0 - 2, px, sxt, ps);
            float64 u2 = ld_u(x, st, t0 - 1, px, sxt, ps);
            int5 cx = {px + t0 * sxt, 0, 0, 0, 0};
            int5 co = {(s * T + t0) * C + c0, 0, 0, 0, 0};
            int t = t0;
            for (; t + 7 < t1; t += 8) {
                float64 xv[8];
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    xv[k] = v_f32_ld_tnsr_b(cx, x);
                    cx[0] += sxt;
                }
                float64 y[8];
                y[0] = conv_silu(w0, w1, w2, w3, u0, u1, u2, xv[0]);
                y[1] = conv_silu(w0, w1, w2, w3, u1, u2, xv[0], xv[1]);
                y[2] = conv_silu(w0, w1, w2, w3, u2, xv[0], xv[1], xv[2]);
                #pragma unroll
                for (int k = 3; k < 8; k++) {
                    y[k] = conv_silu(w0, w1, w2, w3, xv[k - 3], xv[k - 2], xv[k - 1], xv[k]);
                }
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    v_f32_st_tnsr(co, out, y[k]);
                    co[0] += C;
                }
                u0 = xv[5];
                u1 = xv[6];
                u2 = xv[7];
            }
            for (; t < t1; t++) {
                const float64 x0 = v_f32_ld_tnsr_b(cx, x);
                cx[0] += sxt;
                v_f32_st_tnsr(co, out, conv_silu(w0, w1, w2, w3, u0, u1, u2, x0));
                co[0] += C;
                u0 = u1;
                u1 = u2;
                u2 = x0;
            }
            if (t1 == T) {
                // the new state: u[T + k] = (u[T - 3 + k] as indexed above), scattered to h[k + 3 c]
                for (int k = 0; k < 3; k++) {
                    const float64 v = ld_u(x, st, T - 3 + k, px, sxt, ps);
                    int5 ct = {3 * (s * C + c0) + k, 0, 0, 0, 0};
                    for (int i = 0; i < 64; i++) {
                        v_f32_st_tnsr_partial(ct, tail, v, 0, i);
                        ct[0] += 3;
                    }
                }
            }
            if (++tbk == ntb) {
                tbk = 0;
                if (++cb == ncb) {
                    cb = 0;
                    s++;
                }
            }
        }
    }
}

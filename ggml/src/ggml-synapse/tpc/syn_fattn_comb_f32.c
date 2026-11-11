// FLASH_ATTN_EXT output from the partials of syn_fattn.
// part: per (b, h, i, chunk) [acc (nv * 64) | m (64 lanes) | l (64 lanes)]; out contiguous
// [D, H, nq, B] (ggml flash-attn output layout), D = value head size.
// has_sinks: one more logit per head (sinks [H], f32) in the softmax, with no value.
// -DGATE: out = attn * sigmoid(gate), gate a view (element offset, strides per head / query / batch)
// -DOUT_BF16: bf16 output
// -DWIDE: D up to 512, else 256
// tensors: 0 part  [1 gate]  sinks  dep | out
#include "syn_common.h"

#if defined(WIDE)
#define NV 8
#else
#define NV 4
#endif

#if defined(GATE)
void main(tensor part, tensor gate, tensor sinks, tensor dep, tensor out, int nmem, int D, int nq, int H, int B, int nch,
          int has_sinks, int offg, int sgh, int sgi, int sgb)
#else
void main(tensor part, tensor sinks, tensor dep, tensor out, int nmem, int D, int nq, int H, int B, int nch,
          int has_sinks)
#endif
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    if (has_sinks) {
        scalar_cache_reset();
    }
    const int nv = (D + 63) >> 6;
    const int pstride = nv * 64 + 128;
    const int nitems = B * H * nq;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int h = it % H;
            const int bi = it / H;          // b * nq + i
            const int i = bi % nq;
            const int b = bi / nq;
            const int pb = ((b * H + h) * nq + i) * nch * pstride;
            float sk = -1e30f;
            if (has_sinks) {
                int5 cs = {h, 0, 0, 0, 0};
                sk = s_f32_ld_g(gen_addr(cs, sinks));
            }
            float64 M = sk;
            for (int c = 0; c < nch; c++) {
                M = v_f32_max_b(M, ld_f32(part, pb + c * pstride + nv * 64));
            }
            float64 acc[NV];
            #pragma unroll
            for (int u = 0; u < NV; u++) {
                acc[u] = 0.0f;
            }
            float64 l = has_sinks ? nl_exp(sk - M) : 0.0f;
            for (int c = 0; c < nch; c++) {
                const int pc = pb + c * pstride;
                const float64 w = nl_exp(ld_f32(part, pc + nv * 64) - M);
                l = v_f32_mac_b(ld_f32(part, pc + nv * 64 + 64), w, l);
                #pragma unroll
                for (int u = 0; u < NV; u++) {
                    if (u < nv) acc[u] = v_f32_mac_b(ld_f32(part, pc + 64 * u), w, acc[u]);
                }
            }
            const float64 il = v_f32_sel_grt_f32_b(l, 0.0f, nl_recip(v_f32_max_b(l, 1e-30f)), 0.0f);
            const int po = ((b * nq + i) * H + h) * D;
            #pragma unroll
            for (int u = 0; u < NV; u++) {
                if (u < nv) {
                    float64 y = acc[u] * il;
#if defined(GATE)
                    y = y * nl_sigm(ld_f32(gate, offg + h * sgh + i * sgi + b * sgb + 64 * u));
#endif
                    ST_OUT(out, po + 64 * u, y, D - 64 * u);
                }
            }
        }
    }
}

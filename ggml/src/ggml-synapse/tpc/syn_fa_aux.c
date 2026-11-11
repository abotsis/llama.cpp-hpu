// Attention helpers (-DT_F16 / -DT_BF16: element type):
//  -DMEMCPY: out[0, n) = a[off, off + n) (n % 128 == 0)
//            tensors: 0 a  dep | out
//  -DQBD   : block-diagonal queries out [D Hkv, nq H]: column c = iq + nq h holds q[:, iq, h] in rows
//            hk D .. hk D + D - 1 (hk = h / hpg), zeros elsewhere. q: f32 view (offset, strides).
//            tensors: 0 q  1 k  2 v  dep | out (k, v: the K / V caches, not read: the dep orders this kernel
//            after the cache writes)
#include "syn_common.h"

#if defined(T_BF16)
#define VT bfloat128
#define VLD(c, t) v_bf16_ld_tnsr_b(c, t)
#define VST(c, t, v) v_bf16_st_tnsr(c, t, v)
#define VSTP(c, t, v, n) v_bf16_st_tnsr_partial(c, t, v, n, 0)
#define CVT(a, b) cvt_bf16(a, b)
#else
#define VT half128
#define VLD(c, t) v_f16_ld_tnsr_b(c, t)
#define VST(c, t, v) v_f16_st_tnsr(c, t, v)
#define VSTP(c, t, v, n) v_f16_st_tnsr_partial(c, t, v, n, 0)
#define CVT(a, b) cvt_f16(a, b)
#endif

#if defined(MEMCPY)
void main(tensor a, tensor dep, tensor out, int nmem, int n, int off)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nv = n >> 7;
    const int nitems = (nv + 3) >> 2;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int v0 = it << 2;
            int5 c0 = {off + (v0 << 7), 0, 0, 0, 0};
            int5 c1 = c0, c2 = c0, c3 = c0;
            c1[0] += 128; c2[0] += 256; c3[0] += 384;
            const VT x0 = VLD(c0, a), x1 = VLD(c1, a), x2 = VLD(c2, a), x3 = VLD(c3, a);
            int5 d = {v0 << 7, 0, 0, 0, 0};
            VST(d, out, x0);
            if (v0 + 1 < nv) { d[0] += 128; VST(d, out, x1); }
            if (v0 + 2 < nv) { d[0] += 128; VST(d, out, x2); }
            if (v0 + 3 < nv) { d[0] += 128; VST(d, out, x3); }
        }
    }
}
#else
void main(tensor q, tensor k, tensor v, tensor dep, tensor out, int nmem, int D, int Hkv, int H, int nq, int hpg,
          int offq, int sq1, int sq2)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const float64 lane = LANE_F32();
    const int R = D * Hkv;
    const int nch = (R + 127) >> 7;
    const int nitems = nq * H * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int c = it / nch, ch = it - c * nch;
            const int h = c / nq, iq = c - h * nq;
            const int b0 = (h / hpg) * D;                 // first row of the block
            const int r0 = ch << 7;
            const int pq = offq + iq * sq1 + h * sq2 + r0 - b0;
            // lanes of rows r0 + l (lo) and r0 + 64 + l (hi) inside [b0, b0 + D)
            const float lo0 = (float) (b0 - r0), hi0 = (float) (b0 + D - r0);
            float64 lo = ld_f32(q, pq), hi = ld_f32(q, pq + 64);
            lo = v_f32_sel_geq_f32_b(lane, lo0, v_f32_sel_less_f32_b(lane, hi0, lo, 0.0f), 0.0f);
            hi = v_f32_sel_geq_f32_b(lane, lo0 - 64.0f, v_f32_sel_less_f32_b(lane, hi0 - 64.0f, hi, 0.0f), 0.0f);
            int5 d = {c * R + r0, 0, 0, 0, 0};
            const int left = R - r0;
            if (left >= 128) VST(d, out, CVT(lo, hi)); else VSTP(d, out, CVT(lo, hi), left - 1);
        }
    }
}
#endif

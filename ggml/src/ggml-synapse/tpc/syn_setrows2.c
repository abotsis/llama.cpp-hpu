// Two SET_ROWS (the K and V cache writes of one attention layer): for p in {0, 1},
// dst_p[:, idx_p[i]] = DST(src_p[:, i]), rows of ne0 elements, i < n. Both destinations are
// side-effect writes (inputs 0 and 1). idx: i32, or i64 read as i32 pairs (is64).
// tensors: 0 dk  1 dv  2 sk  3 sv  4 ik  5 iv  6 dep | 7 sink
#include "syn_common.h"

#if defined(DST_F16)
#define STROW(t, c, lo, hi, nn) { const half128 y_ = cvt_f16(lo, hi); if ((nn) >= 128) v_f16_st_tnsr(c, t, y_); else v_f16_st_tnsr_partial(c, t, y_, (nn) - 1, 0); }
#elif defined(DST_BF16)
#define STROW(t, c, lo, hi, nn) { const bfloat128 y_ = cvt_bf16(lo, hi); if ((nn) >= 128) v_bf16_st_tnsr(c, t, y_); else v_bf16_st_tnsr_partial(c, t, y_, (nn) - 1, 0); }
#else
#define STROW(t, c, lo, hi, nn) { ST_F32(t, c[0], lo, nn); if ((nn) > 64) ST_F32(t, c[0] + 64, hi, (nn) - 64); }
#endif

void main(tensor dk, tensor dv, tensor sk, tensor sv, tensor ik, tensor iv, tensor dep, tensor sink,
          int nmem, int chunk, int ne0, int n, int is64, float inv_nch,
          int offdk, int sdk, int offsk, int ssk, int offik,
          int offdv, int sdv, int offsv, int ssv, int offiv)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nch = (ne0 + 127) >> 7;
    const int nitems = 2 * n * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_CHUNK(nitems, chunk, m, i0, i1)
        #pragma unroll 1
        for (int it = i0; it < i1; it++) {
            const int pv = it >= n * nch;
            const int x = pv ? it - n * nch : it;
            const int r = FDIV(x, inv_nch);
            const int e0 = (x - r * nch) << 7;
            const int nn = ne0 - e0;
            if (pv) {
                int5 ci = {is64 ? 2 * (offiv + r) : offiv + r, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, iv));
                int5 cs = {offsv + r * ssv + e0, 0, 0, 0, 0};
                const float64 lo = v_f32_ld_tnsr_b(cs, sv);
                cs[0] += 64;
                const float64 hi = v_f32_ld_tnsr_b(cs, sv);
                int5 cd = {offdv + row * sdv + e0, 0, 0, 0, 0};
                STROW(dv, cd, lo, hi, nn)
            } else {
                int5 ci = {is64 ? 2 * (offik + r) : offik + r, 0, 0, 0, 0};
                const int row = s_i32_ld_g((__global int *) gen_addr(ci, ik));
                int5 cs = {offsk + r * ssk + e0, 0, 0, 0, 0};
                const float64 lo = v_f32_ld_tnsr_b(cs, sk);
                cs[0] += 64;
                const float64 hi = v_f32_ld_tnsr_b(cs, sk);
                int5 cd = {offdk + row * sdk + e0, 0, 0, 0, 0};
                STROW(dk, cd, lo, hi, nn)
            }
        }
    }
    if (ioff[0] == 0) {
        int5 z = {0, 0, 0, 0, 0};
        v_f32_st_tnsr(z, sink, 0.0f);
    }
}

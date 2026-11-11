// ARGSORT of long rows: bitonic sort of (key, index) pairs, descending (asc: keys negated), ties by
// index. Rows padded to N = 2^m >= 4096 elements with (-inf, index >= n). Element i = 64 v + lane.
//  -DINIT: per 4096-element block: load, then all stages of sizes 2 .. 4096
//  -DGLOB: one stage (s, j >= 4096) over the rows: compare-exchange vectors v and v + j / 64
//  -DLOC : per block, the stages j = 2048 .. 1 of size s >= 8192
//  else  : out[r * n + p] = index p of the sorted row (i32), p < n
// tensors: INIT 0 a dep | V I;  GLOB / LOC 0 V 1 I dep | V' I';  OUT 0 I dep | out
#include "syn_common.h"

#if defined(INIT) || defined(LOC)
__local__ float64 X[64];
__local__ float64 XI[64];
#endif

// before(a, b): a sorts first (larger key; equal keys: smaller index), as 1 / 0
static inline float64 before(float64 a, float64 ai, float64 b, float64 bi)
{
    return v_f32_sel_grt_f32_b(a, b, (float64) 1.0f,
                               v_f32_sel_eq_f32_b(a, b, v_f32_sel_less_f32_b(ai, bi, (float64) 1.0f, (float64) 0.0f), (float64) 0.0f));
}

// compare-exchange of x (first) and y (second) for direction d (1: first sorts first)
#define CX(x, xi, y, yi, d)                                                     \
    {                                                                           \
        const float64 b_ = before(x, xi, y, yi);                                \
        const float64 k_ = v_f32_sel_eq_f32_b(b_, d, (float64) 1.0f, (float64) 0.0f); \
        const float64 nx_ = v_f32_sel_eq_f32_b(k_, 1.0f, x, y);                 \
        const float64 nxi_ = v_f32_sel_eq_f32_b(k_, 1.0f, xi, yi);              \
        y = v_f32_sel_eq_f32_b(k_, 1.0f, y, x);                                 \
        yi = v_f32_sel_eq_f32_b(k_, 1.0f, yi, xi);                              \
        x = nx_;                                                                \
        xi = nxi_;                                                              \
    }

#if defined(INIT) || defined(LOC)
// lane l takes lane l ^ j (j < 64)
static inline float64 perm(float64 x, int j, float64 lp16)
{
    if (j == 32) {
        return v_f32_mov_dual_group_all_b(x, 0xFFFFFFFF, 2, 3, 0, 1, WR_ALL, undef_f64());
    }
    if (j == 16) {
        return v_f32_mov_dual_group_all_b(x, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, undef_f64());
    }
    float64 r = x;
    #pragma unroll
    for (int e = 0; e < 16; e++) {
        const uchar256 ie = SHUF_IDX(e);
        r = v_f32_sel_eq_f32_b(lp16, (float) (e ^ j), SHUF(x, ie), r);
    }
    return r;
}

// one stage (s, j) on the block at element base `base`
static inline void stage(int s, int j, int base, int64 li, float64 lp16)
{
    if (j >= 64) {
        const int jv = j >> 6;
        for (int v = 0; v < 64; v++) {
            if (v & jv) {
                continue;
            }
            const float64 d = ((base + 64 * v) & s) == 0 ? 1.0f : 0.0f;
            float64 x = X[v], xi = XI[v], y = X[v + jv], yi = XI[v + jv];
            CX(x, xi, y, yi, d)
            X[v] = x; XI[v] = xi; X[v + jv] = y; XI[v + jv] = yi;
        }
        return;
    }
    // within vectors: the lane with bit j clear is the pair's first element
    const float64 low = v_f32_sel_eq_i32_b(li & j, (int64) 0, (float64) 1.0f, (float64) 0.0f);
    const float64 dl = v_f32_sel_eq_i32_b(li & s, (int64) 0, (float64) 1.0f, (float64) 0.0f);
    for (int v = 0; v < 64; v++) {
        const float64 d = s < 64 ? dl : (((base + 64 * v) & s) == 0 ? (float64) 1.0f : (float64) 0.0f);
        const float64 x = X[v], xi = XI[v];
        const float64 p = perm(x, j, lp16), pi = perm(xi, j, lp16);
        // keep self when (self first) == (self sorts first under d)
        const float64 wf = v_f32_sel_eq_f32_b(low, d, (float64) 1.0f, (float64) 0.0f);
        const float64 b = before(x, xi, p, pi);
        const float64 k = v_f32_sel_eq_f32_b(b, wf, (float64) 1.0f, (float64) 0.0f);
        X[v] = v_f32_sel_eq_f32_b(k, 1.0f, x, p);
        XI[v] = v_f32_sel_eq_f32_b(k, 1.0f, xi, pi);
    }
}
#endif

#if defined(INIT)
void main(tensor a, tensor dep, tensor V, tensor I,
          int nmem, int n, int N, int ne1, int ne2, int ne3, int off0, int s01, int s02, int s03, int desc)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int64 li = (int64) read_lane_id_4b_b();
    const float64 lane = v_convert_i32_to_f32_b(li, SW_RHNE, 0, 1, 0);
    const float64 lp16 = v_convert_i32_to_f32_b(li & 15, SW_RHNE, 0, 1, 0);
    const float64 ninf = as_float64((int64) 0xFF800000);
    const int nb = N >> 12;
    const int nitems = ne1 * ne2 * ne3 * nb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nb, blk = it - r * nb;
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + r1 * s01 + r2 * s02 + r3 * s03;
            const int base = blk << 12;
            for (int v = 0; v < 64; v++) {
                const int e = base + 64 * v;
                float64 x = ld_f32(a, pa + e);
                if (!desc) {
                    x = -x;
                }
                X[v] = v_f32_sel_less_f32_b(lane, (float) (n - e), x, ninf);
                XI[v] = lane + (float) e;
            }
            for (int s = 2; s <= 4096; s <<= 1) {
                for (int j = s >> 1; j > 0; j >>= 1) {
                    stage(s, j, base, li, lp16);
                }
            }
            for (int v = 0; v < 64; v++) {
                int5 c = {r * N + base + 64 * v, 0, 0, 0, 0};
                v_f32_st_tnsr(c, V, X[v]);
                v_f32_st_tnsr(c, I, XI[v]);
            }
        }
    }
}
#elif defined(LOC)
void main(tensor V, tensor I, tensor dep, tensor V2, tensor I2, int nmem, int N, int rows, int s)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int64 li = (int64) read_lane_id_4b_b();
    const float64 lp16 = v_convert_i32_to_f32_b(li & 15, SW_RHNE, 0, 1, 0);
    const int nb = N >> 12;
    const int nitems = rows * nb;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nb, blk = it - r * nb;
            const int base = blk << 12;
            for (int v = 0; v < 64; v++) {
                const int p = r * N + base + 64 * v;
                X[v] = ld_f32(V, p);
                XI[v] = ld_f32(I, p);
            }
            for (int j = 2048; j > 0; j >>= 1) {
                stage(s, j, base, li, lp16);
            }
            for (int v = 0; v < 64; v++) {
                int5 c = {r * N + base + 64 * v, 0, 0, 0, 0};
                v_f32_st_tnsr(c, V2, X[v]);
                v_f32_st_tnsr(c, I2, XI[v]);
            }
        }
    }
}
#elif defined(GLOB)
void main(tensor V, tensor I, tensor dep, tensor V2, tensor I2, int nmem, int N, int rows, int s, int j)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int jv = j >> 6;
    const int nh = N >> 7;   // vector pairs per row
    const int nitems = rows * nh;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nh, q = it - r * nh;
            const int v = (q / jv) * 2 * jv + (q % jv);
            const float64 d = ((64 * v) & s) == 0 ? 1.0f : 0.0f;
            const int pa = r * N + 64 * v, pb = pa + 64 * jv;
            float64 x = ld_f32(V, pa), xi = ld_f32(I, pa), y = ld_f32(V, pb), yi = ld_f32(I, pb);
            CX(x, xi, y, yi, d)
            int5 c = {pa, 0, 0, 0, 0};
            v_f32_st_tnsr(c, V2, x);
            v_f32_st_tnsr(c, I2, xi);
            c[0] = pb;
            v_f32_st_tnsr(c, V2, y);
            v_f32_st_tnsr(c, I2, yi);
        }
    }
}
#else
void main(tensor I, tensor dep, tensor out, int nmem, int n, int N, int rows)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nc = (n + 63) >> 6;
    const int nitems = rows * nc;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int r = it / nc, c = it - r * nc;
            const int64 v = v_convert_f32_to_i32_b(ld_f32(I, r * N + 64 * c), SW_RHNE, 0, 1, 0);
            int5 co = {r * n + 64 * c, 0, 0, 0, 0};
            const int left = n - 64 * c;
            if (left >= 64) v_i32_st_tnsr(co, out, v); else v_i32_st_tnsr_partial(co, out, v, left - 1, 0);
        }
    }
}
#endif

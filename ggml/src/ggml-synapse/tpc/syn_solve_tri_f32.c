// SOLVE_TRI: X = A^-1 B for lower-triangular A (left, non-unit diagonal: the only variant ggml_solve_tri builds), f32.
// A {N, N, B2, B3} contiguous, B {K, N, B2, B3} contiguous (RHS columns), out X {K, N, B2, B3} contiguous.
// Forward substitution per (batch, 64 RHS columns), all N rows in one pass (N <= SRN):
//   X[r, c] = (B[r, c] - sum_{t < r} A[r, t] * X[t, c]) / A[r, r]
// The solved rows stay in local memory (never re-read from global). A zero diagonal gives inf / NaN as the CPU.
// tensors: 0 A  1 B  2 dep | 3 X
// params: nmem, N, K, B2, B3, offA, offB
#include "syn_common.h"

#define SRN 256
__local__ float64 XV[SRN];

void main(tensor A, tensor B, tensor dep, tensor X, int nmem, int N, int K, int B2, int B3, int offA, int offB)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nch = (K + 63) >> 6;
    const int nitems = B2 * B3 * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int ch = it % nch;
            const int bt = it / nch;                   // batch r2 + B2 r3
            const int c0 = ch << 6;
            const int ncol = K - c0 < 64 ? K - c0 : 64;
            const int pa = offA + bt * N * N;
            const int pb = offB + bt * N * K + c0;
            const int px = bt * N * K + c0;
            for (int i = 0; i < N; i++) {
                float64 acc = 0.0f;
                for (int t = 0; t < i; t++) {
                    int5 ca = {pa + i * N + t, 0, 0, 0, 0};
                    const float a = s_f32_ld_g((__global float *) gen_addr(ca, A));
                    acc = v_f32_mac_b(XV[t], a, acc);
                }
                int5 cd = {pa + i * N + i, 0, 0, 0, 0};
                const float diag = s_f32_ld_g((__global float *) gen_addr(cd, A));
                const float64 x = (ld_f32(B, pb + i * K) - acc) * (1.0f / diag);
                XV[i] = x;
                ST_F32(X, px + i * K, x, ncol);
            }
        }
    }
}

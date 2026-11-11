// COUNT_EQUAL: out = |a == b| over same-shape I32 inputs, I64 result (as two I32 words).
// -DPARTIAL: one work item per member; member m counts the equal elements of its flat range
// [a, b) of [0, total) (SPLIT_RANGE over total; the host keeps every member's range below 2^31
// elements so unsigned I32 counts cannot overflow) and writes its unsigned count to partial[m].
// No floats, no atomics, no 2^24 ceiling: pure unsigned I32 adds and compares.
// FINAL: a single work item (index space 1, host passes nmem = 1) adds the partial counts in
// ascending order with an explicit two-word unsigned carry accumulator (lo / hi u32;
// nl = lo + p, carry = nl < lo, hi += carry) and writes the low word at out[0] and the high
// word at out[1] (the backend's I64-as-I32-pairs representation). Scalar loads of the partials
// after a scalar_cache_reset (the partial kernel wrote them with vector stores).
// tensors: PARTIAL 0 a  1 b | 2 partial (nmem u32)     FINAL 0 partial | 1 out (I64: 2 dwords)
// params: PARTIAL nmem, total / FINAL nmem, np (the number of partials)  - 2 dwords
#include "syn_common.h"

#if defined(PARTIAL)
void main(tensor a, tensor b, tensor dep, tensor partial, int nmem, int total)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const uint64 lane = (uint64) LANE_I32();     // lane index 0..63 in every lane
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(total, nmem, m, r0, r1e)
        if (r0 >= r1e) {
            continue;
        }
        uint64 acc = 0;
        for (int e = r0; e < r1e; e += 64) {
            const int n = r1e - e < 64 ? r1e - e : 64;
            int5 c = {e, 0, 0, 0, 0};
            const uint64 x = v_u32_ld_tnsr_b(c, a);
            const uint64 y = v_u32_ld_tnsr_b(c, b);
            // per-lane equality; the masked tail contributes nothing
            uint64 eq = v_u32_sel_eq_u32_b(x, y, (uint64) 1, (uint64) 0);
            eq = v_u32_sel_less_u32_b(lane, (uint64) n, eq, (uint64) 0);
            acc = v_u32_add_b(eq, acc);
        }
        // reduce the 64 u32 lanes
        acc = v_u32_add_b(acc, v_u32_mov_dual_group_all_b(acc, 0xFFFFFFFF, 1, 0, 3, 2, WR_ALL, (uint64) 0));
        acc = v_u32_add_b(acc, v_u32_mov_dual_group_all_b(acc, 0xFFFFFFFF, 2, 3, 0, 1, WR_ALL, (uint64) 0));
        DECL_IDX(IDX)
        uint64 s = v_u32_shuffle_b(acc, IDX[0], 0, (uint64) 0);
        _Pragma("unroll")
        for (int e_ = 1; e_ < 16; e_++) {
            s = v_u32_add_b(s, v_u32_shuffle_b(acc, IDX[e_], 0, (uint64) 0));
        }
        // every lane holds the range's count; store lane 0 as partial[m]
        int5 c = {m, 0, 0, 0, 0};
        v_u32_st_tnsr_partial(c, partial, s, 0, 0);
    }
}
#else
void main(tensor partial, tensor dep, tensor out, int nmem, int np)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    if (ioff[0] != 0) {
        return;
    }
    // two-word unsigned accumulator: value = hi * 2^32 + lo
    unsigned int lo = 0, hi = 0;
    for (int m = 0; m < np; m++) {
        int5 c = {m, 0, 0, 0, 0};
        const unsigned int p = s_u32_ld_g((__global unsigned int *) gen_addr(c, partial));
        const unsigned int nl = lo + p;
        // unsigned wrap: the new lo is below the old lo -> carry into hi
        hi += nl < lo ? 1u : 0u;
        lo = nl;
    }
    // the I64 result as two I32 words: low at 0, high at 1
    int5 c0 = {0, 0, 0, 0, 0};
    s_u32_st_g((__global unsigned int *) gen_addr(c0, out), lo);
    int5 c1 = {1, 0, 0, 0, 0};
    s_u32_st_g((__global unsigned int *) gen_addr(c1, out), hi);
}
#endif

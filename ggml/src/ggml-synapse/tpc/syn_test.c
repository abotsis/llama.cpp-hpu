// scratch kernel for on-device semantics probes (ktest harness)
#include "syn_common.h"

void main(tensor in, tensor dep, tensor out, int nmem, int mode)
{
    const int5 ioff = get_index_space_offset();
    if (ioff[0] != 0) {
        return;
    }
    int5 c0 = {0, 0, 0, 0, 0};
    if (mode == 0) {
        // int8 load -> float256 (all)
        const char256 c = v_i8_ld_tnsr_b(c0, in);
        const float256 f = v_convert_i8_to_f32_all_b(c);
        ST_F32(out, 0, f.v1, 64); ST_F32(out, 64, f.v2, 64); ST_F32(out, 128, f.v3, 64); ST_F32(out, 192, f.v4, 64);
    } else if (mode == 1) {
        const char256 c = v_i8_ld_tnsr_b(c0, in);
        const float256 f = convert_char256_to_float256(c, SW_LINEAR);
        ST_F32(out, 0, f.v1, 64); ST_F32(out, 64, f.v2, 64); ST_F32(out, 128, f.v3, 64); ST_F32(out, 192, f.v4, 64);
    } else if (mode == 2) {
        // float256 (v_k lane = 64k + lane - 128) -> int8 store
        float256 g;
        g.v1 = LANE_F32() - 128.0f; g.v2 = LANE_F32() - 64.0f; g.v3 = LANE_F32(); g.v4 = LANE_F32() + 64.0f;
        const char256 c = v_convert_f32_to_i8_all_b(g, SW_RHNE);
        v_i8_st_tnsr(c0, out, c);
    } else if (mode == 3) {
        float256 g;
        g.v1 = LANE_F32() - 128.0f; g.v2 = LANE_F32() - 64.0f; g.v3 = LANE_F32(); g.v4 = LANE_F32() + 64.0f;
        const char256 c = convert_float256_to_char256(g, SW_RHNE | SW_LINEAR);
        v_i8_st_tnsr(c0, out, c);
    } else if (mode == 5) {
        // partial store with source offset 32, size 3, at coordinate 0
        v_f32_st_tnsr_partial(c0, out, LANE_F32(), 3, 32);
    } else if (mode == 6) {
        // int8 partial store: lanes 32..33 at coordinate 8
        float256 g;
        g.v1 = LANE_F32(); g.v2 = 0.0f; g.v3 = 0.0f; g.v4 = 0.0f;
        const char256 c = convert_float256_to_char256(g, SW_RHNE | SW_LINEAR);
        int5 c8 = {8, 0, 0, 0, 0};
        v_i8_st_tnsr_partial(c8, out, c, 1, 32);
    } else if (mode == 4) {
        // scalar f16 load from an int8 tensor at byte 2
        int5 cc = {2, 0, 0, 0, 0};
        const half hv = s_f16_ld_g(gen_addr(cc, in));
        ST_F32(out, 0, (float64) (float) hv, 64);
    }
}

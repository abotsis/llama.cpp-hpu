// LIGHTNING_INDEXER, from the f32 product s[kv, h + H t, st] = k_kv . q_{h,t} (one block of
// qb tokens starting at t0): out[kv, t, st] = sum_h relu(s) w[h, t0 + t, st] + mask[kv, t0 + t, st % nm3].
// out [nkv, qb, 1, ns] contiguous.
// tensors: 0 s  1 w  2 mask | 3 out
#include "syn_common.h"

void main(tensor s, tensor w, tensor mask, tensor dep, tensor out,
          int nmem, int nkv, int H, int qb, int ns, int t0,
          int offw, int sw1, int sw3, int offm, int sm1, int sm3, int nm3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    const int nch = (nkv + 63) >> 6;
    const int nitems = ns * qb * nch;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int ch = it % nch, rest = it / nch;
            const int t = rest % qb, st = rest / qb;
            const int kv0 = ch << 6;
            const float64 wv = ld_f32(w, offw + (t0 + t) * sw1 + st * sw3);
            const int ps = (st * qb + t) * H * nkv + kv0;
            float64 a0 = 0.0f, a1 = 0.0f;
            for (int h = 0; h < H; h += 2) {
                float64 wd;
                if (h < 16) wd = BCAST_DG(wv, 0);
                else if (h < 32) wd = BCAST_DG(wv, 1);
                else if (h < 48) wd = BCAST_DG(wv, 2);
                else wd = BCAST_DG(wv, 3);
                const float64 x0 = v_f32_max_b(ld_f32(s, ps + h * nkv), 0.0f);
                a0 = v_f32_mac_b(x0, SHUF(wd, IDX[h & 15]), a0);
                if (h + 1 < H) {
                    const float64 x1 = v_f32_max_b(ld_f32(s, ps + (h + 1) * nkv), 0.0f);
                    a1 = v_f32_mac_b(x1, SHUF(wd, IDX[(h + 1) & 15]), a1);
                }
            }
            int5 cm = {offm + (t0 + t) * sm1 + (st % nm3) * sm3 + kv0, 0, 0, 0, 0};
            const float64 mk = f16_to_f32(v_f16_ld_tnsr_b(cm, mask)).v1;
            ST_F32(out, (st * qb + t) * nkv + kv0, (a0 + a1) + mk, nkv - kv0);
        }
    }
}

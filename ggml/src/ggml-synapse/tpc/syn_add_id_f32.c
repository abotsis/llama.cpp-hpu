// ADD_ID: out[:, i1, i2] = a[:, i1, i2] + b[:, ids[i1, i2]], f32. a is a view; b rows of b's
// stride sb1 from offb; ids i32 view; out contiguous [ne0, ne1, ne2].
// tensors: 0 a  1 b  2 ids | 3 out
#include "syn_common.h"

void main(tensor a, tensor b, tensor ids, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int offa, int sa1, int sa2, int offb, int sb1,
          int offi, int si0, int si1)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    scalar_cache_reset();
    const int nrows = ne1 * ne2;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int i2 = r / ne1;
            const int i1 = r - i2 * ne1;
            int5 ci = {offi + i1 * si0 + i2 * si1, 0, 0, 0, 0};
            const int e = s_i32_ld_g((__global int *) gen_addr(ci, ids));
            const int pa = offa + i1 * sa1 + i2 * sa2;
            const int pb = offb + e * sb1;
            for (int c = 0; c < ne0; c += 64) {
                ST_F32(out, r * ne0 + c, ld_f32(a, pa + c) + ld_f32(b, pb + c), ne0 - c);
            }
        }
    }
}

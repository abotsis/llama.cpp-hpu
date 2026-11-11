// out = repeat(a) to [ne0, ne1, ne2, ne3] (contiguous), f32: element i of each dim reads
// a's element i mod sne_d. a is a view (element offset + strides of dims 1..3, dim 0 contiguous).
// tensors: 0 a | 1 out
#include "syn_common.h"

void main(tensor a, tensor dep, tensor out,
          int nmem, int ne0, int ne1, int ne2, int ne3, int sne0, int sne1, int sne2, int sne3,
          int off0, int s01, int s02, int s03)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int nrows = ne1 * ne2 * ne3;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int r0, r1e;
        SPLIT_RANGE(nrows, nmem, m, r0, r1e)
        for (int r = r0; r < r1e; r++) {
            const int r3 = r / (ne1 * ne2);
            const int rr = r - r3 * ne1 * ne2;
            const int r2 = rr / ne1;
            const int r1 = rr - r2 * ne1;
            const int pa = off0 + (r1 % sne1) * s01 + (r2 % sne2) * s02 + (r3 % sne3) * s03;
            const int po = r * ne0;
            for (int k = 0; k < ne0; k += sne0) {
                for (int e = 0; e < sne0; e += 64) {
                    ST_F32(out, po + k + e, ld_f32(a, pa + e), sne0 - e);
                }
            }
        }
    }
}

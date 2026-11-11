// Dependency join: no data, one token output written after all four inputs exist.
// tensors: 0..3 any | 4 token
#include "syn_common.h"

void main(tensor a, tensor b, tensor c, tensor d, tensor tok, int nmem)
{
    const int5 ioff = get_index_space_offset();
    if (ioff[0] == 0) {
        int5 z = {0, 0, 0, 0, 0};
        v_f32_st_tnsr(z, tok, 0.0f);
    }
}

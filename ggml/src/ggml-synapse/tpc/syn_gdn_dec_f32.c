// Gated delta net layer of qwen35-style models, one token per sequence:
//   q, k = rms_norm(.) * qk_scale  (the l2 norm as built by the graph), per q/k head
//   beta = sigmoid(b[h]);  eg = exp(softplus(a[h] + dt[h]) * A[h])
//   state rows M[j] (ggml layout, row j contiguous over the key index), read from the cache at
//   row idx[s] and written back at row drow + s (side-effect write, input 0):
//     M[j] *= eg;  d = (v[j] - M[j].k) * beta;  M[j] += d k;  o[j] = M[j].q * (1/sqrt(S))
//   out = rms_norm(o) * w * silu(z), or * sigmoid(z) with zsig (qwen4exp)   (f32, or bf16 with -DOUT_BF16)
// S = 128. Views are element offsets; q/k/v/z/a/b heads are contiguous (stride per head).
// tensors: 0 R  1 idx  2 y (conv output: q|k|v)  3 a  4 b (gate sources)  5 dt  6 A
//          7 z  8 w  9 dep | 10 out
#include "syn_common.h"

#define DS 128

__local__ float64 SV[2 * DS];

void main(tensor R, tensor idx, tensor y, tensor ta, tensor tb, tensor dt, tensor A, tensor z, tensor w,
          tensor dep, tensor out,
          int nmem, int H, int ns, int Hk, int srow, int offi, int drow,
          int offq, int offk, int offv, int sy,
          int offa, int sa, int offb, int sb, int offdt, int offA,
          int offz, int sz, int offw,
          float eps_qk, float qk_scale, float eps_o, int zsig)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const float inv = 1.0f / (float) DS;
    const float oscale = 0.08838834764831845f;   // 1/sqrt(128)
    const int nitems = ns * H;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int s = it / H;
            const int h = it - s * H;
            int hk = h;
            while (hk >= Hk) hk -= Hk;
            int5 ci = {offi + s, 0, 0, 0, 0};
            const int srcrow = s_i32_ld_g((__global int *) gen_addr(ci, idx));
            const int psrc = srcrow * srow + h * DS * DS;
            const int pdst = (drow + s) * srow + h * DS * DS;
            const int py = s * sy;
            // q, k with their norms
            float64 q0 = ld_f32(y, py + offq + hk * DS), q1 = ld_f32(y, py + offq + hk * DS + 64);
            float64 k0 = ld_f32(y, py + offk + hk * DS), k1 = ld_f32(y, py + offk + hk * DS + 64);
            float64 sq, sk;
            NL_REDUCE_ADD(v_f32_mac_b(q0, q0, q1 * q1), IDX, sq);
            NL_REDUCE_ADD(v_f32_mac_b(k0, k0, k1 * k1), IDX, sk);
            const float64 rq = nl_rsqrt(sq * inv + eps_qk) * (qk_scale * oscale);
            const float64 rk = nl_rsqrt(sk * inv + eps_qk) * qk_scale;
            q0 = q0 * rq; q1 = q1 * rq;
            k0 = k0 * rk; k1 = k1 * rk;
            const float64 v0 = ld_f32(y, py + offv + h * DS), v1 = ld_f32(y, py + offv + h * DS + 64);
            // gates
            const float64 av = BCAST0(ld_f32(ta, s * sa + offa + h), IDX);
            const float64 bv = BCAST0(ld_f32(tb, s * sb + offb + h), IDX);
            const float64 dtv = BCAST0(ld_f32(dt, offdt + h), IDX);
            const float64 Av = BCAST0(ld_f32(A, offA + h), IDX);
            const float64 eg = nl_exp(nl_softplus(av + dtv) * Av);
            const float64 beta = nl_sigm(bv);
            // d_j = (v_j - M_j.k) * beta for every decayed row
            float64 dv0 = 0.0f, dv1 = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < DS; j += 4) {
                const int pr = psrc + j * DS;
                const float64 a00 = ld_f32(R, pr) * eg,           a01 = ld_f32(R, pr + 64) * eg;
                const float64 a10 = ld_f32(R, pr + DS) * eg,      a11 = ld_f32(R, pr + DS + 64) * eg;
                const float64 a20 = ld_f32(R, pr + 2 * DS) * eg,  a21 = ld_f32(R, pr + 2 * DS + 64) * eg;
                const float64 a30 = ld_f32(R, pr + 3 * DS) * eg,  a31 = ld_f32(R, pr + 3 * DS + 64) * eg;
                SV[2 * j] = a00;     SV[2 * j + 1] = a01;
                SV[2 * j + 2] = a10; SV[2 * j + 3] = a11;
                SV[2 * j + 4] = a20; SV[2 * j + 5] = a21;
                SV[2 * j + 6] = a30; SV[2 * j + 7] = a31;
                float64 d0, d1, d2, d3;
                NL_REDUCE4_ADD(v_f32_mac_b(a00, k0, a01 * k1), v_f32_mac_b(a10, k0, a11 * k1),
                               v_f32_mac_b(a20, k0, a21 * k1), v_f32_mac_b(a30, k0, a31 * k1), IDX, d0, d1, d2, d3)
                const float jf = (float) (j & 63);
                float64 dv = j < 64 ? dv0 : dv1;
                dv = v_f32_sel_eq_f32_b(lane, jf, d0, dv);
                dv = v_f32_sel_eq_f32_b(lane, jf + 1.0f, d1, dv);
                dv = v_f32_sel_eq_f32_b(lane, jf + 2.0f, d2, dv);
                dv = v_f32_sel_eq_f32_b(lane, jf + 3.0f, d3, dv);
                if (j < 64) dv0 = dv; else dv1 = dv;
            }
            dv0 = (v0 - dv0) * beta;
            dv1 = (v1 - dv1) * beta;
            // M_j += d_j k, o_j = M_j.q, rows stored back
            float64 o0 = 0.0f, o1 = 0.0f;
            #pragma unroll 1
            for (int j = 0; j < DS; j += 4) {
                const int jj = j & 63;
                const float64 dd = j < 64 ? dv0 : dv1;
                float64 db;
                if (jj < 16) db = BCAST_DG(dd, 0);
                else if (jj < 32) db = BCAST_DG(dd, 1);
                else if (jj < 48) db = BCAST_DG(dd, 2);
                else db = BCAST_DG(dd, 3);
                const int e = jj & 15;
                const float64 d0 = SHUF(db, IDX[e]);
                const float64 d1 = SHUF(db, IDX[e + 1]);
                const float64 d2 = SHUF(db, IDX[e + 2]);
                const float64 d3 = SHUF(db, IDX[e + 3]);
                const float64 a00 = v_f32_mac_b(k0, d0, SV[2 * j]),     a01 = v_f32_mac_b(k1, d0, SV[2 * j + 1]);
                const float64 a10 = v_f32_mac_b(k0, d1, SV[2 * j + 2]), a11 = v_f32_mac_b(k1, d1, SV[2 * j + 3]);
                const float64 a20 = v_f32_mac_b(k0, d2, SV[2 * j + 4]), a21 = v_f32_mac_b(k1, d2, SV[2 * j + 5]);
                const float64 a30 = v_f32_mac_b(k0, d3, SV[2 * j + 6]), a31 = v_f32_mac_b(k1, d3, SV[2 * j + 7]);
                const int pw = pdst + j * DS;
                ST_F32(R, pw, a00, 64);           ST_F32(R, pw + 64, a01, 64);
                ST_F32(R, pw + DS, a10, 64);      ST_F32(R, pw + DS + 64, a11, 64);
                ST_F32(R, pw + 2 * DS, a20, 64);  ST_F32(R, pw + 2 * DS + 64, a21, 64);
                ST_F32(R, pw + 3 * DS, a30, 64);  ST_F32(R, pw + 3 * DS + 64, a31, 64);
                float64 r0, r1, r2, r3;
                NL_REDUCE4_ADD(v_f32_mac_b(a00, q0, a01 * q1), v_f32_mac_b(a10, q0, a11 * q1),
                               v_f32_mac_b(a20, q0, a21 * q1), v_f32_mac_b(a30, q0, a31 * q1), IDX, r0, r1, r2, r3)
                const float jf = (float) jj;
                float64 o = j < 64 ? o0 : o1;
                o = v_f32_sel_eq_f32_b(lane, jf, r0, o);
                o = v_f32_sel_eq_f32_b(lane, jf + 1.0f, r1, o);
                o = v_f32_sel_eq_f32_b(lane, jf + 2.0f, r2, o);
                o = v_f32_sel_eq_f32_b(lane, jf + 3.0f, r3, o);
                if (j < 64) o0 = o; else o1 = o;
            }
            // gated rms norm
            float64 so;
            NL_REDUCE_ADD(v_f32_mac_b(o0, o0, o1 * o1), IDX, so);
            const float64 ro = nl_rsqrt(so * inv + eps_o);
            const int pz = s * sz + offz + h * DS;
            const float64 z0 = ld_f32(z, pz), z1 = ld_f32(z, pz + 64);
            const float64 y0 = o0 * ro * ld_f32(w, offw) * (zsig ? nl_sigm(z0) : nl_silu(z0));
            const float64 y1 = o1 * ro * ld_f32(w, offw + 64) * (zsig ? nl_sigm(z1) : nl_silu(z1));
            const int po = (s * H + h) * DS;
#if defined(OUT_BF16)
            int5 co = {po, 0, 0, 0, 0};
            v_bf16_st_tnsr(co, out, cvt_bf16(y0, y1));
#else
            ST_F32(out, po, y0, 64);
            ST_F32(out, po + 64, y1, 64);
#endif
        }
    }
}

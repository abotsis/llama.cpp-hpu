// Gated delta net of a short batch (T tokens per sequence: MTP verify), the multi-token form of
// syn_gdn_dec_f32 (same graph pattern, S = 128):
// -DDEC: per token: q, k rms-normed (* qk_scale), beta = sigmoid(b), eg = exp(softplus(a + dt) * A);
//        for the state rows M[j] (ggml layout, contiguous over the key index):
//          M[j] *= eg;  d = (v[j] - M[j].k) * beta;  M[j] += d k;  o[j] = M[j].q / sqrt(S)
//        The rows are read from the cache at row idx[s]; the state after token T-1-r is written at row
//        drow + s + r * srs for r < K (side-effect write, input 0).
//        out: the raw o [S, H, T, ns] (the gated norm is -DONORM).
//        y (conv output) token t of sequence s at s * sy + t * syt; a / b: s * sa + t * sat + h.
//        tensors: 0 R  1 idx  2 y  3 a  4 b  5 dt  6 A  7 dep | 8 out
// -DONORM: out = rms_norm(o) * w * silu(z) (sigmoid(z) with zsig) per (token, head) row of 128; f32, or bf16
//        with -DOUT_BF16. z row (s, t, h) at offz + s * sz + t * szt + h * S.
//        tensors: 0 o  1 z  2 w  3 dep | 4 out
#include "syn_common.h"

#define DS 128
#define RQ 32

#if defined(DEC)
__local__ float64 SV[2 * RQ];

void main(tensor R, tensor idx, tensor y, tensor ta, tensor tb, tensor dt, tensor A, tensor dep, tensor out,
          int nmem, int H, int ns, int Hk, int srow, int offi, int drow,
          int offq, int offk, int offv, int sy, int syt,
          int offa, int sa, int sat, int offb, int sb, int sbt, int offdt, int offA,
          int T, int K, int srs, float eps_qk, float qk_scale)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const float inv = 1.0f / (float) DS;
    const float oscale = 0.08838834764831845f;   // 1/sqrt(128)
    const int nq = DS / RQ;
    const int nitems = ns * H * nq;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int sh = it / nq;
            const int j0 = (it - sh * nq) * RQ;
            const int s = sh / H;
            const int h = sh - s * H;
            int hk = h;
            while (hk >= Hk) hk -= Hk;
            int5 ci = {offi + s, 0, 0, 0, 0};
            const int srcrow = s_i32_ld_g((__global int *) gen_addr(ci, idx));
            const int psrc = srcrow * srow + (h * DS + j0) * DS;
            const int hb = j0 & 64;           // the 64-row half holding the item's rows
            const int lb = j0 & 63;           // their first lane in it
            #pragma unroll 1
            for (int j = 0; j < RQ; j++) {
                SV[2 * j] = ld_f32(R, psrc + j * DS);
                SV[2 * j + 1] = ld_f32(R, psrc + j * DS + 64);
            }
            const float64 dtv = BCAST0(ld_f32(dt, offdt + h), IDX);
            const float64 Av = BCAST0(ld_f32(A, offA + h), IDX);
            #pragma unroll 1
            for (int t = 0; t < T; t++) {
                const int py = s * sy + t * syt;
                float64 q0 = ld_f32(y, py + offq + hk * DS), q1 = ld_f32(y, py + offq + hk * DS + 64);
                float64 k0 = ld_f32(y, py + offk + hk * DS), k1 = ld_f32(y, py + offk + hk * DS + 64);
                float64 sq, sk;
                NL_REDUCE_ADD(v_f32_mac_b(q0, q0, q1 * q1), IDX, sq);
                NL_REDUCE_ADD(v_f32_mac_b(k0, k0, k1 * k1), IDX, sk);
                const float64 rq = nl_rsqrt(sq * inv + eps_qk) * (qk_scale * oscale);
                const float64 rk = nl_rsqrt(sk * inv + eps_qk) * qk_scale;
                q0 = q0 * rq; q1 = q1 * rq;
                k0 = k0 * rk; k1 = k1 * rk;
                const float64 vv = ld_f32(y, py + offv + h * DS + hb);
                const float64 av = BCAST0(ld_f32(ta, s * sa + t * sat + offa + h), IDX);
                const float64 bv = BCAST0(ld_f32(tb, s * sb + t * sbt + offb + h), IDX);
                const float64 eg = nl_exp(nl_softplus(av + dtv) * Av);
                const float64 beta = nl_sigm(bv);
                // decay, d_j = (v_j - M_j.k) * beta
                float64 dv = 0.0f;
                #pragma unroll 1
                for (int j = 0; j < RQ; j += 4) {
                    const float64 a00 = SV[2 * j] * eg,       a01 = SV[2 * j + 1] * eg;
                    const float64 a10 = SV[2 * j + 2] * eg,   a11 = SV[2 * j + 3] * eg;
                    const float64 a20 = SV[2 * j + 4] * eg,   a21 = SV[2 * j + 5] * eg;
                    const float64 a30 = SV[2 * j + 6] * eg,   a31 = SV[2 * j + 7] * eg;
                    SV[2 * j] = a00;     SV[2 * j + 1] = a01;
                    SV[2 * j + 2] = a10; SV[2 * j + 3] = a11;
                    SV[2 * j + 4] = a20; SV[2 * j + 5] = a21;
                    SV[2 * j + 6] = a30; SV[2 * j + 7] = a31;
                    float64 d0, d1, d2, d3;
                    NL_REDUCE4_ADD(v_f32_mac_b(a00, k0, a01 * k1), v_f32_mac_b(a10, k0, a11 * k1),
                                   v_f32_mac_b(a20, k0, a21 * k1), v_f32_mac_b(a30, k0, a31 * k1), IDX, d0, d1, d2, d3)
                    const float jf = (float) (lb + j);
                    dv = v_f32_sel_eq_f32_b(lane, jf, d0, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 1.0f, d1, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 2.0f, d2, dv);
                    dv = v_f32_sel_eq_f32_b(lane, jf + 3.0f, d3, dv);
                }
                dv = (vv - dv) * beta;
                // M_j += d_j k, o_j = M_j.q; snapshot rows for the last K tokens
                const int r = T - 1 - t;
                const int pdst = (drow + s + r * srs) * srow + (h * DS + j0) * DS;
                float64 o = 0.0f;
                #pragma unroll 1
                for (int j = 0; j < RQ; j += 4) {
                    const int jj = lb + j;
                    float64 db;
                    if (jj < 16) db = BCAST_DG(dv, 0);
                    else if (jj < 32) db = BCAST_DG(dv, 1);
                    else if (jj < 48) db = BCAST_DG(dv, 2);
                    else db = BCAST_DG(dv, 3);
                    const int e = jj & 15;
                    const float64 d0 = SHUF(db, IDX[e]);
                    const float64 d1 = SHUF(db, IDX[e + 1]);
                    const float64 d2 = SHUF(db, IDX[e + 2]);
                    const float64 d3 = SHUF(db, IDX[e + 3]);
                    const float64 a00 = v_f32_mac_b(k0, d0, SV[2 * j]),     a01 = v_f32_mac_b(k1, d0, SV[2 * j + 1]);
                    const float64 a10 = v_f32_mac_b(k0, d1, SV[2 * j + 2]), a11 = v_f32_mac_b(k1, d1, SV[2 * j + 3]);
                    const float64 a20 = v_f32_mac_b(k0, d2, SV[2 * j + 4]), a21 = v_f32_mac_b(k1, d2, SV[2 * j + 5]);
                    const float64 a30 = v_f32_mac_b(k0, d3, SV[2 * j + 6]), a31 = v_f32_mac_b(k1, d3, SV[2 * j + 7]);
                    SV[2 * j] = a00;     SV[2 * j + 1] = a01;
                    SV[2 * j + 2] = a10; SV[2 * j + 3] = a11;
                    SV[2 * j + 4] = a20; SV[2 * j + 5] = a21;
                    SV[2 * j + 6] = a30; SV[2 * j + 7] = a31;
                    if (r < K) {
                        const int pw = pdst + j * DS;
                        ST_F32(R, pw, a00, 64);           ST_F32(R, pw + 64, a01, 64);
                        ST_F32(R, pw + DS, a10, 64);      ST_F32(R, pw + DS + 64, a11, 64);
                        ST_F32(R, pw + 2 * DS, a20, 64);  ST_F32(R, pw + 2 * DS + 64, a21, 64);
                        ST_F32(R, pw + 3 * DS, a30, 64);  ST_F32(R, pw + 3 * DS + 64, a31, 64);
                    }
                    float64 r0, r1, r2, r3;
                    NL_REDUCE4_ADD(v_f32_mac_b(a00, q0, a01 * q1), v_f32_mac_b(a10, q0, a11 * q1),
                                   v_f32_mac_b(a20, q0, a21 * q1), v_f32_mac_b(a30, q0, a31 * q1), IDX, r0, r1, r2, r3)
                    const float jf = (float) jj;
                    o = v_f32_sel_eq_f32_b(lane, jf, r0, o);
                    o = v_f32_sel_eq_f32_b(lane, jf + 1.0f, r1, o);
                    o = v_f32_sel_eq_f32_b(lane, jf + 2.0f, r2, o);
                    o = v_f32_sel_eq_f32_b(lane, jf + 3.0f, r3, o);
                }
                int5 co = {((s * T + t) * H + h) * DS + j0, 0, 0, 0, 0};
                v_f32_st_tnsr_partial(co, out, o, RQ - 1, lb);
            }
        }
    }
}
#elif defined(ONORM)
void main(tensor o, tensor z, tensor w, tensor dep, tensor out,
          int nmem, int H, int T, int ns, int offz, int sz, int szt, int offw, float eps_o, int zsig)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float inv = 1.0f / (float) DS;
    const float64 w0 = ld_f32(w, offw), w1 = ld_f32(w, offw + 64);
    const int nitems = ns * T * H;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            const int st = it / H;
            const int h = it - st * H;
            const int s = st / T;
            const int t = st - s * T;
            const int po = it * DS;
            const float64 o0 = ld_f32(o, po), o1 = ld_f32(o, po + 64);
            float64 so;
            NL_REDUCE_ADD(v_f32_mac_b(o0, o0, o1 * o1), IDX, so);
            const float64 ro = nl_rsqrt(so * inv + eps_o);
            const int pz = offz + s * sz + t * szt + h * DS;
            const float64 z0 = ld_f32(z, pz), z1 = ld_f32(z, pz + 64);
            const float64 y0 = o0 * ro * w0 * (zsig ? nl_sigm(z0) : nl_silu(z0));
            const float64 y1 = o1 * ro * w1 * (zsig ? nl_sigm(z1) : nl_silu(z1));
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
#endif

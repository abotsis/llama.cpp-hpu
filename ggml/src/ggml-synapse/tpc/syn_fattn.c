// FLASH_ATTN_EXT partials: per (batch b, query head h, query i, key range c) it writes
// [acc (nvv * 64) | running max m (64 lanes) | sum l (64 lanes)] to the partial tensor;
// syn_fattn_comb combines them. gsz (1, 2 or 4): a group of query heads that share one KV head.
// q: f32 view [Dk, nq, H, B]; k: KV-type view [Dk, nkv, Hkv, Bkv]; v: [Dv, nkv, Hkv, Bkv];
// mask: f16 view [nkv, nq_pad, M2, M3] (has_mask). Head sizes are multiples of 32 (q lanes past
// Dk are zero; acc lanes past Dv are never read back).
// Scores: s = q.k * scale (softcap: cap * tanh(s)) + slope_h * mask (alibi: slope table [H]).
// Built per KV type: -DKV_F16 / -DKV_BF16 / -DKV_F32 / -DKV_Q8 (q8_0 blocks, offsets in bytes);
// -DWIDE: head sizes up to 576 (k) / 512 (v), else 256.
// kvl (flags bit 10, nq <= 64): per query row the number of KV rows to attend
// (host table from the mask).
// tensors: 0 q  1 k  2 v  3 mask  4 slopes  5 kvl | 6 part
#include "syn_common.h"

#if defined(WIDE)
// even: 16-bit rows load in vector pairs (an odd bound of 9 dropped the last vector of Dk = 576)
#define NVK 10
#define NVV 8
#else
#define NVK 4
#define NVV 4
#endif

#if defined(KV_Q8)
// 64 elements = two q8_0 blocks at byte offset p: quants of block a in lanes 0..31, block b in 32..63.
// The scales come from one f16 load at p (block a's in lane 0, block b's in lane 17).
static inline float64 ld_q8(tensor t, int p, float64 lane, uchar256 i0, uchar256 i1)
{
    int5 ca = {p + 2, 0, 0, 0, 0};
    int5 cb = {p + 4, 0, 0, 0, 0};
    int5 cs = {p, 0, 0, 0, 0};
    const float64 qa = convert_char256_to_float256(v_i8_ld_tnsr_b(ca, t), SW_LINEAR).v1;
    const float64 qb = convert_char256_to_float256(v_i8_ld_tnsr_b(cb, t), SW_LINEAR).v1;
    const float64 sc = f16_to_f32(v_f16_ld_tnsr_b(cs, t)).v1;
    const float64 da = SHUF(BCAST_DG(sc, 0), i0);
    const float64 db = SHUF(BCAST_DG(sc, 1), i1);
    return v_f32_sel_less_f32_b(lane, 32.0f, qa * da, qb * db);
}
#define LDKV(t, p, X, n, M)                                                 \
    {                                                                       \
        _Pragma("unroll")                                                   \
        for (int u_ = 0; u_ < (M); u_++) {                                  \
            if (u_ < (n)) X[u_] = ld_q8(t, (p) + 68 * u_, lane, IDX[0], IDX[1]);            \
        }                                                                   \
    }
#elif defined(KV_F16) || defined(KV_BF16)
#if defined(KV_F16)
#define LD128(c_, t) f16_to_f32(v_f16_ld_tnsr_b(c_, t))
#else
#define LD128(c_, t) bf16_to_f32(v_bf16_ld_tnsr_b(c_, t))
#endif
#define LDKV(t, p, X, n, M)                                                 \
    {                                                                       \
        _Pragma("unroll")                                                   \
        for (int u_ = 0; u_ < (M); u_ += 2) {                               \
            if (u_ < (n)) {                                                 \
                int5 c_ = {(p) + 64 * u_, 0, 0, 0, 0};                      \
                const float128 a_ = LD128(c_, t);                           \
                X[u_] = a_.v1;                                              \
                if (u_ + 1 < (M)) X[u_ + 1] = a_.v2;                        \
            }                                                               \
        }                                                                   \
    }
#else
#define LDKV(t, p, X, n, M)                                                 \
    {                                                                       \
        _Pragma("unroll")                                                   \
        for (int u_ = 0; u_ < (M); u_++) {                                  \
            if (u_ < (n)) X[u_] = ld_f32(t, (p) + 64 * u_);                 \
        }                                                                   \
    }
#endif

__local__ float64 QS[4 * NVK];
__local__ float64 ACC[4 * NVV];

// four K (or V) rows
#if defined(KV_Q8)
__local__ float64 KB[4 * NVK];
#define DECL_ROWS(M)
#define LOAD_ROWS(t, p0, p1, p2, p3, n, M)                                  \
    {                                                                       \
        _Pragma("unroll 1")                                                 \
        for (int r_ = 0; r_ < 4; r_++) {                                    \
            const int pr_ = r_ == 0 ? (p0) : r_ == 1 ? (p1) : r_ == 2 ? (p2) : (p3); \
            _Pragma("unroll 1")                                             \
            for (int u_ = 0; u_ < (n); u_++) {                              \
                KB[r_ * NVK + u_] = ld_q8(t, pr_ + 68 * u_, lane, IDX[0], IDX[1]);          \
            }                                                               \
        }                                                                   \
    }
#define ROW(r, u) KB[(r) * NVK + (u)]
#else
#define DECL_ROWS(M) float64 R0_[M], R1_[M], R2_[M], R3_[M];
#define LOAD_ROWS(t, p0, p1, p2, p3, n, M)                                  \
    {                                                                       \
        LDKV(t, p0, R0_, n, M)                                              \
        LDKV(t, p1, R1_, n, M)                                              \
        LDKV(t, p2, R2_, n, M)                                              \
        LDKV(t, p3, R3_, n, M)                                              \
    }
#define ROW(r, u) R##r##_[u]
#endif

// dkv = Dk | Dv << 16; flags = gsz | has_mask << 8 | alibi << 9 | kvl << 10 (packed: <= 30 scalar params)
void main(tensor q, tensor k, tensor v, tensor mask, tensor slopes, tensor kvl, tensor dep, tensor part,
          int nmem, int dkv, int nq, int H, int B, int hpg, int nkv, int chunk,
          int offq, int sq1, int sq2, int sq3,
          int offk, int sk1, int sk2, int sk3,
          int offv, int sv1, int sv2, int sv3,
          int offm, int sm1, int sm2, int sm3, int M2, int M3,
          float scale, float softcap, int flags)
{
    const int Dk = dkv & 0xFFFF, Dv = dkv >> 16;
    const int gsz = flags & 0xFF, has_mask = (flags >> 8) & 1, alibi = (flags >> 9) & 1, use_kvl = (flags >> 10) & 1;
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    DECL_IDX(IDX)
    scalar_cache_reset();
    const float64 lane = LANE_F32();
    const int nvk = (Dk + 63) >> 6;
    const int nvv = (Dv + 63) >> 6;
    const int ngrp = H / gsz;
    const int nch = (nkv + chunk - 1) / chunk;
    const int nitems = B * ngrp * nq * nch;
    const int pstride = nvv * 64 + 128;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(nitems, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int x = it;
            const int c = x % nch;      x /= nch;
            const int i = x % nq;       x /= nq;
            const int hg = x % ngrp;
            const int b = x / ngrp;
            const int h0 = hg * gsz;
            const int hk = h0 / hpg;
            const int j0 = c * chunk;
            int j1 = j0 + chunk;
            if (j1 > nkv) j1 = nkv;
            if (use_kvl) {
                int5 cl = {i, 0, 0, 0, 0};
                const int kl = s_i32_ld_g((__global int *) gen_addr(cl, kvl));
                if (j1 > kl) j1 = kl;
            }
            float sl[4];
            for (int g = 0; g < 4; g++) {
                sl[g] = 1.0f;
                if (alibi && g < gsz) {
                    int5 cs = {h0 + g, 0, 0, 0, 0};
                    sl[g] = s_f32_ld_g(gen_addr(cs, slopes));
                }
                #pragma unroll
                for (int u = 0; u < NVK; u++) {
                    float64 qv = 0.0f;
                    if (g < gsz && u < nvk) {
                        qv = ld_f32(q, offq + i * sq1 + (h0 + g) * sq2 + b * sq3 + 64 * u) * scale;
                        qv = v_f32_sel_less_f32_b(lane, (float) (Dk - 64 * u), qv, 0.0f);
                    }
                    QS[g * NVK + u] = qv;
                }
                #pragma unroll
                for (int u = 0; u < NVV; u++) {
                    ACC[g * NVV + u] = 0.0f;
                }
            }
            float64 mr[4], lr[4];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                mr[g] = -1e30f;
                lr[g] = 0.0f;
            }
            const int pk0 = offk + hk * sk2 + b * sk3;
            const int pv0 = offv + hk * sv2 + b * sv3;
            for (int jb = j0; jb < j1; jb += 64) {
                int je = jb + 64;
                if (je > j1) je = j1;
                float64 S[4];
                #pragma unroll
                for (int g = 0; g < 4; g++) S[g] = -1e30f;
                for (int j = jb; j < je; j += 4) {
                    // keys past je are scored but masked below
                    DECL_ROWS(NVK)
                    const int r1 = j + 1 < je ? j + 1 : j, r2 = j + 2 < je ? j + 2 : j, r3 = j + 3 < je ? j + 3 : j;
                    LOAD_ROWS(k, pk0 + j * sk1, pk0 + r1 * sk1, pk0 + r2 * sk1, pk0 + r3 * sk1, nvk, NVK)
                    const float jl = (float) (j - jb);
                    for (int g = 0; g < gsz; g++) {
                        float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                        #pragma unroll
                        for (int u = 0; u < NVK; u++) {
                            if (u < nvk) {
                                const float64 qq = QS[g * NVK + u];
                                a0 = v_f32_mac_b(qq, ROW(0, u), a0);
                                a1 = v_f32_mac_b(qq, ROW(1, u), a1);
                                a2 = v_f32_mac_b(qq, ROW(2, u), a2);
                                a3 = v_f32_mac_b(qq, ROW(3, u), a3);
                            }
                        }
                        float64 s0, s1, s2, s3;
                        NL_REDUCE4_ADD(a0, a1, a2, a3, IDX, s0, s1, s2, s3)
                        float64 sg = S[g];
                        sg = v_f32_sel_eq_f32_b(lane, jl, s0, sg);
                        sg = v_f32_sel_eq_f32_b(lane, jl + 1.0f, s1, sg);
                        sg = v_f32_sel_eq_f32_b(lane, jl + 2.0f, s2, sg);
                        sg = v_f32_sel_eq_f32_b(lane, jl + 3.0f, s3, sg);
                        S[g] = sg;
                    }
                }
                // block softmax update per head
                float64 P[4];
                for (int g = 0; g < gsz; g++) {
                    float64 s = S[g];
                    if (softcap != 0.0f) {
                        s = softcap * nl_tanh(s);
                    }
                    if (has_mask) {
                        const int hh = h0 + g;
                        const int pm = offm + i * sm1 + (hh % M2) * sm2 + (b % M3) * sm3 + jb;
                        int5 cm = {pm, 0, 0, 0, 0};
                        const float128 mv = f16_to_f32(v_f16_ld_tnsr_b(cm, mask));
                        s = v_f32_mac_b(mv.v1, sl[g], s);
                    }
                    s = v_f32_sel_less_f32_b(lane, (float) (je - jb), s, -1e30f);
                    float64 mb;
                    NL_REDUCE_MAX(s, IDX, mb);
                    const float64 mn = v_f32_max_b(mr[g], mb);
                    const float64 corr = nl_exp(mr[g] - mn);
                    const float64 pp = v_f32_sel_less_f32_b(lane, (float) (je - jb), nl_exp(s - mn), 0.0f);
                    float64 ps;
                    NL_REDUCE_ADD(pp, IDX, ps);
                    lr[g] = v_f32_mac_b(lr[g], corr, ps);
                    mr[g] = mn;
                    P[g] = pp;
                    #pragma unroll
                    for (int u = 0; u < NVV; u++) {
                        ACC[g * NVV + u] = ACC[g * NVV + u] * corr;
                    }
                }
                // acc += p_j * v_j (padded keys have p = 0)
                for (int j = jb; j < je; j += 4) {
                    DECL_ROWS(NVV)
                    // rows past the block reuse row j (their p is 0; garbage could be NaN)
                    const int r1 = j + 1 < je ? j + 1 : j, r2 = j + 2 < je ? j + 2 : j, r3 = j + 3 < je ? j + 3 : j;
                    LOAD_ROWS(v, pv0 + j * sv1, pv0 + r1 * sv1, pv0 + r2 * sv1, pv0 + r3 * sv1, nvv, NVV)
                    const int jj = j - jb;
                    const int e = jj & 15;
                    const int dg = jj >> 4;
                    for (int g = 0; g < gsz; g++) {
                        float64 pb;
                        if (dg == 0) pb = BCAST_DG(P[g], 0);
                        else if (dg == 1) pb = BCAST_DG(P[g], 1);
                        else if (dg == 2) pb = BCAST_DG(P[g], 2);
                        else pb = BCAST_DG(P[g], 3);
                        const float64 p0 = SHUF(pb, IDX[e]);
                        const float64 p1 = SHUF(pb, IDX[e + 1]);
                        const float64 p2 = SHUF(pb, IDX[e + 2]);
                        const float64 p3 = SHUF(pb, IDX[e + 3]);
                        #pragma unroll
                        for (int u = 0; u < NVV; u++) {
                            if (u < nvv) {
                                float64 acc = ACC[g * NVV + u];
                                acc = v_f32_mac_b(p0, ROW(0, u), acc);
                                acc = v_f32_mac_b(p1, ROW(1, u), acc);
                                acc = v_f32_mac_b(p2, ROW(2, u), acc);
                                acc = v_f32_mac_b(p3, ROW(3, u), acc);
                                ACC[g * NVV + u] = acc;
                            }
                        }
                    }
                }
            }
            for (int g = 0; g < gsz; g++) {
                const int po = ((((b * H) + h0 + g) * nq + i) * nch + c) * pstride;
                #pragma unroll
                for (int u = 0; u < NVV; u++) {
                    if (u < nvv) ST_F32(part, po + 64 * u, ACC[g * NVV + u], 64);
                }
                ST_F32(part, po + nvv * 64, mr[g], 64);
                ST_F32(part, po + nvv * 64 + 64, lr[g], 64);
            }
        }
    }
}

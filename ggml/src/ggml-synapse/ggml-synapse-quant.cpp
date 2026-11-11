/*
 * ggml-synapse-quant.cpp - device layouts of quantized weights.
 *
 * Planar layout of a <= 8-bit type (exactly the native size, a permutation of the native bits):
 * rows in blocks of 128, per block and superblock (one native block of SB elements per row):
 *   [f16 params: nf16 planes of 128 rows x 2 B][byte params: nbp planes of 128 B]
 *   [P4: SB/4 x 256 B][P2: SB/8 x 256 B][P1: SB/16 x 256 B][P8: SB/2 x 256 B]
 * P8 (q8_0): the signed codes, one per byte, 2 k-steps per 256 B: row r of k = 2 m + o at lane
 * 4 (r & 63) + 2 (r >> 6) + o of group m (all 256 lanes, in the lane order of one 256 B nibble group).
 * Code planes hold the element codes (q = p4 | p2 << 4 | ... ) of chunks of 4 k-steps per 256 B group:
 * row r of k = 4 c + j sits in lane 4 (r & 63) + 2 (r >> 6) + (j & 1), in the low part of the byte
 * for j < 2 (P4: lo nibble; P2: bits 0-1 of its chunk pair; P1: bit 0 of its chunk quad) and the high
 * part for j >= 2.
 * The params are the native param bytes (f16 scales / packed scale bytes), one plane per byte.
 *
 * FP8 (from Q8_0): Gaudi2 fp8_143 (e4m3, bias 7, +-240) with a per-row f32 scale max|row| / 240.
 *   rows: [M rows x K B][pad to 128][M x f32]
 *   planar (experts): per 128-row block [128 x f32 scales][K x 128 B: row r of k at
 *   byte 256 (k / 2) + 4 (r & 63) + 2 (r >> 6) + (k & 1)]
 */
#include "ggml-synapse-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

/* ---------------- planar types ---------------- */

static const synapse_qt k_qts[] = {
    //  type             sb   g   nf16 nbp p4 p2 p1 block  f16 offsets  byte param offsets (first), p8
    { GGML_TYPE_Q4_0,    32,  32, 1, 0,  1, 0, 0,  18,  { 0, 0 },    0, 0 },
    { GGML_TYPE_Q4_1,    32,  32, 2, 0,  1, 0, 0,  20,  { 0, 2 },    0, 0 },
    { GGML_TYPE_Q5_0,    32,  32, 1, 0,  1, 0, 1,  22,  { 0, 0 },    0, 0 },
    { GGML_TYPE_Q5_1,    32,  32, 2, 0,  1, 0, 1,  24,  { 0, 2 },    0, 0 },
    { GGML_TYPE_Q2_K,    256, 16, 2, 16, 0, 1, 0,  84,  { 80, 82 },  0, 0 },
    { GGML_TYPE_Q3_K,    256, 16, 1, 12, 0, 1, 1,  110, { 108, 0 },  96, 0 },
    { GGML_TYPE_Q4_K,    256, 32, 2, 12, 1, 0, 0,  144, { 0, 2 },    4, 0 },
    { GGML_TYPE_Q5_K,    256, 32, 2, 12, 1, 0, 1,  176, { 0, 2 },    4, 0 },
    { GGML_TYPE_Q6_K,    256, 16, 1, 16, 1, 1, 0,  210, { 208, 0 },  192, 0 },
    { GGML_TYPE_IQ4_NL,  32,  32, 1, 0,  1, 0, 0,  18,  { 0, 0 },    0, 0 },
    { GGML_TYPE_IQ4_XS,  256, 32, 1, 6,  1, 0, 0,  136, { 0, 0 },    2, 0 },
    { GGML_TYPE_MXFP4,   32,  32, 0, 1,  1, 0, 0,  17,  { 0, 0 },    0, 0 },
    { GGML_TYPE_NVFP4,   64,  16, 0, 4,  1, 0, 0,  36,  { 0, 0 },    0, 0 },
    { GGML_TYPE_TQ2_0,   256, 256, 1, 0, 0, 1, 0,  66,  { 64, 0 },   0, 0 },
    { GGML_TYPE_Q1_0,    128, 128, 1, 0, 0, 0, 1,  18,  { 0, 0 },    0, 0 },
    { GGML_TYPE_Q2_0,    64,  64, 1, 0,  0, 1, 0,  18,  { 0, 0 },    0, 0 },
    { GGML_TYPE_Q8_0,    32,  32, 1, 0,  0, 0, 0,  34,  { 0, 0 },    0, 1 },   // (p8; experts only: weight_kind)
};

const synapse_qt * synapse_qt_get(ggml_type t) {
    for (const auto & q : k_qts) {
        if (q.type == t) {
            return &q;
        }
    }
    return nullptr;
}

struct qt_frag {
    int byte, shift, nbits, cshift;
};

/* native bit fragments of element e of a superblock -> code bits; returns the fragment count */
static int qt_frags(ggml_type t, int e, qt_frag f[2]) {
    switch (t) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
            f[0] = { 2 + e % 16, e < 16 ? 0 : 4, 4, 0 };
            return 1;
        case GGML_TYPE_Q4_1:
            f[0] = { 4 + e % 16, e < 16 ? 0 : 4, 4, 0 };
            return 1;
        case GGML_TYPE_MXFP4:
            f[0] = { 1 + e % 16, e < 16 ? 0 : 4, 4, 0 };
            return 1;
        case GGML_TYPE_Q5_0:
            f[0] = { 6 + e % 16, e < 16 ? 0 : 4, 4, 0 };
            f[1] = { 2 + e / 8, e % 8, 1, 4 };
            return 2;
        case GGML_TYPE_Q5_1:
            f[0] = { 8 + e % 16, e < 16 ? 0 : 4, 4, 0 };
            f[1] = { 4 + e / 8, e % 8, 1, 4 };
            return 2;
        case GGML_TYPE_Q2_K:
            f[0] = { 16 + (e / 128) * 32 + e % 32, 2 * ((e % 128) / 32), 2, 0 };
            return 1;
        case GGML_TYPE_TQ2_0:
            f[0] = { (e / 128) * 32 + e % 32, 2 * ((e % 128) / 32), 2, 0 };
            return 1;
        case GGML_TYPE_Q3_K:
            f[0] = { 32 + (e / 128) * 32 + e % 32, 2 * ((e % 128) / 32), 2, 0 };
            f[1] = { e % 32, e / 32, 1, 2 };
            return 2;
        case GGML_TYPE_Q4_K: {
            const int c = e / 64, r = e % 64;
            f[0] = { 16 + c * 32 + r % 32, r < 32 ? 0 : 4, 4, 0 };
            return 1;
        }
        case GGML_TYPE_Q5_K: {
            const int c = e / 64, r = e % 64;
            f[0] = { 48 + c * 32 + r % 32, r < 32 ? 0 : 4, 4, 0 };
            f[1] = { 16 + r % 32, 2 * c + (r < 32 ? 0 : 1), 1, 4 };
            return 2;
        }
        case GGML_TYPE_Q6_K: {
            const int n = e / 128, w = e % 128, l = w % 32, qq = w / 32;
            f[0] = { n * 64 + l + ((qq & 1) ? 32 : 0), qq < 2 ? 0 : 4, 4, 0 };
            f[1] = { 128 + n * 32 + l, 2 * qq, 2, 4 };
            return 2;
        }
        case GGML_TYPE_IQ4_XS: {
            const int ib = e / 32, r = e % 32;
            f[0] = { 8 + ib * 16 + r % 16, r < 16 ? 0 : 4, 4, 0 };
            return 1;
        }
        case GGML_TYPE_NVFP4: {
            const int s = e / 16, r = e % 16;
            f[0] = { 4 + s * 8 + r % 8, r < 8 ? 0 : 4, 4, 0 };
            return 1;
        }
        case GGML_TYPE_Q1_0:
            f[0] = { 2 + e / 8, e % 8, 1, 0 };
            return 1;
        case GGML_TYPE_Q2_0:
            f[0] = { 2 + e / 4, 2 * (e % 4), 2, 0 };
            return 1;
        case GGML_TYPE_Q8_0:
            f[0] = { 2 + e, 0, 8, 0 };
            return 1;
        default:
            GGML_ABORT("unsupported planar type");
    }
}

/* per superblock of one 128-row block: byte offsets of the regions */
struct qt_regions {
    int f16, bp, p4, p2, p1, p8, end;
};

static qt_regions qt_layout(const synapse_qt * q) {
    qt_regions r;
    r.f16 = 0;
    r.bp = q->nf16 * 256;
    r.p4 = r.bp + q->nbp * 128;
    r.p2 = r.p4 + (q->p4 ? q->sb / 4 * 256 : 0);
    r.p1 = r.p2 + (q->p2 ? q->sb / 8 * 256 : 0);
    r.p8 = r.p1 + (q->p1 ? q->sb / 16 * 256 : 0);
    r.end = r.p8 + (q->p8 ? q->sb / 2 * 256 : 0);
    GGML_ASSERT(r.end == 128 * q->block);
    return r;
}

/* byte lane of row r (0..127) and k-step j (0..3) of a code chunk */
static inline int qt_lane(int j, int r) {
    return 4 * (r & 63) + 2 * (r >> 6) + (j & 1);
}

/* bit positions of the planes inside a code */
static void qt_code_bits(const synapse_qt * q, int & s2, int & s1) {
    s2 = q->p4 ? 4 : 0;
    s1 = q->p4 ? 4 : (q->p2 ? 2 : 0);
}

/* per type, element -> its native bit fragments (qt_frags as tables) */
struct qt_tab {
    int nf;
    uint8_t byte[2][256], shift[2][256], mask[2][256], cshift[2][256];
};

static const qt_tab & qt_tab_get(const synapse_qt * q) {
    static qt_tab tabs[sizeof(k_qts) / sizeof(k_qts[0])];
    static std::once_flag once[sizeof(k_qts) / sizeof(k_qts[0])];
    const size_t i = q - k_qts;
    std::call_once(once[i], [&] {
        qt_tab & t = tabs[i];
        t = {};
        for (int e = 0; e < q->sb; e++) {
            qt_frag f[2];
            t.nf = qt_frags(q->type, e, f);
            for (int k = 0; k < t.nf; k++) {
                t.byte[k][e] = (uint8_t) f[k].byte;
                t.shift[k][e] = (uint8_t) f[k].shift;
                t.mask[k][e] = (uint8_t) ((1 << f[k].nbits) - 1);
                t.cshift[k][e] = (uint8_t) f[k].cshift;
            }
        }
    });
    return tabs[i];
}

/* code plane of w bits per code (n = 8 / w codes per byte): byte of group m and lane 4 i + 2 h + o holds
 * the codes of k = 2 n m + 2 t + o (t = 0..n-1) of row i + 64 h at bits w t */
template <int w>
static void qt_pack_plane(const uint8_t * codes, int sb, int sbit, uint8_t * dst) {
    constexpr int n = 8 / w;
    constexpr uint8_t mask = (1 << w) - 1;
    for (int m = 0; m < sb / (2 * n); m++) {
        uint8_t * g = dst + 256 * m;
        for (int o = 0; o < 2; o++) {
            const uint8_t * col[n];
            for (int t = 0; t < n; t++) {
                col[t] = codes + (size_t) (2 * n * m + 2 * t + o) * 128;
            }
            for (int r = 0; r < 128; r++) {
                uint8_t v = 0;
                for (int t = 0; t < n; t++) {
                    v |= ((col[t][r] >> sbit) & mask) << (w * t);
                }
                g[4 * (r & 63) + 2 * (r >> 6) + o] = v;
            }
        }
    }
}

/* one superblock of 128 rows (row r at src + r * row_stride, rows >= nrows are zero) -> planar */
static void qt_encode_sb(const synapse_qt * q, const uint8_t * src, size_t row_stride, int nrows, uint8_t * dst) {
    const qt_regions L = qt_layout(q);
    const qt_tab & T = qt_tab_get(q);
    if (nrows < 128) {
        memset(dst, 0, L.end);
    }
    alignas(64) uint8_t codes[256 * 128];   // [k][row]
    if (nrows < 128) {
        memset(codes, 0, (size_t) q->sb * 128);
    }
    for (int r = 0; r < nrows; r++) {
        const uint8_t * b = src + r * row_stride;
        for (int f = 0; f < q->nf16; f++) {
            memcpy(dst + L.f16 + f * 256 + 2 * r, b + q->f16_off[f], 2);
        }
        for (int p = 0; p < q->nbp; p++) {
            dst[L.bp + p * 128 + r] = b[q->bp_off + p];
        }
        if (T.nf == 1) {
            for (int e = 0; e < q->sb; e++) {
                codes[e * 128 + r] = (b[T.byte[0][e]] >> T.shift[0][e]) & T.mask[0][e];
            }
        } else {
            for (int e = 0; e < q->sb; e++) {
                codes[e * 128 + r] = (((b[T.byte[0][e]] >> T.shift[0][e]) & T.mask[0][e]) << T.cshift[0][e]) |
                                     (((b[T.byte[1][e]] >> T.shift[1][e]) & T.mask[1][e]) << T.cshift[1][e]);
            }
        }
    }
    int s2, s1;
    qt_code_bits(q, s2, s1);
    if (q->p4) {
        qt_pack_plane<4>(codes, q->sb, 0, dst + L.p4);
    }
    if (q->p2) {
        qt_pack_plane<2>(codes, q->sb, s2, dst + L.p2);
    }
    if (q->p1) {
        qt_pack_plane<1>(codes, q->sb, s1, dst + L.p1);
    }
    if (q->p8) {
        qt_pack_plane<8>(codes, q->sb, 0, dst + L.p8);
    }
}

static void qt_decode_sb(const synapse_qt * q, const uint8_t * src, uint8_t * dst, size_t row_stride, int nrows) {
    const qt_regions L = qt_layout(q);
    int s2, s1;
    qt_code_bits(q, s2, s1);
    for (int r = 0; r < nrows; r++) {
        uint8_t * b = dst + r * row_stride;
        memset(b, 0, q->block);
        for (int f = 0; f < q->nf16; f++) {
            memcpy(b + q->f16_off[f], src + L.f16 + f * 256 + 2 * r, 2);
        }
        for (int p = 0; p < q->nbp; p++) {
            b[q->bp_off + p] = src[L.bp + p * 128 + r];
        }
        for (int e = 0; e < q->sb; e++) {
            const int c4 = e / 4, j = e % 4, part = j >> 1, lane = qt_lane(j, r);
            uint8_t c = 0;
            if (q->p4) {
                c |= (src[L.p4 + 256 * c4 + lane] >> (4 * part)) & 15;
            }
            if (q->p2) {
                c |= ((src[L.p2 + 256 * (c4 / 2) + lane] >> ((c4 % 2) * 4 + 2 * part)) & 3) << s2;
            }
            if (q->p1) {
                c |= ((src[L.p1 + 256 * (c4 / 4) + lane] >> ((c4 % 4) * 2 + part)) & 1) << s1;
            }
            if (q->p8) {
                c = src[L.p8 + 256 * (e / 2) + qt_lane(e & 1, r)];
            }
            qt_frag fr[2];
            const int nf = qt_frags(q->type, e, fr);
            for (int k = 0; k < nf; k++) {
                b[fr[k].byte] |= ((c >> fr[k].cshift) & ((1 << fr[k].nbits) - 1)) << fr[k].shift;
            }
        }
    }
}

static void parallel_for(int64_t n, const std::function<void(int64_t)> & f) {
    const int nth = (int) std::min<int64_t>(n, std::min<unsigned>(32, std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::thread> th;
    for (int t = 1; t < nth; t++) {
        th.emplace_back([&, t] {
            for (int64_t i = t; i < n; i += nth) {
                f(i);
            }
        });
    }
    for (int64_t i = 0; i < n; i += std::max(1, nth)) {
        f(i);
    }
    for (auto & x : th) {
        x.join();
    }
}

size_t synapse_qt_planar_size(const synapse_qt * q, int64_t K, int64_t M, int64_t nmat) {
    return (size_t) nmat * ((M + 127) / 128) * (K / q->sb) * 128 * q->block;
}

void synapse_qt_to_planar(const synapse_qt * q, const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat) {
    const int64_t nsb = K / q->sb, nrb = (M + 127) / 128;
    const size_t row = (size_t) nsb * q->block, sbb = (size_t) 128 * q->block;
    parallel_for(nmat * nrb, [&](int64_t mb) {
        const int64_t m = mb / nrb, rb = mb % nrb;
        const int nrows = (int) std::min<int64_t>(128, M - rb * 128);
        const uint8_t * s = src + ((size_t) m * M + rb * 128) * row;
        uint8_t * d = dst + (size_t) mb * nsb * sbb;
        for (int64_t sb = 0; sb < nsb; sb++) {
            qt_encode_sb(q, s + sb * q->block, row, nrows, d + sb * sbb);
        }
    });
}

void synapse_qt_from_planar(const synapse_qt * q, const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat) {
    const int64_t nsb = K / q->sb, nrb = (M + 127) / 128;
    const size_t row = (size_t) nsb * q->block, sbb = (size_t) 128 * q->block;
    parallel_for(nmat * nrb, [&](int64_t mb) {
        const int64_t m = mb / nrb, rb = mb % nrb;
        const int nrows = (int) std::min<int64_t>(128, M - rb * 128);
        uint8_t * d = dst + ((size_t) m * M + rb * 128) * row;
        const uint8_t * s = src + (size_t) mb * nsb * sbb;
        for (int64_t sb = 0; sb < nsb; sb++) {
            qt_decode_sb(q, s + sb * sbb, d + sb * q->block, row, nrows);
        }
    });
}

/* ---------------- fp8_143 ---------------- */

uint8_t synapse_fp8_enc(float x) {
    // e4m3, exponent bias 7, exponent 15 reserved (IEEE-style specials): max 240; round to nearest even.
    // Integer rounding of the f32 bits (bit-identical to the frexp / nearbyint form for every float)
    uint32_t u;
    memcpy(&u, &x, 4);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80);
    u &= 0x7fffffffu;
    if (u > 0x7f800000u) {
        return sign;                                   // NaN
    }
    if (u >= 0x43700000u) {
        return sign | 0x77;                            // >= 240: saturated (exponent 14, mantissa 7)
    }
    if (u < 0x3c800000u) {
        float a;                                       // < 2^-6: subnormal, steps of 2^-9 (8 steps: 2^-6)
        memcpy(&a, &u, 4);
        return sign | (uint8_t) (int) std::nearbyint(a * 512.0f);
    }
    const uint32_t r = (u + 0x7ffffu + ((u >> 20) & 1u)) & ~0xfffffu;   // 3 mantissa bits, ties to even
    return sign | (uint8_t) ((((int) (r >> 23) - 127 + 7) << 3) | (int) ((r >> 20) & 7u));
}

float synapse_fp8_dec(uint8_t b) {
    const int ex = (b >> 3) & 15, man = b & 7;
    const float v = ex == 0 ? std::ldexp((float) man, -9) : std::ldexp(1.0f + man / 8.0f, ex - 7);
    return (b & 0x80) ? -v : v;
}

size_t synapse_fp8_rows_size(int64_t K, int64_t M) {
    return ((size_t) M * K + 127) / 128 * 128 + (size_t) M * 4;
}

size_t synapse_fp8_planar_size(int64_t K, int64_t M, int64_t nmat) {
    return (size_t) nmat * ((M + 127) / 128) * (512 + (size_t) K * 128);
}

/* q8_0 rows -> f32 */
static void q8_row(const uint8_t * b, float * f, int64_t K) {
    for (int64_t j = 0; j < K / 32; j++) {
        const float d = GGML_FP16_TO_FP32(*(const ggml_fp16_t *) (b + 34 * j));
        for (int i = 0; i < 32; i++) {
            f[32 * j + i] = d * (float) (int8_t) b[34 * j + 2 + i];
        }
    }
}

static float fp8_row(const float * f, uint8_t * q, int64_t K, int64_t stride) {
    float amax = 0.0f;
    for (int64_t k = 0; k < K; k++) {
        amax = std::max(amax, std::fabs(f[k]));
    }
    const float s = amax > 0.0f ? amax / 240.0f : 1.0f;
    for (int64_t k = 0; k < K; k++) {
        q[k * stride] = synapse_fp8_enc(f[k] / s);
    }
    return s;
}

void synapse_q8_to_fp8_rows(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M) {
    const size_t rb = (size_t) K / 32 * 34;
    float * sc = (float *) (dst + ((size_t) M * K + 127) / 128 * 128);
    parallel_for(M, [&](int64_t m) {
        std::vector<float> f(K);
        q8_row(src + m * rb, f.data(), K);
        sc[m] = fp8_row(f.data(), dst + (size_t) m * K, K, 1);
    });
}

void synapse_q8_to_fp8_planar(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat) {
    const size_t rb = (size_t) K / 32 * 34, bb = 512 + (size_t) K * 128;
    const int64_t nrb = (M + 127) / 128;
    memset(dst, 0, synapse_fp8_planar_size(K, M, nmat));
    parallel_for(nmat * nrb, [&](int64_t mb) {
        const int64_t m = mb / nrb, b = mb % nrb;
        uint8_t * d = dst + (size_t) mb * bb;
        std::vector<float> f(K);
        std::vector<uint8_t> q(K);
        for (int r = 0; r < 128 && b * 128 + r < M; r++) {
            q8_row(src + ((size_t) m * M + b * 128 + r) * rb, f.data(), K);
            ((float *) d)[r] = fp8_row(f.data(), q.data(), K, 1);
            for (int64_t k = 0; k < K; k++) {
                d[512 + 256 * (k / 2) + qt_lane((int) (k & 1), r)] = q[k];
            }
        }
    });
}

/* fp8 -> q8_0 (reading weights back) */
static void q8_from_f32(const float * f, uint8_t * b, int64_t K) {
    for (int64_t j = 0; j < K / 32; j++) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            amax = std::max(amax, std::fabs(f[32 * j + i]));
        }
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        *(ggml_fp16_t *) (b + 34 * j) = GGML_FP32_TO_FP16(d);
        for (int i = 0; i < 32; i++) {
            b[34 * j + 2 + i] = (uint8_t) (int8_t) std::lround(f[32 * j + i] * id);
        }
    }
}

void synapse_fp8_rows_to_q8(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M) {
    const size_t rb = (size_t) K / 32 * 34;
    const float * sc = (const float *) (src + ((size_t) M * K + 127) / 128 * 128);
    parallel_for(M, [&](int64_t m) {
        std::vector<float> f(K);
        for (int64_t k = 0; k < K; k++) {
            f[k] = synapse_fp8_dec(src[(size_t) m * K + k]) * sc[m];
        }
        q8_from_f32(f.data(), dst + m * rb, K);
    });
}

void synapse_fp8_planar_to_q8(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat) {
    const size_t rb = (size_t) K / 32 * 34, bb = 512 + (size_t) K * 128;
    const int64_t nrb = (M + 127) / 128;
    parallel_for(nmat * nrb, [&](int64_t mb) {
        const int64_t m = mb / nrb, b = mb % nrb;
        const uint8_t * s = src + (size_t) mb * bb;
        std::vector<float> f(K);
        for (int r = 0; r < 128 && b * 128 + r < M; r++) {
            for (int64_t k = 0; k < K; k++) {
                f[k] = synapse_fp8_dec(s[512 + 256 * (k / 2) + qt_lane((int) (k & 1), r)]) * ((const float *) s)[r];
            }
            q8_from_f32(f.data(), dst + ((size_t) m * M + b * 128 + r) * rb, K);
        }
    });
}

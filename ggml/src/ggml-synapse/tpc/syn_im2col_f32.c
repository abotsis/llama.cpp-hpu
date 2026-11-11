// im2col: image [IW, IH, (ID,) IC * N] f32 -> packed taps. The kernel tensor supplies geometry
// only, not values. Channel is OUTSIDE the tap dims in flattened dim 0 (CPU layout):
//   out element (ic, taps..., o) at tap_flat + T * o, tap_flat = ic * (taps per channel) + tap.
// Each output owns one tap and gathers the image at output*stride + tap*dilation - padding,
// writing exact zero outside the bounds (CPU ggml_compute_forward_im2col[_3d]).
// ND=2 (default): out [KW*KH*IC, OW, OH, N]; params {s0,s1,p0,p1,d0,d1,is_2D}. is_2D=1: image
//   [IW, IH, IC, N], KH taps over dim1. is_2D=0 (1D): out [K*IC, OL, N, 1], image [IW, IC, N],
//   KH = 1 (host passes 1), OH = 1.
// The image is a view: element offset offi + strides sb1 (y), sb2 (channel; 3D: z), sb3 (batch; 3D: c + IC n).
// ND=3: out [KW*KH*KD*IC, OW, OH, OD*N], params {s0,s1,s2,p0,p1,p2,d0,d1,d2}; image
//   [IW,IH,ID,IC*N]; output dim3 = od + OD*n (depth fastest), decoded here.
// DIRECT: the CONV result itself (F32 accumulation), no im2col output:
//   DIRECT 2D: knl [KW,KH,IC,OC], img [IW,IH,IC,N], out [OW,OH,OC,N], params
//     {s0,s1,p0,p1,d0,d1,KW,KH,IC,OC} (s2/p2/d2 unused).
//   DIRECT 3D (-DND3): knl [KW,KH,KD,IC*OC], img [IW,IH,ID,IC*N], out [OW,OH,OD,OC*N], params
//     {s0,s1,s2,p0,p1,p2,d0,d1,d2,KW,KH,KD,IC,OC}.
// Every output element is written exactly once.
// tensors: 0 img  1 knl (DIRECT only, unused otherwise) | 2 out
#include "syn_common.h"

// -DOUT_F16: the packed taps are stored as f16
#if defined(OUT_F16)
#define ST_OUT(c0, v) { int5 c_ = {(c0), 0, 0, 0, 0}; v_f16_st_tnsr_partial(c_, out, cvt_f16((v), (v)), 0, 0); }
#else
#define ST_OUT(c0, v) ST_F32(out, (c0), (v), 1)
#endif

#if defined(DIRECT)
void main(tensor img, tensor knl, tensor dep, tensor out,
          int nmem,
          int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2,
          int KW, int KH, int KD, int IC, int OC,
          int IW, int IH, int ID, int N, int OW, int OH, int OD)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
#if defined(ND3)
    const int items = OW * OH * OD * N * OC;
#else
    const int items = OW * OH * N * OC;
#endif
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            float64 acc = 0.0f;
#if defined(ND3)
            int r = it;
            const int oc = r % OC; r /= OC;
            const int n  = r % N;  r /= N;
            const int od = r % OD; r /= OD;
            const int oh = r % OH; r /= OH;
            const int ow = r;
            // img [IW,IH,ID,IC*N]: (x,y,z,c,n) = x + IW y + IW IH z + IW IH ID (c + IC n)
            const int pw = (n * IC) * IW * IH * ID;
            // knl [KW,KH,KD,IC*OC]: (x,y,z,c,oc) = x + KW y + KW KH z + KW KH KD (c + IC oc)
            const int pk = oc * IC * KW * KH * KD;
            for (int ic = 0; ic < IC; ic++) {
                const int pwi = pw + ic * IW * IH * ID;
                const int pki = pk + ic * KW * KH * KD;
                for (int ikd = 0; ikd < KD; ikd++) {
                    const int iid = od * s2 + ikd * d2 - p2;
                    if (iid < 0 || iid >= ID) continue;
                    for (int ikh = 0; ikh < KH; ikh++) {
                        const int iih = oh * s1 + ikh * d1 - p1;
                        if (iih < 0 || iih >= IH) continue;
                        const int prow = pwi + iid * IW * IH + iih * IW;
                        const int pkrow = pki + (ikd * KH + ikh) * KW;
                        for (int ikw = 0; ikw < KW; ikw++) {
                            const int iiw = ow * s0 + ikw * d0 - p0;
                            if (iiw < 0 || iiw >= IW) continue;
                            acc = v_f32_mac_b(ld_f32(img, prow + iiw), ld_f32(knl, pkrow + ikw), acc);
                        }
                    }
                }
            }
            // out [OW,OH,OD,OC*N]: (x,y,z,c,n) = x + OW y + OW OH z + OW OH OD c + OW OH OD OC n
            const int po = ow + OW * oh + OW * OH * od + OW * OH * OD * (oc + OC * n);
            ST_F32(out, po, acc, 1);
#else
            int r = it;
            const int oc = r % OC; r /= OC;
            const int n  = r % N;  r /= N;
            const int oh = r % OH; r /= OH;
            const int ow = r;
            // img [IW,IH,IC,N]: (x,y,c,n) = x + IW y + IW IH (c + IC n)
            const int pw = n * IC * IW * IH;
            // knl [KW,KH,IC,OC]: (x,y,c,oc) = x + KW y + KW KH (c + IC oc)
            const int pk = oc * IC * KW * KH;
            for (int ic = 0; ic < IC; ic++) {
                const int pwi = pw + ic * IW * IH;
                const int pki = pk + ic * KW * KH;
                for (int ikh = 0; ikh < KH; ikh++) {
                    const int iih = oh * s1 + ikh * d1 - p1;
                    if (iih < 0 || iih >= IH) continue;
                    const int prow = pwi + iih * IW;
                    const int pkrow = pki + ikh * KW;
                    for (int ikw = 0; ikw < KW; ikw++) {
                        const int iiw = ow * s0 + ikw * d0 - p0;
                        if (iiw < 0 || iiw >= IW) continue;
                        acc = v_f32_mac_b(ld_f32(img, prow + iiw), ld_f32(knl, pkrow + ikw), acc);
                    }
                }
            }
            // out [OW,OH,OC,N]: (x,y,c,n) = x + OW y + OW OH c + OW OH OC n
            const int po = ow + OW * oh + OW * OH * (oc + OC * n);
            ST_F32(out, po, acc, 1);
#endif
        }
    }
}
#else // im2col packing
#if defined(ND3)
void main(tensor img, tensor dep, tensor out,
          int nmem,
          int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2,
          int KW, int KH, int KD, int IC,
          int IW, int IH, int ID, int N, int OW, int OH, int OD,
          int offi, int sb1, int sb2, int sb3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int T = KW * KH * KD * IC; // out dim0 extent
    const int KH_KW = KH * KW;
    const int KD_KH_KW = KD * KH_KW;
    const int OR = OW * OH * OD * N; // output positions (dims 1..3 flat)
    const int items = T * OR;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int r = it;
            const int ikw = r % KW; r /= KW;
            const int ikh = r % KH; r /= KH;
            const int ikd = r % KD; r /= KD;
            const int ic  = r % IC; r /= IC;
            const int o   = r;      // flat [OW, OH, OD*N]
            const int o3  = o / (OW * OH);
            const int orr = o - o3 * OW * OH;
            const int oh  = orr / OW;
            const int ow  = orr - oh * OW;
            const int n   = o3 / OD;
            const int od  = o3 - n * OD;
            const int iiw = ow * s0 + ikw * d0 - p0;
            const int iih = oh * s1 + ikh * d1 - p1;
            const int iid = od * s2 + ikd * d2 - p2;
            float64 v = 0.0f;
            if (iiw >= 0 && iiw < IW && iih >= 0 && iih < IH && iid >= 0 && iid < ID) {
                // img [IW,IH,ID,IC*N] view: (x,y,z,c,n) = off + x + sb1 y + sb2 z + sb3 (c + IC n)
                const int p = offi + (n * IC + ic) * sb3 + iid * sb2 + iih * sb1 + iiw;
                v = ld_f32(img, p);
            }
            // out flat = tap_flat + T * o, tap_flat = ic*KD_KH_KW + ikd*KH_KW + ikh*KW + ikw
            ST_OUT(o * T + (ic * KD_KH_KW + ikd * KH_KW + ikh * KW + ikw), v);
        }
    }
}
#else
void main(tensor img, tensor dep, tensor out,
          int nmem,
          int s0, int s1, int p0, int p1, int d0, int d1, int is_2D,
          int KW, int KH, int IC,
          int IW, int IH, int OW, int OH, int N,
          int offi, int sb1, int sb2, int sb3)
{
    const int5 ioff = get_index_space_offset();
    const int5 iend = ioff + get_index_space_size();
    const int T = KW * KH * IC; // out dim0 extent
    const int KH_KW = KH * KW;
    const int OR = is_2D ? OW * OH * N : OW * N;
    const int items = T * OR;
    for (int m = ioff[0]; m < iend[0]; m++) {
        int i0, i1;
        SPLIT_RANGE(items, nmem, m, i0, i1)
        for (int it = i0; it < i1; it++) {
            int r = it;
            const int ikw = r % KW; r /= KW;
            const int ikh = r % KH; r /= KH;
            const int ic  = r % IC; r /= IC;
            const int o   = r;      // flat [OW, OH, N] (2D) or [OL, N] (1D)
            float64 v = 0.0f;
            if (is_2D) {
                const int o3  = o / (OW * OH);
                const int orr = o - o3 * OW * OH;
                const int oh  = orr / OW;
                const int ow  = orr - oh * OW;
                const int iiw = ow * s0 + ikw * d0 - p0;
                const int iih = oh * s1 + ikh * d1 - p1;
                if (iiw >= 0 && iiw < IW && iih >= 0 && iih < IH) {
                    // img [IW,IH,IC,N] view: (x,y,c,n) = off + x + sb1 y + sb2 c + sb3 n
                    v = ld_f32(img, offi + o3 * sb3 + ic * sb2 + iih * sb1 + iiw);
                }
            } else {
                const int ow = o % OW;
                const int n  = o / OW;
                const int iiw = ow * s0 + ikw * d0 - p0;
                if (iiw >= 0 && iiw < IW) {
                    // img [IW, IC, N] view: (x, c, n) = off + x + sb2 c + sb3 n (the host passes the 1D strides so)
                    v = ld_f32(img, offi + n * sb3 + ic * sb2 + iiw);
                }
            }
            // out flat = tap_flat + T * o, tap_flat = ic*KH_KW + ikh*KW + ikw
            ST_OUT(o * T + (ic * KH_KW + ikh * KW + ikw), v);
        }
    }
}
#endif
#endif

# ggml-synapse: SynapseAI (Intel Gaudi2) backend.
#
# Builds two artifacts:
#   1. ggml-synapse             - the backend library
#   2. ggml_synapse_kernels     - the backend TPC kernel library plugin
#                                 (libggml_synapse_kernels.so), installed
#                                 beside the backend and resolved relative
#                                 to the loaded backend at run time.
#
# Requires:
#   - habanalabs SynapseAI headers/libs (/usr/include/habanalabs + /usr/lib/habanalabs)
#   - habanatools glue headers (/usr/lib/habanatools/include)
#   - tpc-clang on PATH, objcopy for kernel blob embedding
#
# Linux x86-64/Gaudi2 only: other platforms are a configure error so the
# objcopy objects are never built for the wrong ISA.

if (NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux") OR NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"))
    message(FATAL_ERROR "GGML_SYNAPSE requires Linux x86-64 (Gaudi2)")
endif()

find_program(TPC_CLANG tpc-clang REQUIRED)
find_program(OBJCOPY objcopy REQUIRED)

if (NOT EXISTS /usr/include/habanalabs/synapse_api.h)
    message(FATAL_ERROR "GGML_SYNAPSE: SynapseAI headers not found (/usr/include/habanalabs)")
endif()
if (NOT EXISTS /usr/lib/habanatools/include/tpc_kernel_lib_interface.h)
    message(FATAL_ERROR "GGML_SYNAPSE: habanatools glue headers not found (/usr/lib/habanatools/include)")
endif()
if (NOT EXISTS /usr/lib/habanalabs/libSynapse.so)
    message(FATAL_ERROR "GGML_SYNAPSE: libSynapse.so not found (/usr/lib/habanalabs)")
endif()

set(SYNAPSE_INCLUDE_DIRS
    /usr/include/habanalabs
    /usr/lib/habanatools/include
)

# ---- TPC kernels -------------------------------------------------------
# name|source|defines (one source builds several variants). The guid of kernel <name> is
# <name>_f32; the glue table is generated from this list.
set(TPC_KERNELS
    "syn_binary|syn_binary_f32.c|"
    "syn_addrms_bf|syn_addrms_bf.c|"
    "syn_addrms_f8|syn_addrms_bf.c|-DOUT_FP8"
    "syn_normrope_c|syn_normrope_c.c|"
    "syn_swiglu_bf|syn_swiglu_bf.c|"
    "syn_swiglu_f8|syn_swiglu_bf.c|-DOUT_FP8"
    "syn_binary_bf|syn_binary_f32.c|-DOUT_BF16"
    "syn_binary_add|syn_binary_f32.c|-DOPC=0"
    "syn_binary_sub|syn_binary_f32.c|-DOPC=1"
    "syn_binary_mul|syn_binary_f32.c|-DOPC=2"
    "syn_binary_div|syn_binary_f32.c|-DOPC=3"
    "syn_binary_swiglu|syn_binary_f32.c|-DOPC=4"
    "syn_binary_geglu|syn_binary_f32.c|-DOPC=5"
    "syn_unary|syn_unary_f32.c|"
    "syn_unary_bf|syn_unary_f32.c|-DOUT_BF16"
    "syn_unary_flat|syn_unary_f32.c|-DFLAT"
    "syn_unary_flat_bf|syn_unary_f32.c|-DFLAT -DOUT_BF16"
    "syn_unary_silu|syn_unary_f32.c|-DFLAT -DOPC=0"
    "syn_unary_sigm|syn_unary_f32.c|-DFLAT -DOPC=1"
    "syn_unary_gelu|syn_unary_f32.c|-DFLAT -DOPC=6"
    "syn_unary_scale|syn_unary_f32.c|-DFLAT -DOPC=8"
    "syn_unaryx|syn_unaryx_f32.c|"
    "syn_glux|syn_glux_f32.c|"
    "syn_repeat|syn_repeat_f32.c|"
    "syn_rowsum|syn_rowsum_f32.c|"
    "syn_argmax|syn_rowsum_f32.c|-DARGMAX"
    "syn_rowmax|syn_rowsum_f32.c|-DROWMAX"
    "syn_argsort|syn_argsort_f32.c|"
    "syn_router|syn_router_f32.c|"
    "syn_mv|syn_mv_f32.c|"
    "syn_mv_bf|syn_mv_f32.c|-DW_BF16"
    "syn_moe_sum|syn_moe_sum_f32.c|"
    "syn_moe_sumi|syn_moe_sum_f32.c|-DIND"
    "syn_topk_thr|syn_topk_f32.c|-DTHR"
    "syn_topk_pick|syn_topk_f32.c|"
    "syn_topk_pickv|syn_topk_f32.c|-DVALS"
    "syn_topk_picki|syn_topk_f32.c|-DIND"
    "syn_bsort_init|syn_bsort_f32.c|-DINIT"
    "syn_bsort_glob|syn_bsort_f32.c|-DGLOB"
    "syn_bsort_loc|syn_bsort_f32.c|-DLOC"
    "syn_bsort_out|syn_bsort_f32.c|-DOUT"
    "syn_add_id|syn_add_id_f32.c|"
    "syn_mmid|syn_mmid_f32.c|"
    "syn_qmv_q4_0|syn_qmm.c|-DQT_Q4_0"
    "syn_qdq_q4_0|syn_qdq.c|-DQT_Q4_0"
    "syn_qmv_q4_1|syn_qmm.c|-DQT_Q4_1"
    "syn_qdq_q4_1|syn_qdq.c|-DQT_Q4_1"
    "syn_qmv_q5_0|syn_qmm.c|-DQT_Q5_0"
    "syn_qdq_q5_0|syn_qdq.c|-DQT_Q5_0"
    "syn_qmv_q5_1|syn_qmm.c|-DQT_Q5_1"
    "syn_qdq_q5_1|syn_qdq.c|-DQT_Q5_1"
    "syn_qmv_q2_k|syn_qmm.c|-DQT_Q2_K"
    "syn_qdq_q2_k|syn_qdq.c|-DQT_Q2_K"
    "syn_qmv_q3_k|syn_qmm.c|-DQT_Q3_K"
    "syn_qdq_q3_k|syn_qdq.c|-DQT_Q3_K"
    "syn_qmv_q4_k|syn_qmm.c|-DQT_Q4_K"
    "syn_qdq_q4_k|syn_qdq.c|-DQT_Q4_K"
    "syn_qdqp_q2_k|syn_qdq.c|-DQT_Q2_K -DPAGED"
    "syn_qdqp_q3_k|syn_qdq.c|-DQT_Q3_K -DPAGED"
    "syn_qdqp_q4_k|syn_qdqp_q4k.c|"
    "syn_qdqp_q5_k|syn_qdqp_q5k.c|"
    "syn_qdqp_q6_k|syn_qdq.c|-DQT_Q6_K -DPAGED"
    "syn_qmv_q5_k|syn_qmm.c|-DQT_Q5_K"
    "syn_qdq_q5_k|syn_qdq.c|-DQT_Q5_K"
    "syn_qmv_q6_k|syn_qmm.c|-DQT_Q6_K"
    "syn_qdq_q6_k|syn_qdq.c|-DQT_Q6_K"
    "syn_qmv_iq4_nl|syn_qmm.c|-DQT_IQ4_NL"
    "syn_qdq_iq4_nl|syn_qmm.c|-DQT_IQ4_NL -DDEQ"
    "syn_qmv_iq4_xs|syn_qmm.c|-DQT_IQ4_XS"
    "syn_qdq_iq4_xs|syn_qmm.c|-DQT_IQ4_XS -DDEQ"
    "syn_qmv_mxfp4|syn_qmm.c|-DQT_MXFP4"
    "syn_qdq_mxfp4|syn_qdq.c|-DQT_MXFP4"
    "syn_qmv_nvfp4|syn_qmm.c|-DQT_NVFP4"
    "syn_qdq_nvfp4|syn_qdq.c|-DQT_NVFP4"
    "syn_qmv_tq2_0|syn_qmm.c|-DQT_TQ2_0"
    "syn_qdq_tq2_0|syn_qdq.c|-DQT_TQ2_0"
    "syn_qmv_q1_0|syn_qmm.c|-DQT_Q1_0"
    "syn_qdq_q1_0|syn_qdq.c|-DQT_Q1_0"
    "syn_qmv_q2_0|syn_qmm.c|-DQT_Q2_0"
    "syn_qdq_q2_0|syn_qdq.c|-DQT_Q2_0"
    "syn_qmv_q8_0|syn_qmm.c|-DQT_Q8_0"
    "syn_qdq_q8_0|syn_qdq.c|-DQT_Q8_0"
    "syn_qdqp_q8_0|syn_qdq.c|-DQT_Q8_0 -DPAGED"
    "syn_qmv8_q8_0|syn_qmm.c|-DQT_Q8_0 -DI8"
    "syn_qmv_fp8|syn_qmm.c|-DQT_FP8"
    "syn_qdq_fp8|syn_qdq.c|-DQT_FP8"
    "syn_moe_gather|syn_moe_f32.c|-DGATHER"
    "syn_moe_sel|syn_moe_f32.c|"
    "syn_moe_cnt|syn_moe_page.c|-DCNT"
    "syn_moe_tot|syn_moe_page.c|-DTOT"
    "syn_moe_plan|syn_moe_page.c|-DPLAN"
    "syn_moe_base|syn_moe_page.c|-DBASE"
    "syn_moe_glu|syn_moe_page.c|-DGLU"
    "syn_moe_asg|syn_moe_page.c|-DASG"
    "syn_moe_gx|syn_moe_page.c|-DGX"
    "syn_moe_scat|syn_moe_page.c|"
    "syn_fp8q|syn_fp8q.c|"
    "syn_ksum|syn_ksum_f32.c|"
    "syn_xq8|syn_xq8_f32.c|"
    "syn_xq8c|syn_xq8_f32.c|-DXQ"
    "syn_qmv8_q4_0|syn_qmm.c|-DQT_Q4_0 -DI8"
    "syn_qmv8_q4_1|syn_qmm.c|-DQT_Q4_1 -DI8"
    "syn_qmv8_q5_0|syn_qmm.c|-DQT_Q5_0 -DI8"
    "syn_qmv8_q5_1|syn_qmm.c|-DQT_Q5_1 -DI8"
    "syn_qmv8_q2_k|syn_qmm.c|-DQT_Q2_K -DI8"
    "syn_qmv8_q3_k|syn_qmm.c|-DQT_Q3_K -DI8"
    "syn_qmv8_q4_k|syn_qmm.c|-DQT_Q4_K -DI8"
    "syn_qmv8_q5_k|syn_qmm.c|-DQT_Q5_K -DI8"
    "syn_qmv8_q6_k|syn_qmm.c|-DQT_Q6_K -DI8"
    "syn_qmv8_tq2_0|syn_qmm.c|-DQT_TQ2_0 -DI8"
    "syn_qmv8_q1_0|syn_qmm.c|-DQT_Q1_0 -DI8"
    "syn_qmv8_q2_0|syn_qmm.c|-DQT_Q2_0 -DI8"
    "syn_qmv8_mxfp4|syn_qmm.c|-DQT_MXFP4 -DI8"
    "syn_qmv8_nvfp4|syn_qmm.c|-DQT_NVFP4 -DI8"
    "syn_fp8amax|syn_fp8q.c|-DAMAX"
    "syn_fp8q1|syn_fp8q.c|-DONE"
    "syn_fp8qt|syn_fp8q.c|-DTOK"
    "syn_fp8cols|syn_fp8q.c|-DCOLS"
    "syn_fp8rc|syn_fp8q.c|-DCOLS -DROWS"
    "syn_fp8qr|syn_fp8q.c|-DROWQ"
    "syn_norm|syn_norm_f32.c|"
    "syn_norm_bf|syn_norm_f32.c|-DOUT_BF16"
    "syn_normrope|syn_norm_f32.c|-DROPE"
    "syn_normrope_bf|syn_norm_f32.c|-DROPE -DOUT_BF16"
    "syn_addnorm|syn_norm_f32.c|-DADD"
    "syn_addnorm_bf|syn_norm_f32.c|-DADD -DOUT_BF16"
    "syn_addnorm_moe|syn_norm_f32.c|-DADD -DMOE"
    "syn_addnorm_moe_bf|syn_norm_f32.c|-DADD -DMOE -DOUT_BF16"
    "syn_addnorm_moe_i|syn_norm_f32.c|-DADD -DMOE -DIND"
    "syn_addnorm_moe_bf_i|syn_norm_f32.c|-DADD -DMOE -DIND -DOUT_BF16"
    "syn_cpy_f32_f32|syn_cpy.c|-DSRC_F32 -DDST_F32"
    "syn_cpy_f32_f16|syn_cpy.c|-DSRC_F32 -DDST_F16"
    "syn_cpy_f32_bf16|syn_cpy.c|-DSRC_F32 -DDST_BF16"
    "syn_cpy_f32_f16_kv|syn_cpy.c|-DSRC_F32 -DDST_F16 -DKV"
    "syn_cpy_f32_bf16_kv|syn_cpy.c|-DSRC_F32 -DDST_BF16 -DKV"
    "syn_cpy_f16_f32|syn_cpy.c|-DSRC_F16 -DDST_F32"
    "syn_cpy_bf16_f32|syn_cpy.c|-DSRC_BF16 -DDST_F32"
    "syn_cpy_f16_f16|syn_cpy.c|-DSRC_F16 -DDST_F16"
    "syn_cpy_f16_bf16|syn_cpy.c|-DSRC_F16 -DDST_BF16"
    "syn_cpy_bf16_f16|syn_cpy.c|-DSRC_BF16 -DDST_F16"
    "syn_cpy_bf16_bf16|syn_cpy.c|-DSRC_BF16 -DDST_BF16"
    "syn_cpy_i32_i32|syn_cpy.c|-DSRC_I32 -DDST_I32"
    "syn_cpy_f32_i32|syn_cpy.c|-DSRC_F32 -DDST_I32"
    "syn_cpy_i32_f32|syn_cpy.c|-DSRC_I32 -DDST_F32"
    "syn_get_rows_f32|syn_get_rows.c|-DSRC_F32"
    "syn_get_rows_f16|syn_get_rows.c|-DSRC_F16"
    "syn_get_rows_bf16|syn_get_rows.c|-DSRC_BF16"
    "syn_get_rows_i32|syn_get_rows.c|-DSRC_I32"
    "syn_pad|syn_pad_f32.c|"
    "syn_pad_circ|syn_pad_f32.c|-DCIRC"
    "syn_cumsum|syn_cumsum_f32.c|"
    "syn_arange|syn_cumsum_f32.c|-DARANGE"
    "syn_concat|syn_concat_f32.c|"
    "syn_ssm_conv_c|syn_ssm_conv_c.c|"
    "syn_ssm_convx|syn_ssm_convx_f32.c|"
    "syn_ssm_scan|syn_ssm_scan_f32.c|"
    "syn_wkv6|syn_wkv_f32.c|-DWKV6"
    "syn_gla|syn_wkv_f32.c|-DGLA"
    "syn_wkv7|syn_wkv_f32.c|-DWKV7"
    "syn_gdn_ck_prep|syn_gdn_ck.c|-DPREP"
    "syn_gdn_ck_solve|syn_gdn_ck.c|-DSOLVE"
    "syn_gdn_ck_step|syn_gdn_ck.c|-DSTEP"
    "syn_gdn_ck_out|syn_gdn_ck.c|-DOUT"
    "syn_gdn|syn_gdn_f32.c|"
    "syn_gdn_conv|syn_gdn_conv_f32.c|"
    "syn_gdn_dec|syn_gdn_dec_f32.c|"
    "syn_gdn_dec_bf|syn_gdn_dec_f32.c|-DOUT_BF16"
    "syn_gdn_full|syn_gdn_full_f32.c|"
    "syn_gdn_full_bf|syn_gdn_full_f32.c|-DOUT_BF16"
    "syn_gdn_conv_mt|syn_gdn_conv_f32.c|-DMT"
    "syn_gdn_mt|syn_gdn_mt_f32.c|-DDEC"
    "syn_gdn_onorm|syn_gdn_mt_f32.c|-DONORM"
    "syn_gdn_onorm_bf|syn_gdn_mt_f32.c|-DONORM -DOUT_BF16"
    "syn_rope|syn_rope_f32.c|"
    "syn_rope_h|syn_rope_f32.c|-DIO_F16"
    "syn_ropen|syn_rope_f32.c|-DNORM"
    "syn_ropen_h|syn_rope_f32.c|-DNORM -DIO_F16"
    "syn_rope_bf|syn_rope_f32.c|-DOUT_BF16"
    "syn_ropen_bf|syn_rope_f32.c|-DNORM -DOUT_BF16"
    "syn_fattn_f16|syn_fattn.c|-DKV_F16"
    "syn_fattn_bf16|syn_fattn.c|-DKV_BF16"
    "syn_fattn_f32|syn_fattn.c|-DKV_F32"
    "syn_fattn_q8|syn_fattn.c|-DKV_Q8"
    "syn_fattn_w_f16|syn_fattn.c|-DKV_F16 -DWIDE"
    "syn_fattn_w_bf16|syn_fattn.c|-DKV_BF16 -DWIDE"
    "syn_fattn_w_f32|syn_fattn.c|-DKV_F32 -DWIDE"
    "syn_fattn_w_q8|syn_fattn.c|-DKV_Q8 -DWIDE"
    "syn_fattn_comb|syn_fattn_comb_f32.c|"
    "syn_fattn_comb_w|syn_fattn_comb_f32.c|-DWIDE"
    "syn_fattn_dec|syn_fattn_dec.c|"
    "syn_fattn_decg|syn_fattn_dec.c|-DGATE_OUT"
    "syn_fattn_dec64|syn_fattn_dec.c|-DND=1"
    "syn_fattn_dec128|syn_fattn_dec.c|-DND=2"
    "syn_memcpy_h|syn_fa_aux.c|-DMEMCPY -DT_F16"
    "syn_memcpy_b|syn_fa_aux.c|-DMEMCPY -DT_BF16"
    "syn_fa_qbd_h|syn_fa_aux.c|-DT_F16"
    "syn_fa_qbd_b|syn_fa_aux.c|-DT_BF16"
    "syn_fattn_combg_c|syn_fattn_combg_c.c|"
    "syn_fattn_combg|syn_fattn_comb_f32.c|-DGATE"
    "syn_fattn_combg_bf|syn_fattn_comb_f32.c|-DGATE -DOUT_BF16"
    "syn_softmax_f16|syn_softmax.c|-DMASK_F16"
    "syn_softmax_f32|syn_softmax.c|-DMASK_F32"
    "syn_softmax_f16_h|syn_softmax.c|-DMASK_F16 -DOUT_F16"
    "syn_softmax_f16_b|syn_softmax.c|-DMASK_F16 -DOUT_BF16"
    "syn_softmax_f32_h|syn_softmax.c|-DMASK_F32 -DOUT_F16"
    "syn_softmax_f32_b|syn_softmax.c|-DMASK_F32 -DOUT_BF16"
    "syn_softmax_u_h|syn_softmax_u.c|-DOUT_F16"
    "syn_softmax_u_b|syn_softmax_u.c|-DOUT_BF16"
    "syn_fa_norm|syn_fa_norm.c|"
    "syn_fat_sm_h|syn_fa_tile.c|-DSM -DOUT_F16"
    "syn_fat_sm_b|syn_fa_tile.c|-DSM -DOUT_BF16"
    "syn_fat_sm_hs|syn_fa_tile.c|-DSM -DSINKS -DOUT_F16"
    "syn_fat_sm_bs|syn_fa_tile.c|-DSM -DSINKS -DOUT_BF16"
    "syn_fat_mrg|syn_fa_tile.c|-DMRG"
    "syn_fat_mask|syn_fa_tile.c|-DMASK"
    "syn_fat_pmrg|syn_fa_tile.c|-DPMRG"
    "syn_join|syn_join_f32.c|"
    "syn_test|syn_test.c|"
    "syn_topk_mask_f16|syn_topk_mask.c|-DF16"
    "syn_topk_mask_f32|syn_topk_mask.c|-DF32"
    "syn_topk_mask_g_f16|syn_topk_mask.c|-DF16 -DGEN"
    "syn_topk_mask_g_f32|syn_topk_mask.c|-DF32 -DGEN"
    "syn_hc_pre|syn_hc.c|-DPRE"
    "syn_hc_comb|syn_hc_comb_f32.c|"
    "syn_hc_post|syn_hc.c|-DPOST"
    "syn_hc_postn|syn_hc.c|-DPOST -DNORM"
    "syn_hc_postn_bf|syn_hc.c|-DPOST -DNORM -DOUT_BF16"
    "syn_wr_f32|syn_wr.c|-DDST_F32"
    "syn_wr_i32|syn_wr.c|-DDST_I32"
    "syn_wr_f16|syn_wr.c|-DDST_F16"
    "syn_wr_bf16|syn_wr.c|-DDST_BF16"
    "syn_wr_f16s|syn_wr.c|-DDST_F16 -DSRC_SAME"
    "syn_wr_bf16s|syn_wr.c|-DDST_BF16 -DSRC_SAME"
    "syn_setrows2_f32|syn_setrows2.c|-DDST_F32"
    "syn_setrows2_f16|syn_setrows2.c|-DDST_F16"
    "syn_setrows2_bf16|syn_setrows2.c|-DDST_BF16"
    "syn_setrows_f32|syn_wr.c|-DDST_F32 -DSET_ROWS"
    "syn_setrows_f16|syn_wr.c|-DDST_F16 -DSET_ROWS"
    "syn_setrows_bf16|syn_wr.c|-DDST_BF16 -DSET_ROWS"
    "syn_setrows_q8|syn_setrows_q8.c|"
    "syn_setrows_q4_0|syn_setrows_q4.c|-DQ4_0"
    "syn_setrows_q4_1|syn_setrows_q4.c|-DQ4_1"
    "syn_setrows_q5_0|syn_setrows_q4.c|-DQ5_0"
    "syn_setrows_q5_1|syn_setrows_q4.c|-DQ5_1"
    "syn_setrows_iq4_nl|syn_setrows_q4.c|-DIQ4_NL"
    "syn_dqrows_q4_0|syn_dqrows.c|-DQ4_0"
    "syn_dqrows_q4_1|syn_dqrows.c|-DQ4_1"
    "syn_dqrows_q5_0|syn_dqrows.c|-DQ5_0"
    "syn_dqrows_q5_1|syn_dqrows.c|-DQ5_1"
    "syn_dqrows_q8_0|syn_dqrows.c|-DQ8_0"
    "syn_dqrows32_q4_0|syn_dqrows.c|-DQ4_0 -DOUT_F32"
    "syn_dqrows32_q4_1|syn_dqrows.c|-DQ4_1 -DOUT_F32"
    "syn_dqrows32_q5_0|syn_dqrows.c|-DQ5_0 -DOUT_F32"
    "syn_dqrows32_q5_1|syn_dqrows.c|-DQ5_1 -DOUT_F32"
    "syn_dqrows32_q8_0|syn_dqrows.c|-DQ8_0 -DOUT_F32"
    "syn_cpy_f32_q8_0|syn_setrows_q8.c|-DCPY"
    "syn_cpy_f32_q4_0|syn_setrows_q4.c|-DQ4_0 -DCPY"
    "syn_cpy_f32_q4_1|syn_setrows_q4.c|-DQ4_1 -DCPY"
    "syn_cpy_f32_q5_0|syn_setrows_q4.c|-DQ5_0 -DCPY"
    "syn_cpy_f32_q5_1|syn_setrows_q4.c|-DQ5_1 -DCPY"
    "syn_cpy_f32_iq4_nl|syn_setrows_q4.c|-DIQ4_NL -DCPY"
    "syn_lid|syn_lid_f32.c|"
    "syn_outprod|syn_outprod_f32.c|"
    "syn_set|syn_set_f32.c|"
    "syn_set_i32|syn_set_f32.c|-DSRC_I32"
    "syn_set_acc|syn_set_f32.c|-DACC"
    "syn_count_eq|syn_count_eq.c|-DPARTIAL"
    "syn_count_eq_final|syn_count_eq.c|"
    "syn_pool|syn_pool_f32.c|"
    "syn_pool2d|syn_pool_f32.c|-DPOOL2D"
    "syn_im2col|syn_im2col_f32.c|"
    "syn_im2col3d|syn_im2col_f32.c|-DND3"
    "syn_im2col_direct|syn_im2col_f32.c|-DDIRECT"
    "syn_im2col_direct3d|syn_im2col_f32.c|-DDIRECT -DND3"
    "syn_conv2d_dw|syn_conv2d_dw_f32.c|"
    "syn_conv2d_dw_f16|syn_conv2d_dw_f32.c|-DW_F16"
    "syn_conv2d_dw_cwhn|syn_conv2d_dw_f32.c|-DCWHN"
    "syn_conv2d_dw_cwhn_f16|syn_conv2d_dw_f32.c|-DCWHN -DW_F16"
    "syn_conv_transpose|syn_conv_transpose_f32.c|"
    "syn_conv_transpose_f16|syn_conv_transpose_f32.c|-DW_F16"
    "syn_conv_transpose2d|syn_conv_transpose_f32.c|-DND2"
    "syn_conv_transpose2d_f16|syn_conv_transpose_f32.c|-DND2 -DW_F16"
    "syn_col2im|syn_col2im_f32.c|"
    "syn_col2im_f16|syn_col2im_f32.c|-DSRC_F16"
    "syn_col2im_bf16|syn_col2im_f32.c|-DSRC_BF16"
    "syn_reflect|syn_index_f32.c|"
    "syn_roll|syn_index_f32.c|-DROLL"
    "syn_upscale_nearest|syn_upscale_f32.c|-DNEAREST"
    "syn_upscale_bilinear|syn_upscale_f32.c|-DBILINEAR"
    "syn_upscale_bicubic|syn_upscale_f32.c|-DBICUBIC"
    "syn_upscale_aa|syn_upscale_f32.c|-DBILINEAR -DANTIALIAS"
    "syn_fattn_general|syn_fattn_general_f32.c|"
    "syn_ssm_scan_gen|syn_ssm_scan_general_f32.c|"
    "syn_ssm_scan_gen_red|syn_ssm_scan_general_f32.c|-DREDUCE"
    "syn_wkv6_gen|syn_wkv_general_f32.c|-DWKV6"
    "syn_gla_gen|syn_wkv_general_f32.c|-DGLA"
    "syn_wkv_gen_red|syn_wkv_general_f32.c|-DREDUCE"
    "syn_wkv7_gen|syn_wkv_general_f32.c|-DWKV7"
    "syn_timestep|syn_timestep_f32.c|"
    "syn_rope_tab|syn_rope_tab_f32.c|"
    "syn_groupnorm|syn_groupnorm_f32.c|"
    "syn_diag|syn_diag_f32.c|-DDIAG"
    "syn_tri|syn_diag_f32.c|-DTRI"
    "syn_mask|syn_diag_f32.c|-DMASK"
    "syn_solve_tri|syn_solve_tri_f32.c|"
)

set(TPC_BLOB_DIR ${CMAKE_CURRENT_BINARY_DIR}/tpc_blobs)
file(GLOB TPC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/tpc/*.h)
set(KERNEL_TABLE_CONTENT "")

foreach(entry ${TPC_KERNELS})
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 kernel)
    list(GET parts 1 ksrc)
    list(LENGTH parts nparts)
    set(kdefs "")
    if (nparts GREATER 2)
        list(GET parts 2 kdefs_str)
        separate_arguments(kdefs UNIX_COMMAND "${kdefs_str}")
    endif()
    set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tpc/${ksrc})
    # sources that #include another kernel source
    set(KERNEL_EXTRA_DEPS "")
    if (ksrc STREQUAL "syn_qdq.c")
        set(KERNEL_EXTRA_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/tpc/syn_qmm.c)
    endif()
    set(KERNEL_ELF ${TPC_BLOB_DIR}/${kernel}.o)
    set(KERNEL_BIN ${TPC_BLOB_DIR}/${kernel}_bin.o)
    add_custom_command(
        OUTPUT ${KERNEL_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TPC_BLOB_DIR}
        COMMAND ${TPC_CLANG} -march=gaudi2 -O2 -x c++ ${kdefs} -c ${KERNEL_SRC} -I${CMAKE_CURRENT_SOURCE_DIR}/tpc -o ${KERNEL_ELF}
        DEPENDS ${KERNEL_SRC} ${TPC_HEADERS} ${KERNEL_EXTRA_DEPS}
        COMMENT "tpc-clang: ${kernel}")
    add_custom_command(
        OUTPUT ${KERNEL_BIN}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${TPC_BLOB_DIR}/flat
        COMMAND ${CMAKE_COMMAND} -E copy ${KERNEL_ELF} ${TPC_BLOB_DIR}/flat/${kernel}_bin
        COMMAND ${CMAKE_COMMAND} -E chdir ${TPC_BLOB_DIR}/flat ${OBJCOPY} -I binary -O elf64-x86-64 -B i386:x86-64 ${kernel}_bin ${KERNEL_BIN}
        DEPENDS ${KERNEL_ELF}
        COMMENT "objcopy: ${kernel}")
    list(APPEND TPC_KERNEL_BINS ${KERNEL_BIN})
    string(APPEND KERNEL_TABLE_CONTENT "K(${kernel})\n")
endforeach()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/glue_gen)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/glue_gen/kernel_table.h.tmp "${KERNEL_TABLE_CONTENT}")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/glue_gen/kernel_table.h.tmp
               ${CMAKE_CURRENT_BINARY_DIR}/glue_gen/kernel_table.h COPYONLY)

add_library(ggml_synapse_kernels SHARED glue/glue.cpp ${TPC_KERNEL_BINS})
target_include_directories(ggml_synapse_kernels PRIVATE
    ${SYNAPSE_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}/glue_gen)
target_compile_features(ggml_synapse_kernels PRIVATE cxx_std_17)
set_target_properties(ggml_synapse_kernels PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    CXX_VISIBILITY_PRESET default)

# ---- backend library -----------------------------------------------------
set(GGML_SOURCES_SYNAPSE
    ggml-synapse.cpp
    ggml-synapse-graph.cpp
    ggml-synapse-quant.cpp
    ggml-synapse-proxy.cpp
)

ggml_add_backend_library(ggml-synapse ${GGML_SOURCES_SYNAPSE})
# hlml: the card list and memory figures (as hl-smi), without opening the cards
target_link_libraries(ggml-synapse PRIVATE Synapse hcl hlml)
target_link_directories(ggml-synapse PRIVATE /usr/lib/habanalabs)
target_include_directories(ggml-synapse PRIVATE ${SYNAPSE_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ggml-synapse PRIVATE cxx_std_17)

add_dependencies(ggml-synapse ggml_synapse_kernels)

# one process per card when several cards are visible, started by the backend on the first use of a card
add_executable(ggml-synapse-worker ggml-synapse-worker.cpp)
target_link_libraries(ggml-synapse-worker PRIVATE ggml-synapse)
set_target_properties(ggml-synapse-worker PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# install the kernel plugin beside the backend so it is discoverable
# relative to the loaded backend library
if (GGML_BACKEND_DL)
    install(TARGETS ggml_synapse_kernels
            LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    install(TARGETS ggml_synapse_kernels
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

message(STATUS "Synapse: backend + kernels enabled (tpc-clang=${TPC_CLANG})")

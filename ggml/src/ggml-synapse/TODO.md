# SYNAPSE backend TODO

## MUL_MAT / MUL_MAT_ID src1 precision

Upstream #31896 added `ggml_prec_set_src(op, prec, 1)`: the backend must not convert src1 below the requested rank (F32 > BF16 > F16 > Q8 > Q4). The GEMM here converts src1 to the weight type (F16 for F16 weights) or down to Q8 rank for quantized weights (int8 GEMV, fp8 rows). With F16 weights and src1 values outside the F16 range the result is NaN (`MUL_MAT_PREC_SRC1_F32` / `MUL_MAT_PREC_SRC1_BF16`, `type_a=f16`).

For now `supports_op` declines these ops (`mm_src1_prec_ok` in `ggml-synapse-graph.cpp`), so they run on the CPU. Current users do not hit this: llama only asks for Q8, mtmd asks for F32 with F32 x F32.

To do: honor the request on the device instead of declining. Steps:
- F16 weights, BF16 request: run the GEMM in BF16 (convert the F16 weight, keep src1 BF16).
- F16 / BF16 weights, F32 request: run the GEMM in F32.
- Quantized weights, F16 / BF16 request: skip the int8 GEMV and the fp8 rows activation quantization, use the BF16 dequant path.

## FLASH_ATTN_EXT over K/V rows

Upstream #29510 added `ggml_flash_attn_ext_rows`: `src[5]` is an I32 list of K/V cache rows per q slice (`[kv, n_slices]`, -1 for padding), K/V are the whole cache shared by all slices. `fa_config_ok` declines it, so llama's probe turns `fused_kv_rows` off and llama attends to the whole unified cache with a dense mask (the path from before the merge, same speed for one sequence).

To do (only worth it for several sequences in one unified cache, f.ex. server slots): gather the listed rows of K and V into dense `[D, kv, Hkv, n_slices]` copies in the cache type (F16 / BF16; the get_rows kernel needs a same-type output variant), then run the normal lowering with one q batch per slice. Quantized caches (q8_0) need a raw block gather. Without it each decode step pays for the whole cache instead of the sequence's own cells.

## Partial offload of quantized models gives NaN

With quantized weights left on the CPU (`-ngl 0`, or a memory fit that keeps layers on the CPU), the ops that `offload_op` sends to the card for batches of 32 or more give NaN logits (Qwen3.5-0.8B Q8_0, Llama-3.2-1B Q8_0; in the process and in a card worker; also in the 2026-11-09 build). BF16 models are not affected. To do: find the op (`GGML_SYNAPSE_NOT` bisect with `-ngl 0`) and fix or decline it in `offload_op`.

## Card worker processes

- A graph that fails to lower in a worker aborts the whole process (`proxy_wait_seq`): the parent queues `CMD_COMPUTE` without waiting. A clean error needs the parent to wait for the first compute of each new graph (about 40 lines; with tensor split the first compiles of the cards then run one after the other).
- Prefill in a worker is slower than in the process (Qwen3.5-0.8B BF16 pp512 about 21-23k vs 26k tok/s; decode the same).
- Fusion plans must only walk producers that are nodes of this graph (`order`): in a worker the nodes of other splits arrive without srcs. The FA rope plan and `match_gdn` check this; the other plans (`norm_plans` reading `pl.w`, the router, MoE sum) are not audited yet.

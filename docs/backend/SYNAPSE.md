# llama.cpp for Intel Gaudi2 (SYNAPSE)

## Background

The SYNAPSE backend runs ggml graphs on Intel Gaudi2 accelerators through the SynapseAI software stack. The backend lowers each ggml graph to a Synapse graph, compiles it with the Synapse graph compiler, and launches the compiled recipe. Matrix multiplications run on the MME (matrix engine). Most other ops run as TPC (tensor processor core) kernels that the backend ships in its own kernel library.

## Software and Hardware Support

| Item | Supported |
| --- | --- |
| Accelerator | Intel Gaudi2 |
| OS / CPU | Linux, x86-64 |
| SynapseAI | 1.24.1 (`habanalabs-graph`, `habanalabs-thunk`) |
| TPC compiler | `tpc-clang` 2.7.33 |

Other platforms fail at configure time.

## Data Types Supported

| Data Type | Status |
| --- | --- |
| F32 | Supported |
| F16 | Supported |
| BF16 | Supported |
| Q8_0 | Supported, stored as FP8 rows with a scale per row |
| Other quantized types | Supported. Types with a device layout (see `ggml-synapse-quant.cpp`) stay quantized in device memory and are dequantized during the matrix multiplication. Other types are dequantized to BF16 at first use. |

For the op support status, see the SYNAPSE column in [docs/ops.md](../ops.md). Training ops (backward ops, optimizers, cross-entropy loss) are not supported.

## Build

Requirements:

- SynapseAI headers and libraries in `/usr/include/habanalabs` and `/usr/lib/habanalabs`
- the habanatools glue headers in `/usr/lib/habanatools/include`
- `tpc-clang` and `objcopy` in `PATH`

```bash
cmake -B build -DGGML_SYNAPSE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The build makes two libraries:

- `libggml-synapse.so`: the backend
- `libggml_synapse_kernels.so`: the TPC kernel library, loaded from the directory of the backend library

It also builds `ggml-synapse-worker`, the per-card worker process (see [Multiple Cards](#multiple-cards)).

## Run

The devices are the visible cards in `hl-smi` index order: `SYNAPSE0`, `SYNAPSE1`, and so on. List them with `--list-devices` (the cards are not opened for this; a card in use by another process shows 0 MiB free) and select them with `-dev`, as with other backends:

```bash
./build/bin/llama-cli --list-devices
./build/bin/llama-cli -m model.gguf -ngl 99 -fa on -dev SYNAPSE3
```

Without `-dev`, llama.cpp uses all visible cards. `HABANA_VISIBLE_MODULES` (a comma list of module ids, the `module_id` column of `hl-smi`) limits the visible cards; they are numbered from `SYNAPSE0` again:

```bash
HABANA_VISIBLE_MODULES=7 ./build/bin/llama-cli -m model.gguf -ngl 99 -fa on
```

One process can open only one Gaudi card. With one visible card the backend uses it in the process itself. With several visible cards each card that is used runs in its own `ggml-synapse-worker` process, started on the first use of the card, and the main process talks to the workers through shared memory. Worker mode costs some prefill speed; for one card, make only that card visible.

### Multiple Cards

Select several cards, and use the layer or tensor split:

```bash
./build/bin/llama-cli -m model.gguf -ngl 99 -fa on -sm tensor -dev SYNAPSE0,SYNAPSE1
```

With `-sm tensor`, the workers exchange partial results with HCCL. `GGML_SYNAPSE_COMM=host` sends them through the host instead.

## Recipe Cache

Graph compilation is slow, so the backend keeps each compiled recipe:

- in memory, for the life of the process
- on disk, in `~/.cache/llama.cpp/synapse` (set another directory with `GGML_SYNAPSE_CACHE`, or `0` to turn the disk cache off)

The cache key covers the lowered program, the Synapse and TPC library versions, the backend kernel library, and the environment variables that the graph compiler reads. A change to any of these compiles the graphs again.

A recipe is specific to its tensor shapes. The first batch with a new shape compiles a new recipe. To fill the cache before use, run `llama-graph-sweep` (`tools/graph-sweep`) with the same arguments as the server.

### Tiled Attention

The attention shape depends on the KV length. To avoid a compile at each new KV length, the backend runs most attention as a fixed-size window recipe that it launches once per window of the KV cache. The graph is cut into segments at each tiled attention op. This applies to decode and to prefill batches, for F16 / BF16 caches with one valid key range (or two, for a wrapped sliding-window cache) per query row. Other attention uses the normal lowering.

Set `GGML_SYNAPSE_FA_TILE` to the window size in keys (default `4096`), or to `0` to turn tiled attention off.

## Environment Variables

The main variables:

| Variable | Default | Description |
| --- | --- | --- |
| `HABANA_VISIBLE_MODULES` | all | Comma list of module ids of the visible cards |
| `GGML_SYNAPSE_CACHE` | `~/.cache/llama.cpp/synapse` | Recipe cache directory, `0`: no disk cache |
| `GGML_SYNAPSE_FA_TILE` | `4096` | Tiled attention window in keys, `0`: off |
| `GGML_SYNAPSE_FA_TILE_MME_BATCH` | `1` | `0`: no tiled attention for batches that cannot use the native SDPA (head size above 256) |
| `GGML_SYNAPSE_FA_SDPA_SOFTMAX` | `0` | Softmax mode of the native SDPA, `4`: F32 softmax (more exact, slower prefill) |
| `GGML_SYNAPSE_NATIVE` | `1` | `0`: no device weight layouts, quantized weights are dequantized to BF16 |
| `GGML_SYNAPSE_Q8_BF16` | `0` | `1`: Q8_0 weights as BF16 copies instead of FP8 rows |
| `GGML_SYNAPSE_COMM` | HCCL | `host`: collectives through the host |
| `GGML_SYNAPSE_NOT` | unset | Comma list of ops that the backend refuses, so that they run on the CPU |
| `GGML_SYNAPSE_NO_FUSE` | off | Turn off the fused lowerings |
| `GGML_SYNAPSE_VERBOSE` | off | Log weight conversions, compiles and graph summaries |
| `GGML_SYNAPSE_DUMP_PROG` | unset | `1`: print each lowered program, `<prefix>`: write one file per program |
| `GGML_SYNAPSE_LOG_UNSUPPORTED` | `0` | `1`: log each rejected op once |
| `GGML_SYNAPSE_TIMING` | off | Print host and device time per graph |

The backend also sets these variables when it registers, unless you set them yourself:

- SynapseAI graph compiler variables (`ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS`, `PT_HPU_SDPA_MAX_TENSOR_BUFFER_SIZE`, `WORKSPACE_EPOCH_SIZE_PERCISION`, and `CHECK_SECTION_OVERLAP_CHECK` unless `GGML_SYNAPSE_SECTION_CHECK` is set)
- llama.cpp and ggml variables that reduce the number of distinct graph shapes: `LLAMA_KV_PAD_GEOM=1` (KV length padded in geometric steps), `LLAMA_UBATCH_PAD=1` (prompt ubatches padded to a few sizes) and `GGML_SCHED_ALLOC_FRESH=1` (each graph gets its own allocation plan). Set any of them to `0` to turn it off.

## Testing

Use the standard tools with a SYNAPSE device:

```bash
./build/bin/test-backend-ops test -b SYNAPSE0
./build/bin/test-llama-archs -b SYNAPSE0
```

With all cards visible, `-b SYNAPSE<N>` runs the tests in a worker process on card N; with `HABANA_VISIBLE_MODULES` set to one card they run in the process. Tools that use every device (ctest, `test-llama-context`) need `HABANA_VISIBLE_MODULES` on a shared machine.

Run large suites one op at a time (`-o <op>`): an abort in one op stops the process.

## Known Issues

- Two `FLASH_ATTN_EXT` cases with very large query values (`q_range=20`) exceed the test tolerance with the default BF16 SDPA softmax. They pass with `GGML_SYNAPSE_FA_SDPA_SOFTMAX=4`.
- The first run of each new shape compiles a recipe. Use the recipe cache or `llama-graph-sweep` for servers.
- One process can use only one card directly. Multiple cards always use worker processes.

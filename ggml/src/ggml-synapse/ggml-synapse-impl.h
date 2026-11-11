#pragma once

#include "ggml-synapse.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <synapse_api.h>

#include <sched.h>

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// error handling
// ---------------------------------------------------------------------------

[[noreturn]] void ggml_synapse_abort(const char * stmt, const char * func,
                                     const char * file, int line, synStatus st);

#define SYNAPSE_CHECK(stmt)                                                              \
    do {                                                                                 \
        const synStatus st__ = (stmt);                                                   \
        if (st__ != synSuccess) {                                                        \
            ggml_synapse_abort(#stmt, __func__, __FILE__, __LINE__, st__);               \
        }                                                                                \
    } while (0)

#define GGML_SYNAPSE_MAX_DEVICES 32

// ---------------------------------------------------------------------------
// type mapping (defined in ggml-synapse.cpp)
// ---------------------------------------------------------------------------

synDataType syn_data_type(enum ggml_type t);

// ---------------------------------------------------------------------------
// per-device context: one per acquired Gaudi card, shared by all backends and
// buffers on that device.
// ---------------------------------------------------------------------------

struct synapse_recipe;
struct synapse_prepared;

// ---------------------------------------------------------------------------
// device layouts of quantized weights (ggml-synapse-quant.cpp)
// ---------------------------------------------------------------------------

// a < 8-bit type in the planar layout (see ggml-synapse-quant.cpp)
struct synapse_qt {
    ggml_type type;
    int sb;          // elements per superblock (the native block)
    int g;           // elements per scale group
    int nf16, nbp;   // f16 param planes, byte param planes
    int p4, p2, p1;  // code planes
    int block;       // native bytes per superblock and row
    int f16_off[2];  // native offsets of the f16 params
    int bp_off;      // native offset of the first byte param
    int p8;          // 8-bit code plane (q8_0: the codes as signed bytes)
};
const synapse_qt * synapse_qt_get(ggml_type t);
size_t synapse_qt_planar_size(const synapse_qt * q, int64_t K, int64_t M, int64_t nmat);
void synapse_qt_to_planar(const synapse_qt * q, const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat);
void synapse_qt_from_planar(const synapse_qt * q, const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat);

uint8_t synapse_fp8_enc(float x);
float   synapse_fp8_dec(uint8_t b);
size_t  synapse_fp8_rows_size(int64_t K, int64_t M);
size_t  synapse_fp8_planar_size(int64_t K, int64_t M, int64_t nmat);
void synapse_q8_to_fp8_rows(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M);
void synapse_q8_to_fp8_planar(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat);
void synapse_fp8_rows_to_q8(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M);
void synapse_fp8_planar_to_q8(const uint8_t * src, uint8_t * dst, int64_t K, int64_t M, int64_t nmat);

// device format of a quantized weight
enum synapse_wkind {
    SYN_W_PLANAR    = 1,   // planar < 8-bit codes
    SYN_W_FP8_ROWS  = 2,   // fp8 rows + per-row scales
    SYN_W_FP8_PLANAR = 3,  // fp8 128-row blocks + per-row scales (experts)
    SYN_W_BF16      = 4,   // bf16 rows (q8_0 dequantized once)
};

struct synapse_wfmt {
    int       kind;
    ggml_type type;
    uint64_t  addr;       // device address of the converted data
    uint64_t  scales;     // FP8_ROWS: per-row scales
    int64_t   K, M, nmat;
    uint64_t  src_bytes;  // size of the source tensor
    bool      in_place;   // converted inside the weight's own memory at load
    bool      own;        // a separate allocation (freed on drop), else pool memory or in place
};

struct ggml_backend_synapse_device_context {
    int32_t     device;   // card index
    synDeviceId sdev;     // acquired synapse device id
    int         card;
    bool        failed = false;

    // all work is ordered on one stream
    synStreamHandle stream = nullptr;

    // the backends of the device share the state below (prepared graphs, arena, workspace, derived weights):
    // graph_compute and the buffer calls that drop derived weights hold this lock
    std::mutex compute_mtx;

    // device workspace shared by all recipes (single stream)
    uint64_t  ws_addr = 0;
    uint64_t  ws_cap  = 0;
    // the arena all prepared graphs share (launches are stream-ordered; nothing in it outlives one)
    uint64_t  arena_addr = 0;
    uint64_t  arena_cap  = 0;

    // pinned staging for synchronous host<->device copies
    std::mutex hst_mtx;
    void *     hst_ptr = nullptr;
    uint64_t   hst_cap = 0;

    // compiled recipes by structural signature
    std::mutex recipes_mtx;
    std::unordered_map<std::string, std::shared_ptr<synapse_recipe>> recipes;

    // prepared launches by graph fingerprint
    std::unordered_map<uint64_t, std::shared_ptr<synapse_prepared>> prepared;
    // per graph object: the node identities its prepared launch was found for
    struct graph_memo {
        struct node_id {
            const ggml_tensor * t;
            const void * data;
            const ggml_tensor * src0;
            int op, flags;
            int64_t ne[GGML_MAX_DIMS];
            size_t nb[GGML_MAX_DIMS];
        };
        std::vector<node_id> nodes;
        std::shared_ptr<synapse_prepared> prep;
    };
    std::unordered_map<const ggml_cgraph *, graph_memo> graph_memos;

    // pinned ring for small non-blocking uploads (slot reused after its copy completed)
    static constexpr int ring_n = 64;
    static constexpr uint64_t ring_slot = 64 << 10;
    char * ring = nullptr;
    bool ring_used[ring_n] = {};
    int ring_i = 0;
    std::vector<uint64_t> up_src, up_dst, up_size;   // queued uploads
    void flush_uploads();
    void flush_uploads_locked();
    void copy_h2d_async_small(uint64_t dst, const void * src, uint64_t nbytes);

    // host copy of small tensors written through set_tensor (device address -> bytes):
    // the lowering reads graph inputs such as rope positions from here
    std::mutex shadow_mtx;
    std::map<uint64_t, std::vector<uint8_t>> shadow;   // ordered: reads of sub-ranges
    void shadow_set(uint64_t addr, const void * data, size_t size);
    void shadow_drop(uint64_t addr, size_t size);
    void shadow_drop_locked(uint64_t addr, size_t size);
    bool shadow_get(uint64_t addr, void * dst, size_t size);

    // quantized weights dequantized to bf16 device constants at first use, keyed by (data, layout)
    struct wdq_entry {
        uint64_t addr;
        uint64_t src_bytes;
        bool     own;   // a separate allocation (freed on drop), else pool memory
    };
    std::map<std::pair<const void *, int>, wdq_entry> wdq;
    void wdq_drop(uint64_t addr, size_t size);
    void wcat_drop(uint64_t addr, size_t size);

    // concatenated weights for merged GEMMs: key -> (addr, source ranges)
    std::map<std::string, std::pair<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>>> wcat;
    // layout 0: rows as stored ([K, rows]); layout 1: per 2-D matrix of a 3-D tensor, blocks of
    // 128 rows transposed ([128 rows, K, row blocks, matrices]) for MUL_MAT_ID;
    // layout 2: rows as stored, f32 (exact, for row gathers)
    uint64_t dequant_weight(const ggml_tensor * w, int layout = 0);

    // quantized weights in a device format: converted at load (weight buffers, in place) or at
    // first use (a copy); keyed by the tensor's data address
    std::map<uint64_t, synapse_wfmt> wfmt;
    struct staging {
        std::vector<uint8_t> data;
        size_t received = 0;
    };
    std::map<uint64_t, staging> wstage;   // weights uploaded in pieces, converted when complete
    void *   wpin[2] = { nullptr, nullptr };   // pinned buffers for weight conversion uploads
    size_t   wpin_cap = 0;
    // convert native rows of a planar weight (whole 128-row blocks) and upload them to dst
    void     upload_planar(const ggml_tensor * t, uint64_t dst, const uint8_t * src, size_t size);
    std::mutex wfmt_mtx;
    // the planned device format of t (kind 0: none); in_place: whether it fits t's own memory
    static int  weight_kind(const ggml_tensor * t, bool & in_place);
    // set_tensor of a weight-buffer tensor: true when consumed (converted in place, or staged)
    bool weight_set(const ggml_tensor * t, const void * data, size_t offset, size_t size);
    // get_tensor of a converted weight: true when served (converted back)
    bool weight_get(const ggml_tensor * t, void * data, size_t offset, size_t size);
    // the device format of weight w, converting a copy at first use; nullptr: none
    const synapse_wfmt * weight_format(const ggml_tensor * w);
    // a 3-D q8_0 weight of batched MUL_MATs (MLA k_b / v_b) as fp8 rows (converted at first use); nullptr: none
    const synapse_wfmt * weight_format_rows(const ggml_tensor * w);
    void wfmt_drop(uint64_t addr, size_t size, bool all);
    uint64_t one_f32 = 0;      // a device f32 1.0 (fp8 GEMM activation scale)

    uint64_t dev_malloc(uint64_t size);
    // bump pool for derived weights (merged GEMM weights, re-arranged constants)
    std::vector<std::pair<uint64_t, uint64_t>> pool;   // (base, size)
    uint64_t pool_used = 0;
    uint64_t pool_alloc(uint64_t size);
    uint64_t pool_region(uint64_t addr) const;
    void     pool_release_unused();

    // pooled copies of small f32 weights
    static constexpr uint64_t smallw_cap = 64ull << 20;
    uint64_t smallw_base = 0;
    uint64_t smallw_used = 0;
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> smallw;   // source addr -> (offset, size)
    uint64_t smallw_get(uint64_t src, uint64_t size);
    void     dev_free(uint64_t addr);
    void *   stage(uint64_t nbytes);
    void     copy_h2d(uint64_t dst, const void * src, uint64_t nbytes);
    void     copy_d2h(void * dst, uint64_t src, uint64_t nbytes);

    void     memset_async(uint64_t addr, uint8_t value, uint64_t nbytes);

    // device work enqueued since the last stream sync (ring uploads do not count); set_pending after the
    // enqueue, under hst_mtx: else the sync of another backend can clear it before it waited for that work
    std::atomic<bool> pending{false};
    void set_pending();

    // pinned host allocations (host buffer type)
    std::mutex pinned_mtx;
    std::map<uintptr_t, size_t> pinned;
    void pinned_add(void * p, size_t n);
    void pinned_drop(void * p);
    bool is_pinned(const void * p, size_t n);
    void     sync();

    ggml_backend_synapse_device_context(int32_t device);
    ~ggml_backend_synapse_device_context();
};

// process-wide synapse init (refcounted)
void ggml_synapse_acquire();
void ggml_synapse_release();

// GC_KERNEL_PATH setup; must run before the first synInitialize
void ggml_synapse_register_tpc_plugin();

// visible cards (HLML, hl-smi index order, HABANA_VISIBLE_MODULES applied): card N is device SYNAPSE<N>
int  ggml_synapse_card_count();
int  ggml_synapse_card_module(int card);                              // -1 if no such card
const char * ggml_synapse_card_pci(int card);                         // nullptr if unknown
const char * ggml_synapse_card_description(int card);
bool ggml_synapse_card_memory(int card, size_t * free, size_t * total);   // false if unknown
bool ggml_synapse_card_affinity(int card, cpu_set_t * set);           // the CPUs near the card
synDeviceId ggml_synapse_acquire_card(int card);

// ---------------------------------------------------------------------------
// lowering (ggml-synapse-graph.cpp)
// ---------------------------------------------------------------------------

enum ggml_status ggml_synapse_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);
void ggml_synapse_prepared_free(ggml_backend_synapse_device_context * dctx);

bool ggml_synapse_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

// ---------------------------------------------------------------------------
// card worker processes (ggml-synapse-proxy.cpp)
// ---------------------------------------------------------------------------

// one proxy device per visible card; its worker process starts when the card is first used
bool ggml_synapse_proxy_init(ggml_backend_reg_t reg, std::vector<ggml_backend_dev_t> & devices);
// collectives of the card workers (ggml_backend_comm_*), nullptr outside multi-card mode
void * ggml_synapse_proxy_proc_address(const char * name);
void ggml_synapse_trace(const char * what);
// an INFO log line (llama.cpp shows these at -lv 4); in a card worker it goes to the parent process, whose logger
// applies the verbosity (a worker's own stderr would print every level)
void ggml_synapse_log_info(const char * fmt, ...) __attribute__((format(printf, 1, 2)));

// weight upload time (ns) and bytes: planar convert / wait for the DMA, whole-tensor convert / upload
struct synapse_load_stat { uint64_t conv_ns, wait_ns, bytes, whole_conv_ns, whole_h2d_ns, whole_bytes; };
extern synapse_load_stat g_load_stat;

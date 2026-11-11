/*
 * ggml-synapse: native SynapseAI (Intel Gaudi2) backend for ggml.
 *
 * Registers an ordinary ggml backend (registry name SYNAPSE, devices
 * SYNAPSE0..N). Device resources are owned per device context: one acquired
 * synapse device, one ordered stream, a device workspace and a pinned
 * staging slab shared by all buffers and backends on that card.
 */
#include "ggml-synapse.h"

#include "ggml-synapse-impl.h"

#include <synapse_api.h>
#include <hlml.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <thread>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

void ggml_synapse_abort(const char * stmt, const char * func,
                        const char * file, int line, synStatus st) {
    char desc[STATUS_DESCRIPTION_MAX_SIZE] = {0};
    synStatusGetBriefDescription(st, desc, sizeof(desc));
    GGML_LOG_ERROR("Synapse error in %s at %s:%d\n", func, file, line);
    GGML_LOG_ERROR("  %s -> status %d (%s)\n", stmt, (int) st, desc);
    GGML_ABORT("Synapse API failure");
}

/* ---------------- type mapping ---------------- */

synDataType syn_data_type(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return syn_type_single;
        case GGML_TYPE_BF16: return syn_type_bf16;
        case GGML_TYPE_F16:  return syn_type_fp16;
        case GGML_TYPE_I32:  return syn_type_int32;
        case GGML_TYPE_I64:  return syn_type_int64;
        case GGML_TYPE_I8:   return syn_type_int8;
        default:             GGML_ABORT("Synapse: no type mapping for %s", ggml_type_name(t)); return syn_type_single;
    }
}

/* ---------------- process-wide init ---------------- */

static std::mutex g_init_mtx;
static int        g_init_refs = 0;

/* Synapse keeps process-lifetime GC state: synDestroy followed by
 * synInitialize does not reliably re-register TPC plugins, so the runtime is
 * initialized once and never torn down. Acquire/release only track the
 * initialization handshake. */
void ggml_synapse_acquire() {
    std::lock_guard<std::mutex> lock(g_init_mtx);
    if (g_init_refs++ == 0) {
        // the graph compiler reads GC_KERNEL_PATH inside synInitialize
        ggml_synapse_register_tpc_plugin();
        // the GC's in-place reuse suggestion aliases an in-place node's
        // input and output in one section slot; our kernels are not written
        // for that (the pair gets zeroed) - off before synInitialize (GCFG)
        setenv("ENABLE_INPLACE_REUSE_FOR_SUGGESTIONS", "false", 0);
        // SDPA tensor buffer limit and workspace size precision for long graphs
        setenv("PT_HPU_SDPA_MAX_TENSOR_BUFFER_SIZE", "2147483647", 0);
        setenv("WORKSPACE_EPOCH_SIZE_PERCISION", "0.2", 0);
        // GGML_SYNAPSE_SECTION_CHECK=1: Synapse's section overlap check stays on
        if (!getenv("GGML_SYNAPSE_SECTION_CHECK")) {
            setenv("ENABLE_EXPERIMENTAL_FLAGS", "true", 0);
            setenv("CHECK_SECTION_OVERLAP_CHECK", "0", 0);
        }
        SYNAPSE_CHECK(synInitialize());
    }
}

void ggml_synapse_release() {
    std::lock_guard<std::mutex> lock(g_init_mtx);
    if (g_init_refs > 0) {
        --g_init_refs;
    }
}

/* ---------------- device enumeration ---------------- */

/* the visible cards in hl-smi index order, listed by HLML (no card is opened for this);
 * HABANA_VISIBLE_MODULES=<module ids> keeps only those cards */
struct synapse_card_info {
    int           module = -1;   // Synapse module id (synDeviceAcquireByModuleId)
    std::string   pci;           // PCI bus id, empty if unknown
    std::string   description;
    hlml_device_t hl = nullptr;  // nullptr: HLML not available
};

static std::mutex g_hlml_mtx;

static const std::vector<synapse_card_info> & synapse_cards() {
    static std::vector<synapse_card_info> cards;
    static std::once_flag once;
    std::call_once(once, [] {
        std::vector<int> keep;
        const char * vis = getenv("HABANA_VISIBLE_MODULES");
        for (const char * p = vis ? vis : ""; *p;) {
            char * end = nullptr;
            const long m = strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            keep.push_back((int) m);
            p = *end ? end + 1 : end;
        }
        auto visible = [&](int module) {
            return vis == nullptr || std::find(keep.begin(), keep.end(), module) != keep.end();
        };
        unsigned n = 0;
        if (hlml_init() == HLML_SUCCESS && hlml_device_get_count(&n) == HLML_SUCCESS) {
            for (unsigned i = 0; i < n; i++) {
                synapse_card_info c;
                unsigned module = 0;
                if (hlml_device_get_handle_by_index(i, &c.hl) != HLML_SUCCESS ||
                    hlml_device_get_module_id(c.hl, &module) != HLML_SUCCESS || !visible((int) module)) {
                    continue;
                }
                c.module = (int) module;
                hlml_pci_info_t pci = {};
                if (hlml_device_get_pci_info(c.hl, &pci) == HLML_SUCCESS) {
                    c.pci = pci.bus_id;
                    std::transform(c.pci.begin(), c.pci.end(), c.pci.begin(), ::tolower);
                }
                char name[64] = { 0 };
                hlml_device_get_name(c.hl, name, sizeof(name));
                c.description = std::string("Intel Gaudi2 ") + name + " (module " + std::to_string(c.module) +
                                (c.pci.empty() ? "" : ", " + c.pci) + ")";
                cards.push_back(c);
            }
            return;
        }
        // no HLML: the module ids Synapse reports, without PCI id and memory figures
        ggml_synapse_acquire();
        uint32_t ids[GGML_SYNAPSE_MAX_DEVICES];
        uint32_t m = GGML_SYNAPSE_MAX_DEVICES;
        if (synDeviceGetModuleIDs(ids, &m) == synSuccess) {
            for (uint32_t i = 0; i < m; i++) {
                if (visible((int) ids[i])) {
                    synapse_card_info c;
                    c.module = (int) ids[i];
                    c.description = "Intel Gaudi2 (module " + std::to_string(c.module) + ")";
                    cards.push_back(c);
                }
            }
        }
    });
    return cards;
}

int ggml_synapse_card_count() {
    return (int) std::min<size_t>(synapse_cards().size(), GGML_SYNAPSE_MAX_DEVICES);
}

int ggml_synapse_card_module(int card) {
    return card >= 0 && card < ggml_synapse_card_count() ? synapse_cards()[card].module : -1;
}

const char * ggml_synapse_card_pci(int card) {
    return card >= 0 && card < ggml_synapse_card_count() && !synapse_cards()[card].pci.empty() ?
               synapse_cards()[card].pci.c_str() : nullptr;
}

const char * ggml_synapse_card_description(int card) {
    return card >= 0 && card < ggml_synapse_card_count() ? synapse_cards()[card].description.c_str() : "Intel Gaudi2";
}

bool ggml_synapse_card_memory(int card, size_t * free, size_t * total) {
    if (card < 0 || card >= ggml_synapse_card_count() || !synapse_cards()[card].hl) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_hlml_mtx);
    hlml_memory_t m = {};
    if (hlml_device_get_memory_info(synapse_cards()[card].hl, &m) != HLML_SUCCESS) {
        return false;
    }
    *free  = (size_t) m.free;
    *total = (size_t) m.total;
    return true;
}

bool ggml_synapse_card_affinity(int card, cpu_set_t * set) {
    if (card < 0 || card >= ggml_synapse_card_count() || !synapse_cards()[card].hl) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_hlml_mtx);
    unsigned long mask[CPU_SETSIZE / (8 * sizeof(unsigned long))] = { 0 };
    const unsigned n = sizeof(mask) / sizeof(mask[0]);
    if (hlml_device_get_cpu_affinity(synapse_cards()[card].hl, n, mask) != HLML_SUCCESS) {
        return false;
    }
    CPU_ZERO(set);
    int count = 0;
    for (unsigned i = 0; i < n; i++) {
        for (unsigned b = 0; b < 8 * sizeof(unsigned long); b++) {
            if (mask[i] & (1ul << b)) {
                CPU_SET((int) (i * 8 * sizeof(unsigned long) + b), set);
                count++;
            }
        }
    }
    return count > 0;
}

/* returns the acquired device id, or -1 on failure (0 is a valid id) */
synDeviceId ggml_synapse_acquire_card(int card) {
    const int module = ggml_synapse_card_module(card);
    if (module < 0) {
        GGML_LOG_ERROR("%s: no card %d\n", __func__, card);
        return -1;
    }
    synDeviceId sdev = 0;
    const synStatus st = synDeviceAcquireByModuleId(&sdev, (synModuleId) module);
    if (st != synSuccess) {
        char desc[STATUS_DESCRIPTION_MAX_SIZE] = {0};
        synStatusGetBriefDescription(st, desc, sizeof(desc));
        GGML_LOG_ERROR("%s: " GGML_SYNAPSE_NAME "%d (module %d): synDeviceAcquireByModuleId status %d (%s), card in use?\n",
                       __func__, card, module, (int) st, desc);
        return -1;
    }
    return (synDeviceId) sdev;
}

/* ---------------- device context ---------------- */

uint64_t ggml_backend_synapse_device_context::dev_malloc(uint64_t size) {
    uint64_t addr = 0;
    SYNAPSE_CHECK(synDeviceMalloc(sdev, size ? size : 128, 0, 0, &addr));
    return addr;
}

uint64_t ggml_backend_synapse_device_context::pool_alloc(uint64_t size) {
    size = (size + 255) & ~255ull;
    if (pool.empty() || pool_used + size > pool.back().second) {
        const uint64_t cap = std::max<uint64_t>(size, 2ull << 30);
        pool.push_back({ dev_malloc(cap), cap });
        pool_used = 0;
    }
    const uint64_t addr = pool.back().first + pool_used;
    pool_used += size;
    return addr;
}

// derived weights live in the pool until no cached entry refers to it (a freed model): then it goes
void ggml_backend_synapse_device_context::pool_release_unused() {
    if (pool.empty() || !wcat.empty()) {
        return;
    }
    for (const auto & kv : wdq) {
        if (!kv.second.own) {
            return;
        }
    }
    for (const auto & kv : wfmt) {
        if (!kv.second.own && !kv.second.in_place) {
            return;
        }
    }
    sync();
    for (auto & r : pool) {
        dev_free(r.first);
    }
    pool.clear();
    pool_used = 0;
}

uint64_t ggml_backend_synapse_device_context::smallw_get(uint64_t src, uint64_t size) {
    auto it = smallw.find(src);
    if (it != smallw.end() && it->second.second == size) {
        return it->second.first;
    }
    if (!smallw_base) {
        smallw_base = dev_malloc(smallw_cap);
    }
    const uint64_t asz = (size + 255) & ~255ull;
    GGML_ASSERT(smallw_used + asz <= smallw_cap && "small weight pool exhausted");
    const uint64_t off = smallw_used;
    smallw_used += asz;
    flush_uploads();
    SYNAPSE_CHECK(synMemCopyAsync(stream, src, size, smallw_base + off, DRAM_TO_DRAM));
    sync();
    smallw[src] = { off, size };
    return off;
}

uint64_t ggml_backend_synapse_device_context::pool_region(uint64_t addr) const {
    for (const auto & r : pool) {
        if (addr >= r.first && addr < r.first + r.second) {
            return r.first;
        }
    }
    return 0;
}

void ggml_backend_synapse_device_context::dev_free(uint64_t addr) {
    if (addr) {
        SYNAPSE_CHECK(synDeviceFree(sdev, addr, 0));
    }
}

// pinned staging; the caller holds hst_mtx for the whole copy
void * ggml_backend_synapse_device_context::stage(uint64_t nbytes) {
    if (hst_cap < nbytes) {
        if (hst_ptr) {
            synHostFree(sdev, hst_ptr, 0);
            hst_ptr = nullptr;
        }
        const uint64_t cap = ((nbytes + (1u << 26) - 1) / (1u << 26)) * (1u << 26);
        SYNAPSE_CHECK(synHostMalloc(sdev, cap, 0, &hst_ptr));
        hst_cap = cap;
    }
    return hst_ptr;
}

static bool g_timing = getenv("GGML_SYNAPSE_TIMING") != nullptr;
static bool g_trace = getenv("GGML_SYNAPSE_TRACE") != nullptr;
static bool g_trace_agg = g_trace && atoi(getenv("GGML_SYNAPSE_TRACE")) == 2;

// GGML_SYNAPSE_TRACE=2: aggregate the time between consecutive trace points, printed at exit
struct trace_agg {
    std::map<std::pair<std::string, std::string>, std::vector<double>> sum;
    std::string prev;
    int64_t t_prev = 0;
    ~trace_agg() {
        for (auto & kv : sum) {
            auto & v = kv.second;
            std::sort(v.begin(), v.end());
            double tot = 0;
            for (double x : v) {
                tot += x;
            }
            fprintf(stderr, "synapse: %-22s -> %-22s n %6zu median %8.1f us mean %8.1f us total %9.1f ms\n", kv.first.first.c_str(),
                    kv.first.second.c_str(), v.size(), v[v.size() / 2], tot / v.size(), tot / 1000.0);
        }
    }
};

void ggml_synapse_trace(const char * what) {
    if (g_trace_agg) {
        static trace_agg agg;
        const int64_t t = ggml_time_us();
        if (!agg.prev.empty()) {
            agg.sum[{ agg.prev, what }].push_back(t - agg.t_prev);
        }
        agg.prev = what;
        agg.t_prev = t;
        return;
    }
    if (g_timing || g_trace) {
        static int64_t t0 = ggml_time_us();
        fprintf(stderr, "synapse: p%d t=%.3f %s\n", (int) getpid(), (ggml_time_us() - t0) / 1000.0, what);
    }
}

synapse_load_stat g_load_stat = {};

void ggml_backend_synapse_device_context::copy_h2d(uint64_t dst, const void * src, uint64_t nbytes) {
    std::lock_guard<std::mutex> lock(hst_mtx);
    flush_uploads_locked();
    const int64_t t0 = g_timing ? ggml_time_us() : 0;
    struct tlog { int64_t t0; uint64_t n; ~tlog() { if (g_timing) fprintf(stderr, "synapse: h2d %llu B %.3f ms\n", (unsigned long long) n, (ggml_time_us() - t0) / 1000.0); } } tl{ t0, nbytes };
    const uint64_t chunk = 64ull << 20;
    const char * p = (const char *) src;
    while (nbytes > 0) {
        const uint64_t n = std::min(nbytes, chunk);
        void * stag = stage(n);
        std::memcpy(stag, p, n);
        SYNAPSE_CHECK(synMemCopyAsync(stream, (uint64_t)(uintptr_t) stag, n, dst, HOST_TO_DRAM));
        SYNAPSE_CHECK(synStreamSynchronize(stream));
        p += n;
        dst += n;
        nbytes -= n;
    }
}

void ggml_backend_synapse_device_context::copy_h2d_async_small(uint64_t dst, const void * src, uint64_t nbytes) {
    if (nbytes > ring_slot) {
        copy_h2d(dst, src, nbytes);
        return;
    }
    std::lock_guard<std::mutex> lock(hst_mtx);
    if (!ring) {
        SYNAPSE_CHECK(synHostMalloc(sdev, ring_n * ring_slot, 0, (void **) &ring));
    }
    const int i = ring_i;
    ring_i = (ring_i + 1) % ring_n;
    if (ring_used[i]) {
        // the ring wrapped without a stream sync in between (rare): drain before reuse
        flush_uploads_locked();
        SYNAPSE_CHECK(synStreamSynchronize(stream));
        std::fill(std::begin(ring_used), std::end(ring_used), false);
    }
    memcpy(ring + i * ring_slot, src, nbytes);
    // queued; flush_uploads submits it
    up_src.push_back((uint64_t) (uintptr_t) (ring + i * ring_slot));
    up_dst.push_back(dst);
    up_size.push_back(nbytes);
    ring_used[i] = true;
}

void ggml_backend_synapse_device_context::flush_uploads_locked() {
    if (up_src.empty()) {
        return;
    }
    // slots stay used until the next stream sync
    SYNAPSE_CHECK(synMemCopyAsyncMultiple(stream, up_src.data(), up_size.data(), up_dst.data(), HOST_TO_DRAM, up_src.size()));
    up_src.clear();
    up_dst.clear();
    up_size.clear();
}

void ggml_backend_synapse_device_context::flush_uploads() {
    std::lock_guard<std::mutex> lock(hst_mtx);
    flush_uploads_locked();
}

void ggml_backend_synapse_device_context::pinned_add(void * p, size_t n) {
    std::lock_guard<std::mutex> lock(pinned_mtx);
    pinned[(uintptr_t) p] = n;
}

void ggml_backend_synapse_device_context::pinned_drop(void * p) {
    std::lock_guard<std::mutex> lock(pinned_mtx);
    pinned.erase((uintptr_t) p);
}

bool ggml_backend_synapse_device_context::is_pinned(const void * p, size_t n) {
    std::lock_guard<std::mutex> lock(pinned_mtx);
    auto it = pinned.upper_bound((uintptr_t) p);
    if (it == pinned.begin()) {
        return false;
    }
    --it;
    return (uintptr_t) p >= it->first && (uintptr_t) p + n <= it->first + it->second;
}

void ggml_backend_synapse_device_context::copy_d2h(void * dst, uint64_t src, uint64_t nbytes) {
    flush_uploads();
    if (is_pinned(dst, nbytes)) {
        SYNAPSE_CHECK(synMemCopyAsync(stream, src, nbytes, (uint64_t) (uintptr_t) dst, DRAM_TO_HOST));
        sync();
        return;
    }
    std::lock_guard<std::mutex> lock(hst_mtx);
    const int64_t t0 = g_timing ? ggml_time_us() : 0;
    struct tlog { int64_t t0; uint64_t n; ~tlog() { if (g_timing) fprintf(stderr, "synapse: d2h %llu B %.3f ms\n", (unsigned long long) n, (ggml_time_us() - t0) / 1000.0); } } tl{ t0, nbytes };
    const uint64_t chunk = 64ull << 20;
    char * p = (char *) dst;
    while (nbytes > 0) {
        const uint64_t n = std::min(nbytes, chunk);
        void * stag = stage(n);
        SYNAPSE_CHECK(synMemCopyAsync(stream, src, n, (uint64_t)(uintptr_t) stag, DRAM_TO_HOST));
        SYNAPSE_CHECK(synStreamSynchronize(stream));
        std::memcpy(p, stag, n);
        p += n;
        src += n;
        nbytes -= n;
    }
}

void ggml_backend_synapse_device_context::memset_async(uint64_t addr, uint8_t value, uint64_t nbytes) {
    flush_uploads();
    uint64_t off = addr;
    while (nbytes > 0) {
        const uint64_t chunk = std::min<uint64_t>(nbytes, 1u << 31);
        SYNAPSE_CHECK(synMemsetD8Async(off, value, chunk, stream));
        off   += chunk;
        nbytes -= chunk;
    }
    set_pending();
}

void ggml_backend_synapse_device_context::set_pending() {
    std::lock_guard<std::mutex> lock(hst_mtx);
    pending = true;
}

void ggml_backend_synapse_device_context::sync() {
    std::lock_guard<std::mutex> lock(hst_mtx);
    flush_uploads_locked();
    static const int64_t spin_us = getenv("GGML_SYNAPSE_SPIN_US") ? atoll(getenv("GGML_SYNAPSE_SPIN_US")) : 5000;
    synStatus st = spin_us > 0 ? synStreamQuery(stream) : synBusy;
    if (st == synBusy && spin_us > 0) {
        const int64_t t0 = ggml_time_us();
        while ((st = synStreamQuery(stream)) == synBusy && ggml_time_us() - t0 < spin_us) {
        }
    }
    if (st == synBusy) {
        st = synStreamSynchronize(stream);
    }
    SYNAPSE_CHECK(st);
    std::fill(std::begin(ring_used), std::end(ring_used), false);   // every upload has landed
    pending = false;
}

/* ---------------- quantized weights in device formats ---------------- */

static bool native_weights() {
    static const bool off = getenv("GGML_SYNAPSE_NATIVE") && atoi(getenv("GGML_SYNAPSE_NATIVE")) == 0;
    return !off;
}

int ggml_backend_synapse_device_context::weight_kind(const ggml_tensor * t, bool & in_place) {
    in_place = false;
    if (!native_weights() || !ggml_is_contiguous(t) || t->ne[3] != 1) {
        return 0;
    }
    const int64_t K = t->ne[0], M = t->ne[1], nmat = t->ne[2];
    const size_t nb = ggml_nbytes(t);
    if (t->type == GGML_TYPE_Q8_0) {
        if (nmat == 1) {
            // fp8 rows (fp8_inc: in place). GGML_SYNAPSE_Q8_BF16=1: bf16 rows instead (exact to
            // bf16 rounding; fp8 rows lose the per-32 block scales and quantize the activation per tensor: ACC-1),
            // a copy of twice the size, fp8 rows when memory is short (weight_format)
            static const bool q8_bf16 = getenv("GGML_SYNAPSE_Q8_BF16") && atoi(getenv("GGML_SYNAPSE_Q8_BF16")) != 0;
            if (q8_bf16) {
                return SYN_W_BF16;
            }
            in_place = synapse_fp8_rows_size(K, M) <= nb;
            return SYN_W_FP8_ROWS;
        }
        // experts: the exact int8 planar layout (codes + per-32 scales, the native size: in place).
        // GGML_SYNAPSE_Q8_PLANAR=0: fp8 128-row blocks with per-row scales (less exact)
        static const bool q8_planar = !getenv("GGML_SYNAPSE_Q8_PLANAR") || atoi(getenv("GGML_SYNAPSE_Q8_PLANAR")) != 0;
        if (q8_planar && K % 32 == 0) {
            in_place = M % 128 == 0;
            return SYN_W_PLANAR;
        }
        in_place = M % 128 == 0 && synapse_fp8_planar_size(K, M, nmat) <= nb;
        return SYN_W_FP8_PLANAR;
    }
    const synapse_qt * q = synapse_qt_get(t->type);
    if (q && K % q->sb == 0) {
        in_place = M % 128 == 0;   // the planar layout is a permutation of the native bytes
        return SYN_W_PLANAR;
    }
    return 0;
}

/* native bytes -> device format bytes (dst holds `size_of` bytes) */
static void wconvert(int kind, const ggml_tensor * t, const uint8_t * src, uint8_t * dst) {
    const int64_t K = t->ne[0], M = t->ne[1], nmat = t->ne[2];
    switch (kind) {
        case SYN_W_FP8_ROWS:   synapse_q8_to_fp8_rows(src, dst, K, M * nmat); break;
        case SYN_W_FP8_PLANAR: synapse_q8_to_fp8_planar(src, dst, K, M, nmat); break;
        case SYN_W_BF16: {
            const ggml_to_float_t to_f32 = ggml_get_type_traits(t->type)->to_float;
            const size_t rs = ggml_row_size(t->type, K);
            std::vector<float> row(K);
            for (int64_t r = 0; r < M * nmat; r++) {
                to_f32(src + r * rs, row.data(), K);
                ggml_fp32_to_bf16_row(row.data(), (ggml_bf16_t *) dst + r * K, K);
            }
            break;
        }
        default:               synapse_qt_to_planar(synapse_qt_get(t->type), src, dst, K, M, nmat); break;
    }
}

static size_t wsize(int kind, const ggml_tensor * t) {
    const int64_t K = t->ne[0], M = t->ne[1], nmat = t->ne[2];
    switch (kind) {
        case SYN_W_FP8_ROWS:   return synapse_fp8_rows_size(K, M * nmat);
        case SYN_W_FP8_PLANAR: return synapse_fp8_planar_size(K, M, nmat);
        case SYN_W_BF16:       return (size_t) (K * M * nmat) * sizeof(ggml_bf16_t);
        default:               return synapse_qt_planar_size(synapse_qt_get(t->type), K, M, nmat);
    }
}

static synapse_wfmt wentry(int kind, const ggml_tensor * t, uint64_t addr, bool in_place, bool own) {
    synapse_wfmt f;
    f.kind = kind;
    f.type = t->type;
    f.addr = addr;
    f.K = t->ne[0];
    f.M = t->ne[1];
    f.nmat = t->ne[2];
    f.scales = kind == SYN_W_FP8_ROWS ? addr + ((uint64_t) f.M * f.nmat * f.K + 127) / 128 * 128 : 0;
    f.src_bytes = ggml_nbytes(t);
    f.in_place = in_place;
    f.own = own;
    return f;
}

// The planar layout permutes bytes within 128-row blocks (same size, same offset), so whole blocks
// convert on their own
void ggml_backend_synapse_device_context::upload_planar(const ggml_tensor * t, uint64_t dst, const uint8_t * src, size_t size) {
    const synapse_qt * q = synapse_qt_get(t->type);
    const size_t row = ggml_row_size(t->type, t->ne[0]);
    const size_t rbb = row * 128;
    const size_t piece = std::max<size_t>(rbb, ((size_t) 64 << 20) / rbb * rbb);
    std::lock_guard<std::mutex> lock(hst_mtx);
    flush_uploads_locked();
    if (wpin_cap < piece) {
        for (int i = 0; i < 2; i++) {
            if (wpin[i]) {
                synHostFree(sdev, wpin[i], 0);
            }
            SYNAPSE_CHECK(synHostMalloc(sdev, std::max(piece, (size_t) 64 << 20), 0, &wpin[i]));
        }
        wpin_cap = std::max(piece, (size_t) 64 << 20);
    }
    int b = 0;
    for (size_t off = 0; off < size; off += piece, b ^= 1) {
        const size_t n = std::min(piece, size - off);
        const int64_t t0 = ggml_time_us();
        synapse_qt_to_planar(q, src + off, (uint8_t *) wpin[b], t->ne[0], (int64_t) (n / row), 1);
        const int64_t t1 = ggml_time_us();
        SYNAPSE_CHECK(synStreamSynchronize(stream));
        SYNAPSE_CHECK(synMemCopyAsync(stream, (uint64_t) (uintptr_t) wpin[b], n, dst + off, HOST_TO_DRAM));
        g_load_stat.conv_ns += (t1 - t0) * 1000;
        g_load_stat.wait_ns += (ggml_time_us() - t1) * 1000;
        g_load_stat.bytes += n;
    }
    const int64_t t2 = ggml_time_us();
    SYNAPSE_CHECK(synStreamSynchronize(stream));
    g_load_stat.wait_ns += (ggml_time_us() - t2) * 1000;
}

bool ggml_backend_synapse_device_context::weight_set(const ggml_tensor * t, const void * data, size_t offset, size_t size) {
    bool in_place;
    const int kind = weight_kind(t, in_place);
    if (!kind || !in_place) {
        return false;
    }
    const uint64_t addr = (uint64_t) (uintptr_t) t->data;
    const size_t nb = ggml_nbytes(t);
    // planar pieces of whole 128-row blocks
    const size_t rbb = ggml_row_size(t->type, t->ne[0]) * 128;
    if (kind == SYN_W_PLANAR && offset % rbb == 0 && (size % rbb == 0 || offset + size == nb)) {
        bool first, last;
        {
            std::lock_guard<std::mutex> lock(wfmt_mtx);
            auto it = wstage.find(addr);
            if (it != wstage.end() && !it->second.data.empty()) {
                goto staged;   // pieces of this tensor already accumulate
            }
            first = it == wstage.end();
        }
        if (first) {
            wfmt_drop(addr, nb, true);   // (also drops staging entries: before this tensor's)
        }
        {
            std::lock_guard<std::mutex> lock(wfmt_mtx);
            staging & st = wstage[addr];
            st.received += size;
            last = st.received >= nb;
            if (last) {
                wstage.erase(addr);
            }
        }
        upload_planar(t, addr + offset, (const uint8_t *) data, size);
        if (last) {
            std::lock_guard<std::mutex> lock(wfmt_mtx);
            wfmt[addr] = wentry(kind, t, addr, true, false);
        }
        return true;
    }
staged:
    std::vector<uint8_t> src;
    {
        std::lock_guard<std::mutex> lock(wfmt_mtx);
        staging & st = wstage[addr];
        if (st.data.size() != nb) {
            st.data.assign(nb, 0);
            st.received = 0;
        }
        memcpy(st.data.data() + offset, data, size);
        st.received += size;
        if (st.received < nb) {
            return true;
        }
        src.swap(st.data);
        wstage.erase(addr);
    }
    // the whole tensor is here: convert in place (the device format fits the native size)
    const int64_t t0 = ggml_time_us();
    std::vector<uint8_t> dst(nb, 0);
    wconvert(kind, t, src.data(), dst.data());
    wfmt_drop(addr, nb, true);
    const int64_t t1 = ggml_time_us();
    copy_h2d(addr, dst.data(), nb);
    g_load_stat.whole_conv_ns += (t1 - t0) * 1000;
    g_load_stat.whole_h2d_ns += (ggml_time_us() - t1) * 1000;
    g_load_stat.whole_bytes += nb;
    std::lock_guard<std::mutex> lock(wfmt_mtx);
    wfmt[addr] = wentry(kind, t, addr, true, false);
    if (getenv("GGML_SYNAPSE_VERBOSE")) {
        GGML_LOG_INFO("%s: %s %s [%lld x %lld x %lld] -> %s in place\n", __func__, t->name, ggml_type_name(t->type),
                      (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                      kind == SYN_W_FP8_ROWS ? "fp8 rows" : kind == SYN_W_FP8_PLANAR ? "fp8 planar" : "planar");
    }
    return true;
}

bool ggml_backend_synapse_device_context::weight_get(const ggml_tensor * t, void * data, size_t offset, size_t size) {
    const uint64_t addr = (uint64_t) (uintptr_t) t->data;
    synapse_wfmt f;
    {
        std::lock_guard<std::mutex> lock(wfmt_mtx);
        auto it = wfmt.find(addr);
        if (it == wfmt.end() || !it->second.in_place) {
            return false;
        }
        f = it->second;
    }
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> dev(nb), nat(nb);
    copy_d2h(dev.data(), addr, nb);
    switch (f.kind) {
        case SYN_W_FP8_ROWS:   synapse_fp8_rows_to_q8(dev.data(), nat.data(), f.K, f.M * f.nmat); break;
        case SYN_W_FP8_PLANAR: synapse_fp8_planar_to_q8(dev.data(), nat.data(), f.K, f.M, f.nmat); break;
        default:               synapse_qt_from_planar(synapse_qt_get(f.type), dev.data(), nat.data(), f.K, f.M, f.nmat); break;
    }
    memcpy(data, nat.data() + offset, size);
    return true;
}

const synapse_wfmt * ggml_backend_synapse_device_context::weight_format(const ggml_tensor * w) {
    const uint64_t addr = (uint64_t) (uintptr_t) w->data;
    {
        std::lock_guard<std::mutex> lock(wfmt_mtx);
        auto it = wfmt.find(addr);
        if (it != wfmt.end()) {
            return it->second.src_bytes == ggml_nbytes(w) && it->second.type == w->type ? &it->second : nullptr;
        }
    }
    bool in_place;
    const int kind0 = weight_kind(w, in_place);
    if (!kind0) {
        return nullptr;
    }
    const bool weights = w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    // q8_0 outside model weights (tests, adapters) keeps the exact bf16 path unless asked for
    static const bool fp8_all = getenv("GGML_SYNAPSE_FP8_ALL") != nullptr;
    if (kind0 != SYN_W_PLANAR && !weights && !fp8_all) {
        return nullptr;
    }
    // a converted copy (padded row blocks)
    int kind = kind0;
    if (kind == SYN_W_BF16) {
        // the bf16 copy only with room left (default 8 GiB: a new 2 GiB pool block + graph workspaces;
        // GGML_SYNAPSE_Q8_BF16_ROOM_GB): else fp8 rows
        static const uint64_t room = (uint64_t) (getenv("GGML_SYNAPSE_Q8_BF16_ROOM_GB") ? atoll(getenv("GGML_SYNAPSE_Q8_BF16_ROOM_GB")) : 8) << 30;
        uint64_t mfree = 0, mtotal = 0;
        const uint64_t need = wsize(SYN_W_BF16, w) + room;
        if (synDeviceGetMemoryInfo(sdev, &mfree, &mtotal) != synSuccess || mfree < need) {
            static bool told = false;
            if (!told) {
                told = true;
                GGML_LOG_WARN("%s: %.1f GiB free: q8_0 weights from %s on as fp8 rows (less exact than bf16)\n", __func__,
                              mfree / 1073741824.0, w->name);
            }
            kind = SYN_W_FP8_ROWS;
        }
    }
    const size_t nb = ggml_nbytes(w), sz = wsize(kind, w);
    std::vector<uint8_t> src(nb), dst(sz, 0);
    copy_d2h(src.data(), addr, nb);
    wconvert(kind, w, src.data(), dst.data());
    const uint64_t dev = weights ? pool_alloc(sz) : dev_malloc(sz);
    copy_h2d(dev, dst.data(), sz);
    std::lock_guard<std::mutex> lock(wfmt_mtx);
    auto & e = wfmt[addr];
    e = wentry(kind, w, dev, false, !weights);
    return &e;
}

// 3-D q8_0 weights load as int8 planar, the experts' format (the loader cannot tell a MUL_MAT_ID's experts from
// a batched MUL_MAT's per-head matrices); the first batched MUL_MAT converts them to fp8 rows (one row per
// output row of every matrix) in place like the 2-D q8_0 weights: the native bytes are read back, no extra memory.
// Outside model weights (tests) a converted copy with GGML_SYNAPSE_FP8_ALL, else none (the exact bf16 path).
const synapse_wfmt * ggml_backend_synapse_device_context::weight_format_rows(const ggml_tensor * w) {
    static const bool q8_bf16 = getenv("GGML_SYNAPSE_Q8_BF16") && atoi(getenv("GGML_SYNAPSE_Q8_BF16")) != 0;
    static const bool fp8_all = getenv("GGML_SYNAPSE_FP8_ALL") != nullptr;
    if (!native_weights() || q8_bf16 || w->type != GGML_TYPE_Q8_0 || w->ne[3] != 1 || !ggml_is_contiguous(w)) {
        return nullptr;
    }
    const uint64_t addr = (uint64_t) (uintptr_t) w->data;
    const size_t nb = ggml_nbytes(w);
    {
        std::lock_guard<std::mutex> lock(wfmt_mtx);
        auto it = wfmt.find(addr);
        if (it != wfmt.end() && it->second.kind == SYN_W_FP8_ROWS && it->second.src_bytes == nb && it->second.type == w->type) {
            return &it->second;
        }
    }
    const bool weights = w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    const int64_t K = w->ne[0], R = w->ne[1] * w->ne[2];
    const size_t sz = synapse_fp8_rows_size(K, R);
    if ((weights && sz > nb) || (!weights && !fp8_all)) {
        return nullptr;
    }
    std::vector<uint8_t> nat(nb), dst(weights ? nb : sz, 0);
    if (!weight_get(w, nat.data(), 0, nb)) {
        copy_d2h(nat.data(), addr, nb);
    }
    synapse_q8_to_fp8_rows(nat.data(), dst.data(), K, R);
    if (weights) {
        // (no wfmt_drop: it frees the prepared graphs, and this runs inside a graph's lowering. No program
        // reads these bytes in the planar layout: only weight_get, which follows the entry replaced below)
        copy_h2d(addr, dst.data(), nb);
    }
    const uint64_t dev = weights ? addr : dev_malloc(sz);
    if (!weights) {
        copy_h2d(dev, dst.data(), sz);
    }
    std::lock_guard<std::mutex> lock(wfmt_mtx);
    auto & e = wfmt[addr];
    e = wentry(SYN_W_FP8_ROWS, w, dev, weights, !weights);
    if (getenv("GGML_SYNAPSE_VERBOSE")) {
        GGML_LOG_INFO("%s: %s q8_0 [%lld x %lld x %lld] -> fp8 rows%s\n", __func__, w->name, (long long) w->ne[0],
                      (long long) w->ne[1], (long long) w->ne[2], weights ? " in place" : "");
    }
    return &e;
}

void ggml_backend_synapse_device_context::wfmt_drop(uint64_t addr, size_t size, bool all) {
    std::vector<synapse_wfmt> gone;
    {
        std::lock_guard<std::mutex> lock(wfmt_mtx);
        for (auto it = wfmt.begin(); it != wfmt.end();) {
            if (it->first < addr + size && addr < it->first + it->second.src_bytes && (all || !it->second.in_place)) {
                gone.push_back(it->second);
                it = wfmt.erase(it);
            } else {
                ++it;
            }
        }
        if (all) {
            for (auto it = wstage.begin(); it != wstage.end();) {
                it = it->first < addr + size && addr <= it->first ? wstage.erase(it) : std::next(it);
            }
        }
    }
    if (!gone.empty()) {
        ggml_synapse_prepared_free(this);
        for (const auto & f : gone) {
            if (f.own) {
                dev_free(f.addr);
            }
        }
    }
}

uint64_t ggml_backend_synapse_device_context::dequant_weight(const ggml_tensor * w, int layout) {
    auto it = wdq.find({ w->data, layout });
    if (it != wdq.end()) {
        return it->second.addr;
    }
    const auto * tr = ggml_get_type_traits(w->type);
    GGML_ASSERT(w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32 || tr->to_float != nullptr);
    GGML_ASSERT(layout != 1 || ggml_is_contiguous(w));
    const int64_t K = w->ne[0];            // row length
    const int64_t N = ggml_nrows(w);       // rows
    const int64_t M = w->ne[1];            // rows per matrix
    const int64_t nrb = (M + 127) / 128;   // layout 1: blocks of 128 rows per matrix (zero padded)
    const uint64_t esz = layout == 2 ? 4 : 2;
    const size_t src_bytes = ggml_nbytes(w);
    const size_t row_bytes = ggml_row_size(w->type, K);
    // output rows per slab and per matrix
    const int64_t orows = layout == 1 ? (N / M) * nrb * 128 : N;
    const uint64_t nbytes = (uint64_t) orows * K * esz;

    // weights of a model live as long as the context: pool memory; others (tests, adapters) are
    // freed when their source is rewritten
    const bool own = w->buffer == nullptr || ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    const uint64_t addr = own ? dev_malloc(nbytes) : pool_alloc(nbytes);

    // slabs of whole matrices (layout 1) or of ~64 MB of output rows
    const int64_t slab = layout == 1 ? M * std::max<int64_t>(1, (64ll << 20) / std::max<int64_t>(1, nrb * 128 * K * 2))
                                     : std::max<int64_t>(1, (64ll << 20) / std::max<int64_t>(1, K * (int64_t) esz));
    std::vector<uint8_t> raw;
    std::vector<uint8_t> out;
    const int nth = (int) std::min<unsigned>(32, std::max(1u, std::thread::hardware_concurrency()));
    uint64_t dst = addr;
    for (int64_t r0 = 0; r0 < N; r0 += slab) {
        const int64_t nr = std::min(slab, N - r0);
        const int64_t nout = layout == 1 ? (nr / M) * nrb * 128 : nr;
        raw.resize(nr * row_bytes);
        out.assign((size_t) nout * K * esz, 0);
        if (!weight_get(w, raw.data(), r0 * row_bytes, nr * row_bytes)) {
            copy_d2h(raw.data(), (uint64_t) (uintptr_t) w->data + r0 * row_bytes, nr * row_bytes);
        }
        auto work = [&](int t) {
            std::vector<float> f(K);
            for (int64_t r = t; r < nr; r += nth) {
                const uint8_t * src = raw.data() + r * row_bytes;
                if (w->type == GGML_TYPE_BF16) {
                    ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, f.data(), K);
                } else if (w->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, f.data(), K);
                } else if (w->type == GGML_TYPE_F32) {
                    memcpy(f.data(), src, K * 4);
                } else {
                    tr->to_float(src, f.data(), K);
                }
                if (layout == 0) {
                    ggml_fp32_to_bf16_row(f.data(), (ggml_bf16_t *) out.data() + r * K, K);
                } else if (layout == 2) {
                    memcpy(out.data() + r * K * 4, f.data(), K * 4);
                } else {
                    // row g of matrix e: block g / 128, lane g % 128 of [lane, k]
                    const int64_t e = r / M, g = r % M;
                    uint16_t * blk = (uint16_t *) out.data() + ((e * nrb + g / 128) * K) * 128;
                    for (int64_t k = 0; k < K; k++) {
                        blk[k * 128 + g % 128] = GGML_FP32_TO_BF16(f[k]).bits;
                    }
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nth; t++) {
            th.emplace_back(work, t);
        }
        work(0);
        for (auto & x : th) {
            x.join();
        }
        copy_h2d(dst, out.data(), (uint64_t) nout * K * esz);
        dst += (uint64_t) nout * K * esz;
    }
    wdq[{ w->data, layout }] = { addr, src_bytes, own };
    if (getenv("GGML_SYNAPSE_VERBOSE")) {
        GGML_LOG_INFO("%s: dequant %s %s [%lld x %lld] layout %d (%llu MB)\n", __func__, w->name, ggml_type_name(w->type),
                      (long long) N, (long long) K, layout, (unsigned long long) (nbytes >> 20));
    }
    return addr;
}

void ggml_backend_synapse_device_context::shadow_set(uint64_t addr, const void * data, size_t size) {
    std::lock_guard<std::mutex> lock(shadow_mtx);
    auto it = shadow.find(addr);
    if (it != shadow.end() && it->second.size() == size) {
        memcpy(it->second.data(), data, size);   // the same input tensor rewritten
        return;
    }
    shadow_drop_locked(addr, size);
    // (64 MiB: prefill attention masks, read by the tiled attention's window tables)
    if (size <= (64u << 20)) {
        shadow[addr].assign((const uint8_t *) data, (const uint8_t *) data + size);
    }
}

void ggml_backend_synapse_device_context::wdq_drop(uint64_t addr, size_t size) {
    for (auto it = wdq.begin(); it != wdq.end();) {
        const uint64_t a = (uint64_t) (uintptr_t) it->first.first;
        if (a < addr + size && addr < a + it->second.src_bytes) {
            ggml_synapse_prepared_free(this);
            if (it->second.own) {
                dev_free(it->second.addr);
            }
            it = wdq.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_backend_synapse_device_context::wcat_drop(uint64_t addr, size_t size) {
    for (auto it = smallw.begin(); it != smallw.end();) {
        if (it->first < addr + size && addr < it->first + it->second.second) {
            ggml_synapse_prepared_free(this);
            it = smallw.erase(it);   // pool space is not reclaimed
        } else {
            ++it;
        }
    }
    for (auto it = wcat.begin(); it != wcat.end();) {
        bool hit = false;
        for (const auto & r : it->second.second) {
            if (r.first < addr + size && addr < r.first + r.second) {
                hit = true;
            }
        }
        if (hit) {
            ggml_synapse_prepared_free(this);
            it = wcat.erase(it);   // pool memory is not reclaimed
        } else {
            ++it;
        }
    }
}

void ggml_backend_synapse_device_context::shadow_drop_locked(uint64_t addr, size_t size) {
    // entries hold at most 1 MiB: only those starting within that much below addr can overlap
    const uint64_t lo = addr > (1u << 20) ? addr - (1u << 20) : 0;
    for (auto it = shadow.lower_bound(lo); it != shadow.end() && it->first < addr + size;) {
        if (addr < it->first + it->second.size()) {
            it = shadow.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_backend_synapse_device_context::shadow_drop(uint64_t addr, size_t size) {
    std::lock_guard<std::mutex> lock(shadow_mtx);
    shadow_drop_locked(addr, size);
}

bool ggml_backend_synapse_device_context::shadow_get(uint64_t addr, void * dst, size_t size) {
    std::lock_guard<std::mutex> lock(shadow_mtx);
    auto it = shadow.upper_bound(addr);
    if (it == shadow.begin()) {
        return false;
    }
    --it;
    if (addr + size > it->first + it->second.size()) {
        return false;
    }
    memcpy(dst, it->second.data() + (addr - it->first), size);
    return true;
}

ggml_backend_synapse_device_context::ggml_backend_synapse_device_context(int32_t device_)
        : device(device_) {
    ggml_synapse_acquire();

    // SYNAPSE<N> is visible card N (synapse_cards); the driver allows one card per process
    sdev = ggml_synapse_acquire_card(device);
    if (sdev == (synDeviceId) -1) {
        ggml_synapse_release();
        failed = true;
        return;
    }
    card = device;
    if (synStreamCreateGeneric(&stream, sdev, 0) != synSuccess) {
        synDeviceRelease(sdev);
        failed = true;
        return;
    }
}

ggml_backend_synapse_device_context::~ggml_backend_synapse_device_context() {
    if (stream) {
        synStreamDestroy(stream);
        stream = nullptr;
    }
    if (ws_addr) {
        synDeviceFree(sdev, ws_addr, 0);
        ws_addr = 0;
    }
    if (arena_addr) {
        synDeviceFree(sdev, arena_addr, 0);
        arena_addr = 0;
    }
    prepared.clear();
    graph_memos.clear();
    recipes.clear();
    if (ring) {
        synHostFree(sdev, ring, 0);
        ring = nullptr;
    }
    for (auto & kv : wdq) {
        if (kv.second.own) {
            synDeviceFree(sdev, kv.second.addr, 0);
        }
    }
    wdq.clear();
    wcat.clear();
    for (auto & r : pool) {
        synDeviceFree(sdev, r.first, 0);
    }
    pool.clear();
    if (smallw_base) {
        synDeviceFree(sdev, smallw_base, 0);
        smallw_base = 0;
    }
    if (hst_ptr) {
        synHostFree(sdev, hst_ptr, 0);
        hst_ptr = nullptr;
    }
    synDeviceRelease(sdev);
    ggml_synapse_release();
}

/* per-device context registry: all backends on one card share one context */
static std::mutex g_dctx_mtx;
static ggml_backend_synapse_device_context * g_dctx[GGML_SYNAPSE_MAX_DEVICES] = {};

/* Returning from main with an acquired card aborts inside the Synapse
 * runtime's own exit teardown. Release everything in order via atexit: the
 * handler runs before the static destructors Synapse registered at load. */
static void ggml_synapse_teardown_atexit() {
    std::lock_guard<std::mutex> lock(g_dctx_mtx);
    for (int i = 0; i < GGML_SYNAPSE_MAX_DEVICES; i++) {
        if (g_dctx[i] != nullptr) {
            delete g_dctx[i];
            g_dctx[i] = nullptr;
        }
    }
    synDestroy();
}

static ggml_backend_synapse_device_context * synapse_device_ctx(int32_t device) {
    GGML_ASSERT(device >= 0 && device < GGML_SYNAPSE_MAX_DEVICES);
    std::lock_guard<std::mutex> lock(g_dctx_mtx);
    if (g_dctx[device] == nullptr) {
        g_dctx[device] = new ggml_backend_synapse_device_context(device);
        static bool exit_registered = false;
        if (!exit_registered) {
            exit_registered = true;
            std::atexit(ggml_synapse_teardown_atexit);
        }
    }
    return g_dctx[device];
}

// the device context if the card is open already (queries must not open it)
static ggml_backend_synapse_device_context * synapse_device_ctx_open(int32_t device) {
    std::lock_guard<std::mutex> lock(g_dctx_mtx);
    return device >= 0 && device < GGML_SYNAPSE_MAX_DEVICES ? g_dctx[device] : nullptr;
}

// true when the cards run in worker processes (several visible cards)
static bool g_workers = false;

/* ---------------- buffer interface ---------------- */

struct ggml_backend_synapse_buffer_context {
    ggml_backend_synapse_device_context * dctx;
    uint64_t dev_ptr;
    uint64_t raw_ptr;   // address returned by dev_malloc
    size_t   size;
};

static const char * ggml_backend_synapse_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) buft->context;
    static std::string names[GGML_SYNAPSE_MAX_DEVICES];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    names[dctx->device] = std::string(GGML_SYNAPSE_NAME) + std::to_string(dctx->device);
    return names[dctx->device].c_str();
}

bool ggml_backend_buft_is_synapse(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_synapse_buffer_type_name;
}

static void ggml_backend_synapse_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(ctx->dctx->compute_mtx);
    ctx->dctx->sync();   // queued uploads may target this buffer
    ggml_synapse_prepared_free(ctx->dctx);
    ctx->dctx->shadow_drop(ctx->dev_ptr, ctx->size);
    ctx->dctx->wdq_drop(ctx->dev_ptr, ctx->size);
    ctx->dctx->wcat_drop(ctx->dev_ptr, ctx->size);
    ctx->dctx->wfmt_drop(ctx->dev_ptr, ctx->size, true);
    ctx->dctx->pool_release_unused();
    ctx->dctx->dev_free(ctx->raw_ptr);
    delete ctx;
}

static void * ggml_backend_synapse_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    return (void *)(uintptr_t) ctx->dev_ptr;
}

static enum ggml_status ggml_backend_synapse_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_synapse_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                   const void * data, size_t offset, size_t size) {
    if (g_trace && !g_trace_agg) {
        char w[160];
        snprintf(w, sizeof(w), "buf_set %s %zu", tensor->name, size);
        ggml_synapse_trace(w);
    } else {
        ggml_synapse_trace("buf_set");
    }
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    const uint64_t dst = (uint64_t)(uintptr_t) tensor->data + offset;
    const enum ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buffer);
    std::unique_lock<std::mutex> lock(ctx->dctx->compute_mtx, std::defer_lock);
    if (usage != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        lock.lock();
        // derived weights (dequantized / merged / converted copies) of this range are stale now
        ctx->dctx->wdq_drop(dst, size);
        ctx->dctx->wcat_drop(dst, size);
        ctx->dctx->wfmt_drop(dst, size, false);
    }
    // quantized model weights are converted to their device format as they load
    if (usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && ctx->dctx->weight_set(tensor, data, offset, size)) {
        return;
    }
    // host copies are kept for graph inputs (read by the lowering), never for weights
    if (usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        ctx->dctx->shadow_set(dst, data, size);
    } else {
        ctx->dctx->shadow_drop(dst, size);
    }
    ctx->dctx->copy_h2d_async_small(dst, data, size);
}

static void ggml_backend_synapse_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                   void * data, size_t offset, size_t size) {
    if (g_trace && !g_trace_agg) {
        char w[160];
        snprintf(w, sizeof(w), "buf_get %s %zu", tensor->name, size);
        ggml_synapse_trace(w);
    } else {
        ggml_synapse_trace("buf_get");
    }
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    if (ctx->dctx->weight_get(tensor, data, offset, size)) {
        return;   // a converted weight, read back in its ggml format
    }
    ctx->dctx->copy_d2h(data, (uint64_t)(uintptr_t) tensor->data + offset, size);
}

static void ggml_backend_synapse_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                      size_t offset, size_t size) {
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        // weights can be stored in a converted device format: write them like set_tensor does
        const std::vector<uint8_t> host(size, value);
        ggml_backend_synapse_buffer_set_tensor(buffer, tensor, host.data(), offset, size);
        return;
    }
    const uint64_t dst = (uint64_t)(uintptr_t) tensor->data + offset;
    std::unique_lock<std::mutex> lock(ctx->dctx->compute_mtx, std::defer_lock);
    if (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        lock.lock();
        ctx->dctx->wdq_drop(dst, size);
        ctx->dctx->wcat_drop(dst, size);
        ctx->dctx->wfmt_drop(dst, size, false);
    }
    ctx->dctx->shadow_drop(dst, size);
    ctx->dctx->memset_async(dst, value, size);
}

static void ggml_backend_synapse_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_synapse_buffer_context * ctx = (ggml_backend_synapse_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(ctx->dctx->compute_mtx);
    ctx->dctx->shadow_drop(ctx->dev_ptr, ctx->size);
    ctx->dctx->wfmt_drop(ctx->dev_ptr, ctx->size, true);
    ctx->dctx->memset_async(ctx->dev_ptr, value, ctx->size);
    ctx->dctx->sync();
}

static const ggml_backend_buffer_i ggml_backend_synapse_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_synapse_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_synapse_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_synapse_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_synapse_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_synapse_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_synapse_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_synapse_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_synapse_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) buft->context;

    const size_t alignment = 128;
    size = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    const uint64_t raw_ptr = dctx->dev_malloc(size + alignment);
    if (raw_ptr == 0) {
        GGML_LOG_ERROR("%s: synDeviceMalloc failed (%zu bytes)\n", __func__, size);
        return nullptr;
    }
    const uint64_t dev_ptr = (raw_ptr + alignment - 1) & ~(uint64_t) (alignment - 1);

    ggml_backend_synapse_buffer_context * ctx = new ggml_backend_synapse_buffer_context;
    ctx->dctx    = dctx;
    ctx->dev_ptr = dev_ptr;
    ctx->raw_ptr = raw_ptr;
    ctx->size    = size;

    return ggml_backend_buffer_init(buft, ggml_backend_synapse_buffer_interface, ctx, size);
}

static size_t ggml_backend_synapse_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_synapse_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_synapse_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_synapse_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_synapse_buffer_type_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ ggml_backend_synapse_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_synapse_buffer_type_get_alloc_size,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ NULL,
};

// the buffer type of an in-process device (opens the card)
static ggml_backend_buffer_type_t synapse_buffer_type(ggml_backend_dev_t dev, int device) {
    static ggml_backend_buffer_type bufts[GGML_SYNAPSE_MAX_DEVICES];
    static bool ok[GGML_SYNAPSE_MAX_DEVICES] = {};
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    if (!ok[device]) {
        bufts[device] = {
            /* .iface   = */ ggml_backend_synapse_buffer_type_interface,
            /* .device  = */ dev,
            /* .context = */ synapse_device_ctx(device),
        };
        ok[device] = true;
    }
    return &bufts[device];
}

ggml_backend_buffer_type_t ggml_backend_synapse_buffer_type(int device) {
    ggml_backend_reg_t reg = ggml_backend_synapse_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, device));
}

/* ---------------- backend interface ---------------- */

static const char * ggml_backend_synapse_name(ggml_backend_t backend) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    static std::string names[GGML_SYNAPSE_MAX_DEVICES];
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    names[dctx->device] = std::string(GGML_SYNAPSE_NAME) + std::to_string(dctx->device);
    return names[dctx->device].c_str();
}

static void ggml_backend_synapse_free(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // device contexts are process-lifetime; nothing to free per backend
}

static void ggml_backend_synapse_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor,
                                                  const void * data, size_t offset, size_t size) {
    ggml_synapse_trace("async_set");
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    if (tensor->buffer && ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        std::lock_guard<std::mutex> lock(dctx->compute_mtx);
        if (dctx->weight_set(tensor, data, offset, size)) {
            return;   // staged and converted synchronously (the caller's pinned chunk may be reused)
        }
    }
    if (!tensor->buffer || ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        dctx->shadow_set((uint64_t)(uintptr_t) tensor->data + offset, data, size);
    } else {
        dctx->shadow_drop((uint64_t)(uintptr_t) tensor->data + offset, size);
    }
    dctx->copy_h2d_async_small((uint64_t)(uintptr_t) tensor->data + offset, data, size);
}

static void ggml_backend_synapse_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor,
                                                  void * data, size_t offset, size_t size) {
    ggml_synapse_trace("async_get");
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    if (dctx->weight_get(tensor, data, offset, size)) {
        return;
    }
    dctx->flush_uploads();
    if (dctx->is_pinned(data, size)) {
        // completed by ggml_backend_synchronize
        SYNAPSE_CHECK(synMemCopyAsync(dctx->stream, (uint64_t)(uintptr_t) tensor->data + offset, size,
                                      (uint64_t)(uintptr_t) data, DRAM_TO_HOST));
        dctx->set_pending();
        return;
    }
    dctx->copy_d2h(data, (uint64_t)(uintptr_t) tensor->data + offset, size);
}

static void ggml_backend_synapse_synchronize(ggml_backend_t backend) {
    ggml_synapse_trace("sync");
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    // GGML_SYNAPSE_STRICT_SYNC=1: always sync the stream
    static const bool strict = getenv("GGML_SYNAPSE_STRICT_SYNC") != nullptr;
    if (dctx->pending || strict) {
        dctx->sync();
        ggml_synapse_trace("sync waited");
    }
}

static void ggml_synapse_prof_dump(void);
static ggml_backend_synapse_device_context * g_prof_dctx = nullptr;

static enum ggml_status ggml_backend_synapse_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    /* profiling aid: GGML_SYNAPSE_PROF=<tsv path> captures one TEF trace
     * over all graphs computed after the env is set, dumped at process
     * exit; GGML_SYNAPSE_PROF_SKIP=N
     * starts it at the (N+1)-th graph */
    static bool prof_init = false;
    static long prof_skip = getenv("GGML_SYNAPSE_PROF_SKIP") ? atol(getenv("GGML_SYNAPSE_PROF_SKIP")) : 0;
    if (!prof_init && prof_skip > 0) {
        prof_skip--;
    } else if (!prof_init) {
        prof_init = true;
        if (getenv("GGML_SYNAPSE_PROF")) {
            uint32_t bytes = 0;
            ggml_backend_synapse_device_context * dctx =
                (ggml_backend_synapse_device_context *) backend->context;
            if (synProfilerQueryRequiredMemory(dctx->sdev, &bytes) == synSuccess && bytes) {
                uint64_t buf = 0;
                if (synDeviceMalloc(dctx->sdev, bytes, 0, 0, &buf) == synSuccess) {
                    synProfilerSetUserBuffer(dctx->sdev, (void *)(uintptr_t) buf);
                }
            }
            g_prof_dctx = dctx;
            synProfilerStart(synTraceAll, dctx->sdev);
            atexit(ggml_synapse_prof_dump);
        }
    }
    return ggml_synapse_graph_compute(backend, cgraph);
}

static void ggml_synapse_prof_dump(void) {
    const char * path = getenv("GGML_SYNAPSE_PROF");
    if (!path || !g_prof_dctx) {
        return;
    }
    ggml_backend_synapse_device_context * dctx = g_prof_dctx;
    dctx->sync();
    synProfilerStop(synTraceAll, dctx->sdev);
    size_t bytes = 0, entries = 0;
    if (synProfilerGetTrace(synTraceAll, dctx->sdev, synTraceFormatTEF, nullptr, &bytes,
                            &entries) != synSuccess) {
        return;
    }
    std::vector<unsigned char> buf(bytes);
    if (synProfilerGetTrace(synTraceAll, dctx->sdev, synTraceFormatTEF, buf.data(), &bytes,
                            &entries) != synSuccess) {
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        return;
    }
    fprintf(f, "ph\tengine_type\tengine_index\tts\tdur\tname\tcategory\tmeta\n");
    auto * ev = reinterpret_cast<synTraceEvent *>(buf.data());
    for (size_t i = 0; i < entries; i++) {
        auto & e = ev[i];
        fprintf(f, "%c\t%u\t%u\t%.3Lf\t%.3Lf\t%s\t%s\t%s\n", e.type ? e.type : '?', e.engineType,
                e.engineIndex, (long double) e.timestamp, (long double) e.duration,
                e.name ? e.name : "", e.category ? e.category : "",
                (e.type == 'M' && e.arguments.name) ? e.arguments.name : "");
    }
    fclose(f);
}

static void ggml_backend_synapse_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    SYNAPSE_CHECK(synEventRecord((synEventHandle) event->context, dctx->stream));
}

static void ggml_backend_synapse_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;
    SYNAPSE_CHECK(synStreamWaitEvent(dctx->stream, (synEventHandle) event->context, 0));
}

static const ggml_backend_i ggml_backend_synapse_interface = {
    /* .get_name                = */ ggml_backend_synapse_name,
    /* .free                    = */ ggml_backend_synapse_free,
    /* .set_tensor_async        = */ ggml_backend_synapse_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_synapse_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_synapse_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_synapse_graph_compute,
    /* .event_record            = */ ggml_backend_synapse_event_record,
    /* .event_wait              = */ ggml_backend_synapse_event_wait,
    /* .graph_optimize          = */ NULL,
};

/* ---------------- device interface ---------------- */

struct ggml_backend_synapse_device_context_pub {
    int32_t device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_synapse_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_synapse_device_context_pub * ctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_synapse_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_synapse_device_context_pub * ctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_synapse_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_synapse_device_context_pub * pctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    ggml_backend_synapse_device_context * dctx = synapse_device_ctx_open(pctx->device);
    if (!dctx || dctx->failed) {
        // not open (or in use elsewhere): the driver's figures, a card held by another process shows 0 free
        if (!ggml_synapse_card_memory(pctx->device, free, total)) {
            *free = *total = 0;
        }
        return;
    }
    uint64_t vals[2] = { 0, 0 };
    const synDeviceAttribute attrs[] = { DEVICE_ATTRIBUTE_DRAM_SIZE, DEVICE_ATTRIBUTE_DRAM_FREE_SIZE };
    synDeviceGetAttribute(vals, attrs, 2, dctx->sdev);
    *free  = (size_t) vals[1];
    *total = (size_t) vals[0];
}

static enum ggml_backend_dev_type ggml_backend_synapse_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_synapse_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_synapse_device_get_name(dev);
    props->description = ggml_backend_synapse_device_get_description(dev);
    props->type        = ggml_backend_synapse_device_get_type(dev);
    props->device_id   = ggml_synapse_card_pci(((ggml_backend_synapse_device_context_pub *) dev->context)->device);
    ggml_backend_synapse_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ true,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_synapse_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_synapse_device_context_pub * dctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    return ggml_backend_synapse_init(dctx->device);
}

static bool ggml_backend_synapse_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_synapse_device_context_pub * ctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    return ggml_backend_buft_is_synapse(buft) && buft->device == dev;
}

static ggml_backend_buffer_type_t ggml_backend_synapse_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_synapse_device_context_pub * ctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    return synapse_buffer_type(dev, ctx->device);
}

/* pinned host buffers */
static const char * ggml_backend_synapse_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_SYNAPSE_NAME "_Host";
}

static void ggml_backend_synapse_host_buffer_free(ggml_backend_buffer_t buffer) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) buffer->buft->context;
    dctx->pinned_drop(buffer->context);
    synHostFree(dctx->sdev, buffer->context, 0);
}

static ggml_backend_buffer_t ggml_backend_synapse_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) buft->context;
    void * ptr = nullptr;
    if (synHostMalloc(dctx->sdev, size ? size : 64, 0, &ptr) != synSuccess) {
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    dctx->pinned_add(ptr, size);
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_synapse_host_buffer_free;
    return buffer;
}

static size_t ggml_backend_synapse_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static bool ggml_backend_synapse_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static ggml_backend_buffer_type_t ggml_backend_synapse_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    if (getenv("GGML_SYNAPSE_NO_HOSTBUF")) {
        return nullptr;
    }
    ggml_backend_synapse_device_context_pub * pctx = (ggml_backend_synapse_device_context_pub *) dev->context;
    static ggml_backend_buffer_type buft;
    static bool init = false;
    if (!init) {
        init = true;
        buft = {
            /* .iface   = */ {
                /* .get_name         = */ ggml_backend_synapse_host_buffer_type_name,
                /* .alloc_buffer     = */ ggml_backend_synapse_host_buffer_type_alloc_buffer,
                /* .alloc_buffer_n   = */ NULL,
                /* .get_alignment    = */ ggml_backend_synapse_host_buffer_type_get_alignment,
                /* .get_max_size     = */ NULL,
                /* .get_alloc_size   = */ NULL,
                /* .get_alloc_size_n = */ NULL,
                /* .is_host          = */ ggml_backend_synapse_host_buffer_type_is_host,
            },
            /* .device  = */ dev,
            /* .context = */ synapse_device_ctx(pctx->device),
        };
    }
    return &buft;
}

// ops on host-resident weights are offloaded for batches >= 32 only (the CUDA policy)
static bool ggml_backend_synapse_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int64_t min_batch = 32;
    if (op->op == GGML_OP_GET_ROWS) {
        return false;
    }
    return ggml_synapse_supports_op(dev, op) && op->ne[1] >= min_batch;
}

static ggml_backend_event_t ggml_backend_synapse_device_event_new(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    synEventHandle ev = nullptr;
    synEventCreate(&ev, 0, 0);
    return new ggml_backend_event{ ggml_backend_reg_dev_get(ggml_backend_synapse_reg(), 0), ev };
}

static void ggml_backend_synapse_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    synEventDestroy((synEventHandle) event->context);
    delete event;
}

static void ggml_backend_synapse_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    synEventSynchronize((synEventHandle) event->context);
}

static const ggml_backend_device_i ggml_backend_synapse_device_interface = {
    /* .get_name                = */ ggml_backend_synapse_device_get_name,
    /* .get_description         = */ ggml_backend_synapse_device_get_description,
    /* .get_memory              = */ ggml_backend_synapse_device_get_memory,
    /* .get_type                = */ ggml_backend_synapse_device_get_type,
    /* .get_props               = */ ggml_backend_synapse_device_get_props,
    /* .init_backend            = */ ggml_backend_synapse_device_init,
    /* .get_buffer_type         = */ ggml_backend_synapse_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_synapse_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_synapse_supports_op,
    /* .supports_buft           = */ ggml_backend_synapse_supports_buft,
    /* .offload_op              = */ ggml_backend_synapse_offload_op,
    /* .event_new               = */ ggml_backend_synapse_device_event_new,
    /* .event_free              = */ ggml_backend_synapse_device_event_free,
    /* .event_synchronize       = */ ggml_backend_synapse_device_event_synchronize,
};

/* ---------------- registry ---------------- */

struct ggml_backend_synapse_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_synapse_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_SYNAPSE_NAME;
}

static size_t ggml_backend_synapse_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_synapse_reg_context * ctx = (ggml_backend_synapse_reg_context *) reg->context;
    /* GGML_SYNAPSE=0 disables the backend entirely (CPU reference runs) */
    const char * e = getenv("GGML_SYNAPSE");
    if (e && atoi(e) == 0) {
        return 0;
    }
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_synapse_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_synapse_reg_context * ctx = (ggml_backend_synapse_reg_context *) reg->context;
    GGML_ASSERT(ctx && index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_synapse_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    static const bool no_comm = getenv("GGML_SYNAPSE_NO_COMM") != nullptr;
    return no_comm ? nullptr : ggml_synapse_proxy_proc_address(name);
}

static const ggml_backend_reg_i ggml_backend_synapse_reg_interface = {
    /* .get_name          = */ ggml_backend_synapse_reg_get_name,
    /* .get_device_count  = */ ggml_backend_synapse_reg_get_device_count,
    /* .get_device        = */ ggml_backend_synapse_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_synapse_reg_get_proc_address,
};

static ggml_guid_t ggml_backend_synapse_guid() {
    static ggml_guid guid = { 0xb2, 0x81, 0x4a, 0x77, 0x33, 0x5c, 0x48, 0x9d,
                              0xa6, 0x02, 0x51, 0xe8, 0x77, 0x0b, 0x93, 0x4f };
    return &guid;
}

static bool ggml_synapse_usable_generation() {
    // Gaudi2 only; other generations (Gaudi3 etc.) are rejected so the build
    // never emits wrong ISA objects or runs on an unsupported card.
    return true;   // synDeviceGaudi2 enumeration already filters the type
}

ggml_backend_reg_t ggml_backend_synapse_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            initialized = true;
            if (!ggml_synapse_usable_generation()) {
                reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                        /* .iface       = */ ggml_backend_synapse_reg_interface,
                                        /* .context     = */ nullptr };
                return &reg;
            }
            ggml_backend_synapse_reg_context * ctx = new ggml_backend_synapse_reg_context;
            // llama's KV padding in geometric steps by default here (LLAMA_KV_PAD_GEOM=0: off)
            setenv("LLAMA_KV_PAD_GEOM", "1", 0);
            // and a prompt ubatch of one sequence padded to one of a few sizes (llama-batch.cpp llama_ubatch_pad_size;
            // LLAMA_UBATCH_PAD=0: off)
            setenv("LLAMA_UBATCH_PAD", "1", 0);
            // each graph allocated from a plan of its own (ggml-backend.cpp; GGML_SCHED_ALLOC_FRESH=0: off)
            setenv("GGML_SCHED_ALLOC_FRESH", "1", 0);
            /* the driver opens one card per process (a second synDeviceAcquire fails with status 20):
             * with several visible cards each one runs in a worker process (ggml-synapse-proxy.cpp),
             * started when the card is first used; this process does not use Synapse itself */
            const int n = ggml_synapse_card_count();
            if (n > 1) {
                g_workers = true;
                if (!ggml_synapse_proxy_init(&reg, ctx->devices)) {
                    GGML_LOG_ERROR("%s: card worker setup failed; no " GGML_SYNAPSE_NAME " devices\n", __func__);
                    ctx->devices.clear();
                }
                reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                        /* .iface       = */ ggml_backend_synapse_reg_interface,
                                        /* .context     = */ ctx };
                return &reg;
            }
            // one visible card: used in this process (opened on first use)
            if (n == 1) {
                ggml_synapse_acquire();   // process-lifetime synInitialize
            }
            for (int32_t i = 0; i < n; ++i) {
                ggml_backend_synapse_device_context_pub * dctx = new ggml_backend_synapse_device_context_pub;
                dctx->device = i;
                dctx->name = GGML_SYNAPSE_NAME + std::to_string(i);
                dctx->description = ggml_synapse_card_description(i);
                ggml_backend_dev_t dev = new ggml_backend_device{
                    /* .iface   = */ ggml_backend_synapse_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dctx,
                };
                ctx->devices.push_back(dev);
            }
            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_synapse_reg_interface,
                                    /* .context     = */ ctx };
        }
    }
    return &reg;
}

ggml_backend_t ggml_backend_synapse_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_synapse_reg();
    if (g_workers) {
        // the card runs in its worker process
        if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
            GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
            return nullptr;
        }
        return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, device), nullptr);
    }
    ggml_synapse_acquire();   // ensure synInitialize ran before any query
    if (device < 0 || device >= ggml_synapse_card_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        ggml_synapse_release();
        return nullptr;
    }
    ggml_backend_synapse_device_context * ctx = synapse_device_ctx(device);
    if (ctx->failed) {
        ggml_synapse_release();
        return nullptr;
    }
    ggml_backend_t backend = new ggml_backend{
        /* .guid      = */ ggml_backend_synapse_guid(),
        /* .interface = */ ggml_backend_synapse_interface,
        /* .device    = */ ggml_backend_reg_dev_get(reg, device),
        /* .context   = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_synapse(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_synapse_guid());
}

bool ggml_backend_buffer_is_synapse(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && ggml_backend_buft_is_synapse(ggml_backend_buffer_get_type(buffer));
}

int ggml_backend_synapse_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_synapse_reg());
}

GGML_BACKEND_DL_IMPL(ggml_backend_synapse_reg)

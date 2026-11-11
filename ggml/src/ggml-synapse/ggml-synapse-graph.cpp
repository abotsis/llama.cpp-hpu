/*
 * ggml-synapse-graph.cpp - lowers a ggml (split) graph into a Synapse recipe.
 *
 * Memory model:
 *  - every ggml "root" (the end of a view_src chain) maps to one Synapse tensor:
 *      external roots (weights, caches, graph inputs) -> persistent input bound at their address
 *      roots produced in the graph                     -> internal tensor (GC managed), or a
 *                                                         persistent output when read outside
 *  - views are kernel parameters (element offset + strides) over a flat 1-D tensor; reshape
 *    nodes (logical) give the GEMMs their 2-D shapes
 *  - writes into a view of an existing root (SET_ROWS into the KV cache, CPY into recurrent
 *    state, in-place ops) are side-effect kernels that store through their first input; they
 *    are ordered against the other readers/writers of that root with control dependencies
 *    (the GC only orders by data flow). A persistent "sink" output keeps them alive.
 *  - a graph output that overlaps a persistent input (the allocator reused the memory) goes
 *    through an arena slot and a device copy after the launch
 */
#include "ggml-synapse-impl.h"

#include <synapse_api.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

/* ---------------- hashing ---------------- */

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h = (h ^ p[i]) * 1099511628211ull;
    }
    return h;
}

static std::string hex128(const std::string & s) {
    const uint64_t a = fnv1a(s.data(), s.size());
    const uint64_t b = fnv1a(s.data(), s.size(), 0x9e3779b97f4a7c15ull);
    char buf[40];
    snprintf(buf, sizeof(buf), "%016" PRIx64 "%016" PRIx64, a, b);
    return buf;
}

/* kernel plugin + SDK identity */
static const std::string & synapse_env_digest() {
    static std::string digest;
    static std::once_flag once;
    std::call_once(once, [] {
        std::string s = "ggml-synapse-2;";
        for (const char * p : { "/usr/lib/habanalabs/libSynapse.so", "/usr/lib/habanalabs/libtpc_kernels.so" }) {
            std::error_code ec;
            if (fs::exists(p, ec)) {
                s += p;
                s += ":" + std::to_string((long long) fs::last_write_time(p, ec).time_since_epoch().count()) + ";";
            }
        }
        Dl_info info{};
        if (dladdr((void *) &ggml_synapse_register_tpc_plugin, &info) && info.dli_fname) {
            const fs::path plugin = fs::path(info.dli_fname).parent_path() / "libggml_synapse_kernels.so";
            std::ifstream f(plugin, std::ios::binary);
            if (f) {
                std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                s += hex128(bytes);
            }
        }
        for (char ** e = environ; *e; ++e) {
            for (const char * k : { "ENABLE_", "GC_", "SYN_", "RUN_TPC", "SRAM_", "DISABLE_", "MAX_TPC", "PT_HPU_", "WORKSPACE_" }) {
                if (strncmp(*e, k, strlen(k)) == 0) {
                    s += *e;
                    s += ";";
                }
            }
        }
        digest = hex128(s);
    });
    return digest;
}

static std::string synapse_cache_dir() {
    const char * env = getenv("GGML_SYNAPSE_CACHE");
    if (env && strcmp(env, "0") == 0) {
        return "";
    }
    if (env && env[0]) {
        return env;
    }
    const char * home = getenv("HOME");
    if (home && home[0]) {
        return std::string(home) + "/.cache/llama.cpp/synapse";
    }
    return "";
}

/* ---------------- TPC plugin registration ---------------- */

/* The graph compiler discovers custom TPC kernels through GC_KERNEL_PATH; the companion
 * plugin is resolved next to the loaded backend library and the stock kernel library always
 * follows it. Must run before the first synInitialize. */
void ggml_synapse_register_tpc_plugin() {
    std::string plugin;
    Dl_info info{};
    if (dladdr((void *) &ggml_synapse_register_tpc_plugin, &info) && info.dli_fname) {
        const fs::path dir = fs::path(info.dli_fname).parent_path();
        std::error_code ec;
        const fs::path p = dir / "libggml_synapse_kernels.so";
        if (fs::exists(p, ec)) {
            plugin = fs::absolute(p).string();
        }
    }
    // the shell profile on these hosts may export a GC_KERNEL_PATH for another project:
    // override it (GGML_SYNAPSE_KERNEL_PATH appends extra libraries explicitly)
    std::string paths = plugin;
    const char * extra = getenv("GGML_SYNAPSE_KERNEL_PATH");
    if (extra && extra[0]) {
        paths = paths.empty() ? std::string(extra) : paths + ":" + extra;
    }
    const char * stock = "/usr/lib/habanalabs/libtpc_kernels.so";
    paths = paths.empty() ? std::string(stock) : paths + ":" + stock;
    setenv("GC_KERNEL_PATH", paths.c_str(), 1);
    if (plugin.empty()) {
        GGML_LOG_WARN("%s: kernel plugin not found next to the backend library\n", __func__);
    }
}

/* ---------------- lowered program ---------------- */

namespace {

enum lt_kind {
    LT_INTERNAL = 0,
    LT_IN       = 1,   // persistent input at an absolute device address
    LT_OUT      = 2,   // persistent output at an absolute device address
    LT_ARENA    = 3,   // persistent at an arena offset (sinks, staged outputs, host tables)
};

struct lt_t {
    synDataType dt   = syn_type_single;
    unsigned    dims = 1;
    uint64_t    sz[4] = { 1, 1, 1, 1 };
    int         kind = LT_INTERNAL;
    uint64_t    addr = 0;        // LT_IN/LT_OUT: device address, LT_ARENA: arena offset
    uint64_t    nbytes = 0;
    int         base = -1;       // reshape alias of lt `base` (dependency tracking)
    uint64_t    region = 0;      // base of the allocation holding an LT_IN tensor (0: own section)
    int         sec = -1;        // persistent section group
    uint64_t    sec_off = 0;     // offset in the section
};

struct ln_t {
    std::string          guid;
    std::vector<int>     ins;
    std::vector<int>     outs;
    std::vector<uint8_t> params;
    std::vector<int>     deps;   // control dependencies: nodes that must run first
    std::string          src;    // the ggml node lowered (GGML_SYNAPSE_DUMP_PROG)
};

struct host_table {
    uint64_t arena_off = 0;
    uint64_t nbytes = 0;
    std::function<bool(void *)> fill;   // writes nbytes; false: input not host-known
};

struct post_copy {
    uint64_t arena_off;
    uint64_t dst;
    uint64_t nbytes;
};

struct program {
    std::vector<lt_t>       lts;
    std::vector<ln_t>       nodes;
    std::vector<host_table> tables;
    std::vector<post_copy>  posts;
    uint64_t                arena_size = 0;
};

} // namespace

struct synapse_recipe {
    synRecipeHandle r = nullptr;
    uint64_t ws_size = 0;
    bool from_disk = false;   // deserialized from the recipe cache, else compiled
    double build_ms = 0;      // graph build + compile (or deserialize)
    std::vector<int>      persist;   // lt index of each launch entry
    std::vector<std::string> names;
    std::vector<uint64_t> ids;
    ~synapse_recipe() {
        if (r) {
            synRecipeDestroy(r);
        }
    }
};

// one launch of a tiled program: a segment, or (tile >= 0) the windows of tiled FA `tile` (launch: the template)
struct synapse_step {
    std::shared_ptr<synapse_recipe> rec;
    std::vector<synLaunchTensorInfoExt> launch;
    std::vector<std::pair<size_t, uint64_t>> launch_arena;
    int tile = -1;
    // tile steps: the window recipe and launch template per window size of the group
    std::vector<std::shared_ptr<synapse_recipe>> wrec;
    std::vector<std::vector<synLaunchTensorInfoExt>> wlaunch;
};
struct synapse_tile {
    int group = 0;
    uint64_t qbd_off = 0, acc_off[3] = { 0, 0, 0 }, ml_off[3] = { 0, 0, 0 };   // A, B, F
    uint64_t sinks_off = 0;
    uint64_t kaddr = 0, vaddr = 0, krow = 0, vrow = 0;
    // launch entries: qbd kw vw tab acc_in ml_in ml_out acc_out [sinks]
    int idx[9] = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
};
struct synapse_tile_group {
    uint64_t tab_off = 0, entry = 256;
    std::shared_ptr<std::vector<int64_t>> win;   // per window of the current launch: start cell, size index
};

struct synapse_prepared {
    std::shared_ptr<synapse_recipe> rec;
    // tiled decode attention (rec unused): steps in order, the windows' tables and state
    std::vector<synapse_step> steps;
    std::vector<synapse_tile> tiles;
    std::vector<synapse_tile_group> tile_groups;
    uint64_t ws_max = 0;
    std::shared_ptr<int> tile_bad;
    std::shared_ptr<synapse_prepared> fallback;   // untiled, for launches whose mask is not one range per row
    std::vector<synLaunchTensorInfoExt> launch;
    std::vector<lt_t> launch_info;    // the launch tensors' lowering (for diagnostics)
    uint64_t arena_size = 0;
    uint64_t arena = 0;               // the shared arena the addresses below point into
    std::vector<std::pair<size_t, uint64_t>> launch_arena;   // launch entry, arena offset
    std::vector<uint64_t> post_off;
    void *   host = nullptr;          // pinned mirror of the host tables
    uint64_t host_size = 0;
    std::vector<host_table> tables;
    std::vector<post_copy> posts;
    std::vector<uint64_t> post_src, post_dst, post_size;
    ggml_backend_synapse_device_context * dctx = nullptr;
    ~synapse_prepared() {
        free(host);
    }
    void rebase(uint64_t base) {
        for (const auto & la : launch_arena) {
            launch[la.first].pTensorAddress = base + la.second;
        }
        for (auto & st : steps) {
            for (const auto & la : st.launch_arena) {
                st.launch[la.first].pTensorAddress = base + la.second;
            }
        }
        for (size_t i = 0; i < post_off.size(); i++) {
            post_src[i] = base + post_off[i];
        }
        arena = base;
    }
};

static std::string lt_name(size_t i) {
    return "t" + std::to_string(i);
}

static uint64_t syn_type_size(synDataType dt) {
    switch (dt) {
        case syn_type_single: return 4;
        case syn_type_int32:  return 4;
        case syn_type_bf16:   return 2;
        case syn_type_fp16:   return 2;
        case syn_type_int8:   return 1;
        case syn_type_fp8_143: return 1;
        default:              return 4;
    }
}

/* structural signature: everything the compiled recipe depends on (no addresses) */
static std::string program_signature(const program & p) {
    std::string s;
    s.reserve(64 * (p.lts.size() + p.nodes.size()));
    char buf[160];
    for (const auto & t : p.lts) {
        snprintf(buf, sizeof(buf), "%d:%u:%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ":%d:%d:%" PRIu64 "|", (int) t.dt, t.dims,
                 t.sz[0], t.sz[1], t.sz[2], t.sz[3], t.kind == LT_INTERNAL ? 0 : (t.kind == LT_IN ? 1 : 2), t.sec, t.sec_off);
        s += buf;
    }
    s += "#";
    for (const auto & n : p.nodes) {
        s += n.guid;
        s += ':';
        for (int i : n.ins)  { s += std::to_string(i); s += ','; }
        s += '>';
        for (int o : n.outs) { s += std::to_string(o); s += ','; }
        s += '[';
        s.append((const char *) n.params.data(), n.params.size());
        s += ']';
        for (int d : n.deps) { s += std::to_string(d); s += ','; }
        s += '|';
    }
    return s;
}

/* ---------------- compile ---------------- */

static std::shared_ptr<synapse_recipe> synapse_compile(const program & p, const std::string & sig) {
    // the Synapse graph API is not thread safe: graphs built at the same time fail in synNodeCreate
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    const int64_t t_build = ggml_time_us();
    auto rec = std::make_shared<synapse_recipe>();
    const std::string key = hex128(sig);
    const std::string rname = "ggml_" + key.substr(0, 24);

    synGraphHandle g = nullptr;
    SYNAPSE_CHECK(synGraphCreate(&g, synDeviceGaudi2));

    std::vector<synTensor> th(p.lts.size(), nullptr);
    std::vector<synSectionHandle> secs;
    std::unordered_map<int, synSectionHandle> group_sec;
    std::vector<std::string> names(p.lts.size());
    for (size_t i = 0; i < p.lts.size(); i++) {
        const lt_t & t = p.lts[i];
        names[i] = lt_name(i);
        synTensorDescriptor d;
        d.m_dataType = t.dt;
        d.m_dims = t.dims;
        for (unsigned k = 0; k < SYN_MAX_TENSOR_DIM; k++) {
            d.m_sizes[k] = k < t.dims ? (unsigned) t.sz[k] : 0;
            d.m_minSizes[k] = d.m_sizes[k];
            d.m_strides[k] = 0;
        }
        d.m_name = names[i].c_str();
        synSectionHandle sec = nullptr;
        uint64_t off = 0;
        if (t.kind != LT_INTERNAL) {
            auto gs = t.sec >= 0 ? group_sec.find(t.sec) : group_sec.end();
            if (gs != group_sec.end()) {
                sec = gs->second;
            } else {
                SYNAPSE_CHECK(synSectionCreate(&sec, 0, g));
                SYNAPSE_CHECK(synSectionSetPersistent(sec, true));
                secs.push_back(sec);
                if (t.sec >= 0) {
                    group_sec[t.sec] = sec;
                }
            }
            off = t.sec >= 0 ? t.sec_off : 0;
            rec->persist.push_back((int) i);
            rec->names.push_back(names[i]);
        }
        SYNAPSE_CHECK(synTensorCreate(&th[i], &d, sec, off));
    }

    bool ok = true;
    std::vector<synNodeId> ids(p.nodes.size(), 0);
    for (size_t ni = 0; ni < p.nodes.size() && ok; ni++) {
        const ln_t & n = p.nodes[ni];
        std::vector<synTensor> ins, outs;
        for (int i : n.ins) {
            ins.push_back(th[i]);
        }
        for (int o : n.outs) {
            outs.push_back(th[o]);
        }
        const std::string nname = "n" + std::to_string(ni) + "_" + key.substr(0, 6) + "_" + n.guid;
        const synStatus st = synNodeCreateWithId(g, ins.data(), outs.data(), (uint32_t) ins.size(), (uint32_t) outs.size(),
                                                 n.params.empty() ? nullptr : n.params.data(), (unsigned) n.params.size(),
                                                 n.guid.c_str(), nname.c_str(), &ids[ni], nullptr, nullptr);
        if (st != synSuccess) {
            GGML_LOG_ERROR("%s: synNodeCreate(%s) failed: %d\n", __func__, n.guid.c_str(), (int) st);
            ok = false;
            break;
        }
        for (int d : n.deps) {
            const synNodeId a = ids[d], b = ids[ni];
            if (synNodeDependencySet(g, &a, &b, 1, 1) != synSuccess) {
                GGML_LOG_ERROR("%s: synNodeDependencySet failed\n", __func__);
                ok = false;
                break;
            }
        }
    }

    bool loaded = false;
    std::string cache_path;
    const std::string cdir = synapse_cache_dir();
    if (ok && !cdir.empty()) {
        cache_path = cdir + "/" + hex128(sig + synapse_env_digest()) + ".recipe";
        std::error_code ec;
        if (fs::exists(cache_path, ec) && synRecipeDeSerialize(&rec->r, cache_path.c_str()) == synSuccess) {
            loaded = true;
            rec->from_disk = true;
        }
    }
    if (ok && !loaded) {
        const int64_t t0 = ggml_time_us();
        const synStatus st = synGraphCompile(&rec->r, g, rname.c_str(), nullptr);
        if (st != synSuccess) {
            GGML_LOG_ERROR("%s: synGraphCompile failed: %d (%zu nodes)\n", __func__, (int) st, p.nodes.size());
            if (getenv("GGML_SYNAPSE_COMPILE_DIAG")) {
                for (const ln_t & nd : p.nodes) {
                    std::string s;
                    for (int k = 0; k < 2; k++) {
                        for (int i : k ? nd.outs : nd.ins) {
                            const lt_t & t = p.lts[i];
                            uint64_t ne = 1;
                            for (unsigned d = 0; d < t.dims; d++) {
                                ne *= t.sz[d];
                            }
                            s += (k ? " >" : " ") + std::to_string(ne) + "e";
                            for (unsigned d = 0; d < t.dims; d++) {
                                s += (d ? "x" : "[") + std::to_string(t.sz[d]);
                            }
                            s += "]";
                        }
                    }
                    GGML_LOG_ERROR("  %s:%s\n", nd.guid.c_str(), s.c_str());
                }
            }
            ok = false;
        } else {
            if (getenv("GGML_SYNAPSE_VERBOSE")) {
                GGML_LOG_INFO("%s: compiled %zu nodes in %.1f ms\n", __func__, p.nodes.size(), (ggml_time_us() - t0) / 1000.0);
            }
            if (!cache_path.empty()) {
                std::error_code ec;
                fs::create_directories(cdir, ec);
                const std::string tmp = cache_path + ".tmp." + std::to_string(getpid());
                if (synRecipeSerialize(rec->r, tmp.c_str()) == synSuccess) {
                    fs::rename(tmp, cache_path, ec);
                } else {
                    fs::remove(tmp, ec);
                }
            }
        }
    }

    for (auto t : th) {
        if (t) {
            synTensorDestroy(t);
        }
    }
    for (auto s : secs) {
        synSectionDestroy(s);
    }
    synGraphDestroy(g);
    if (!ok) {
        return nullptr;
    }

    SYNAPSE_CHECK(synWorkspaceGetSize(&rec->ws_size, rec->r));
    if (!rec->names.empty()) {
        std::vector<const char *> nm;
        for (auto & s : rec->names) {
            nm.push_back(s.c_str());
        }
        rec->ids.resize(nm.size());
        SYNAPSE_CHECK(synTensorRetrieveIds(rec->r, nm.data(), rec->ids.data(), (uint32_t) nm.size()));
    }
    rec->build_ms = (ggml_time_us() - t_build) / 1000.0;
    return rec;
}

/* what a graph's shape depends on: the tokens it computes (the widest weight GEMM / attention query batch) and the
   KV length its attention reads */
static void graph_shape(const ggml_cgraph * cg, int64_t & tokens, int64_t & kv) {
    tokens = kv = 0;
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if (t->op == GGML_OP_MUL_MAT && t->src[0]->op == GGML_OP_NONE) {
            tokens = std::max(tokens, t->src[1]->ne[1] * t->src[1]->ne[2] * t->src[1]->ne[3]);
        } else if (t->op == GGML_OP_MUL_MAT_ID) {
            tokens = std::max(tokens, t->src[1]->ne[2]);
        } else if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            tokens = std::max(tokens, t->src[0]->ne[1] * t->src[0]->ne[3]);
            kv = std::max(kv, t->src[1]->ne[1]);
        }
    }
}

/* ---------------- lowering ---------------- */

static bool full_root_weight(const ggml_tensor * w);

namespace {

static synDataType lt_dtype(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return syn_type_single;
        case GGML_TYPE_F16:  return syn_type_fp16;
        case GGML_TYPE_BF16: return syn_type_bf16;
        case GGML_TYPE_I32:  return syn_type_int32;
        case GGML_TYPE_I64:  return syn_type_int32;   // read as i32 pairs
        default:             return syn_type_int8;    // raw bytes
    }
}

/* element size of the flat view of a ggml type */
static int64_t lt_esize(ggml_type t) {
    return t == GGML_TYPE_I64 ? 4 : (int64_t) syn_type_size(lt_dtype(t));
}

/* quantized types the backend dequantizes to bf16 (weights, gathered rows) */
static bool dq_type(ggml_type t) {
    return ggml_is_quantized(t) && ggml_get_type_traits(t)->to_float != nullptr;
}

static const ggml_tensor * root_of(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool is_view_op(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

/* operand: a ggml tensor as a view over a flat lowered tensor (element units) */
struct opnd {
    int     lt = -1;
    int64_t off = 0;
    int64_t ne[4] = { 1, 1, 1, 1 };
    int64_t nb[4] = { 0, 0, 0, 0 };
};

struct params_t {
    std::vector<int32_t> v;
    size_t bytes = 0;   // exact parameter size (native nodes taking a struct), 0: the words in v
    params_t & i(int64_t x) { v.push_back((int32_t) x); return *this; }
    params_t & f(float x) { int32_t y; memcpy(&y, &x, 4); v.push_back(y); return *this; }
    // index space size (params[0]) + items per member, for kernels using SPLIT_CHUNK
    params_t & split(int64_t work, int64_t per);
};

// kernel name of a quantized type: lower-case ggml name (defined here: the lowerer's methods
// use it before the original definition site)
static std::string qt_kname(const char * prefix, ggml_type t) {
    std::string s = prefix;
    for (const char * p = ggml_type_name(t); *p; p++) {
        s += (char) tolower(*p);
    }
    return s + "_f32";
}

static params_t gemm_params(bool ta, bool tb) {
    params_t pr;
    synGEMMParams gp(ta, tb);
    pr.v.resize((sizeof(gp) + 3) / 4);
    memcpy(pr.v.data(), &gp, sizeof(gp));
    pr.bytes = sizeof(gp);
    return pr;
}

static int64_t hc_chunk(int64_t ne0, int64_t T) {
    const int64_t per_tok = (24 + T - 1) / T;
    const int64_t c = ((ne0 + per_tok - 1) / per_tok + 63) / 64 * 64;
    return std::max<int64_t>(64, std::min<int64_t>(1024, c));
}

static int nmem_for(int64_t work, int64_t per = 16) {
    int64_t n = (work + per - 1) / per;
    return (int) std::max<int64_t>(1, std::min<int64_t>(24, n));
}

params_t & params_t::split(int64_t work, int64_t per) {
    const int nm = nmem_for(work, per);
    return i(nm).i((work + nm - 1) / nm);
}

struct lowerer {
    ggml_backend_synapse_device_context * dctx;
    const ggml_cgraph * cg;
    program p;
    std::string err;

    struct root_state {
        int lt = -1;                    // flat lt of the root
        std::unordered_map<std::string, int> shapes;   // reshaped aliases
        bool bf16 = false;              // stored as bf16 (only GEMMs read it)
        int lt_bf = -1;                 // f32 root with a second, bf16 copy for its GEMM readers
        // fp8 rows [ne] + a f32 scale per row [rows] for its fp8 GEMM readers; fp8: stored only so (lt = lt_f8)
        bool fp8 = false;
        int lt_f8 = -1, lt_f8s = -1;
        // remap: the root is rows of mlen elements inside lt, row stride mstride, from roff
        int64_t roff = 0, mlen = 0, mstride = 0;
        int64_t boff = -1;              // >= 0: element offset of the root inside lt
        int dense = -1;                 // de-interleaved copy (for views crossing rows)
    };
    // graph structure
    std::unordered_map<const ggml_tensor *, std::vector<const ggml_tensor *>> cons;
    std::unordered_map<const ggml_tensor *, int> order;
    std::unordered_map<const ggml_tensor *, bool> skip;
    struct norm_plan {
        const ggml_tensor * add = nullptr;
        const ggml_tensor * mul = nullptr;
        const ggml_tensor * w = nullptr;
        const ggml_tensor * rope = nullptr;
        // the add's other operand is a sum of nslices views of one tensor, stride elements apart (an
        // all-gather of tensor-parallel partial sums): read as slices by the kernel
        const ggml_tensor * slice0 = nullptr;
        int nslices = 1;
        int64_t slice_stride = 0;
        int slice_side = 1;
        // MoE layer output: the slices are views of MUL(ebase, wts) (wts [1, nslices, T]: per slice and
        // token weights, read by the kernel instead), the add operand is (slices + extra)
        const ggml_tensor * extra = nullptr;
        const ggml_tensor * wts = nullptr;
        const ggml_tensor * ebase = nullptr;
    };
    std::unordered_map<const ggml_tensor *, norm_plan> norm_plans;
    bool match_chain(norm_plan & pl, const ggml_tensor * top, const ggml_tensor * extra, const ggml_tensor * mid, int side,
                     bool own_top = false, bool unweighted = false) {
        const ggml_tensor * r = top;
        std::vector<const ggml_tensor *> chain, views;
        while (r->op == GGML_OP_ADD && !r->view_src && ((own_top && r == top) || (cons[r].size() == 1 && !is_output[r])) &&
               !skip[r] && order.count(r) && ggml_are_same_shape(r, pl.add)) {
            chain.push_back(r);
            views.push_back(r->src[1]);
            r = r->src[0];
        }
        if (chain.empty()) {
            return false;
        }
        views.push_back(r);
        std::reverse(views.begin(), views.end());
        const ggml_tensor * base = views[0]->view_src;
        bool ok = base && base->type == GGML_TYPE_F32 && views.size() <= 16;
        bool strided = false;
        for (size_t k = 0; ok && k < views.size(); k++) {
            const ggml_tensor * v = views[k];
            const bool rows = v->nb[0] == sizeof(float) && v->ne[2] == 1 && v->ne[3] == 1 && v->nb[1] == views[0]->nb[1];
            strided |= !ggml_is_contiguous(v);
            ok = v->op == GGML_OP_VIEW && v->view_src == base && (ggml_is_contiguous(v) || rows) && ggml_are_same_shape(v, pl.add) &&
                 v->view_offs == views[0]->view_offs + k * (views.size() > 1 ? views[1]->view_offs - views[0]->view_offs : 0) &&
                 cons[v].size() == 1;
        }
        if (!ok || views.size() < 2 || (views[1]->view_offs - views[0]->view_offs) % sizeof(float) != 0) {
            return false;
        }
        const ggml_tensor * wts = nullptr, * ebase = nullptr;
        if (base->op == GGML_OP_MUL && !is_output[base] && !skip[base] && order.count(base) && cons[base].size() == views.size()) {
            const ggml_tensor * e = base->src[0], * w = base->src[1];
            if (ggml_are_same_shape(e, base) && e->type == GGML_TYPE_F32 && ggml_is_contiguous(e) && w->type == GGML_TYPE_F32 &&
                w->ne[0] == 1 && w->ne[1] == base->ne[1] && w->ne[2] == base->ne[2] && w->ne[3] == base->ne[3] &&
                base->ne[1] == (int64_t) views.size()) {
                wts = w;
                ebase = e;
            }
        }
        if (unweighted && !wts && !strided && base->type == GGML_TYPE_F32 && ggml_is_contiguous(base) &&
            cons[base].size() == views.size()) {
            ebase = base;
        } else if ((extra || strided || own_top) && !wts) {
            return false;
        }
        if (wts && views.size() > 32) {
            return false;   // (syn_moe_sum keeps a token's rows and weights in arrays of 32)
        }
        pl.slice0 = views[0];
        pl.nslices = (int) views.size();
        pl.slice_stride = (int64_t) ((views[1]->view_offs - views[0]->view_offs) / sizeof(float));
        pl.slice_side = side;
        pl.extra = extra;
        pl.wts = wts;
        pl.ebase = ebase;
        for (const ggml_tensor * c : chain) {
            skip[c] = true;
        }
        if (mid) {
            skip[mid] = true;
        }
        if (wts) {
            skip[base] = true;
        }
        return true;
    }
    void match_slices(norm_plan & pl, const ggml_tensor * norm) {
        GGML_UNUSED(norm);
        for (int side = 0; side < 2; side++) {
            if (match_chain(pl, pl.add->src[side], nullptr, nullptr, side)) {
                return;
            }
        }
        for (int side = 0; side < 2; side++) {
            const ggml_tensor * m = pl.add->src[side];
            if (m->op != GGML_OP_ADD || m->view_src || cons[m].size() != 1 || is_output[m] || skip[m] || !order.count(m) ||
                !ggml_are_same_shape(m, pl.add)) {
                continue;
            }
            for (int s2 = 0; s2 < 2; s2++) {
                const ggml_tensor * x = m->src[1 - s2];
                if (x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, pl.add) || x->nb[0] != sizeof(float) ||
                    x->ne[2] != 1 || x->ne[3] != 1) {
                    continue;
                }
                if (match_chain(pl, m->src[s2], x, m, side)) {
                    return;
                }
            }
        }
    }
    struct gdn_plan {
        const ggml_tensor * G = nullptr, * Y = nullptr, * xsrc = nullptr, * wconv = nullptr;
        const ggml_tensor * Rr = nullptr, * Rs = nullptr, * idx = nullptr;
        const ggml_tensor * cpy_r = nullptr, * cpy_s = nullptr, * gr_r = nullptr, * gr_s = nullptr;
        const ggml_tensor * cont_r = nullptr;   // a CONT between the conv-state view and its CPY (qwen4exp)
        // rollback slots 1..K-1 of the conv state (n_rs_seq > 0): CPYs of the old conv state (concat columns 0..nc-2)
        std::vector<const ggml_tensor *> cpy_rx;
        const ggml_tensor * qv = nullptr, * kv = nullptr, * vv = nullptr;
        const ggml_tensor * alpha = nullptr, * dt = nullptr, * A = nullptr, * beta = nullptr;
        const ggml_tensor * z = nullptr, * wn = nullptr;
        float eps_qk = 0.0f, qk_scale = 0.0f, eps_o = 0.0f;
        bool zsig = false;   // output gate sigmoid(z) (qwen4exp), else silu(z)
    };
    std::unordered_map<const ggml_tensor *, gdn_plan> gdn_plans;   // keyed by the final MUL
    // keyed by the CONCAT
    struct convx_plan {
        const ggml_tensor * conv = nullptr, * Y = nullptr, * cpy = nullptr;
    };
    std::unordered_map<const ggml_tensor *, convx_plan> convx_plans;
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> rope_fa;   // rope -> FA
    std::unordered_map<const ggml_tensor *, int> fa_q_bf;                   // FA -> bf16 Q [D, nq, H]
    bool fa_sdpa(const ggml_tensor * n);
    void plan_convx(const ggml_tensor * cat);
    bool lower_convx(const ggml_tensor * cat, const convx_plan & pl);
    struct hc_plan {
        const ggml_tensor * p = nullptr;      // post = pb * sigmoid(pa * p); null: the post tensor as is
        float pa = 1.0f, pb = 1.0f;
        const ggml_tensor * norm = nullptr, * mul = nullptr;
        bool bf = false;                      // a bf16 copy of the norm output for its GEMM readers
    };
    std::unordered_map<const ggml_tensor *, hc_plan> hc_plans;   // keyed by the DSV4_HC_POST
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> silu_scale;   // SILU -> its SCALE source
    void plan_hc(const ggml_tensor * n);
    std::unordered_map<std::string, int> rope_tables;
    struct gate_plan {
        const ggml_tensor * mul = nullptr;    // attn * sigmoid(gate)
        const ggml_tensor * gate = nullptr;   // the gate view (before CONT / SIGMOID)
    };
    std::unordered_map<const ggml_tensor *, gate_plan> gate_plans;   // keyed by the FA node
    struct router_plan {
        const ggml_tensor * bias = nullptr, * argsort = nullptr, * weights = nullptr;
        int k = 0;
        float cmin = 0.0f, scale = 1.0f;
        bool norm = false;
        bool smax = false;   // softmax scores (Qwen style): no bias, the argsort reads the probabilities
    };
    std::unordered_map<const ggml_tensor *, router_plan> router_plans;   // keyed by the SIGMOID / SOFT_MAX
    // attention masks equal to another tensor (DSA top-k masks while top-k selects every key)
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> mask_alias;
    const ggml_tensor * mask_of(const ggml_tensor * fa) {
        const ggml_tensor * mk = fa->src[3];
        auto it = mk ? mask_alias.find(mk) : mask_alias.end();
        return it != mask_alias.end() ? it->second : mk;
    }
    void plan_dsa_mask(const ggml_tensor * add);
    // keyed by the final ADD
    std::unordered_map<const ggml_tensor *, norm_plan> moe_sum_plans;
    bool lower_moe_sum(const ggml_tensor * n, const norm_plan & pl);
    void plan_router(const ggml_tensor * sg);
    bool lower_router(const ggml_tensor * sg, const router_plan & pl);
    std::unordered_map<const ggml_tensor *, const ggml_tensor *> setrows_pair;   // second -> first
    bool match_gdn(const ggml_tensor * G, gdn_plan & pl, std::vector<const ggml_tensor *> & members,
                   const ggml_tensor *& anchor);
    bool lower_gdn(const ggml_tensor * n, const gdn_plan & pl);
    bool lower_gdn_mt(const ggml_tensor * n, const gdn_plan & pl, const opnd & ox, const opnd & oi, int lr, int ls,
                      int64_t srow_r, int64_t drow_r);
    std::unordered_map<const ggml_tensor *, root_state> roots;
    std::unordered_map<const ggml_tensor *, bool> produced;   // roots computed in this graph
    std::unordered_map<const ggml_tensor *, bool> is_output;  // roots read outside the graph


    int base_of(int lt) const {
        while (p.lts[lt].base >= 0) {
            lt = p.lts[lt].base;
        }
        return lt;
    }

    int new_lt(synDataType dt, std::initializer_list<uint64_t> sz, int kind = LT_INTERNAL, uint64_t addr = 0) {
        lt_t t;
        t.dt = dt;
        t.dims = (unsigned) sz.size();
        int k = 0;
        uint64_t n = syn_type_size(dt);
        for (uint64_t s : sz) {
            t.sz[k++] = s;
            n *= s;
        }
        t.kind = kind;
        t.addr = addr;
        t.nbytes = n;
        p.lts.push_back(t);
        return (int) p.lts.size() - 1;
    }

    uint64_t arena_alloc(uint64_t nbytes) {
        const uint64_t off = p.arena_size;
        p.arena_size += (nbytes + 255) & ~255ull;
        return off;
    }

    /* Ordering. The GC schedules by data flow only (control dependencies are dropped by its
     * final re-sort), so side-effect writes are ordered with token tensors: every custom kernel
     * has a trailing `dep` input. A writer depends on the outputs of the root's readers since
     * the previous write (WAR) and on that write's sink (WAW); a reader depends on the sink of
     * the last write (RAW). Several tokens are joined by syn_join nodes. */
    std::unordered_map<int, int> last_write_tok;               // base lt -> sink of last writer
    std::unordered_map<int, std::vector<int>> reader_outs;     // base lt -> reader outputs since
    int dummy_tok = -1;
    bool edge_lost = false;   // a native node needed an ordering token: the lowering fails at that node

    int join(std::vector<int> toks) {
        if (toks.empty()) {
            if (dummy_tok < 0) {
                dummy_tok = new_lt(syn_type_single, { 64 }, LT_ARENA);
                p.lts[dummy_tok].addr = arena_alloc(256);
            }
            return dummy_tok;
        }
        while (toks.size() > 1) {
            std::vector<int> next;
            for (size_t i = 0; i < toks.size(); i += 4) {
                if (i + 1 == toks.size()) {
                    next.push_back(toks[i]);
                    continue;
                }
                ln_t n;
                n.guid = "syn_join_f32";
                for (size_t k = 0; k < 4; k++) {
                    n.ins.push_back(toks[std::min(i + k, toks.size() - 1)]);
                }
                const int t = new_lt(syn_type_single, { 64 });
                n.outs = { t };
                const int32_t one = 1;
                n.params.resize(4);
                memcpy(n.params.data(), &one, 4);
                p.nodes.push_back(n);
                next.push_back(t);
            }
            toks = next;
        }
        return toks[0];
    }

    // writes: bitmask of inputs the kernel stores through (side-effect writes)
    int add_node(const char * guid, std::vector<int> ins, std::vector<int> outs, const params_t & prm,
                 unsigned writes = 0) {
        const bool custom = strncmp(guid, "syn_", 4) == 0;
        // 32 scalar params (the glue's MAX_SCALAR_PARAMS) faults the device; 30 is known good
        if (custom && prm.v.size() > 30) {
            err = std::string("too many scalar params for ") + guid;
        }
        std::vector<int> toks;
        for (size_t k = 0; k < ins.size(); k++) {
            const int b = base_of(ins[k]);
            auto lw = last_write_tok.find(b);
            if (lw != last_write_tok.end()) {
                toks.push_back(lw->second);
            }
            if (writes & (1u << k)) {
                for (int r : reader_outs[b]) {
                    toks.push_back(r);
                }
            }
        }
        std::sort(toks.begin(), toks.end());
        toks.erase(std::unique(toks.begin(), toks.end()), toks.end());
        const size_t n_data_ins = ins.size();
        if (custom) {
            ins.push_back(join(toks));
        } else if (!toks.empty()) {
            err = std::string("ordering edge into a native node: ") + guid;
            edge_lost = true;
        }
        ln_t n;
        n.guid = guid;
        n.ins = ins;
        n.outs = outs;
        n.src = cur_src;
        n.params.resize(prm.bytes ? prm.bytes : prm.v.size() * 4);
        memcpy(n.params.data(), prm.v.data(), n.params.size());
        p.nodes.push_back(std::move(n));
        for (size_t k = 0; k < n_data_ins; k++) {
            const int b = base_of(ins[k]);
            if (writes & (1u << k)) {
                last_write_tok[b] = outs[0];
                reader_outs[b].clear();
            } else if (!outs.empty()) {
                reader_outs[b].push_back(outs[0]);
            }
        }
        return (int) p.nodes.size() - 1;
    }

    std::string cur_src;   // the ggml node being lowered, recorded in each node (GGML_SYNAPSE_DUMP_PROG)

    std::vector<int> sinks;
    int sink() {
        const int s = new_lt(syn_type_single, { 64 });
        sinks.push_back(s);
        return s;
    }

    /* native transpose of the dense [a, b] (nb of them, type dt) in lt `src`: a flat lt holding [b, a] x nb */
    int transpose2d(int src, int64_t a, int64_t b, int64_t nb = 1, synDataType dt = syn_type_single) {
        const int in = nb > 1 ? reshape(src, dt, { (uint64_t) a, (uint64_t) b, (uint64_t) nb }) :
                                reshape(src, dt, { (uint64_t) a, (uint64_t) b });
        const int dst = nb > 1 ? new_lt(dt, { (uint64_t) b, (uint64_t) a, (uint64_t) nb }) :
                                 new_lt(dt, { (uint64_t) b, (uint64_t) a });
        synTransposeParams tp = {};
        tp.permutation[0] = TPD_Width;
        tp.permutation[1] = TPD_Channel;
        tp.permutation[2] = TPD_Height;
        tp.tensorDim = nb > 1 ? 3 : 2;
        params_t pt;
        pt.v.resize((sizeof(tp) + 3) / 4);
        memcpy(pt.v.data(), &tp, sizeof(tp));
        pt.bytes = sizeof(tp);
        add_node("transpose", { in }, { dst }, pt);
        return reshape(dst, dt, { (uint64_t) (a * b * nb) });
    }

    /* a view whose dim 0 walks the rows of a value read as [W, rows] (dim-0 stride W; the other dims inside one row
      * or whole rows apart): the same view of a native transpose of the rows read, dim 0 contiguous. */
    std::map<std::tuple<int, int64_t, int64_t, int64_t>, int> tposed;
    bool untranspose(opnd & o) {
        const int64_t W = o.nb[0];
        const int64_t L = (int64_t) p.lts[o.lt].sz[0];
        const synDataType dt = p.lts[o.lt].dt;
        const char * cpy = dt == syn_type_single ? "syn_cpy_f32_f32_f32" : dt == syn_type_int32 ? "syn_cpy_i32_i32_f32" :
                           dt == syn_type_bf16 ? "syn_cpy_bf16_bf16_f32" : dt == syn_type_fp16 ? "syn_cpy_f16_f16_f32" : nullptr;
        if (o.ne[0] <= 1 || W <= 1 || p.lts[o.lt].dims != 1 || !cpy) {
            return false;
        }
        int64_t col = o.off % W;     // last column read
        int64_t nrow = o.ne[0];      // rows read, from row o.off / W
        bool row_dim[4] = { true, false, false, false };
        for (int d = 1; d < 4; d++) {
            if (o.ne[d] <= 1) {
                continue;
            }
            if (o.nb[d] % W == 0) {
                row_dim[d] = true;
                nrow += (o.ne[d] - 1) * (o.nb[d] / W);
            } else if (o.nb[d] < W) {
                col += (o.ne[d] - 1) * o.nb[d];
            } else {
                return false;
            }
        }
        int64_t row0 = o.off / W;
        if (col >= W || (row0 + nrow) * W > L) {
            return false;
        }
        const bool written = last_write_tok.count(base_of(o.lt)) > 0;
        if (!written && L % W == 0 && 2 * nrow * W >= L) {
            // the whole value
            row0 = 0;
            nrow = L / W;
        }
        const auto key = std::make_tuple(o.lt, W, row0, nrow);
        auto it = written ? tposed.end() : tposed.find(key);
        int tl;
        if (it != tposed.end()) {
            tl = it->second;
        } else {
            int src = o.lt;
            if (written || nrow * W < L) {
                // the rows read copied out first: part of a larger value, or a
                // value written earlier in the graph (a native node takes no ordering token, the copy does)
                const int64_t n = nrow * W;
                src = new_lt(dt, { (uint64_t) n });
                params_t pr;
                pr.i(nmem_for((n + 127) / 128, 8)).i(n).i(1).i(1).i(1).i(row0 * W).i(1).i(n).i(n).i(n);
                add_node(cpy, { o.lt }, { src }, pr);
            }
            tl = transpose2d(src, W, nrow, 1, dt);
            if (!written) {
                tposed[key] = tl;
            }
        }
        opnd r = o;
        r.lt = tl;
        r.off = (o.off % W) * nrow + (o.off / W - row0);
        for (int d = 0; d < 4; d++) {
            r.nb[d] = d > 0 && o.ne[d] <= 1 ? 0 : row_dim[d] ? o.nb[d] / W : o.nb[d] * nrow;
        }
        o = r;
        return true;
    }

    /* reshape alias of lt `src` (logical node) */
    int reshape(int src, synDataType dt, std::initializer_list<uint64_t> sz) {
        const int r = new_lt(dt, sz);
        p.lts[r].base = base_of(src);
        ln_t n;
        n.guid = "reshape";
        n.ins = { src };
        n.outs = { r };
        p.nodes.push_back(n);
        return r;
    }

    bool fail(const char * fmt, const ggml_tensor * t) {
        char buf[256];
        snprintf(buf, sizeof(buf), fmt, t ? ggml_op_desc(t) : "?");
        err = buf;
        if (t) {
            err += std::string(" (") + t->name + ")";
        }
        return false;
    }

    /* flat lt of a root, creating the persistent input on first use */
    int small_lt = -1;

    int root_lt(const ggml_tensor * r, bool allow_small = true) {
        auto & rs = roots[r];
        if (rs.lt >= 0 && (allow_small || rs.boff < 0)) {
            return rs.lt;
        }
        if (r->buffer == nullptr || ggml_backend_buffer_is_host(r->buffer)) {
            err = std::string("root not in device memory: ") + r->name;
            return -1;
        }
        static const bool no_small = getenv("GGML_SYNAPSE_NO_SMALLW") != nullptr;
        if (allow_small && !no_small && r->type == GGML_TYPE_F32 && ggml_nbytes(r) <= (1u << 20) &&
            ggml_backend_buffer_get_usage(r->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            const uint64_t off = dctx->smallw_get((uint64_t) (uintptr_t) r->data, ggml_nbytes(r));
            if (small_lt < 0) {
                small_lt = new_lt(syn_type_single, { dctx->smallw_cap / 4 }, LT_IN, dctx->smallw_base);
                p.lts[small_lt].region = 0;
            }
            rs.lt = small_lt;
            rs.boff = (int64_t) (off / 4);
            return rs.lt;
        }
        const uint64_t nb = ggml_nbytes(r);
        const int64_t es = lt_esize(r->type);
        const int lt = new_lt(lt_dtype(r->type), { (uint64_t) (nb / es) }, LT_IN, (uint64_t) (uintptr_t) r->data);
        p.lts[lt].region = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(r->buffer);
        if (rs.lt < 0) {
            rs.lt = lt;
            rs.boff = -1;
        }
        return lt;
    }

    /* the root as a tensor of another shape (reshape alias of the flat lt) */
    int root_shaped(const ggml_tensor * r, synDataType dt, std::initializer_list<uint64_t> sz) {
        const int flat = root_lt(r, false);
        if (flat < 0) {
            return -1;
        }
        std::string key = std::to_string((int) dt);
        for (uint64_t s : sz) {
            key += "," + std::to_string(s);
        }
        auto & rs = roots[r];
        auto it = rs.shapes.find(key);
        if (it != rs.shapes.end()) {
            return it->second;
        }
        const int lt = reshape(flat, dt, sz);
        rs.shapes[key] = lt;
        return lt;
    }

    /* root r was written earlier in this graph (e.g. a KV cache after its SET_ROWS): a native node cannot wait for the write */
    bool written_root(const ggml_tensor * r) {
        auto it = roots.find(r);
        return it != roots.end() && it->second.lt >= 0 && last_write_tok.count(base_of(it->second.lt)) > 0;
    }

    bool get(const ggml_tensor * t, opnd & o) {
        const ggml_tensor * r = root_of(t);
        const int lt = root_lt(r);
        if (lt < 0) {
            return false;
        }
        const int64_t es = lt_esize(r->type);
        if (t->type != r->type && !(ggml_type_size(t->type) == ggml_type_size(r->type))) {
            return fail("view type mismatch: %s", t);
        }
        const int64_t boff = (int64_t) ((const char *) t->data - (const char *) r->data);
        if (boff % es) {
            return fail("unaligned view: %s", t);
        }
        const root_state & rs = roots[r];
        if (rs.bf16) {
            return fail("bf16 root read by a non-GEMM op: %s", t);
        }
        if (rs.fp8) {
            return fail("fp8 root read by a non-GEMM op: %s", t);
        }
        o.lt = lt;
        o.off = boff / es + (rs.boff > 0 ? rs.boff : 0);
        for (int d = 0; d < 4; d++) {
            o.ne[d] = t->ne[d];
            if ((int64_t) t->nb[d] % es) {
                return fail("unaligned stride: %s", t);
            }
            o.nb[d] = (int64_t) t->nb[d] / es;
            if (t->type == GGML_TYPE_I64) {
                // strides in i32 units are doubled; kernels index i64 tensors in i64 units
                o.nb[d] /= 2;
            }
        }
        if (t->type == GGML_TYPE_I64) {
            o.off /= 2;
        }
        if (rs.mlen > 0) {
            opnd m = o;
            const int64_t row = o.off / rs.mlen;
            m.off = rs.roff + row * rs.mstride + (o.off - row * rs.mlen);
            bool cross = false;
            for (int d = 0; d < 4; d++) {
                if (o.ne[d] <= 1) {
                    continue;
                }
                if (o.nb[d] % rs.mlen == 0) {
                    m.nb[d] = o.nb[d] / rs.mlen * rs.mstride;
                } else if (o.nb[d] * (o.ne[d] - 1) >= rs.mlen) {
                    cross = true;
                }
            }
            if (!cross) {
                o = m;
            } else {
                // a view across the rows of a remapped root (e.g. a reshape of Vcur): read a
                // de-interleaved copy of the root
                root_state & rw = roots[r];
                if (rw.dense < 0) {
                    const int64_t nr = ggml_nelements(r) / rs.mlen;
                    rw.dense = new_lt(syn_type_single, { (uint64_t) ggml_nelements(r) });
                    params_t pc;
                    pc.i(nmem_for(nr * ((rs.mlen + 127) / 128), 8)).i(rs.mlen).i(nr).i(1).i(1)
                      .i(rs.roff).i(1).i(rs.mstride).i(rs.mstride * nr).i(rs.mstride * nr);
                    add_node("syn_cpy_f32_f32_f32", { rs.lt }, { rw.dense }, pc);
                }
                o.lt = rw.dense;
            }
        }
        if (o.off > INT32_MAX || (int64_t) p.lts[lt].sz[0] > INT32_MAX) {
            return fail("tensor too large: %s", t);
        }
        return true;
    }

    /* destination of node n: a fresh flat lt (n is a root) or the root of the view n writes */
    int fresh_out(const ggml_tensor * n) {
        const int64_t es = lt_esize(n->type);
        const uint64_t cnt = ggml_nbytes(n) / es;
        int lt;
        if (is_output[n]) {
            lt = new_lt(lt_dtype(n->type), { cnt }, LT_OUT, (uint64_t) (uintptr_t) n->data);
        } else if (want_bf16(n)) {
            lt = new_lt(syn_type_bf16, { cnt });
            roots[n].bf16 = true;
        } else {
            lt = new_lt(lt_dtype(n->type), { cnt });
        }
        roots[n].lt = lt;
        return lt;
    }

    /* w feeds a bf16 GEMM: bf16 weights, or quantized ones without a device format (dequantized) */
    bool bf16_gemm_weight(const ggml_tensor * w) {
        if (w->type == GGML_TYPE_BF16) {
            return true;
        }
        if (!dq_type(w->type)) {
            return false;
        }
        bool in_place;
        const int kind = ggml_backend_synapse_device_context::weight_kind(w, in_place);
        if (kind == SYN_W_BF16 && w->ne[2] == 1 && w->ne[3] == 1 && full_root_weight(w) && !produced.count(root_of(w))) {
            // a bf16 copy (q8_0), unless the memory guard fell back to fp8 rows
            const synapse_wfmt * f = dctx->weight_format(root_of(w));
            return f && f->kind == SYN_W_BF16;
        }
        return kind == 0;
    }

    /* every in-graph use of t is the activation of a bf16 GEMM reading the whole tensor */
    bool gemm_consumers_only(const ggml_tensor * t) {
        auto it = cons.find(t);
        if (it == cons.end() || it->second.empty() || !ggml_is_contiguous(t)) {
            return false;
        }
        for (const ggml_tensor * c : it->second) {
            if (c->op != GGML_OP_MUL_MAT || c->src[1] != t || c->src[0] == t || !bf16_gemm_weight(c->src[0])) {
                return false;
            }
        }
        return true;
    }

    /* w is read by lower_qmm's fp8 GEMM (fp8 rows + per-row scales) */
    bool fp8_gemm_weight(const ggml_tensor * w) {
        if (!dq_type(w->type) || w->ne[2] != 1 || w->ne[3] != 1 || !full_root_weight(w) || produced.count(root_of(w))) {
            return false;
        }
        const synapse_wfmt * f = dctx->weight_format(root_of(w));
        return f && f->kind == SYN_W_FP8_ROWS;
    }

    /* every in-graph use of t is the activation of a bf16 GEMM (fp8: an fp8 GEMM), directly or through
     * contiguous whole-tensor reshapes: t can be produced as bf16 (fp8 rows) */
    bool gemm_only(const ggml_tensor * t, int depth = 0, bool fp8 = false) {
        auto it = cons.find(t);
        if (it == cons.end() || it->second.empty() || depth > 4) {
            return false;
        }
        for (const ggml_tensor * c : it->second) {
            if (c->op == GGML_OP_MUL_MAT && c->src[1] == t && c->src[0] != t && ggml_is_contiguous(t) &&
                (fp8 ? !skip[c] && c->type == GGML_TYPE_F32 && fp8_gemm_weight(c->src[0]) : bf16_gemm_weight(c->src[0]))) {
                continue;
            }
            if ((c->op == GGML_OP_RESHAPE || c->op == GGML_OP_VIEW) && c->data == t->data &&
                ggml_nelements(c) == ggml_nelements(t) && ggml_is_contiguous(c) && gemm_only(c, depth + 1, fp8)) {
                continue;
            }
            return false;
        }
        return true;
    }

    /* fp8 activations with a scale per row;
     * GGML_SYNAPSE_NO_FP8_FUSE=1: a per-tensor quantization per GEMM (syn_fp8q1 / syn_fp8amax + syn_fp8q) */
    static bool fp8_fuse() {
        static const bool off = getenv("GGML_SYNAPSE_NO_FP8_FUSE") != nullptr;
        return !off;
    }

    /* GGML_SYNAPSE_FP8_POW2_SCALE=1: per-token scales rounded up to a power of two, so a value rounds the same in every
     * token as with one scale per tensor (better for phi-2's LayerNorm + bias, a little worse for Llama / Qwen3.5) */
    static int fp8_pow2() {
        static const int on = getenv("GGML_SYNAPSE_FP8_POW2_SCALE") && atoi(getenv("GGML_SYNAPSE_FP8_POW2_SCALE")) != 0;
        return on;
    }

    /* every in-graph reader of n is an fp8 GEMM: n can be produced as fp8 rows + scales */
    bool fp8_readers_only(const ggml_tensor * n) {
        return fp8_fuse() && n->type == GGML_TYPE_F32 && n->view_src == nullptr && gemm_only(n, 0, true);
    }

    bool want_bf16(const ggml_tensor * n) {
        static const bool off = getenv("GGML_SYNAPSE_NO_BF16_OUT") != nullptr;
        switch (n->op) {
            case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV: case GGML_OP_GLU:
            case GGML_OP_UNARY: case GGML_OP_SCALE: case GGML_OP_SQR: case GGML_OP_SQRT:
            case GGML_OP_RMS_NORM: case GGML_OP_L2_NORM: case GGML_OP_NORM:
                break;
            default:
                return false;
        }
        return !off && n->type == GGML_TYPE_F32 && n->view_src == nullptr && !is_output[n] && gemm_only(n);
    }

    /* write contiguous f32/same-type data (lt `src`, n's shape) into the view n of its root */
    bool write_view(const ggml_tensor * n, int src, ggml_type src_type) {
        opnd d;
        if (!get(n, d)) {
            return false;
        }
        if (d.nb[0] != 1) {
            return fail("non-contiguous write view: %s", n);
        }
        const char * g = nullptr;
        switch (n->type) {
            case GGML_TYPE_F32:  g = "syn_wr_f32_f32"; break;
            case GGML_TYPE_F16:  g = src_type == GGML_TYPE_F16 ? "syn_wr_f16s_f32" : "syn_wr_f16_f32"; break;
            case GGML_TYPE_BF16: g = src_type == GGML_TYPE_BF16 ? "syn_wr_bf16s_f32" : "syn_wr_bf16_f32"; break;
            case GGML_TYPE_I32:
                if (src_type != GGML_TYPE_I32) {
                    return fail("write source type: %s", n);
                }
                g = "syn_wr_i32_f32";
                break;
            default: return fail("write type: %s", n);
        }
        if (src_type != GGML_TYPE_F32 && src_type != n->type) {
            return fail("write source type: %s", n);
        }
        const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
        params_t pr;
        pr.i(nmem_for(rows * ((n->ne[0] + 127) / 128))).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
          .i(d.off).i(d.nb[1]).i(d.nb[2]).i(d.nb[3])
          .i(0).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2]);
        add_node(g, { d.lt, src }, { sink() }, pr, true);
        return true;
    }

    /* output of an elementwise-style op: fresh lt, or a temp that is written into the view */
    int begin_out(const ggml_tensor * n) {
        if (n->view_src == nullptr) {
            return fresh_out(n);
        }
        const int64_t es = lt_esize(n->type);
        return new_lt(lt_dtype(n->type), { (uint64_t) (ggml_nelements(n) * (n->type == GGML_TYPE_I64 ? 2 : 1)) });
        (void) es;
    }

    bool end_out(const ggml_tensor * n, int out) {
        if (n->view_src == nullptr) {
            return true;
        }
        return write_view(n, out, n->type);
    }

    /* operand over a contiguous lt holding t's elements */
    static void dense_opnd(const ggml_tensor * t, int lt, opnd & o) {
        o.lt = lt;
        o.off = 0;
        int64_t st = 1;
        for (int d = 0; d < 4; d++) {
            o.ne[d] = t->ne[d];
            o.nb[d] = st;
            st *= t->ne[d];
        }
    }

    /* contiguous copy of operand s (t's shape) by kernel g (syn_cpy_*): the result as a flat lt */
    int cpy_dense(const ggml_tensor * t, const opnd & s, const char * g, synDataType dt) {
        const int lt = new_lt(dt, { (uint64_t) ggml_nelements(t) });
        return cpy_dense(t, s, g, dt, lt);
    }

    /* cpy_dense into a caller-owned lt (e.g. the output lt of the node's result) */
    int cpy_dense(const ggml_tensor * t, const opnd & s, const char * g, synDataType dt, int lt) {
        const int64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
        params_t pr;
        pr.i(nmem_for(s.nb[0] == 1 ? rows * ((t->ne[0] + 127) / 128) : ggml_nelements(t), s.nb[0] == 1 ? 8 : 256))
          .i(t->ne[0]).i(t->ne[1]).i(t->ne[2]).i(t->ne[3])
          .i(s.off).i(t->ne[0] > 1 ? s.nb[0] : 1).i(s.nb[1]).i(s.nb[2]).i(s.nb[3]);
        add_node(g, { s.lt }, { lt }, pr);
        return lt;
    }

    /* f32 operand of t: f16 / bf16 values are converted into a contiguous f32 temp */
    bool get_f32(const ggml_tensor * t, opnd & o) {
        if (t->type == GGML_TYPE_F32) {
            return get(t, o);
        }
        const char * g = t->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : t->type == GGML_TYPE_BF16 ? "syn_cpy_bf16_f32_f32" : nullptr;
        if (!g) {
            return fail("operand type: %s", t);
        }
        opnd s;
        if (!get(t, s)) {
            return false;
        }
        dense_opnd(t, cpy_dense(t, s, g, syn_type_single), o);
        return true;
    }

    /* MUL_MAT_ID activation b as f32: an f16 / bf16 b converted once into a dense f32 copy (shared by
     * the expert products reading it) */
    std::unordered_map<const ggml_tensor *, int> act_f32;
    bool get_act(const ggml_tensor * b, opnd & o) {
        if (b->type == GGML_TYPE_F32) {
            return get(b, o);
        }
        auto it = act_f32.find(b);
        if (it != act_f32.end()) {
            dense_opnd(b, it->second, o);
            return true;
        }
        if (!get_f32(b, o)) {
            return false;
        }
        act_f32[b] = o.lt;
        return true;
    }

    /* output of an op computed in f32: n's own lt (f32), or an f32 temp end_out32 converts */
    int begin_out32(const ggml_tensor * n) {
        if (n->type == GGML_TYPE_F32) {
            return begin_out(n);
        }
        return new_lt(syn_type_single, { (uint64_t) ggml_nelements(n) });
    }

    bool end_out32(const ggml_tensor * n, int out) {
        if (n->type == GGML_TYPE_F32) {
            return end_out(n, out);
        }
        if (n->view_src) {
            return write_view(n, out, GGML_TYPE_F32);
        }
        const char * g = n->type == GGML_TYPE_F16 ? "syn_cpy_f32_f16_f32" : n->type == GGML_TYPE_BF16 ? "syn_cpy_f32_bf16_f32" : nullptr;
        if (!g) {
            return fail("output type: %s", n);
        }
        const int lt = fresh_out(n);
        const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
        params_t pr;
        pr.i(nmem_for(rows * ((n->ne[0] + 127) / 128), 8)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
          .i(0).i(1).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2]);
        add_node(g, { out }, { lt }, pr);
        return true;
    }

    /* operand s (f32, src's shape) repeated to t's shape: a contiguous temp (or lt dst) */
    bool repeat_to(const ggml_tensor * src, const opnd & s, const ggml_tensor * t, opnd & o, int dst = -1) {
        if (src->ne[0] > 1 && s.nb[0] != 1) {
            return fail("strided repeat source: %s", t);
        }
        const int lt = dst >= 0 ? dst : new_lt(syn_type_single, { (uint64_t) ggml_nelements(t) });
        const int64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
        params_t pr;
        pr.i(nmem_for(rows * ((t->ne[0] + 63) / 64), 8)).i(t->ne[0]).i(t->ne[1]).i(t->ne[2]).i(t->ne[3])
          .i(src->ne[0]).i(src->ne[1]).i(src->ne[2]).i(src->ne[3]).i(s.off).i(s.nb[1]).i(s.nb[2]).i(s.nb[3]);
        add_node("syn_repeat_f32", { s.lt }, { lt }, pr);
        dense_opnd(t, lt, o);
        return true;
    }

    bool lower_node(const ggml_tensor * n);
    void plan_fusions();
    std::unordered_map<int, bool> arena_out;   // arena tensors written by the graph
    bool is_arena_out(const lt_t & t) {
        return arena_out.count((int) (&t - p.lts.data())) > 0;
    }
    std::map<int, std::vector<int>> alias_out;   // outputs in the section of the inputs they overwrite (those inputs)
    bool no_out_alias = false;
    std::map<int, int> kv_window;   // K / V rows read in place (lower_fa_mme) -> the cache input holding them
    bool no_kv_view = false;
    bool kv_view_failed = false;
    /* output `o` may overwrite the persistent inputs `over` (ggml reused their memory) in place when
     * nothing writes them, one node writes o without reading them and all their readers come before
     * it by data flow: the compiler then sees one section and keeps that order */
    bool out_alias_ok(int o, const std::vector<int> & over) {
        static const bool off = getenv("GGML_SYNAPSE_NO_OUT_ALIAS") != nullptr;
        if (off || no_out_alias) {
            return false;
        }
        std::unordered_set<int> a;
        for (int k : over) {
            if (k < 0 || p.lts[k].region == 0 || p.lts[k].region != p.lts[over[0]].region) {
                return false;
            }
            a.insert(k);
        }
        for (size_t k = 0; k < p.lts.size(); k++) {
            if (p.lts[k].base >= 0 && a.count(base_of((int) k))) {
                a.insert((int) k);
            }
        }
        int w = -1;
        std::vector<int> readers;
        std::vector<std::vector<int>> prod(p.lts.size());
        for (size_t n = 0; n < p.nodes.size(); n++) {
            const ln_t & nd = p.nodes[n];
            for (int x : nd.outs) {
                if (a.count(x)) {
                    return false;
                }
                if (nd.guid != "reshape" && base_of(x) == o) {
                    if (w >= 0) {
                        return false;
                    }
                    w = (int) n;
                }
                prod[x].push_back((int) n);
            }
            for (int x : nd.ins) {
                if (a.count(x)) {
                    readers.push_back((int) n);
                    break;
                }
            }
        }
        if (w < 0) {
            return false;
        }
        std::vector<char> anc(p.nodes.size(), 0);
        std::vector<int> stack = { w };
        while (!stack.empty()) {
            const int n = stack.back();
            stack.pop_back();
            for (int x : p.nodes[n].ins) {
                for (int q : prod[x]) {
                    if (!anc[q]) {
                        anc[q] = 1;
                        stack.push_back(q);
                    }
                }
            }
        }
        for (int r : readers) {
            if (!anc[r]) {
                return false;
            }
        }
        return true;
    }
    bool lower_mul_mat(const ggml_tensor * n);
    bool lower_mul_mat_batched(const ggml_tensor * n);
    bool lower_qmm(const ggml_tensor * n, const synapse_wfmt * f);
    bool lower_qmmid(const ggml_tensor * n, const synapse_wfmt * f);
    bool lower_moe_dense(const ggml_tensor * n, const synapse_wfmt * f, bool & done);
    bool lower_moe_paged(const ggml_tensor * n, const synapse_wfmt * f, bool & done);
    struct moe_route { int pe, pairs; };   // the pages' experts, the pairs' page rows
    std::map<std::string, moe_route> moe_routes;
    struct paged_out { int y, pairs; };   // [M PS P] f32 page rows, the pairs' rows
    std::unordered_map<const ggml_tensor *, paged_out> paged_rows;   // MUL_MAT_ID results left as page rows
    struct paged_glu_t { int ya = -1, yb = -1, pe = -1; int64_t ff = 0, psh = 0, P = 0, offa = 0, sa = 0, offb = 0, sb = 0; };
    std::unordered_map<const ggml_tensor *, paged_glu_t> paged_glu;   // GLU node -> its gate_up page rows
    std::unordered_map<const ggml_tensor *, int> paged_x;             // MUL_MAT_ID input -> bf16 [K, PS, P] pages
    bool paged_shape(const ggml_tensor * n, const synapse_wfmt * f, int & psh, int64_t & P);
    const ggml_tensor * paged_glu_next(const ggml_tensor * n, int psh, int64_t P, int & side);
    bool paged_glu_tail(const ggml_tensor * g, const ggml_tensor * ids, int64_t ff, int psh, int64_t P);
    bool paged_same(const ggml_tensor * m, const ggml_tensor * ids, int psh, int64_t P);
    bool lower_moe_glu(const ggml_tensor * g, const paged_glu_t & q);
    bool zero_product(const ggml_tensor * n);
    bool lower_lid(const ggml_tensor * n);
    bool lower_gdn_chunked(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov, const opnd & og,
                           const opnd & ob, const opnd & os);
    bool lower_fa_mme(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov, const opnd & om,
                      synDataType dt, int mode);
    bool x_columns(const ggml_tensor * a, opnd & o, int64_t & sx1);
    // int8 x (pairs + group scales / sums) of activation a for the int8 GEMVs
    std::map<std::tuple<int, int64_t, int64_t, int64_t, bool>, std::pair<int, int>> xpreps;
    bool x_prep(const ggml_tensor * a, int & x8, int & xg, bool compact);
    int  wfmt_lt(const ggml_tensor * w, const synapse_wfmt * f, uint64_t addr, synDataType dt, std::initializer_list<uint64_t> sz);
    int mm_activation(const ggml_tensor * a, synDataType adt, const char * cvt);
    bool lower_flash_attn(const ggml_tensor * n);
    // tiled decode attention (GGML_SYNAPSE_FA_TILE=<W keys>): the program is cut after each tiled FA's query prep; the
    // host launches one shared tile recipe per W-key window of the cache between the segments, so no recipe depends on
    // the KV length
    struct fa_tile_t {
        size_t cut = 0;                  // the segment before this FA's windows ends at node `cut`
        int group = -1;                  // window group (mask, W, cells): its tables
        int qbd = -1;                    // [R, C] block-diagonal queries (made persistent by the split)
        int sinks = -1;                  // [H] f32 sink logits (persistent), -1: none
        int acc_f = -1, ml_f = -1;       // the last window's accumulators (arena lts read by the finalize)
        uint64_t acc_ab[2] = { 0, 0 }, ml_ab[2] = { 0, 0 };   // arena offsets of the ping-pong accumulators
        uint64_t kaddr = 0, vaddr = 0;   // K / V rows of cell 0
        int64_t R = 0, Rv = 0, C = 0, nq = 0, H = 0, hpg = 0, Dv = 0, W = 0, D = 0, Hkv = 0;
        bool pre = false;                // prefill: native SDPA per window (qbd: bf16 Q [D, nq, H]), else MME decode
        synDataType dt = syn_type_fp16;
        float scale = 1.0f;
    };
    // FAs reading one mask over caches of one size share the window tables: per window [first, -, lo1 hi1 lo2 hi2 per query],
    // filled from the mask on the host before each launch (win: start cell and size index per window)
    struct tile_group_t {
        const ggml_tensor * mask = nullptr;
        int64_t W = 0, cells = 0, tmax = 0;
        std::vector<int64_t> sizes;      // window sizes W, 4 W, 16 W (up to the cache size; prefill: W, 4 W)
        uint64_t tab_off = 0, entry = 256;   // table entry bytes per window
        std::shared_ptr<std::vector<int64_t>> win = std::make_shared<std::vector<int64_t>>();
    };
    std::vector<fa_tile_t> fa_tiles;
    std::vector<tile_group_t> tile_groups;
    std::shared_ptr<int> tile_bad = std::make_shared<int>(0);    // a mask the tables cannot describe: run untiled
    bool no_tile = false;
    // GGML_SYNAPSE_FA_TILE: window keys (default 4096), 0: off
    static int64_t fa_tile_w() {
        static const int64_t w = getenv("GGML_SYNAPSE_FA_TILE") ? atoll(getenv("GGML_SYNAPSE_FA_TILE")) : 4096;
        return w > 0 ? (w + 255) / 256 * 256 : 0;
    }
    static bool fa_tile_shape_ok(const ggml_tensor * n);
    static bool fa_tile_sdpa(const ggml_tensor * n) {
        const ggml_tensor * q = n->src[0], * v = n->src[2];
        const int64_t D = q->ne[0];
        return q->ne[1] > 31 && D == v->ne[0] && (D == 64 || D == 128 || D == 256) && !n->src[4];
    }
    int lower_fa_tile(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov);
    bool split_segments(std::vector<program> & segs);
    bool lower_rope(const ggml_tensor * n);
    int rope_table_lt(const ggml_tensor * n);
    int kvlen_table_lt(const ggml_tensor * mk, int64_t nq, int64_t nkv);
    int alibi_table_lt(int64_t H, float max_bias);
    int flat_f32_lt(const ggml_tensor * t);
    bool lower_out_prod(const ggml_tensor * n);
    bool lower_set_acc(const ggml_tensor * n);
    bool lower_cpy_quant(const ggml_tensor * n);
    bool lower_cpy_dq(const ggml_tensor * n);
    bool lower_add1(const ggml_tensor * n);
    bool lower_count_equal(const ggml_tensor * n);
    bool lower_group_norm(const ggml_tensor * n);
    bool lower_diag_tri(const ggml_tensor * n);
    bool lower_solve_tri(const ggml_tensor * n);
    bool lower_im2col(const ggml_tensor * n);
    bool lower_conv(const ggml_tensor * n);
    bool lower_conv_2d_dw(const ggml_tensor * n);
    bool lower_conv_transpose(const ggml_tensor * n);
    bool lower_col2im(const ggml_tensor * n);
    bool lower_upscale(const ggml_tensor * n);
    bool lower_reflect(const ggml_tensor * n);
    bool lower_roll(const ggml_tensor * n);
    bool lower_timestep(const ggml_tensor * n);
    bool lower_fattn_general(const ggml_tensor * n);
    bool lower_ssm_scan_general(const ggml_tensor * n);
    bool lower_wkv_general(const ggml_tensor * n);
    int  rope_table_dev_lt(const ggml_tensor * n);

    bool run();
};

} // namespace

/* strides of operand o (of src) for walking dst's shape: broadcast dims get stride 0 */
static void bcast_strides(const ggml_tensor * src, const opnd & o, const ggml_tensor * dst, int64_t s[4]) {
    for (int d = 0; d < 4; d++) {
        s[d] = (src->ne[d] == 1 && dst->ne[d] != 1) ? 0 : o.nb[d];
    }
}

/* merge dim 1 into dim 0 while every operand (strides s[k], dim 0 contiguous) is contiguous across it:
   [1, 16, T] views become [16, T] */
static void merge_dim0(int64_t ne[4], int64_t * const * s, int ns) {
    while (ne[1] > 1 || ne[2] > 1 || ne[3] > 1) {
        bool ok = true;
        for (int k = 0; k < ns; k++) {
            ok &= ne[1] == 1 || s[k][1] == ne[0];
        }
        if (!ok) {
            return;
        }
        ne[0] *= ne[1];
        for (int d = 1; d < 3; d++) {
            ne[d] = ne[d + 1];
            for (int k = 0; k < ns; k++) {
                s[k][d] = s[k][d + 1];
            }
        }
        ne[3] = 1;
        for (int k = 0; k < ns; k++) {
            s[k][3] = s[k][2] * ne[2];
        }
    }
}

/* n's elements as nr rows of L (leading dims merged while every operand is contiguous across them), operand
   k's row r at r * rs[k] (s[k]: strides of dims 1..3, 0 for broadcast); false: no uniform row stride */
static bool ew_rows(const ggml_tensor * n, const int64_t * const * s, int ns, bool merge, int64_t & L, int64_t & nr,
                    int64_t * rs) {
    L = n->ne[0];
    int d = 1;
    for (; merge && d < 4; d++) {
        bool ok = true;
        for (int k = 0; k < ns; k++) {
            ok &= n->ne[d] == 1 || s[k][d] == L;
        }
        if (!ok) {
            break;
        }
        L *= n->ne[d];
    }
    nr = 1;
    for (int k = 0; k < ns; k++) {
        rs[k] = 0;
    }
    for (; d < 4; d++) {
        if (n->ne[d] == 1) {
            continue;
        }
        for (int k = 0; k < ns; k++) {
            if (nr == 1) {
                rs[k] = s[k][d];
            } else if (s[k][d] != rs[k] * nr) {
                return false;
            }
        }
        nr *= n->ne[d];
    }
    return true;
}

static int unary_code(const ggml_tensor * n) {
    switch (ggml_get_unary_op(n)) {
        case GGML_UNARY_OP_SILU:     return 0;
        case GGML_UNARY_OP_SIGMOID:  return 1;
        case GGML_UNARY_OP_SOFTPLUS: return 2;
        case GGML_UNARY_OP_NEG:      return 3;
        case GGML_UNARY_OP_EXP:      return 4;
        case GGML_UNARY_OP_RELU:     return 5;
        case GGML_UNARY_OP_GELU:     return 6;
        case GGML_UNARY_OP_TANH:     return 7;
        case GGML_UNARY_OP_ABS:      return 11;
        default:                     return -1;
    }
}

/* syn_unaryx op of n (unary or plain op), -1: none */
static int unaryx_code(const ggml_tensor * n) {
    switch (n->op) {
        case GGML_OP_LEAKY_RELU: return 7;
        case GGML_OP_LOG:        return 13;
        case GGML_OP_SIN:        return 14;
        case GGML_OP_COS:        return 15;
        case GGML_OP_CLAMP:      return 16;
        case GGML_OP_UNARY:      break;
        default:                 return -1;
    }
    switch (ggml_get_unary_op(n)) {
        case GGML_UNARY_OP_GELU_ERF:    return 0;
        case GGML_UNARY_OP_GELU_QUICK:  return 1;
        case GGML_UNARY_OP_ELU:         return 2;
        case GGML_UNARY_OP_HARDSIGMOID: return 3;
        case GGML_UNARY_OP_HARDSWISH:   return 4;
        case GGML_UNARY_OP_STEP:        return 5;
        case GGML_UNARY_OP_SGN:         return 6;
        case GGML_UNARY_OP_EXPM1:       return 8;
        case GGML_UNARY_OP_FLOOR:       return 9;
        case GGML_UNARY_OP_CEIL:        return 10;
        case GGML_UNARY_OP_ROUND:       return 11;
        case GGML_UNARY_OP_TRUNC:       return 12;
        case GGML_UNARY_OP_XIELU:       return 17;
        default:                        return -1;
    }
}

/* flash attention: the shared valid-operation predicate (support and lowering). Arbitrary
 * positive Dk and Dv under the addressability limits, the existing GQA / batch / mask / sink
 * contracts, independent K/V types in {F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1}
 * (quantized rows keep their block alignment requirement). The F16 mask contract stays. */
static bool fa_config_ok(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2], * mk = op->src[3], * snk = op->src[4];
    const int64_t Dk = q->ne[0], Dv = v->ne[0];
    auto kv_ok = [](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_Q8_0 ||
               t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 || t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1;
    };
    const int64_t blk = std::max(ggml_blck_size(k->type), ggml_blck_size(v->type));
    // src[5]: K/V row lists (ggml_flash_attn_ext_rows), not lowered; llama falls back to the dense mask (see TODO.md)
    return !op->src[5] && kv_ok(k->type) && kv_ok(v->type) && q->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           k->ne[0] == Dk && Dk > 0 && Dv > 0 && k->ne[2] > 0 && q->ne[2] % k->ne[2] == 0 && v->ne[2] == k->ne[2] &&
           k->ne[3] == q->ne[3] && v->ne[3] == q->ne[3] &&
           // quantized rows keep the block alignment requirement
           ((ggml_is_quantized(k->type) && Dk % blk != 0) || (ggml_is_quantized(v->type) && Dv % blk != 0) ?
                false : true) &&
           (!mk || mk->type == GGML_TYPE_F16) &&
           (!snk || (snk->type == GGML_TYPE_F32 && ggml_nelements(snk) >= q->ne[2] && ggml_is_contiguous(snk)));
}

/* extra restrictions of the bounded kernels: width/multiple and type-pair rules. */
static bool fa_tpc_config_ok(const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0], * k = op->src[1], * v = op->src[2];
    const int64_t Dk = q->ne[0], Dv = v->ne[0];
    // other K/V types (and mixed pairs) are converted to f16 first
    auto f16able = [](ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
               t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1;
    };
    const bool kv_ok = (k->type == v->type && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16 ||
                                               k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_Q8_0)) ||
                       (f16able(k->type) && f16able(v->type));
    return kv_ok && Dk % 8 == 0 && Dv % 8 == 0 && Dk <= 576 && Dv <= 512;
}

/* flash attention lowering mode: 0, 1 or 2 (see lower_fa_mme) */
static int64_t fa_mme_budget() {
    static const int64_t b = (getenv("GGML_SYNAPSE_FA_MME_MB") ? atoll(getenv("GGML_SYNAPSE_FA_MME_MB")) : 1024) << 20;
    return b;
}

static int fa_mme(const ggml_tensor * n) {
    static const int64_t qmin = getenv("GGML_SYNAPSE_FA_MME") ? atoll(getenv("GGML_SYNAPSE_FA_MME")) : 32;
    static const int64_t kvmin = getenv("GGML_SYNAPSE_FA_MME_KV") ? atoll(getenv("GGML_SYNAPSE_FA_MME_KV")) : 2048;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2];
    const int64_t D = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    const int64_t nkv = k->ne[1], Hkv = k->ne[2];
    if (ggml_get_op_params_f32(n, 2) != 0.0f) {
        return 0;
    }
    // query rows per KV head (MLA: one KV head, all heads' queries). MLA-shaped attention (one wide
    // KV head) always takes mode 1: the mode 0 one-query path gave wrong results with the heads
    // split over cards (-sm tensor: 4 / 8 cards; open).
    // GGML_SYNAPSE_FA_MLA_TPC=1: allow mode 0 (CORR-1 retest)
    static const bool mla_tpc = getenv("GGML_SYNAPSE_FA_MLA_TPC") != nullptr;
    const bool mla = !mla_tpc && Hkv == 1 && D > 256;
    if (qmin > 0 && (nq * (H / Hkv) >= qmin || mla) && nkv * H * B * 4 <= fa_mme_budget()) {
        return 1;
    }
    const size_t es = ggml_type_size(k->type);
    if (kvmin > 0 && nkv >= kvmin && B == 1 && k->type == v->type && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) &&
        k->nb[0] == es && k->nb[2] == D * es && k->nb[1] == D * Hkv * es && v->nb[0] == es && v->nb[2] == Dv * es &&
        v->nb[1] == Dv * Hkv * es && (D * Hkv) % 128 == 0 && (Dv * Hkv) % 128 == 0 && nkv * nq * H * 4 <= fa_mme_budget()) {
        return 2;
    }
    return 0;
}

/* syn_glux op of a GLU node, -1: none */
static int glux_code(const ggml_tensor * n) {
    switch (ggml_get_glu_op(n)) {
        case GGML_GLU_OP_GEGLU_ERF:   return 0;
        case GGML_GLU_OP_GEGLU_QUICK: return 1;
        case GGML_GLU_OP_SWIGLU_OAI:  return 2;
        case GGML_GLU_OP_SWIGLU_CLAMP: return 3;
        default:                      return -1;
    }
}

/* SSM_SCAN general-path admission: arbitrary valid nc and nr, no state-size cap. */
static bool ssm_scan_general_ok(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[1], * A = op->src[3], * B = op->src[4], * ids = op->src[6];
    if (op->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids)) {
        return false;
    }
    for (int k = 0; k < 6; k++) {
        if (op->src[k]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    const int64_t ng = B->ne[1], nh = x->ne[1];
    // A: [1, nh] (Mamba-2) or [d_state, nh] (Mamba-1)
    return ng > 0 && nh % ng == 0 && (A->ne[0] == 1 || A->ne[0] == op->src[0]->ne[0]) && A->ne[1] == nh;
}

/* the whole-row pool path's shape predicate (windows of k == stride, no padding) */
static bool p2_or_1d_win(const int32_t * pp, ggml_op op) {
    const bool p2 = op == GGML_OP_POOL_2D;
    // params {op, k0, k1, s0, s1, p0, p1} (2D) / {op, k0, s0, p0} (1D)
    const int k = pp[1], st = p2 ? pp[3] : pp[2];
    return k >= 1 && st == k && (p2 ? pp[5] == 0 && pp[6] == 0 && pp[2] == 1 && pp[4] == 1 : pp[3] == 0);
}

/* OUT_PROD: out[m, n, b2, b3] = sum_p a[m, p, b2/r2, b3/r3] * b[n, p, b2, b3],
   r2 = b.ne2 / a.ne2, r3 = b.ne3 / a.ne3 (division-based batch broadcast). */
bool lowerer::lower_out_prod(const ggml_tensor * n) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    opnd oa, ob;
    if (!get_f32(a, oa) || !get_f32(b, ob)) {
        return false;
    }
    // src1 may be a transpose view (the dflash attention output): its dim-0 stride rides the ABI's
    // sb0. src0's dim-0 walk is the kernel's 64-lane m loop: it must be contiguous.
    if (oa.nb[0] != 1) {
        return fail("out_prod transposed operand: %s", n);
    }
    const int64_t M = a->ne[0], K = a->ne[1], N = b->ne[0];
    if (b->ne[1] != K) {
        return fail("out_prod shape: %s", n);
    }
    if (M < 1 || K < 1 || N < 1 || a->ne[2] < 1 || a->ne[3] < 1 || b->ne[2] < 1 || b->ne[3] < 1) {
        return fail("out_prod empty: %s", n);
    }
    if (b->ne[2] % a->ne[2] != 0 || b->ne[3] % a->ne[3] != 0) {
        return fail("out_prod batch broadcast: %s", n);
    }
    const int64_t r2 = b->ne[2] / a->ne[2], r3 = b->ne[3] / a->ne[3];
    const int64_t B2 = b->ne[2], B3 = b->ne[3];
    // dense no-broadcast planes (the dflash QKV shapes): the native batch GEMM. The operands are
    // re-laid out to the Synapse convention: B storage {N, K, batches} = b transposed, A storage
    // {M, K, batches} = a transposed, result {M, N, batches}.
    if (r2 == 1 && r3 == 1 && oa.nb[1] == 1 && oa.nb[2] == M && oa.nb[3] == M * K &&
        ob.nb[1] == 1 && ob.nb[2] == N && ob.nb[3] == N * K && ggml_nbytes(b) / 4 < (1ll << 30)) {
        const int64_t nb = B2 * B3;
        const int lt = a->ne[2] * a->ne[3] == 1 && b->ne[2] * b->ne[3] == 1 ? 1 : 0;
        GGML_UNUSED(lt);
        // a and b are each [x, K, B2, B3] with contiguous rows: flatten to [x, K, nb] and transpose
        // each batch to the {x, K} storage. untranspose handles a dim-0 stride > 1 walk of rows.
        opnd oa2 = oa, ob2 = ob;
        oa2.ne[0] = M;  oa2.ne[1] = K;  oa2.ne[2] = nb;  oa2.ne[3] = 1;
        oa2.nb[0] = 1;  oa2.nb[1] = 1;  oa2.nb[2] = M * K;  oa2.nb[3] = M * K;
        ob2.ne[0] = N;  ob2.ne[1] = K;  ob2.ne[2] = nb;  ob2.ne[3] = 1;
        ob2.nb[0] = 1;  ob2.nb[1] = 1;  ob2.nb[2] = N * K;  ob2.nb[3] = N * K;
        if (M > 1 && !untranspose(oa2)) {
            return fail("out_prod operand a: %s", n);
        }
        if (N > 1 && !untranspose(ob2)) {
            return fail("out_prod operand b: %s", n);
        }
        // the transposes produce [K, x] planes: reshape to the {x, K, batches} storage
        const int la = oa2.lt, lb = ob2.lt;
        const int la3 = reshape(la, syn_type_single, { (uint64_t) K, (uint64_t) M, (uint64_t) nb });
        const int lb3 = reshape(lb, syn_type_single, { (uint64_t) K, (uint64_t) N, (uint64_t) nb });
        const int y2 = new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N, (uint64_t) nb });
        // the weight-like operand a is stored {K, M} after the untranspose: tb = true
        // (as lower_fa_mme's k stored {D, nkv})
        add_node("batch_gemm", { lb3, la3 }, { y2 }, gemm_params(false, true));
        const int ly = reshape(y2, syn_type_single, { (uint64_t) (M * N * nb) });
        if (n->view_src != nullptr) {
            return write_view(n, ly, GGML_TYPE_F32);
        }
        roots[n].lt = ly;
        return true;
    }
    // broadcast / irregular views: the general kernel
    // the kernel's 64 lanes are consecutive m values: it writes them contiguously, so the scratch is
    // [row = (n, b2, b3), column = m]. The [m, row] ggml layout comes from the copy kernel's strided
    // element-wise path (a native transpose takes no ordering token, so it would race the kernel).
    const int64_t nb_out = N * B2 * B3;
    const int out = begin_out(n);
    // 1-D: the copy kernel's strided path addresses it with flat coordinates
    const int scr = new_lt(syn_type_single, { (uint64_t) (nb_out * M) });
    params_t pr;
    pr.i(nmem_for(N * B2 * B3, 1)).i(M).i(N).i(K).i(B2).i(B3).i(r2).i(r3)
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
      .i(ob.off).i(ob.nb[0]).i(ob.nb[1]).i(ob.nb[2]).i(ob.nb[3])
      .f(1.0f / (float) r2).f(1.0f / (float) r3);
    add_node("syn_outprod_f32", { oa.lt, ob.lt }, { scr }, pr);
    // out flat = row * M + m (dim 0 contiguous, batches outermost): the scratch layout exactly
    params_t pc;
    pc.i(nmem_for((int64_t) (M * nb_out), 512)).i(M * nb_out).i(1).i(1).i(1).i(0).i(1).i(1).i(1).i(1);
    add_node("syn_cpy_f32_f32_f32", { scr }, { out }, pc);
    return end_out(n, out);
}

/* SET / ACC. Op params {nb1, nb2, nb3, offset, inplace} in bytes. */
bool lowerer::lower_set_acc(const ggml_tensor * n) {
    const ggml_tensor * dst = n->src[0];
    const ggml_tensor * src = n->src[1];
    const bool acc = n->op == GGML_OP_ACC;
    const int32_t * pp = (const int32_t *) n->op_params;
    const int64_t nb1 = pp[0], nb2 = pp[1], nb3 = pp[2], offset = pp[3];
    const bool inplace = pp[4] != 0;
    if (acc) {
        if (n->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) {
            return fail("acc types: %s", n);
        }
    } else if (n->type == GGML_TYPE_F32) {
        if (dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) {
            return fail("set types: %s", n);
        }
    } else if (n->type == GGML_TYPE_I32) {
        if (dst->type != GGML_TYPE_I32 || src->type != GGML_TYPE_I32) {
            return fail("set types: %s", n);
        }
    } else {
        return fail("set types: %s", n);
    }
    if (ggml_nelements(src) > ggml_nelements(dst)) {
        return fail("set extent: %s", n);
    }
    const int64_t es = 4;
    for (int d = 0; d < 3; d++) {
        if ((d == 0 ? nb1 : d == 1 ? nb2 : nb3) % es != 0) {
            return fail("set stride alignment: %s", n);
        }
    }
    if (offset % es != 0) {
        return fail("set offset alignment: %s", n);
    }
    // the last written byte must stay inside the destination
    const int64_t last = offset + (src->ne[1] > 0 ? (src->ne[1] - 1) * nb1 : 0) +
                         (src->ne[2] > 0 ? (src->ne[2] - 1) * nb2 : 0) +
                         (src->ne[3] > 0 ? (src->ne[3] - 1) * nb3 : 0) +
                         src->ne[0] * es;
    if (last > ggml_nbytes(dst)) {
        return fail("set extent: %s", n);
    }
    // the destination rows the scatter writes: one logical row of src1 per (i1, i2, i3); their
    // destination byte ranges must not overlap each other
    const int64_t rowb = src->ne[0] * es;
    if (nb1 < rowb && src->ne[1] > 1) {
        return fail("set overlapping rows: %s", n);
    }
    if (nb2 < (src->ne[1] > 0 ? (src->ne[1] - 1) * nb1 + rowb : 0) && src->ne[2] > 1) {
        return fail("set overlapping rows: %s", n);
    }
    if (nb3 < (src->ne[2] > 0 ? (src->ne[2] - 1) * nb2 + (src->ne[1] > 0 ? (src->ne[1] - 1) * nb1 + rowb : 0) : 0) &&
        src->ne[3] > 1) {
        return fail("set overlapping rows: %s", n);
    }
    opnd od, os;
    if (!get(n, od) || !get(src, os)) {
        return false;
    }
    if (od.nb[0] != 1 || os.nb[0] != 1) {
        return fail("set layout: %s", n);
    }
    // src1 aliasing the written region: snapshot it first (a plain read of the pre-mutation value)
    int ssrc = os.lt;
    if (base_of(os.lt) == base_of(od.lt)) {
        const int snap = cpy_dense(src, os, acc ? "syn_cpy_f32_f32_f32" :
                                  n->type == GGML_TYPE_I32 ? "syn_cpy_i32_i32_f32" : "syn_cpy_f32_f32_f32",
                                  n->type == GGML_TYPE_I32 ? syn_type_int32 : syn_type_single);
        if (snap < 0) {
            return false;
        }
        ssrc = snap;
    }
    // out-of-place: the result is a fresh tensor: copy src0 (the whole destination) straight into
    // the result's output lt, then scatter through it. In place: the scatter mutates the root.
    int dt;
    if (!inplace) {
        opnd obase;
        if (!get(dst, obase)) {
            return false;
        }
        dt = fresh_out(n);
        if (cpy_dense(dst, obase, n->type == GGML_TYPE_I32 ? "syn_cpy_i32_i32_f32" : "syn_cpy_f32_f32_f32",
                      n->type == GGML_TYPE_I32 ? syn_type_int32 : syn_type_single, dt) < 0) {
            return false;
        }
    } else {
        dt = od.lt;
    }
    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    params_t pr;
    pr.i(nmem_for(rows * ((src->ne[0] + 63) / 64), 4)).i(src->ne[0]).i(src->ne[1]).i(src->ne[2]).i(src->ne[3])
      .i(os.off).i(os.nb[1]).i(os.nb[2]).i(os.nb[3])
      .i(nb1 / es).i(nb2 / es).i(nb3 / es).i(offset / es)
      .i(inplace ? 1 : 0);
    add_node(acc ? "syn_set_acc_f32" : n->type == GGML_TYPE_I32 ? "syn_set_i32_f32" : "syn_set_f32",
             { dt, ssrc }, { sink() }, pr, true);
    if (n->view_src == nullptr) {
        roots[n].lt = dt;
    }
    return true;
}

/* ADD1: the ADD lowering with a scalar src1 (broadcast over dim 0). */
bool lowerer::lower_add1(const ggml_tensor * n) {
    opnd oa;
    if (!get_f32(n->src[0], oa)) {
        return false;
    }
    opnd ob;
    // src1 is a scalar (or a one-element-per-row tensor): read it as a broadcast operand
    if (!get_f32(n->src[1], ob)) {
        return false;
    }
    const int out = begin_out32(n);
    const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
    params_t pr;
    pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(0).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
      .i(ob.off).i(0).i(0).i(0).i(1).i(0);
    add_node("syn_binary_f32", { oa.lt, ob.lt }, { out }, pr);
    return end_out32(n, out);
}

/* COUNT_EQUAL: I32 same-shape inputs, I64 scalar output (two I32 words). */
bool lowerer::lower_count_equal(const ggml_tensor * n) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    if (n->type != GGML_TYPE_I64 || a->type != GGML_TYPE_I32 || b->type != GGML_TYPE_I32 ||
        !ggml_are_same_shape(a, b)) {
        return fail("count_equal types: %s", n);
    }
    opnd oa, ob;
    if (!get(a, oa) || !get(b, ob)) {
        return false;
    }
    int la = oa.lt, lb = ob.lt;
    if (oa.nb[0] != 1 || oa.nb[1] != a->ne[0] || (a->ne[2] > 1 && oa.nb[2] != a->ne[0] * a->ne[1]) ||
        (a->ne[3] > 1 && oa.nb[3] != a->ne[0] * a->ne[1] * a->ne[2])) {
        la = cpy_dense(a, oa, "syn_cpy_i32_i32_f32", syn_type_int32);
        if (la < 0) {
            return false;
        }
    }
    if (ob.nb[0] != 1 || ob.nb[1] != b->ne[0] || (b->ne[2] > 1 && ob.nb[2] != b->ne[0] * b->ne[1]) ||
        (b->ne[3] > 1 && ob.nb[3] != b->ne[0] * b->ne[1] * b->ne[2])) {
        lb = cpy_dense(b, ob, "syn_cpy_i32_i32_f32", syn_type_int32);
        if (lb < 0) {
            return false;
        }
    }
    const int64_t total = ggml_nelements(a);
    // every member's range below 2^31 elements so the unsigned I32 partials cannot overflow
    int64_t np = (total + (1ll << 30) - 1) / (1ll << 30);
    if (np < 1) {
        np = 1;
    }
    if (np > 256) {
        np = 256;
    }
    const int part = new_lt(syn_type_int32, { (uint64_t) np });
    params_t pr;
    pr.i((int32_t) np).i(total);
    add_node("syn_count_eq_f32", { la, lb }, { part }, pr);
    const int out = begin_out(n);
    params_t pf;
    pf.i(1).i((int32_t) np);
    add_node("syn_count_eq_final_f32", { part }, { out }, pf);
    return end_out(n, out);
}

/* GROUP_NORM: F32, groups from op param 0, epsilon from f32 param 1. */
bool lowerer::lower_group_norm(const ggml_tensor * n) {
    const ggml_tensor * s0 = n->src[0];
    if (n->type != GGML_TYPE_F32 || s0->type != GGML_TYPE_F32) {
        return fail("group_norm types: %s", n);
    }
    opnd oa;
    if (!get(s0, oa)) {
        return false;
    }
    if (s0->ne[0] > 1 && oa.nb[0] != 1) {
        return fail("strided dim 0: %s", n);
    }
    const int groups = ggml_get_op_params_i32(n, 0);
    const float eps = ggml_get_op_params_f32(n, 1);
    if (groups < 1 || s0->ne[2] % groups != 0) {
        return fail("group_norm groups: %s", n);
    }
    const int out = begin_out(n);
    const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
    params_t pr;
    pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
      .i(groups).i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(eps);
    add_node("syn_groupnorm_f32", { oa.lt }, { out }, pr);
    return end_out(n, out);
}

/* DIAG / TRI / DIAG_MASK_INF. */
bool lowerer::lower_diag_tri(const ggml_tensor * n) {
    opnd oa;
    if (!get(n->src[0], oa)) {
        return false;
    }
    if (n->type != GGML_TYPE_F32 || n->src[0]->type != GGML_TYPE_F32) {
        return fail("diag types: %s", n);
    }
    const int out = begin_out(n);
    const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
    if (n->op == GGML_OP_DIAG) {
        if (n->src[0]->ne[1] != 1 || n->ne[0] != n->src[0]->ne[0] || n->ne[1] != n->src[0]->ne[0]) {
            return fail("diag shape: %s", n);
        }
        params_t pr;
        pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(n->ne[0]).i(n->ne[2]).i(n->ne[3])
          .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
        add_node("syn_diag_f32", { oa.lt }, { out }, pr);
        return end_out(n, out);
    }
    if (n->op == GGML_OP_TRI) {
        const int type = ggml_get_op_params_i32(n, 0);
        if (type < 0 || type > 3) {
            return fail("tri type: %s", n);
        }
        params_t pr;
        pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(n->ne[0]).i(n->ne[2]).i(n->ne[3])
          .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(type);
        add_node("syn_tri_f32", { oa.lt }, { out }, pr);
        return end_out(n, out);
    }
    // GGML_OP_DIAG_MASK_INF: n_past from op param 0
    const int n_past = ggml_get_op_params_i32(n, 0);
    if (n_past < 0 || n->src[0]->ne[1] != n->ne[1]) {
        return fail("diag_mask shape: %s", n);
    }
    params_t pr;
    pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(n->ne[0]).i(n->ne[2]).i(n->ne[3])
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(n_past);
    add_node("syn_mask_f32", { oa.lt }, { out }, pr);
    return end_out(n, out);
}

/* SOLVE_TRI: contiguous F32 A {N, N, B2, B3}, B {K, N, B2, B3}, lower / left / non-unit. */
bool lowerer::lower_solve_tri(const ggml_tensor * n) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    if (n->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return fail("solve_tri types: %s", n);
    }
    if (a->ne[0] != a->ne[1] || a->ne[1] != b->ne[1] || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3]) {
        return fail("solve_tri shape: %s", n);
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return fail("solve_tri layout: %s", n);
    }
    opnd oa, ob;
    if (!get(a, oa) || !get(b, ob)) {
        return false;
    }
    const int64_t N = a->ne[0], K = b->ne[0], B2 = a->ne[2], B3 = a->ne[3];
    if (N > 256) {
        return fail("solve_tri size: %s", n);
    }
    // one pass of forward substitution per (batch, 64 RHS columns), all N rows
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(B2 * B3 * ((K + 63) / 64), 1)).i(N).i(K).i(B2).i(B3).i(oa.off).i(ob.off);
    add_node("syn_solve_tri_f32", { oa.lt, ob.lt }, { out }, pr);
    return end_out(n, out);
}

/* IM2COL / IM2COL_3D: the kernel tensor supplies geometry only. F16 results are packed in f32 and converted. */
bool lowerer::lower_im2col(const ggml_tensor * n) {
    const ggml_tensor * knl = n->src[0];
    const ggml_tensor * img = n->src[1];
    if ((n->type != GGML_TYPE_F32 && n->type != GGML_TYPE_F16) || img->type != GGML_TYPE_F32) {
        return fail("im2col types: %s", n);
    }
    const int32_t * pp = (const int32_t *) n->op_params;
    opnd oi;
    if (!get(img, oi)) {
        return false;
    }
    if (img->ne[0] > 1 && oi.nb[0] != 1) {
        return fail("strided dim 0: %s", n);
    }
    const int out = begin_out32(n);
    if (n->op == GGML_OP_IM2COL_3D) {
        const int64_t IC = pp[9];
        const int64_t N = img->ne[3] / IC;
        const int64_t ID = img->ne[2], IH = img->ne[1], IW = img->ne[0];
        const int64_t KD = knl->ne[2], KH = knl->ne[1], KW = knl->ne[0];
        const int64_t OD = n->ne[3] / N, OH = n->ne[2], OW = n->ne[1];
        params_t pr;
        pr.i(nmem_for((KW * KH * KD * IC) * (OW * OH * OD * N), 1))
          .i(pp[0]).i(pp[1]).i(pp[2]).i(pp[3]).i(pp[4]).i(pp[5]).i(pp[6]).i(pp[7]).i(pp[8])
          .i(KW).i(KH).i(KD).i(IC)
          .i(IW).i(IH).i(ID).i(N).i(OW).i(OH).i(OD)
          .i(oi.off).i(oi.nb[1]).i(oi.nb[2]).i(oi.nb[3]);
        add_node("syn_im2col3d_f32", { oi.lt }, { out }, pr);
        return end_out32(n, out);
    }
    const bool is_2D = pp[6] != 0;
    // 1D: image [IW, IC, N]; 2D: image [IW, IH, IC, N]
    const int64_t IC = is_2D ? img->ne[2] : img->ne[1];
    const int64_t KH = is_2D ? knl->ne[1] : 1;
    const int64_t KW = knl->ne[0];
    const int64_t IW = img->ne[0], IH = is_2D ? img->ne[1] : 1;
    const int64_t OW = n->ne[1], OH = is_2D ? n->ne[2] : 1;
    const int64_t N = is_2D ? img->ne[3] : img->ne[2];
    // the kernel reads (x, y, c, n) at off + x + sb1 y + sb2 c + sb3 n
    const int64_t sb1 = is_2D ? oi.nb[1] : 0, sb2 = is_2D ? oi.nb[2] : oi.nb[1], sb3 = is_2D ? oi.nb[3] : oi.nb[2];
    params_t pr;
    pr.i(nmem_for((KW * KH * IC) * (OW * OH * N), 1))
      .i(pp[0]).i(pp[1]).i(pp[2]).i(pp[3]).i(pp[4]).i(pp[5]).i(is_2D ? 1 : 0)
      .i(KW).i(KH).i(IC)
      .i(IW).i(IH).i(OW).i(OH).i(N)
      .i(oi.off).i(sb1).i(sb2).i(sb3);
    add_node("syn_im2col_f32", { oi.lt }, { out }, pr);
    return end_out32(n, out);
}

/* CONV_2D / CONV_3D: im2col packing + native F32 GEMM (packed taps up to 512 MiB), else the DIRECT kernel */
bool lowerer::lower_conv(const ggml_tensor * n) {
    const ggml_tensor * knl = n->src[0];
    const ggml_tensor * img = n->src[1];
    if (n->type != GGML_TYPE_F32 || img->type != GGML_TYPE_F32) {
        return fail("conv types: %s", n);
    }
    if (knl->type != GGML_TYPE_F32 && knl->type != GGML_TYPE_F16 && knl->type != GGML_TYPE_BF16) {
        return fail("conv kernel type: %s", n);
    }
    const bool is3d = n->op == GGML_OP_CONV_3D;
    const int32_t * pp = (const int32_t *) n->op_params;
    opnd oi, ok;
    if (!get(img, oi) || !get_f32(knl, ok)) {
        return false;
    }
    if ((img->ne[0] > 1 && oi.nb[0] != 1) || oi.nb[1] != img->ne[0] || oi.nb[2] != img->ne[0] * img->ne[1] ||
        oi.nb[3] != img->ne[0] * img->ne[1] * img->ne[2]) {
        // a permuted image (CWHN) or any other strided view: copied dense first
        const int dense = cpy_dense(img, oi, "syn_cpy_f32_f32_f32", syn_type_single);
        dense_opnd(img, dense, oi);
    }
    int64_t KW, KH, KD, IC, OC, IW, IH, ID, N, OW, OH, OD, s0, s1, s2, p0, p1, p2, d0, d1, d2;
    if (is3d) {
        s0 = pp[0]; s1 = pp[1]; s2 = pp[2]; p0 = pp[3]; p1 = pp[4]; p2 = pp[5];
        d0 = pp[6]; d1 = pp[7]; d2 = pp[8];
        IC = pp[9]; N = pp[10]; OC = pp[11];
        KW = knl->ne[0]; KH = knl->ne[1]; KD = knl->ne[2];
        IW = img->ne[0]; IH = img->ne[1]; ID = img->ne[2];
        OW = n->ne[0]; OH = n->ne[1]; OD = n->ne[2];
    } else {
        s0 = pp[0]; s1 = pp[1]; s2 = 1; p0 = pp[2]; p1 = pp[3]; p2 = 0;
        d0 = pp[4]; d1 = pp[5]; d2 = 1;
        IC = knl->ne[2]; OC = knl->ne[3];
        KW = knl->ne[0]; KH = knl->ne[1];
        IW = img->ne[0]; IH = img->ne[1];
        ID = 1; KD = 1;
        OW = n->ne[0]; OH = n->ne[1];
        N = img->ne[3]; OD = 1;
    }
    const int64_t R = KW * KH * KD * IC;      // reduction width: taps (kw fastest, then kh, kd, ic)
    const int64_t P = OW * OH * OD;           // output positions per batch
    // im2col + GEMM while the packed taps fit 512 MiB, else the DIRECT kernel
    if (R * P * N <= (128ll << 20) && R * P * N < (1ll << 31) && OC * P * N < (1ll << 31)) {
        // packed [R, P N]: column p + P n (the IM2COL / IM2COL_3D kernels' layout)
        const int packed = new_lt(syn_type_single, { (uint64_t) (R * P * N) });
        params_t pr;
        if (is3d) {
            pr.i(nmem_for(R * P * N, 1))
              .i(s0).i(s1).i(s2).i(p0).i(p1).i(p2).i(d0).i(d1).i(d2)
              .i(KW).i(KH).i(KD).i(IC)
              .i(IW).i(IH).i(ID).i(N).i(OW).i(OH).i(OD)
              .i(oi.off).i(oi.nb[1]).i(oi.nb[2]).i(oi.nb[3]);
            add_node("syn_im2col3d_f32", { oi.lt }, { packed }, pr);
        } else {
            pr.i(nmem_for(R * P * N, 1))
              .i(s0).i(s1).i(p0).i(p1).i(d0).i(d1).i(1)
              .i(KW).i(KH).i(IC)
              .i(IW).i(IH).i(OW).i(OH).i(N)
              .i(oi.off).i(oi.nb[1]).i(oi.nb[2]).i(oi.nb[3]);
            add_node("syn_im2col_f32", { oi.lt }, { packed }, pr);
        }
        // the kernel as dense [R, OC]: element r + R oc (ggml's [KW, KH, (KD,) IC, OC] / [KW, KH, KD, IC OC] order)
        int kd = ok.lt;
        const lt_t & kb = p.lts[base_of(ok.lt)];
        const bool dense = ok.off == 0 && kb.dims == 1 && kb.sz[0] == (uint64_t) (R * OC) && (KW <= 1 || ok.nb[0] == 1) &&
                           ok.nb[1] == KW && ok.nb[2] == KW * KH && ok.nb[3] == KW * KH * (is3d ? KD : IC);
        if (!dense) {
            kd = cpy_dense(knl, ok, "syn_cpy_f32_f32_f32", syn_type_single);
        }
        // y [OC, P N] = W^T applied to every packed column
        const int y = new_lt(syn_type_single, { (uint64_t) OC, (uint64_t) (P * N) });
        add_node("gemm", { reshape(packed, syn_type_single, { (uint64_t) R, (uint64_t) (P * N) }),
                           reshape(kd, syn_type_single, { (uint64_t) R, (uint64_t) OC }) },
                 { y }, gemm_params(false, true));
        // out [P, OC, N] (2D: [OW, OH, OC, N]; 3D: [OW, OH, OD, OC N]): per batch [OC, P] transposed
        const int64_t tot = OC * P * N;
        const int lt2 = OC == 1 ? reshape(y, syn_type_single, { (uint64_t) tot }) :
                        transpose2d(reshape(y, syn_type_single, { (uint64_t) tot }), OC, P, N, syn_type_single);
        const int out = begin_out(n);
        params_t pc;
        pc.i(nmem_for((tot + 127) / 128, 8)).i(tot).i(1).i(1).i(1).i(0).i(1).i(tot).i(tot).i(tot);
        add_node("syn_cpy_f32_f32_f32", { lt2 }, { out }, pc);
        return end_out(n, out);
    }
    // one reduction row exceeds the budget: DIRECT streaming
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(OW * OH * (is3d ? OD : 1) * OC * N, 1))
      .i(s0).i(s1).i(s2).i(p0).i(p1).i(p2).i(d0).i(d1).i(d2)
      .i(KW).i(KH).i(KD).i(IC).i(OC)
      .i(IW).i(IH).i(ID).i(N).i(OW).i(OH).i(OD)
      .i(oi.off).i(ok.off);
    add_node(is3d ? "syn_im2col_direct3d_f32" : "syn_im2col_direct_f32", { oi.lt, ok.lt }, { out }, pr);
    return end_out(n, out);
}

/* CONV_2D_DW: depthwise, WHCN and CWHN strides. */
bool lowerer::lower_conv_2d_dw(const ggml_tensor * n) {
    const ggml_tensor * knl = n->src[0];
    const ggml_tensor * img = n->src[1];
    if (n->type != GGML_TYPE_F32 || img->type != GGML_TYPE_F32) {
        return fail("conv_2d_dw types: %s", n);
    }
    if (knl->type != GGML_TYPE_F32 && knl->type != GGML_TYPE_F16) {
        return fail("conv_2d_dw kernel type: %s", n);
    }
    opnd oi, ok;
    if (!get(img, oi) || !get(knl, ok)) {
        return false;
    }
    // the constructor's CWHN layout: nb0 = C * es, nb2 = es (contiguous channels); the result has the same order
    const bool cwhn = !ggml_is_contiguous(img) && ggml_is_contiguous_channels(img);
    if (cwhn && ok.nb[3] != 1) {
        return fail("conv_2d_dw CWHN kernel without contiguous channels: %s", n);
    }
    const int32_t * pp = (const int32_t *) n->op_params;
    const int64_t C = img->ne[2], N = img->ne[3];
    const int64_t SW = img->ne[0], SH = img->ne[1], DW = n->ne[0], DH = n->ne[1];
    const int64_t KW = knl->ne[0], KH = knl->ne[1];
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(cwhn ? DW * DH * N * ((C + 63) / 64) : ggml_nelements(n), cwhn ? 4 : 64))
      .i(C).i(N).i(SW).i(SH).i(DW).i(DH).i(KW).i(KH)
      .i(pp[0]).i(pp[1]).i(pp[2]).i(pp[3]).i(pp[4]).i(pp[5])
      .i(oi.off).i(oi.nb[0]).i(oi.nb[1]).i(oi.nb[2]).i(oi.nb[3])
      .i(ok.off).i(ok.nb[0]).i(ok.nb[1]).i(ok.nb[3]);
    add_node(cwhn ? (knl->type == GGML_TYPE_F16 ? "syn_conv2d_dw_cwhn_f16_f32" : "syn_conv2d_dw_cwhn_f32") :
                    (knl->type == GGML_TYPE_F16 ? "syn_conv2d_dw_f16_f32" : "syn_conv2d_dw_f32"),
             { ok.lt, oi.lt }, { out }, pr);
    return end_out(n, out);
}

/* CONV_TRANSPOSE_1D / 2D: gather per output. */
bool lowerer::lower_conv_transpose(const ggml_tensor * n) {
    const ggml_tensor * knl = n->src[0];
    const ggml_tensor * img = n->src[1];
    if (n->type != GGML_TYPE_F32 || img->type != GGML_TYPE_F32) {
        return fail("conv_transpose types: %s", n);
    }
    if (knl->type != GGML_TYPE_F32 && knl->type != GGML_TYPE_F16) {
        return fail("conv_transpose kernel type: %s", n);
    }
    opnd oi, ok;
    if (!get(img, oi) || !get(knl, ok)) {
        return false;
    }
    if (img->ne[0] > 1 && oi.nb[0] != 1) {
        dense_opnd(img, cpy_dense(img, oi, "syn_cpy_f32_f32_f32", syn_type_single), oi);
    }
    if (knl->ne[0] > 1 && ok.nb[0] != 1) {
        if (knl->type == GGML_TYPE_F16) {
            dense_opnd(knl, cpy_dense(knl, ok, "syn_cpy_f16_f16_f32", syn_type_fp16), ok);
        } else {
            dense_opnd(knl, cpy_dense(knl, ok, "syn_cpy_f32_f32_f32", syn_type_single), ok);
        }
    }
    const bool nd2 = n->op == GGML_OP_CONV_TRANSPOSE_2D;
    const int s = ggml_get_op_params_i32(n, 0);
    const int out = begin_out(n);
    params_t pr;
    if (nd2) {
        const int64_t KW = knl->ne[0], KH = knl->ne[1], OC = knl->ne[2], IC = knl->ne[3];
        const int64_t IW = img->ne[0], IH = img->ne[1], B = img->ne[3];
        pr.i(nmem_for(ggml_nelements(n), 64)).i(s)
          .i(KW).i(KH).i(OC).i(IC).i(IW).i(IH).i(B).i(n->ne[0]).i(n->ne[1])
          .i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ok.nb[3])
          .i(oi.off).i(oi.nb[1]).i(oi.nb[2]).i(oi.nb[3]);
        add_node(knl->type == GGML_TYPE_F16 ? "syn_conv_transpose2d_f16_f32" : "syn_conv_transpose2d_f32",
                 { ok.lt, oi.lt }, { out }, pr);
    } else {
        // ggml_conv_transpose_1d: p0 == 0, d0 == 1
        const int64_t K = knl->ne[0], OC = knl->ne[1], IC = knl->ne[2];
        const int64_t T = img->ne[0];
        pr.i(nmem_for(ggml_nelements(n), 64)).i(s)
          .i(K).i(OC).i(IC).i(T).i(n->ne[0])
          .i(ok.off).i(ok.nb[1]).i(ok.nb[2])
          .i(oi.off).i(oi.nb[1]);
        add_node(knl->type == GGML_TYPE_F16 ? "syn_conv_transpose_f16_f32" : "syn_conv_transpose_f32",
                 { ok.lt, oi.lt }, { out }, pr);
    }
    return end_out(n, out);
}

/* COL2IM_1D: columns [K*OC, Tin] -> signal [(Tin-1)*s + K - 2p, OC], F32/F16/BF16. */
bool lowerer::lower_col2im(const ggml_tensor * n) {
    const ggml_tensor * cols = n->src[0];
    if (n->type != cols->type) {
        return fail("col2im types: %s", n);
    }
    if (n->type != GGML_TYPE_F32 && n->type != GGML_TYPE_F16 && n->type != GGML_TYPE_BF16) {
        return fail("col2im types: %s", n);
    }
    if (!ggml_is_contiguous(cols)) {
        return fail("col2im layout: %s", n);
    }
    opnd oc;
    if (!get(cols, oc)) {
        return false;
    }
    const int32_t * pp = (const int32_t *) n->op_params;
    const int s = pp[0];
    const int64_t OC = pp[1];
    const int p = pp[2];
    const int64_t K = cols->ne[0] / OC;
    const int64_t Tin = cols->ne[1];
    const int out = begin_out(n);
    const char * g = n->type == GGML_TYPE_F32 ? "syn_col2im_f32" :
                     n->type == GGML_TYPE_F16 ? "syn_col2im_f16_f32" : "syn_col2im_bf16_f32";
    params_t pr;
    pr.i(nmem_for(n->ne[0] * OC, 1)).i(s).i(OC).i(p).i(K).i(Tin).i(n->ne[0])
      .i(oc.off);
    add_node(g, { oc.lt }, { out }, pr);
    return end_out(n, out);
}

/* CPY F32 -> quantized formats: the CPY variant of the setrows kernels. As the CPU, the source rows (its own
 * shape) are quantized one after the other into the contiguous destination, written in place. */
bool lowerer::lower_cpy_quant(const ggml_tensor * n) {
    const ggml_tensor * src = n->src[0];
    if (n->view_src == nullptr || !ggml_is_contiguous(n) || src->ne[0] % ggml_blck_size(n->type) != 0) {
        return fail("cpy quant layout: %s", n);
    }
    const char * g = n->type == GGML_TYPE_Q8_0 ? "syn_cpy_f32_q8_0_f32" :
                     n->type == GGML_TYPE_Q4_0 ? "syn_cpy_f32_q4_0_f32" :
                     n->type == GGML_TYPE_Q4_1 ? "syn_cpy_f32_q4_1_f32" :
                     n->type == GGML_TYPE_Q5_0 ? "syn_cpy_f32_q5_0_f32" :
                     n->type == GGML_TYPE_Q5_1 ? "syn_cpy_f32_q5_1_f32" :
                     n->type == GGML_TYPE_IQ4_NL ? "syn_cpy_f32_iq4_nl_f32" : nullptr;
    if (!g) {
        return fail("cpy quant type: %s", n);
    }
    opnd os, od;
    if (!get_f32(src, os) || !get(n, od)) {
        return false;
    }
    if (os.nb[0] != 1) {
        dense_opnd(src, cpy_dense(src, os, "syn_cpy_f32_f32_f32", syn_type_single), os);
    }
    // od.off: the destination's byte offset (raw byte lt)
    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    params_t pr;
    pr.i(nmem_for(rows * ((src->ne[0] + 63) / 64), 4)).i(src->ne[0]).i(src->ne[1]).i(src->ne[2]).i(src->ne[3])
      .i(od.off).i(os.off).i(os.nb[1]).i(os.nb[2]).i(os.nb[3]);
    add_node(g, { od.lt, os.lt }, { sink() }, pr, true);
    return true;
}

/* CPY quantized -> F32: the dqrows32 family (F32 output, not the f16 one). Dense dequant, then
 * the existing view writer for a noncontiguous destination. */
bool lowerer::lower_cpy_dq(const ggml_tensor * n) {
    const ggml_tensor * src = n->src[0];
    opnd os;
    if (!get(src, os)) {
        return false;
    }
    if (src->ne[0] % ggml_blck_size(src->type) != 0) {
        return fail("cpy dq row: %s", n);
    }
    // dequantized in the source's shape: the flat element order the destination takes
    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    const int lt = new_lt(syn_type_single, { (uint64_t) ggml_nelements(n) });
    params_t pr;
    pr.i(nmem_for(rows * ((src->ne[0] + 63) / 64), 8)).i(src->ne[0]).i(src->ne[1]).i(src->ne[2]).i(src->ne[3])
      .i(os.off).i(os.nb[1]).i(os.nb[2]).i(os.nb[3]);
    add_node(qt_kname("syn_dqrows32_", src->type).c_str(), { os.lt }, { lt }, pr);
    if (n->view_src != nullptr) {
        return write_view(n, lt, GGML_TYPE_F32);
    }
    roots[n].lt = lt;
    return true;
}

/* UPSCALE: modes NEAREST / BILINEAR / BICUBIC, flags ALIGN_CORNERS / ANTIALIAS. */
bool lowerer::lower_upscale(const ggml_tensor * n) {
    const ggml_tensor * s0 = n->src[0];
    if (n->type != GGML_TYPE_F32 || s0->type != GGML_TYPE_F32) {
        return fail("upscale types: %s", n);
    }
    opnd oa;
    if (!get(s0, oa)) {
        return false;
    }
    if (s0->ne[0] > 1 && oa.nb[0] != 1) {
        // a transposed source: copied dense first
        dense_opnd(s0, cpy_dense(s0, oa, "syn_cpy_f32_f32_f32", syn_type_single), oa);
    }
    const int32_t mode = ggml_get_op_params_i32(n, 0);
    const int m = mode & 0xFF;
    const bool aa = (mode & 512) != 0;
    if (aa && m != 1) {
        return fail("upscale antialias: %s", n);
    }
    const char * g = m == 0 ? "syn_upscale_nearest_f32" : m == 1 ? (aa ? "syn_upscale_aa_f32" : "syn_upscale_bilinear_f32") :
                     "syn_upscale_bicubic_f32";
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(n->ne[1] * n->ne[2] * n->ne[3], 1))
      .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
      .i(s0->ne[0]).i(s0->ne[1])
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
      .i(mode).i(s0->ne[2]).i(s0->ne[3]);
    add_node(g, { oa.lt }, { out }, pr);
    return end_out(n, out);
}

/* PAD_REFLECT_1D and ROLL. */
bool lowerer::lower_reflect(const ggml_tensor * n) {
    const ggml_tensor * s0 = n->src[0];
    if (n->type != GGML_TYPE_F32 || s0->type != GGML_TYPE_F32) {
        return fail("pad_reflect types: %s", n);
    }
    opnd oa;
    if (!get(s0, oa)) {
        return false;
    }
    if (s0->ne[0] > 1 && oa.nb[0] != 1) {
        return fail("strided dim 0: %s", n);
    }
    const int32_t * pp = (const int32_t *) n->op_params;
    const int p0 = pp[0], p1 = pp[1];
    if (p0 < 0 || p1 < 0 || p0 >= s0->ne[0] || p1 >= s0->ne[0]) {
        return fail("pad_reflect pads: %s", n);
    }
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(n->ne[1] * n->ne[2] * n->ne[3], 1))
      .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
      .i(p0).i(p1);
    add_node("syn_reflect_f32", { oa.lt }, { out }, pr);
    return end_out(n, out);
}

bool lowerer::lower_roll(const ggml_tensor * n) {
    const ggml_tensor * s0 = n->src[0];
    if (n->type != GGML_TYPE_F32 || s0->type != GGML_TYPE_F32) {
        return fail("roll types: %s", n);
    }
    opnd oa;
    if (!get(s0, oa)) {
        return false;
    }
    const int32_t * pp = (const int32_t *) n->op_params;
    for (int d = 0; d < 4; d++) {
        const int64_t sh = pp[d];
        if (sh >= s0->ne[d] || -sh >= s0->ne[d]) {
            return fail("roll shift: %s", n);
        }
    }
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(ggml_nelements(n) / 256, 1))
      .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
      .i(oa.off).i(oa.nb[0]).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
      .i(pp[0]).i(pp[1]).i(pp[2]).i(pp[3]);
    add_node("syn_roll_f32", { oa.lt }, { out }, pr);
    return end_out(n, out);
}

/* TIMESTEP_EMBEDDING. */
bool lowerer::lower_timestep(const ggml_tensor * n) {
    const ggml_tensor * s0 = n->src[0];
    if (n->type != GGML_TYPE_F32 || s0->type != GGML_TYPE_F32) {
        return fail("timestep types: %s", n);
    }
    opnd oa;
    if (!get(s0, oa)) {
        return false;
    }
    const int dim = ggml_get_op_params_i32(n, 0);
    const int max_period = ggml_get_op_params_i32(n, 1);
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(s0->ne[0], 1)).i(dim).i(max_period).i(s0->ne[0]).i(oa.off);
    add_node("syn_timestep_f32", { oa.lt }, { out }, pr);
    return end_out(n, out);
}

/* FLASH_ATTN_EXT general device path: arbitrary Dk/Dv, mixed K/V floating types, quantized rows. */
bool lowerer::lower_fattn_general(const ggml_tensor * n) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = mask_of(n), * snk = n->src[4];
    const float scale = ggml_get_op_params_f32(n, 0);
    const float max_bias = ggml_get_op_params_f32(n, 1);
    const float softcap = ggml_get_op_params_f32(n, 2);
    const int64_t Dk = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    const int64_t nkv = k->ne[1], Hkv = k->ne[2];
    if (q->type != GGML_TYPE_F32 || n->type != GGML_TYPE_F32) {
        return fail("flash_attn general types: %s", n);
    }
    opnd oq, ok, ov, om;
    // the kernel reads Q, K and V dense (offsets only): a copy of a strided f32 view
    auto dense_f32 = [&](const ggml_tensor * t, opnd & o) -> bool {
        if (!get_f32(t, o)) {
            return false;
        }
        int64_t st = 1;
        bool dense = true;
        for (int d = 0; d < 4; d++) {
            dense = dense && (t->ne[d] <= 1 || o.nb[d] == st);
            st *= t->ne[d];
        }
        if (!dense) {
            dense_opnd(t, cpy_dense(t, o, "syn_cpy_f32_f32_f32", syn_type_single), o);
        }
        return true;
    };
    if (!dense_f32(q, oq)) {
        return false;
    }
    // quantized K / V rows: the F32-output dqrows family (never the F16-output syn_dqrows_)
    auto dens_f32 = [&](const ggml_tensor * t, opnd & o) -> bool {
        if (ggml_is_quantized(t->type)) {
            if (t->ne[0] % ggml_blck_size(t->type) != 0) {
                return fail("flash_attn general quant row: %s", n);
            }
            opnd ot;
            if (!get(t, ot)) {
                return false;
            }
            const int64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
            const int lt = new_lt(syn_type_single, { (uint64_t) ggml_nelements(t) });
            params_t pr;
            pr.i(nmem_for(rows * ((t->ne[0] + 63) / 64), 8)).i(t->ne[0]).i(t->ne[1]).i(t->ne[2]).i(t->ne[3])
              .i(ot.off).i(ot.nb[1]).i(ot.nb[2]).i(ot.nb[3]);
            add_node(qt_kname("syn_dqrows32_", t->type).c_str(), { ot.lt }, { lt }, pr);
            dense_opnd(t, lt, o);
            return true;
        }
        return get_f32(t, o);
    };
    if (!dens_f32(k, ok) || !dens_f32(v, ov)) {
        return false;
    }
    if (mk) {
        // the mask contract is F16 (the shared predicate keeps it); read as f16 rows
        if (mk->type != GGML_TYPE_F16) {
            return fail("flash_attn general mask type: %s", n);
        }
        if (!get(mk, om)) {
            return false;
        }
        if (om.nb[0] != 1) {
            // densify the mask rows (dim 0 must be contiguous for the scalar reads)
            const int lt = new_lt(syn_type_fp16, { (uint64_t) ggml_nelements(mk) });
            params_t pr;
            pr.i(nmem_for(mk->ne[1] * mk->ne[2] * mk->ne[3] * ((mk->ne[0] + 63) / 64), 8))
              .i(mk->ne[0]).i(mk->ne[1]).i(mk->ne[2]).i(mk->ne[3])
              .i(om.off).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]);
            add_node("syn_cpy_f16_f16_f32", { om.lt }, { lt }, pr);
            om.lt = lt;
            om.off = 0;
            om.nb[0] = 1;
            om.nb[1] = mk->ne[0];
            om.nb[2] = mk->ne[0] * mk->ne[1];
            om.nb[3] = mk->ne[0] * mk->ne[1] * mk->ne[2];
        }
    }
    // the kernel reads q / k / v dense [D, n, H, B] from their offsets: views (e.g. K / V windows of a cache) copied
    auto densify = [&](const ggml_tensor * t, opnd & o) {
        if ((t->ne[0] <= 1 || o.nb[0] == 1) && o.nb[1] == t->ne[0] && o.nb[2] == t->ne[0] * t->ne[1] &&
            o.nb[3] == t->ne[0] * t->ne[1] * t->ne[2]) {
            return;
        }
        dense_opnd(t, cpy_dense(t, o, "syn_cpy_f32_f32_f32", syn_type_single), o);
    };
    densify(q, oq);
    densify(k, ok);
    densify(v, ov);
    const int lsl = max_bias > 0.0f ? alibi_table_lt(H, max_bias) : oq.lt;
    const int lsk = snk ? flat_f32_lt(snk) : oq.lt;
    if (lsk < 0) {
        return false;
    }
    // separate Dk and Dv scalars; sc = softcap != 0 ? scale / softcap : scale
    float sc = scale;
    if (softcap != 0.0f) {
        sc = scale / softcap;
    }
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(B * nq * H * ((Dv + 63) / 64), 1))
      .i(Dk).i(Dv).i(nq).i(H).i(nkv).i(B).i(H / Hkv)
      .i(oq.off).i(ok.off).i(ov.off).i(mk ? om.off : 0)
      .i(mk ? om.nb[1] : 0).i(mk ? om.nb[2] : 0).i(mk ? om.nb[3] : 0)
      .i(mk ? mk->ne[2] : 1).i(mk ? mk->ne[3] : 1)
      .f(sc).f(softcap)
      .i((mk != nullptr) | ((max_bias > 0.0f) << 1) | ((snk != nullptr) << 2));
    add_node("syn_fattn_general_f32", { oq.lt, ok.lt, ov.lt, mk ? om.lt : oq.lt, lsl, lsk }, { out }, pr);
    return end_out(n, out);
}

/* SSM_SCAN general path: one STEP node runs every token per (sequence, row, state tile); one REDUCE node sums the
   tile partials into the outputs and copies the K state snapshots */
bool lowerer::lower_ssm_scan_general(const ggml_tensor * n) {
    const ggml_tensor * st = n->src[0], * x = n->src[1], * dt = n->src[2], * A = n->src[3];
    const ggml_tensor * B = n->src[4], * C = n->src[5], * ids = n->src[6];
    opnd os, ox, od, oa, ob, oc, oi;
    if (!get(st, os) || !get(x, ox) || !get(dt, od) || !get(A, oa) || !get(B, ob) || !get(C, oc) ||
        !get(ids, oi)) {
        return false;
    }
    const int64_t nc = st->ne[0], nr = x->ne[0], nh = x->ne[1], nt = x->ne[2], ns = x->ne[3], ng = B->ne[1];
    const int64_t K = ggml_get_op_params_i32(n, 0);
    const bool a1 = A->ne[0] != 1;
    // the kernel reads the state rows, dt and A dense, x / B / C rows contiguous (strides of dims 2 / 3 as params)
    auto dense = [&](const ggml_tensor * t, opnd & o) {
        if ((t->ne[0] <= 1 || o.nb[0] == 1) && o.nb[1] == t->ne[0] && o.nb[2] == t->ne[0] * t->ne[1] &&
            (t->ne[3] <= 1 || o.nb[3] == t->ne[0] * t->ne[1] * t->ne[2])) {
            return;
        }
        dense_opnd(t, cpy_dense(t, o, "syn_cpy_f32_f32_f32", syn_type_single), o);
    };
    dense(dt, od);
    dense(A, oa);
    if ((os.nb[0] != 1 && nc > 1) || os.nb[1] != nc || os.nb[2] != nc * nr || ox.nb[0] != 1 || ox.nb[1] != nr ||
        ob.nb[0] != 1 || ob.nb[1] != nc || oc.nb[0] != 1 || oc.nb[1] != nc) {
        return fail("ssm_scan general layout: %s", n);
    }
    const int64_t R = nr * nh, ntile = (nc + 63) / 64;
    const int part = new_lt(syn_type_single, { (uint64_t) (ns * nt * R * ntile) });
    const int sts = new_lt(syn_type_single, { (uint64_t) (K * ns * R * nc) });
    params_t pr;
    pr.i(nmem_for(ns * R * ntile, 4)).i(nc).i(nr).i(nh).i(ng).i(nt).i(ns).i(K).i(a1 ? 1 : 0)
      .i(os.off).i(ox.off).i(ox.nb[2]).i(ox.nb[3]).i(od.off).i(oa.off)
      .i(ob.off).i(ob.nb[2]).i(ob.nb[3]).i(oc.off).i(oc.nb[2]).i(oc.nb[3]).i(oi.off);
    add_node("syn_ssm_scan_gen_f32", { os.lt, ox.lt, od.lt, oa.lt, ob.lt, oc.lt, oi.lt }, { part, sts }, pr);
    const int out = begin_out(n);
    params_t pf;
    pf.i(nmem_for(ns * R * nt + (K * ns * R * nc + 63) / 64, 4)).i(nc).i(nr).i(nh).i(nt).i(ns).i(R).i(K);
    add_node("syn_ssm_scan_gen_red_f32", { part, sts }, { out }, pf);
    return end_out(n, out);
}

/* RWKV_WKV6 / GATED_LINEAR_ATTN / RWKV_WKV7 general path (any positive S). WKV6 / GLA: one STEP node runs every
   token per (sequence, head, 32-row tile, 64-column chunk) with the state tile in local memory and writes per-token
   row-tile partials and the final state tiles; one REDUCE node sums the partials into the outputs and copies the
   states. WKV7 (independent rows): one node, one item per state row. Every tensor has one writer. */
bool lowerer::lower_wkv_general(const ggml_tensor * n) {
    const int nin = n->op == GGML_OP_RWKV_WKV7 ? 7 : n->op == GGML_OP_RWKV_WKV6 ? 6 : 5;
    std::vector<int> ins;
    for (int k = 0; k < nin; k++) {
        const int lt = flat_f32_lt(n->src[k]);
        if (lt < 0) {
            return false;
        }
        ins.push_back(lt);
    }
    const int64_t S = n->src[1]->ne[0], H = n->src[1]->ne[1], T = n->src[1]->ne[2], ns = n->src[nin - 1]->ne[1];
    if (n->op == GGML_OP_RWKV_WKV7) {
        if (S > 64 * 128) {
            return fail("wkv7 general head size: %s", n);
        }
        const int out = begin_out(n);
        params_t pr;
        pr.i(nmem_for(ns * H * S, 4)).i(S).i(H).i(T).i(ns);
        add_node("syn_wkv7_gen_f32", ins, { out }, pr);
        return end_out(n, out);
    }
    const int64_t nr32 = (S + 31) / 32, nvc = (S + 63) / 64;
    const int part = new_lt(syn_type_single, { (uint64_t) (nr32 * S * H * T) });
    const int sts = new_lt(syn_type_single, { (uint64_t) (S * S * H * ns) });
    params_t ps;
    ps.i(nmem_for(ns * H * nr32 * nvc, 1)).i(S).i(H).i(T).i(ns);
    if (n->op == GGML_OP_GATED_LINEAR_ATTN) {
        ps.f(ggml_get_op_params_f32(n, 0));
    }
    add_node(n->op == GGML_OP_RWKV_WKV6 ? "syn_wkv6_gen_f32" : "syn_gla_gen_f32", ins, { part, sts }, ps);
    const int out = begin_out(n);
    params_t pf;
    pf.i(nmem_for((S * H * T + 63) / 64 + (S * S * H * ns + 63) / 64, 8)).i(S).i(H).i(T).i(ns);
    add_node("syn_wkv_gen_red_f32", { part, sts }, { out }, pf);
    return end_out(n, out);
}

bool lowerer::lower_node(const ggml_tensor * n) {
    {
        auto it = gdn_plans.find(n);
        if (it != gdn_plans.end()) {
            return lower_gdn(n, it->second);
        }
        auto rt = router_plans.find(n);
        if (rt != router_plans.end()) {
            return lower_router(n, rt->second);
        }
        auto ms = moe_sum_plans.find(n);
        if (ms != moe_sum_plans.end()) {
            return lower_moe_sum(n, ms->second);
        }
        auto cx = convx_plans.find(n);
        if (cx != convx_plans.end()) {
            return lower_convx(n, cx->second);
        }
        auto pgl = paged_glu.find(n);
        if (pgl != paged_glu.end()) {
            return lower_moe_glu(n, pgl->second);
        }
    }
    const ggml_tensor * s0 = n->src[0];
    const ggml_tensor * s1 = n->src[1];
    switch (n->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_GLU: {
            int code;
            opnd oa, ob;
            int64_t sa[4], sb[4];
            int bc0 = 0;
            const int gx = n->op == GGML_OP_GLU ? glux_code(n) : -1;
            if (n->op == GGML_OP_GLU) {
                switch (ggml_get_glu_op(n)) {
                    case GGML_GLU_OP_SWIGLU: code = 4; break;
                    case GGML_GLU_OP_GEGLU:  code = 5; break;
                    case GGML_GLU_OP_REGLU:  code = 6; break;
                    default:
                        if (gx < 0) {
                            return fail("glu op: %s", n);
                        }
                        code = -1;
                }
                if (!get_f32(s0, oa)) {
                    return false;
                }
                if (s1) {
                    if (!get_f32(s1, ob)) {
                        return false;
                    }
                } else {
                    // fused input [a | b] (swapped: [b | a])
                    const int64_t nc = n->ne[0];
                    const bool swapped = ggml_get_op_params_i32(n, 1) != 0;
                    ob = oa;
                    oa.off += swapped ? nc : 0;
                    ob.off += swapped ? 0 : nc;
                    s1 = s0;
                }
                for (int d = 0; d < 4; d++) {
                    sa[d] = oa.nb[d];
                    sb[d] = ob.nb[d];
                }
            } else {
                code = n->op == GGML_OP_ADD ? 0 : n->op == GGML_OP_SUB ? 1 : n->op == GGML_OP_MUL ? 2 : 3;
                if (!get_f32(s0, oa) || !get_f32(s1, ob)) {
                    return false;
                }
                // b repeated along a dim (neither 1 nor the full size): materialized first
                bool rpt = false;
                for (int d = 0; d < 4; d++) {
                    rpt |= s1->ne[d] != 1 && s1->ne[d] != n->ne[d];
                }
                if (rpt) {
                    opnd o2;
                    if (!repeat_to(s1, ob, n, o2)) {
                        return false;
                    }
                    ob = o2;
                }
                bcast_strides(s0, oa, n, sa);
                bcast_strides(rpt ? n : s1, ob, n, sb);
                bc0 = !rpt && s1->ne[0] == 1 && n->ne[0] != 1;
            }
            if ((n->ne[0] > 1 && sa[0] != 1) || (!bc0 && n->ne[0] > 1 && sb[0] != 1)) {
                return fail("strided dim 0: %s", n);
            }
            if (code == 4 && n->view_src == nullptr && roots[n].lt < 0 && !is_output[n] && n->ne[0] % 256 == 0 &&
                n->ne[0] <= 16384 && n->ne[2] == 1 && n->ne[3] == 1 && sa[0] == 1 && sb[0] == 1 && fp8_readers_only(n)) {
                // read by fp8 GEMMs only: fp8 rows + a scale per row (token)
                const int q = new_lt(syn_type_fp8_143, { (uint64_t) ggml_nelements(n) });
                const int qs = new_lt(syn_type_single, { (uint64_t) n->ne[1] });
                root_state & rs = roots[n];
                rs.lt = rs.lt_f8 = q;
                rs.lt_f8s = qs;
                rs.fp8 = true;
                params_t pc;
                pc.split(n->ne[1], 1).i(n->ne[0]).i(n->ne[1]).i(oa.off).i(sa[1]).i(ob.off).i(sb[1]).i(fp8_pow2());
                add_node("syn_swiglu_f8_f32", { oa.lt, ob.lt }, { q, qs }, pc);
                return true;
            }
            if (code == 4 && n->view_src == nullptr && roots[n].lt < 0 && want_bf16(n) && n->ne[0] % 256 == 0 &&
                n->ne[2] == 1 && n->ne[3] == 1 && sa[0] == 1 && sb[0] == 1 && !bc0) {
                const int out = fresh_out(n);
                params_t pc;
                pc.split(n->ne[1] * (n->ne[0] / 256), 2).i(n->ne[0]).i(n->ne[1]).i(oa.off).i(sa[1]).i(ob.off).i(sb[1])
                  .f(1.0f / (float) (n->ne[0] / 256));
                add_node("syn_swiglu_bf_f32", { oa.lt, ob.lt }, { out }, pc);
                return true;
            }
            const int out = begin_out32(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            if (gx >= 0) {
                params_t pr;
                pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(gx).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(oa.off).i(sa[1]).i(sa[2]).i(sa[3]).i(ob.off).i(sb[1]).i(sb[2]).i(sb[3])
                  .f(ggml_get_op_params_f32(n, 2)).f(ggml_get_op_params_f32(n, 3));
                add_node("syn_glux_f32", { oa.lt, ob.lt }, { out }, pr);
                return end_out32(n, out);
            }
            const bool obf = n->view_src == nullptr && roots[n].bf16;
            const int64_t * ss[2] = { sa, sb };
            // a, b rows at uniform strides (b may be one row for all or one element per row)
            static const char * const rowk[] = { "syn_binary_add_f32", "syn_binary_sub_f32", "syn_binary_mul_f32",
                                                 "syn_binary_div_f32", "syn_binary_swiglu_f32", "syn_binary_geglu_f32" };
            int64_t L, nr, rs[2];
            if (!obf && code >= 0 && code <= 5 && ggml_nelements(n) >= 65536 && (n->ne[0] == 1 || sa[0] == 1) &&
                ew_rows(n, ss, 2, !bc0, L, nr, rs)) {
                params_t pr;
                pr.i(nmem_for(nr * ((L + 511) / 512), 1)).i(code).i(L).i(nr).i(1).i(1)
                  .i(oa.off).i(rs[0]).i(0).i(0).i(ob.off).i(rs[1]).i(0).i(0).i(bc0).i(1);
                add_node(rowk[code], { oa.lt, ob.lt }, { out }, pr);
                return end_out32(n, out);
            }
            auto dense = [&](const int64_t * st) {
                int64_t expect = n->ne[0];
                for (int d = 1; d < 4; d++) {
                    if (n->ne[d] > 1 && st[d] != expect) {
                        return false;
                    }
                    expect *= n->ne[d];
                }
                return true;
            };
            const int flat = !bc0 && dense(sa) && dense(sb) && (n->ne[0] == 1 || (sa[0] == 1 && sb[0] == 1));
            // [1, 16, T]-style views: rows of the contiguous leading dims
            int64_t ne[4] = { n->ne[0], n->ne[1], n->ne[2], n->ne[3] };
            int64_t * ms[2] = { sa, sb };
            if (!flat && !bc0) {
                merge_dim0(ne, ms, 2);
            }
            params_t pr;
            const int64_t work = flat ? (ggml_nelements(n) + 255) / 256 : ne[1] * ne[2] * ne[3] * ((ne[0] + 63) / 64);
            pr.i(nmem_for(work, flat ? 2 : 8)).i(code).i(ne[0]).i(ne[1]).i(ne[2]).i(ne[3])
              .i(oa.off).i(sa[1]).i(sa[2]).i(sa[3]).i(ob.off).i(sb[1]).i(sb[2]).i(sb[3]).i(bc0).i(flat);
            add_node(obf ? "syn_binary_bf_f32" : "syn_binary_f32", { oa.lt, ob.lt }, { out }, pr);
            return end_out32(n, out);
        }

        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_FILL: {
            int code;
            float f0 = 0.0f, f1 = 0.0f;
            const int ux = unaryx_code(n);
            if (ux >= 0) {
                opnd oa;
                if (!get_f32(s0, oa)) {
                    return false;
                }
                if (n->ne[0] > 1 && oa.nb[0] != 1) {
                    return fail("strided dim 0: %s", n);
                }
                float fx[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                if (n->op == GGML_OP_LEAKY_RELU || n->op == GGML_OP_CLAMP) {
                    fx[0] = ggml_get_op_params_f32(n, 0);
                    fx[1] = ggml_get_op_params_f32(n, 1);
                } else if (ux == 17) {
                    for (int k = 0; k < 4; k++) {
                        fx[k] = ggml_get_op_params_f32(n, k + 1);
                    }
                }
                const int out = begin_out32(n);
                const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
                params_t pr;
                pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(ux).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(fx[0]).f(fx[1]).f(fx[2]).f(fx[3]);
                add_node("syn_unaryx_f32", { oa.lt }, { out }, pr);
                return end_out32(n, out);
            }
            if (n->op == GGML_OP_FILL) {
                code = 12;
                f0 = ggml_get_op_params_f32(n, 0);
            } else if (n->op == GGML_OP_SCALE) {
                code = 8;
                f0 = ggml_get_op_params_f32(n, 0);
                f1 = ggml_get_op_params_f32(n, 1);
            } else if (n->op == GGML_OP_SQR) {
                code = 9;
            } else if (n->op == GGML_OP_SQRT) {
                code = 10;
            } else {
                code = unary_code(n);
            }
            if (code < 0) {
                return fail("unary op: %s", n);
            }
            // silu(scale(x)) (the hyper-connection low-rank activation)
            const ggml_tensor * ua = s0;
            auto sc = silu_scale.find(n);
            if (code == 0 && sc != silu_scale.end()) {
                ua = sc->second->src[0];
                code = 13;
                f0 = ggml_get_op_params_f32(sc->second, 0);
            }
            opnd oa;
            if (!get_f32(ua, oa)) {
                return false;
            }
            if (n->ne[0] > 1 && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int out = begin_out32(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            const int64_t ne = ggml_nelements(n);
            const bool bf = n->view_src == nullptr && roots[n].bf16;
            // rows of the contiguous leading dims at one uniform stride
            int64_t L = n->ne[0], rs = 0, nr = 1;
            int kd = 1;
            for (; kd < 4 && (n->ne[kd] == 1 || oa.nb[kd] == L); kd++) {
                L *= n->ne[kd];
            }
            bool flat = ne >= 65536 && L >= 256;
            for (int d = kd; d < 4 && flat; d++) {
                if (n->ne[d] == 1) {
                    continue;
                }
                if (rs == 0) {
                    rs = oa.nb[d];
                } else if (oa.nb[d] != rs * nr) {
                    flat = false;
                }
                nr *= n->ne[d];
            }
            if (flat) {
                const char * g = bf ? "syn_unary_flat_bf_f32" : code == 0 ? "syn_unary_silu_f32" : code == 1 ? "syn_unary_sigm_f32" :
                                 code == 6 ? "syn_unary_gelu_f32" : code == 8 ? "syn_unary_scale_f32" : "syn_unary_flat_f32";
                params_t pr;
                pr.i(nmem_for(nr * ((L + 255) / 256), 4)).i(code).i(L).i(nr).i(rs).i(oa.off).f(f0).f(f1);
                add_node(g, { oa.lt }, { out }, pr);
                return end_out32(n, out);
            }
            // [1, 16, T]-style views: rows of the contiguous leading dims
            int64_t ne4[4] = { n->ne[0], n->ne[1], n->ne[2], n->ne[3] };
            int64_t sa[4] = { 1, oa.nb[1], oa.nb[2], oa.nb[3] };
            int64_t * ms[1] = { sa };
            merge_dim0(ne4, ms, 1);
            params_t pr;
            pr.i(nmem_for(ne4[1] * ne4[2] * ne4[3] * ((ne4[0] + 63) / 64), 8)).i(code).i(ne4[0]).i(ne4[1]).i(ne4[2]).i(ne4[3])
              .i(oa.off).i(sa[1]).i(sa[2]).i(sa[3]).f(f0).f(f1);
            add_node(bf ? "syn_unary_bf_f32" : "syn_unary_f32", { oa.lt }, { out }, pr);
            GGML_UNUSED(rows);
            return end_out32(n, out);
        }

        case GGML_OP_REPEAT: {
            opnd oa, o2;
            if (!get_f32(s0, oa)) {
                return false;
            }
            const int out = begin_out32(n);
            if (!repeat_to(s0, oa, n, o2, out)) {
                return false;
            }
            return end_out32(n, out);
        }

        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SUM:
        case GGML_OP_ARGMAX:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D: {   // (pools: k == stride windows without padding here, the rest syn_pool)
            opnd oa;
            if (!get(s0, oa)) {
                return false;
            }
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            // SUM_ROWS widening: the F32 kernel over converted rows, the result converted back
            if (n->op == GGML_OP_SUM_ROWS && n->type != GGML_TYPE_F32) {
                const int srcf = cpy_dense(s0, oa, s0->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : "syn_cpy_bf16_f32_f32",
                                           syn_type_single);
                if (srcf < 0) {
                    return false;
                }
                const int out32 = new_lt(syn_type_single, { (uint64_t) ggml_nelements(n) });
                params_t ps;
                ps.i(nmem_for(s0->ne[1] * s0->ne[2] * s0->ne[3], 2)).i(s0->ne[0]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
                  .i(0).i(s0->ne[0]).i(s0->ne[0] * s0->ne[1]).i(s0->ne[0] * s0->ne[1] * s0->ne[2]).f(1.0f);
                add_node("syn_rowsum_f32", { srcf }, { out32 }, ps);
                const char * back = n->type == GGML_TYPE_F16 ? "syn_cpy_f32_f16_f32" : "syn_cpy_f32_bf16_f32";
                const int out = begin_out(n);
                params_t pb;
                pb.i(nmem_for(n->ne[1] * n->ne[2] * n->ne[3] * ((n->ne[0] + 127) / 128), 8))
                  .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(0).i(1).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2]);
                add_node(back, { out32 }, { out }, pb);
                return end_out(n, out);
            }
            // general path: independent kernel width and stride, padding (MAX/AVG only, F32)
            if (n->op == GGML_OP_POOL_1D || n->op == GGML_OP_POOL_2D) {
                const int32_t * pp = (const int32_t *) n->op_params;
                const bool p2 = n->op == GGML_OP_POOL_2D;
                // params {op, k0, k1, s0, s1, p0, p1} (2D) / {op, k0, s0, p0} (1D)
                const int k0 = pp[1], s0_ = p2 ? pp[3] : pp[2], p0 = p2 ? pp[5] : pp[3];
                const int k1 = p2 ? pp[2] : 1, s1_ = p2 ? pp[4] : 1, p1 = p2 ? pp[6] : 0;
                // the window path below reads the rows as one uniform stride
                const bool rows_uniform = oa.nb[2] == oa.nb[1] * s0->ne[1] && oa.nb[3] == oa.nb[2] * s0->ne[2];
                if (!p2_or_1d_win(pp, n->op) || !rows_uniform) {
                    const int out = begin_out(n);
                    params_t pr;
                    pr.i(nmem_for(ggml_nelements(n), 64))
                      .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                      .i(s0->ne[0]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
                      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
                      .i(k0).i(s0_).i(p0).i(k1).i(s1_).i(p1).i(pp[0] == GGML_OP_POOL_MAX ? 0 : 1);
                    add_node(p2 ? "syn_pool2d_f32" : "syn_pool_f32", { oa.lt }, { out }, pr);
                    return end_out(n, out);
                }
            }
            const bool am = n->op == GGML_OP_ARGMAX;
            const bool pool = n->op == GGML_OP_POOL_1D || n->op == GGML_OP_POOL_2D;
            const bool pmax = pool && ggml_get_op_params_i32(n, 0) == GGML_OP_POOL_MAX;
            const int64_t rows = s0->ne[1] * s0->ne[2] * s0->ne[3];
            auto rowsum = [&](const opnd & o, const int64_t * ne, float scale, int out) {
                params_t pr;
                pr.i(nmem_for(ne[1] * ne[2] * ne[3], 2)).i(ne[0]).i(ne[1]).i(ne[2]).i(ne[3])
                  .i(o.off).i(o.nb[1]).i(o.nb[2]).i(o.nb[3]).f(scale);
                add_node(am ? "syn_argmax_f32" : pmax ? "syn_rowmax_f32" : "syn_rowsum_f32", { o.lt }, { out }, pr);
            };
            if (pool && ggml_get_op_params_i32(n, 1) != s0->ne[0]) {
                // each window a row of a view (a contiguous source; a short tail is dropped, as ggml does)
                const int64_t k = ggml_get_op_params_i32(n, 1);
                opnd ov = oa;
                const int64_t ne[4] = { k, n->ne[0], rows, 1 };
                ov.nb[1] = k;
                ov.nb[2] = oa.nb[1];
                ov.nb[3] = oa.nb[1] * rows;
                const int out = begin_out(n);
                rowsum(ov, ne, pmax ? 1.0f : 1.0f / (float) k, out);
                return end_out(n, out);
            }
            if (n->op == GGML_OP_SUM) {
                // row sums, then the sum of those
                const int part = new_lt(syn_type_single, { (uint64_t) rows });
                rowsum(oa, s0->ne, 1.0f, part);
                opnd op2;
                op2.lt = part;
                op2.nb[1] = rows;
                const int64_t ne2[4] = { rows, 1, 1, 1 };
                const int out = begin_out(n);
                rowsum(op2, ne2, 1.0f, out);
                return end_out(n, out);
            }
            const int out = begin_out(n);
            rowsum(oa, s0->ne, n->op == GGML_OP_MEAN || (pool && !pmax) ? 1.0f / (float) s0->ne[0] : 1.0f, out);
            return end_out(n, out);
        }

        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM: {
            opnd oa;
            const bool fused_add = norm_plans.count(n) && norm_plans[n].add;
            if (!fused_add && !get(s0, oa)) {
                return false;
            }
            if (!fused_add && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int code = n->op == GGML_OP_RMS_NORM ? 0 : n->op == GGML_OP_L2_NORM ? 1 : 2;
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            auto pit = norm_plans.find(n);
            if (pit != norm_plans.end()) {
                const norm_plan & pl = pit->second;
                opnd ob, ow;
                if (pl.add) {
                    const ggml_tensor * sa = pl.add->src[pl.slice0 ? 1 - pl.slice_side : 0];
                    const ggml_tensor * sb = pl.slice0 ? pl.slice0 : pl.add->src[1];
                    if (!get(sa, oa)) {
                        return false;
                    }
                    if (pl.wts) {
                        ob = oa;
                        for (int d = 0; d < 4; d++) {
                            ob.nb[d] = (int64_t) (sb->nb[d] / sizeof(float));
                        }
                    } else if (!get(sb, ob)) {
                        return false;
                    }
                    if (oa.nb[0] != 1 || ob.nb[0] != 1) {
                        return fail("strided dim 0: %s", n);
                    }
                } else {
                    ob = oa;
                }
                if (!get(pl.w, ow)) {
                    return false;
                }
                const ggml_tensor * res = pl.rope ? pl.rope : pl.mul;
                int tab = -1;
                if (pl.rope) {
                    tab = rope_table_lt(pl.rope);
                    if (tab < 0) {
                        return false;
                    }
                }
                const bool compact = code == 0 && !pl.rope && n->ne[0] % 256 == 0 && n->ne[0] <= 4096 && n->ne[2] == 1 &&
                                     n->ne[3] == 1;
                // output as fp8 rows + a scale per row. A graph output also gets its f32
                // rows (then the add's sum is not kept)
                const bool sum_free = !pl.add || (!is_output[pl.add] && cons[pl.add].size() == 1);
                if (code == 0 && !pl.rope && !pl.slice0 && n->ne[0] % 256 == 0 && n->ne[0] <= 8192 && n->ne[2] == 1 &&
                    n->ne[3] == 1 && (!is_output[res] || sum_free) && fp8_readers_only(res)) {
                    const int q = new_lt(syn_type_fp8_143, { (uint64_t) ggml_nelements(res) });
                    const int qs = new_lt(syn_type_single, { (uint64_t) rows });
                    int xo, flags = (pl.add ? 0 : 1) | (fp8_pow2() ? 8 : 0);
                    if (is_output[res]) {
                        xo = fresh_out(res);
                        flags |= 2;
                    } else {
                        xo = pl.add ? fresh_out(pl.add) : new_lt(syn_type_single, { 64 });
                        flags |= pl.add ? 0 : 4;
                        roots[res].lt = q;
                        roots[res].fp8 = true;
                    }
                    roots[res].lt_f8 = q;
                    roots[res].lt_f8s = qs;
                    params_t pc;
                    pc.split(rows, 1).i(n->ne[0]).i(rows).i(oa.off).i(oa.nb[1]).i(ob.off).i(ob.nb[1]).i(ow.off)
                      .f(ggml_get_op_params_f32(n, 0)).f(1.0f / (float) n->ne[0]).i(flags);
                    add_node("syn_addrms_f8_f32", { oa.lt, ob.lt, ow.lt }, { q, qs, xo }, pc);
                    return true;
                }
                if (compact && pl.add && !pl.slice0 && is_output[res] && !is_output[pl.add] && cons[pl.add].size() == 1 &&
                    gemm_consumers_only(res)) {
                    // final norm: f32 output + bf16 copy, the sum is not kept
                    const int yo = fresh_out(res);
                    const int ob16 = new_lt(syn_type_bf16, { (uint64_t) ggml_nelements(res) });
                    roots[res].lt_bf = ob16;
                    params_t pc;
                    pc.split(rows, 1).i(n->ne[0]).i(rows).i(oa.off).i(oa.nb[1]).i(ob.off).i(ob.nb[1]).i(ow.off)
                      .f(ggml_get_op_params_f32(n, 0)).f(1.0f / (float) n->ne[0]).i(2);
                    add_node("syn_addrms_bf_f32", { oa.lt, ob.lt, ow.lt }, { ob16, yo }, pc);
                    return true;
                }
                const int out = fresh_out(res);
                const bool bf = roots[res].bf16;
                if (compact && !pl.add && bf) {
                    params_t pc;
                    pc.split(rows, 1).i(n->ne[0]).i(rows).i(oa.off).i(oa.nb[1]).i(oa.off).i(oa.nb[1]).i(ow.off)
                      .f(ggml_get_op_params_f32(n, 0)).f(1.0f / (float) n->ne[0]).i(1 | 4);
                    add_node("syn_addrms_bf_f32", { oa.lt, oa.lt, ow.lt }, { out, new_lt(syn_type_single, { 64 }) }, pc);
                    return true;
                }
                int xo = -1;
                if (pl.add) {
                    xo = fresh_out(pl.add);
                }
                params_t pr;
                pr.i(nmem_for(rows, 1)).i(code).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(ggml_get_op_params_f32(n, 0)).i(1).i(ow.off)
                  .i(ob.off).i(ob.nb[1]).i(ob.nb[2]).i(ob.nb[3]);
                const char * g = pl.add ? (bf ? "syn_addnorm_bf_f32" : "syn_addnorm_f32") : (bf ? "syn_norm_bf_f32" : "syn_norm_f32");
                std::vector<int> outs = { out };
                if (xo >= 0) {
                    outs.push_back(xo);
                }
                if (pl.add && !pl.slice0 && bf && code == 0 && n->ne[0] % 256 == 0 && n->ne[0] <= 4096 && n->ne[2] == 1 &&
                    n->ne[3] == 1) {
                    params_t pc;
                    pc.split(rows, 1).i(n->ne[0]).i(rows).i(oa.off).i(oa.nb[1]).i(ob.off).i(ob.nb[1]).i(ow.off)
                      .f(ggml_get_op_params_f32(n, 0)).f(1.0f / (float) n->ne[0]).i(0);
                    add_node("syn_addrms_bf_f32", { oa.lt, ob.lt, ow.lt }, outs, pc);
                    return true;
                }
                if (pl.rope && !bf && n->ne[0] % 64 == 0 && n->ne[0] <= 1024 && n->ne[3] == 1) {
                    params_t pc;
                    pc.split(rows, 1).i(n->ne[0]).i(n->ne[1]).i(rows).i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(ow.off)
                      .f(ggml_get_op_params_f32(n, 0)).i(((const int32_t *) pl.rope->op_params)[1])
                      .f(1.0f / (float) n->ne[0]).f(1.0f / (float) n->ne[1]);
                    add_node("syn_normrope_c_f32", { oa.lt, ow.lt, tab }, outs, pc);
                    return true;
                }
                if (pl.rope) {
                    pr.i(((const int32_t *) pl.rope->op_params)[1]);
                    add_node(bf ? "syn_normrope_bf_f32" : "syn_normrope_f32", { oa.lt, ob.lt, ow.lt, tab }, outs, pr);
                    return true;
                }
                if (pl.add) {
                    pr.i(pl.nslices).i((int32_t) pl.slice_stride);
                }
                if (pl.wts) {
                    // MoE output: b = the unweighted expert outputs (slice 0's offset), c = the shared expert
                    opnd oe, oc, ov;
                    const auto pg = paged_rows.find(pl.ebase);
                    if (pg != paged_rows.end()) {
                        // the expert outputs as page rows (lower_moe_paged)
                        if (!get(pl.wts, ov) || (pl.extra && !get(pl.extra, oc))) {
                            return false;
                        }
                        if (!pl.extra) {
                            oc = oa;
                        }
                        params_t pm;
                        pm.i(nmem_for(rows, 1)).i(code).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                          .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(ggml_get_op_params_f32(n, 0)).i(1).i(ow.off)
                          .i(0).i(0).i(0).i(0).i(pl.nslices).i(0)
                          .i(oc.off).i(oc.nb[1]).f(pl.extra ? 1.0f : 0.0f).i(ov.off).i(ov.nb[1]).i(ov.nb[2]);
                        add_node(bf ? "syn_addnorm_moe_bf_i_f32" : "syn_addnorm_moe_i_f32",
                                 { oa.lt, pg->second.y, ow.lt, oc.lt, ov.lt, pg->second.pairs }, outs, pm);
                        return true;
                    }
                    if (!get(pl.ebase, oe) || !get(pl.wts, ov) || (pl.extra && !get(pl.extra, oc))) {
                        return false;
                    }
                    if (!pl.extra) {
                        oc = oa;
                    }
                    oe.off += (int64_t) (pl.slice0->view_offs / sizeof(float));
                    // b's row strides: the slices' (token rows of the expert outputs)
                    params_t pm;
                    pm.i(nmem_for(rows, 1)).i(code).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(ggml_get_op_params_f32(n, 0)).i(1).i(ow.off)
                      .i(oe.off).i(ob.nb[1]).i(ob.nb[2]).i(ob.nb[3]).i(pl.nslices).i((int32_t) pl.slice_stride)
                      .i(oc.off).i(oc.nb[1]).f(pl.extra ? 1.0f : 0.0f).i(ov.off).i(ov.nb[1]).i(ov.nb[2]);
                    add_node(bf ? "syn_addnorm_moe_bf_f32" : "syn_addnorm_moe_f32", { oa.lt, oe.lt, ow.lt, oc.lt, ov.lt }, outs, pm);
                    return true;
                }
                add_node(g, { oa.lt, ob.lt, ow.lt }, outs, pr);
                return true;
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(rows, 1)).i(code).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).f(ggml_get_op_params_f32(n, 0)).i(0).i(0)
              .i(0).i(0).i(0).i(0);
            // widening: F16 / BF16 sources run the F32 kernel over converted rows, the result
            // converted back (get_f32 + output conversion; MEAN / SUM stay F32, CUMSUM stays F32)
            if (s0->type != GGML_TYPE_F32) {
                const int srcf = cpy_dense(s0, oa, s0->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : "syn_cpy_bf16_f32_f32",
                                           syn_type_single);
                if (srcf < 0) {
                    return false;
                }
                const int out32 = new_lt(syn_type_single, { (uint64_t) ggml_nelements(n) });
                params_t pr2;
                pr2.i(nmem_for(rows, 1)).i(code).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(0).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2])
                  .f(ggml_get_op_params_f32(n, 0)).i(0).i(0)
                  .i(0).i(0).i(0).i(0);
                add_node("syn_norm_f32", { srcf, srcf, srcf }, { out32 }, pr2);
                const char * back = n->type == GGML_TYPE_F16 ? "syn_cpy_f32_f16_f32" : "syn_cpy_f32_bf16_f32";
                const int64_t rows2 = n->ne[1] * n->ne[2] * n->ne[3];
                params_t pb;
                pb.i(nmem_for(rows2 * ((n->ne[0] + 127) / 128), 8)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(0).i(1).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2]);
                add_node(back, { out32 }, { out }, pb);
                return end_out(n, out);
            }
            add_node(roots[n].bf16 ? "syn_norm_bf_f32" : "syn_norm_f32", { oa.lt, oa.lt, oa.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            opnd oa;
            if (!get(s0, oa)) {
                return false;
            }
            // a transposed source: a native transpose (the kernel's strided path is element-wise)
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                untranspose(oa);
            }
            // CPY writes into its destination (n is a view of src[1]): a side-effect node
            if (n->view_src != nullptr && oa.nb[0] == 1 && ggml_are_same_shape(s0, n) &&
                (s0->type == GGML_TYPE_F32 || s0->type == n->type)) {
                opnd d;
                if (!get(n, d)) {
                    return false;
                }
                if (d.nb[0] == 1) {
                    const char * g = nullptr;
                    switch (n->type) {
                        case GGML_TYPE_F32:  g = "syn_wr_f32_f32"; break;
                        case GGML_TYPE_F16:  g = s0->type == GGML_TYPE_F16 ? "syn_wr_f16s_f32" : "syn_wr_f16_f32"; break;
                        case GGML_TYPE_BF16: g = s0->type == GGML_TYPE_BF16 ? "syn_wr_bf16s_f32" : "syn_wr_bf16_f32"; break;
                        default: break;
                    }
                    if (g) {
                        const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
                        params_t pr;
                        pr.i(nmem_for(rows * ((n->ne[0] + 127) / 128), 8)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                          .i(d.off).i(d.nb[1]).i(d.nb[2]).i(d.nb[3])
                          .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
                        add_node(g, { d.lt, oa.lt }, { sink() }, pr, true);
                        return true;
                    }
                }
            }
            const char * g = nullptr;
            const ggml_type st = s0->type, dt = n->type;
            if (st == GGML_TYPE_F32 && dt == GGML_TYPE_F32) g = "syn_cpy_f32_f32_f32";
            else if (st == GGML_TYPE_F32 && dt == GGML_TYPE_F16) g = "syn_cpy_f32_f16_f32";
            else if (st == GGML_TYPE_F32 && dt == GGML_TYPE_BF16) g = "syn_cpy_f32_bf16_f32";
            else if (st == GGML_TYPE_F16 && dt == GGML_TYPE_F32) g = "syn_cpy_f16_f32_f32";
            else if (st == GGML_TYPE_BF16 && dt == GGML_TYPE_F32) g = "syn_cpy_bf16_f32_f32";
            else if (st == GGML_TYPE_F16 && dt == GGML_TYPE_F16) g = "syn_cpy_f16_f16_f32";
            else if (st == GGML_TYPE_BF16 && dt == GGML_TYPE_BF16) g = "syn_cpy_bf16_bf16_f32";
            else if (st == GGML_TYPE_F16 && dt == GGML_TYPE_BF16) g = "syn_cpy_f16_bf16_f32";
            else if (st == GGML_TYPE_BF16 && dt == GGML_TYPE_F16) g = "syn_cpy_bf16_f16_f32";
            else if (st == GGML_TYPE_I32 && dt == GGML_TYPE_I32) g = "syn_cpy_i32_i32_f32";
            else if (st == GGML_TYPE_F32 && dt == GGML_TYPE_I32) g = "syn_cpy_f32_i32_f32";
            else if (st == GGML_TYPE_I32 && dt == GGML_TYPE_F32) g = "syn_cpy_i32_f32_f32";
            else if (st == GGML_TYPE_F32 && ggml_is_quantized(dt) && n->ne[0] % ggml_blck_size(dt) == 0) {
                // F32 -> quantized: the CPY variant of the setrows kernels (dense rows + a
                // side-effect byte copy into the destination view)
                return lower_cpy_quant(n);
            }
            else if (ggml_is_quantized(st) && dt == GGML_TYPE_F32) {
                // quantized -> F32: the dqrows32 family (F32 output), a dense dequant + the
                // existing view writer for noncontiguous destinations
                return lower_cpy_dq(n);
            }
            if (!g) {
                return fail("cpy types: %s", n);
            }
            // the result in n's own (contiguous) shape; ggml_cpy/dup reshape freely, so walk
            // the source in its own shape and let the output be the flat element order
            const int out = begin_out(n);
            // so size-1 dims do not matter: without them, a dim 0 of size 1 (one element per kernel row) makes way
            // for a contiguous or transposable one
            if (oa.ne[0] == 1) {
                opnd sq = oa;
                int k = 0;
                for (int d = 0; d < 4; d++) {
                    if (oa.ne[d] > 1) {
                        sq.ne[k] = oa.ne[d];
                        sq.nb[k++] = oa.nb[d];
                    }
                }
                for (; k < 4; k++) {
                    sq.ne[k] = 1;
                }
                if (sq.ne[0] > 1 && (sq.nb[0] == 1 || (ggml_nelements(s0) >= 4096 && untranspose(sq)))) {
                    oa = sq;
                }
            }
            // size-1 dims get contiguous strides (the kernel copies a contiguous source as one flat range)
            if (oa.ne[0] == 1) {
                oa.nb[0] = 1;
            }
            for (int d = 1; d < 4; d++) {
                if (oa.ne[d] == 1) {
                    oa.nb[d] = oa.nb[d - 1] * oa.ne[d - 1];
                }
            }
            const int64_t rows = oa.ne[1] * oa.ne[2] * oa.ne[3];
            params_t pr;
            pr.i(nmem_for(oa.nb[0] == 1 ? rows * ((oa.ne[0] + 127) / 128) : ggml_nelements(s0), oa.nb[0] == 1 ? 8 : 256))
              .i(oa.ne[0]).i(oa.ne[1]).i(oa.ne[2]).i(oa.ne[3])
              .i(oa.off).i(oa.nb[0]).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
            add_node(g, { oa.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_CUMSUM: {
            opnd oa;
            if (!get(s0, oa)) {
                return false;
            }
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(ggml_nrows(n), 1)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
            add_node("syn_cumsum_f32", { oa.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_ARANGE: {
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for((n->ne[0] + 63) / 64, 4)).i(n->ne[0]).f(ggml_get_op_params_f32(n, 0)).f(ggml_get_op_params_f32(n, 2));
            add_node("syn_arange_f32", {}, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_PAD: {
            opnd oa;
            if (!get(s0, oa)) {
                return false;
            }
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int32_t * pp = (const int32_t *) n->op_params;
            const int out = begin_out(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            params_t pr;
            pr.i(nmem_for(rows * ((n->ne[0] + 63) / 64), 8)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(pp[0]).i(pp[2]).i(pp[4]).i(pp[6]).i(s0->ne[0]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
            // op_params[8] == 1: circular (positive modulo per dimension, wraps repeatedly)
            add_node(pp[8] == 1 ? "syn_pad_circ_f32" : "syn_pad_f32", { oa.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_GET_ROWS: {
            // one source row: every valid index is 0, the result is the row itself
            if (s0->view_src == nullptr && !is_output[n] && n->type == s0->type && s0->ne[1] == 1 && s0->ne[2] == 1 &&
                s0->ne[3] == 1 && ggml_nelements(n) == ggml_nelements(s0) && roots.count(s0) && roots[s0].lt >= 0 &&
                !roots[s0].bf16 && roots[s0].lt_bf < 0 && roots[s0].mlen == 0 && roots[s0].boff < 0) {
                roots[n] = roots[s0];
                return true;
            }
            opnd oa, oi;
            if (dq_type(s0->type)) {
                // rows of the dequantized bf16 copy (a source written in this graph is not supported)
                const ggml_tensor * r = root_of(s0);
                if (produced.count(r) || (const char *) s0->data != (const char *) r->data || !ggml_is_contiguous(s0)) {
                    return fail("get_rows quantized source: %s", n);
                }
                const uint64_t addr = dctx->dequant_weight(r, 2);
                oa.lt = new_lt(syn_type_single, { (uint64_t) ggml_nelements(r) }, LT_IN, addr);
                p.lts[oa.lt].region = (r->buffer && ggml_backend_buffer_get_usage(r->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ?
                                          dctx->pool_region(addr) : 0;
                oa.off = 0;
                oa.ne[0] = s0->ne[0]; oa.ne[1] = s0->ne[1]; oa.ne[2] = s0->ne[2]; oa.ne[3] = s0->ne[3];
                oa.nb[0] = 1; oa.nb[1] = s0->ne[0]; oa.nb[2] = s0->ne[0] * s0->ne[1]; oa.nb[3] = oa.nb[2] * s0->ne[2];
                if (!get(s1, oi)) {
                    return false;
                }
            } else if (!get(s0, oa) || !get(s1, oi)) {
                return false;
            }
            const char * g = dq_type(s0->type) ? "syn_get_rows_f32_f32" :
                             s0->type == GGML_TYPE_F32 ? "syn_get_rows_f32_f32" :
                             s0->type == GGML_TYPE_F16 ? "syn_get_rows_f16_f32" :
                             s0->type == GGML_TYPE_BF16 ? "syn_get_rows_bf16_f32" :
                             s0->type == GGML_TYPE_I32 ? "syn_get_rows_i32_f32" : nullptr;
            if (!g || n->type != (s0->type == GGML_TYPE_I32 ? GGML_TYPE_I32 : GGML_TYPE_F32) || oa.nb[0] != 1) {
                return fail("get_rows types: %s", n);
            }
            const int out = begin_out(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            params_t pr;
            pr.i(nmem_for(rows * ((n->ne[0] + 127) / 128), 4)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3])
              .i(oi.off).i(oi.nb[0]).i(oi.nb[1]).i(oi.nb[2]).i(s1->type == GGML_TYPE_I64);
            add_node(g, { oa.lt, oi.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_SET_ROWS: {
            auto pit = setrows_pair.find(n);
            if (pit != setrows_pair.end()) {
                const ggml_tensor * f = pit->second;
                opnd dk, sk, ik, dv, sv, iv;
                if (!get(f, dk) || !get(f->src[0], sk) || !get(f->src[1], ik) || !get(n, dv) || !get(s0, sv) || !get(s1, iv)) {
                    return false;
                }
                const char * g = n->type == GGML_TYPE_F32 ? "syn_setrows2_f32_f32" :
                                 n->type == GGML_TYPE_F16 ? "syn_setrows2_f16_f32" : "syn_setrows2_bf16_f32";
                const int64_t rows = s0->ne[1];
                params_t pr;
                pr.split(2 * rows * ((s0->ne[0] + 127) / 128), 2).i(s0->ne[0]).i(rows).i(s1->type == GGML_TYPE_I64)
                  .f(1.0f / (float) ((s0->ne[0] + 127) / 128))
                  .i(dk.off).i(dk.nb[1]).i(sk.off).i(sk.nb[1]).i(ik.off)
                  .i(dv.off).i(dv.nb[1]).i(sv.off).i(sv.nb[1]).i(iv.off);
                add_node(g, { dk.lt, dv.lt, sk.lt, sv.lt, ik.lt, iv.lt }, { sink() }, pr, 0x3);
                return true;
            }
            // n is a view of the destination; src0 = rows (converted to f32), src1 = indices
            opnd d, os, oi;
            if (!get(n, d) || !get_f32(s0, os) || !get(s1, oi)) {
                return false;
            }
            const char * gq = n->type == GGML_TYPE_Q8_0 ? "syn_setrows_q8_f32" :
                              n->type == GGML_TYPE_Q4_0 ? "syn_setrows_q4_0_f32" :
                              n->type == GGML_TYPE_Q4_1 ? "syn_setrows_q4_1_f32" :
                              n->type == GGML_TYPE_Q5_0 ? "syn_setrows_q5_0_f32" :
                              n->type == GGML_TYPE_Q5_1 ? "syn_setrows_q5_1_f32" :
                              n->type == GGML_TYPE_IQ4_NL ? "syn_setrows_iq4_nl_f32" : nullptr;
            if (gq) {
                if (os.nb[0] != 1 || s0->ne[0] % 32) {
                    return fail("set_rows quantized layout: %s", n);
                }
                const int64_t rows = s0->ne[1] * s0->ne[2] * s0->ne[3];
                params_t pr;
                pr.i(nmem_for(rows * ((s0->ne[0] + 63) / 64), 4)).i(s0->ne[0]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
                  .i(d.off).i(d.nb[1]).i(d.nb[2]).i(d.nb[3])
                  .i(os.off).i(os.nb[1]).i(os.nb[2]).i(os.nb[3])
                  .i(oi.off).i(oi.nb[0]).i(oi.nb[1]).i(oi.nb[2]).i(s1->ne[1]).i(s1->ne[2]).i(s1->type == GGML_TYPE_I64);
                add_node(gq, { d.lt, os.lt, oi.lt }, { sink() }, pr, true);
                return true;
            }
            const char * g = n->type == GGML_TYPE_F32 ? "syn_setrows_f32_f32" :
                             n->type == GGML_TYPE_F16 ? "syn_setrows_f16_f32" :
                             n->type == GGML_TYPE_BF16 ? "syn_setrows_bf16_f32" : nullptr;
            if (!g || d.nb[0] != 1 || os.nb[0] != 1) {
                return fail("set_rows types: %s", n);
            }
            const int64_t rows = s0->ne[1] * s0->ne[2] * s0->ne[3];
            params_t pr;
            pr.i(nmem_for(rows * ((s0->ne[0] + 127) / 128), 4)).i(s0->ne[0]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
              .i(d.off).i(d.nb[1]).i(d.nb[2]).i(d.nb[3])
              .i(os.off).i(os.nb[1]).i(os.nb[2]).i(os.nb[3])
              .i(oi.off).i(oi.nb[0]).i(oi.nb[1]).i(oi.nb[2]).i(s1->ne[1]).i(s1->ne[2]).i(s1->type == GGML_TYPE_I64);
            add_node(g, { d.lt, os.lt, oi.lt }, { sink() }, pr, true);
            return true;
        }

        case GGML_OP_CONCAT: {
            const int dim = ggml_get_op_params_i32(n, 0);
            opnd oa, ob;
            if (!get(s0, oa) || !get(s1, ob)) {
                return false;
            }
            // widening: same declared source / result types F32/F16/BF16; non-F32 runs the F32
            // kernel over converted inputs, the result converted back
            if (n->type != GGML_TYPE_F32 && n->type != GGML_TYPE_F16 && n->type != GGML_TYPE_BF16) {
                return fail("concat layout: %s", n);
            }
            if (s0->type != n->type || s1->type != n->type) {
                return fail("concat layout: %s", n);
            }
            if (n->type != GGML_TYPE_F32) {
                const int srcf0 = cpy_dense(s0, oa, s0->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : "syn_cpy_bf16_f32_f32",
                                            syn_type_single);
                const int srcf1 = cpy_dense(s1, ob, s1->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : "syn_cpy_bf16_f32_f32",
                                            syn_type_single);
                if (srcf0 < 0 || srcf1 < 0) {
                    return false;
                }
                opnd fa, fb;
                dense_opnd(s0, srcf0, fa);
                dense_opnd(s1, srcf1, fb);
                const int out32 = new_lt(syn_type_single, { (uint64_t) ggml_nelements(n) });
                params_t p32;
                p32.i(nmem_for(ggml_nelements(n) / 256, 1)).i(dim).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(s0->ne[dim])
                  .i(0).i(fa.nb[1]).i(fa.nb[2]).i(fa.nb[3]).i(0).i(fb.nb[1]).i(fb.nb[2]).i(fb.nb[3]).i(1).i(1);
                add_node("syn_concat_f32", { srcf0, srcf1 }, { out32 }, p32);
                const char * back = n->type == GGML_TYPE_F16 ? "syn_cpy_f32_f16_f32" : "syn_cpy_f32_bf16_f32";
                const int out = begin_out(n);
                params_t pb;
                pb.i(nmem_for(n->ne[1] * n->ne[2] * n->ne[3] * ((n->ne[0] + 127) / 128), 8))
                  .i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(0).i(1).i(n->ne[0]).i(n->ne[0] * n->ne[1]).i(n->ne[0] * n->ne[1] * n->ne[2]);
                add_node(back, { out32 }, { out }, pb);
                return end_out(n, out);
            }
            // a transposed source: a native transpose (the strided path below is element-wise)
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                untranspose(oa);
            }
            if (s1->ne[0] > 1 && ob.nb[0] != 1) {
                untranspose(ob);
            }
            const int sa0 = s0->ne[0] > 1 ? (int) oa.nb[0] : 1;
            const int sb0 = s1->ne[0] > 1 ? (int) ob.nb[0] : 1;
            const int out = begin_out(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            params_t pr;
            const int64_t work = (sa0 != 1 || sb0 != 1) ? ggml_nelements(n) / 256 : rows * ((n->ne[0] + 63) / 64) / 16;
            pr.i(nmem_for(work, 1)).i(dim).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3]).i(s0->ne[dim])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(ob.off).i(ob.nb[1]).i(ob.nb[2]).i(ob.nb[3]).i(sa0).i(sb0);
            add_node("syn_concat_f32", { oa.lt, ob.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_GATED_DELTA_NET: {
            const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2];
            const ggml_tensor * g = n->src[3], * b = n->src[4], * st = n->src[5];
            opnd oq, ok, ov, og, ob, os;
            if (!get(q, oq) || !get(k, ok) || !get(v, ov) || !get(g, og) || !get(b, ob) || !get(st, os)) {
                return false;
            }
            const int64_t S = v->ne[0], H = v->ne[1], T = v->ne[2], ns = v->ne[3];
            const int64_t K = ggml_get_op_params_i32(n, 0);
            if (K < 1 || S % 64 != 0 || S > 256 || q->ne[3] != ns || k->ne[3] != ns ||
                oq.nb[0] != 1 || ok.nb[0] != 1 || ov.nb[0] != 1) {
                return fail("gated_delta_net config: %s", n);
            }
            if (lower_gdn_chunked(n, oq, ok, ov, og, ob, os)) {
                return true;
            }
            if (!err.empty()) {
                return false;
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(ns * H * S / 4, 4)).i(S).i(H).i(T).i(ns).i(q->ne[1]).i(k->ne[1])
              .i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(oq.nb[3])
              .i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ok.nb[3])
              .i(ov.off).i(ov.nb[1]).i(ov.nb[2]).i(ov.nb[3])
              .i(og.off).i(ob.off).i(os.off).i(os.nb[3]).i(g->ne[0] == S).f(1.0f / sqrtf((float) S))
              .i(K).i(0).i(T).i(1).i(S * H * T * ns);
            add_node("syn_gdn_f32", { oq.lt, ok.lt, ov.lt, og.lt, ob.lt, os.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_SOFT_MAX: {
            const ggml_tensor * mk = n->src[1];
            opnd ox, om;
            if (!get(s0, ox)) {
                return false;
            }
            if (mk && !get(mk, om)) {
                return false;
            }
            const float max_bias = ggml_get_op_params_f32(n, 1);
            const ggml_tensor * snk = n->src[2];
            if (ox.nb[0] != 1 || (mk && (om.nb[0] != 1 || (mk->type != GGML_TYPE_F16 && mk->type != GGML_TYPE_F32)))) {
                return fail("soft_max config: %s", n);
            }
            const int lsl = max_bias > 0.0f ? alibi_table_lt(n->ne[2], max_bias) : ox.lt;
            const int lsk = snk ? flat_f32_lt(snk) : ox.lt;
            if (lsk < 0) {
                return false;
            }
            const int out = begin_out(n);
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            params_t pr;
            pr.i(nmem_for(rows, 2)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(ox.off).i(ox.nb[1]).i(ox.nb[2]).i(ox.nb[3]);
            if (mk) {
                pr.i(om.off).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]).i(mk->ne[1]).i(mk->ne[2]).i(mk->ne[3]).i(1);
            } else {
                pr.i(0).i(0).i(0).i(0).i(1).i(1).i(1).i(0);
            }
            pr.f(ggml_get_op_params_f32(n, 0)).i(max_bias > 0.0f).i(snk != nullptr);
            const bool f16 = mk && mk->type == GGML_TYPE_F16;
            add_node(f16 ? "syn_softmax_f16_f32" : "syn_softmax_f32_f32", { ox.lt, mk ? om.lt : ox.lt, lsl, lsk }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K: {
            opnd oa;
            if (!get(s0, oa)) {
                return false;
            }
            if (s0->ne[0] > 1 && oa.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t k = n->ne[0];
            const bool desc = n->op == GGML_OP_TOP_K || ggml_get_op_params_i32(n, 0) == GGML_SORT_ORDER_DESC;
            // np: the leading columns of the argsort that its readers view (MoE top-k)
            int64_t np = k;
            if (n->op == GGML_OP_ARGSORT && !is_output[n] && n->view_src == nullptr && cons.count(n)) {
                np = 0;
                for (const ggml_tensor * c : cons[n]) {
                    const size_t vo = c->op == GGML_OP_VIEW ? *(const size_t *) c->op_params : 0;
                    if (c->op == GGML_OP_VIEW && c->nb[1] == n->nb[1] && vo % 4 == 0 && vo / 4 + c->ne[0] <= k) {
                        np = std::max<int64_t>(np, vo / 4 + c->ne[0]);
                    } else {
                        np = k;
                    }
                }
                np = std::max<int64_t>(np, 1);
            }
            const int64_t rows = s0->ne[1] * s0->ne[2] * s0->ne[3];
            const int64_t nn = s0->ne[0];
            if (n->op == GGML_OP_TOP_K && (nn > 4096 || k > 64)) {
                // radix select: per-row threshold, then an in-order pick of the selected elements
                const int64_t nv = (nn + 63) / 64;
                const int64_t np = k <= 64 ? std::min<int64_t>(nv / 2, 48 / std::max<int64_t>(rows, 1)) : 1;
                const auto topk = [&](int la, int64_t off, int64_t n0, int64_t ne1, int64_t ne2, int64_t ne3,
                                      int64_t s1, int64_t s2, int64_t s3, int64_t parts, bool indep, int ind,
                                      int out, int vals) {
                    const int64_t nr = ne1 * ne2 * ne3, nvr = (n0 + 63) / 64;
                    const int64_t pt = indep ? parts : 1;
                    const int thr = new_lt(syn_type_int32, { (uint64_t) (2 * nr * pt) });
                    const int cnt = new_lt(syn_type_single, { (uint64_t) (2 * nvr * nr) });
                    params_t pr;
                    pr.i(nmem_for(nr * pt, 1)).i(n0).i(k).i(ne1).i(ne2).i(ne3).i(off).i(s1).i(s2).i(s3).i(desc).i(pt);
                    add_node("syn_topk_thr_f32", { la }, { thr, cnt }, pr);
                    pr.v[0] = nmem_for(nr * parts, 1);
                    pr.v.back() = (int32_t) parts;
                    pr.i(indep);
                    if (vals >= 0) {
                        add_node("syn_topk_pickv_f32", { la, thr, cnt }, { out, vals }, pr);
                    } else if (ind >= 0) {
                        add_node("syn_topk_picki_f32", { la, thr, cnt, ind }, { out }, pr);
                    } else {
                        add_node("syn_topk_pick_f32", { la, thr, cnt }, { out }, pr);
                    }
                };
                const int out = begin_out(n);
                if (np >= 4) {
                    const int64_t nc = np * k;
                    const int ci = new_lt(syn_type_int32, { (uint64_t) (rows * nc) });
                    const int cv = new_lt(syn_type_single, { (uint64_t) (rows * nc) });
                    topk(oa.lt, oa.off, nn, s0->ne[1], s0->ne[2], s0->ne[3], oa.nb[1], oa.nb[2], oa.nb[3], np, true, -1, ci, cv);
                    topk(cv, 0, nc, rows, 1, 1, nc, nc * rows, nc * rows,
                         std::max<int64_t>(1, std::min<int64_t>((nc + 63) / 64, 48 / rows)), false, ci, out, -1);
                } else {
                    topk(oa.lt, oa.off, nn, s0->ne[1], s0->ne[2], s0->ne[3], oa.nb[1], oa.nb[2], oa.nb[3],
                         std::max<int64_t>(1, std::min<int64_t>(nv, 48 / std::max<int64_t>(rows, 1))), false, -1, out, -1);
                }
                return end_out(n, out);
            }
            if (nn > 4096) {
                // bitonic sort over rows padded to N = 2^m
                int64_t N = 4096;
                while (N < nn) {
                    N *= 2;
                }
                const uint64_t sz = (uint64_t) (rows * N);
                int V = new_lt(syn_type_single, { sz }), I = new_lt(syn_type_single, { sz });
                params_t pi;
                pi.i(nmem_for(rows * N / 4096, 1)).i(nn).i(N).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
                  .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(desc);
                add_node("syn_bsort_init_f32", { oa.lt }, { V, I }, pi);
                for (int64_t sz2 = 8192; sz2 <= N; sz2 *= 2) {
                    for (int64_t j = sz2 / 2; j >= 4096; j /= 2) {
                        const int V2 = new_lt(syn_type_single, { sz }), I2 = new_lt(syn_type_single, { sz });
                        params_t pg;
                        pg.i(nmem_for(rows * N / 128, 64)).i(N).i(rows).i(sz2).i(j);
                        add_node("syn_bsort_glob_f32", { V, I }, { V2, I2 }, pg);
                        V = V2;
                        I = I2;
                    }
                    const int V2 = new_lt(syn_type_single, { sz }), I2 = new_lt(syn_type_single, { sz });
                    params_t pl;
                    pl.i(nmem_for(rows * N / 4096, 1)).i(N).i(rows).i(sz2);
                    add_node("syn_bsort_loc_f32", { V, I }, { V2, I2 }, pl);
                    V = V2;
                    I = I2;
                }
                const int out = begin_out(n);
                params_t po;
                po.i(nmem_for(rows * ((nn + 63) / 64), 64)).i(nn).i(N).i(rows);
                add_node("syn_bsort_out_f32", { I }, { out }, po);
                return end_out(n, out);
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(rows, 1)).i(s0->ne[0]).i(k).i(np).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
              .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(desc);
            add_node("syn_argsort_f32", { oa.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_ADD_ID: {
            const ggml_tensor * ids = n->src[2];
            opnd oa, ob, oi;
            if (!get(s0, oa) || !get(s1, ob) || !get(ids, oi)) {
                return false;
            }
            if ((s0->ne[0] > 1 && oa.nb[0] != 1) || (s1->ne[0] > 1 && ob.nb[0] != 1)) {
                return fail("strided dim 0: %s", n);
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(n->ne[1] * n->ne[2], 2)).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(oa.off).i(oa.nb[1]).i(oa.nb[2])
              .i(ob.off).i(ob.nb[1]).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
            add_node("syn_add_id_f32", { oa.lt, ob.lt, oi.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_MUL_MAT_ID: {
            // per (token, slot) product over expert weights re-laid out as bf16 row blocks
            const ggml_tensor * as = s0, * b = s1, * ids = n->src[2];
            const ggml_tensor * ar = root_of(as);
            if (as->ne[0] == 0) {
                return zero_product(n);
            }
            if ((const char *) as->data != (const char *) ar->data || ggml_nbytes(as) != ggml_nbytes(ar) ||
                !ggml_is_contiguous(as) || produced.count(ar)) {
                return fail("mul_mat_id weights: %s", n);
            }
            opnd ox, oi;
            const bool paged_in = paged_x.count(b) > 0;   // (b is in page layout only: lower_moe_paged takes it)
            const bool b32 = b->type == GGML_TYPE_F32;    // (f16 / bf16 b: converted by the path that reads it)
            if ((!paged_in && b32 && !get(b, ox)) || !get(ids, oi)) {
                return false;
            }
            if (!paged_in && b32 && b->ne[0] > 1 && ox.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t K = as->ne[0], M = as->ne[1], E = as->ne[2], nu = ids->ne[0], T = ids->ne[1];
            const int64_t nrb = (M + 127) / 128;
            const synapse_wfmt * f = dq_type(as->type) ? dctx->weight_format(ar) : nullptr;
            if (f && (f->kind == SYN_W_FP8_ROWS || f->kind == SYN_W_BF16)) {
                f = nullptr;
            }
            bool done = false;
            const bool okp = lower_moe_paged(n, f, done);
            if (done) {
                return okp;
            }
            if (paged_in) {
                return fail("paged MUL_MAT_ID input not taken: %s", n);
            }
            const bool ok = lower_moe_dense(n, f, done);
            if (done) {
                return ok;
            }
            if (f) {
                return lower_qmmid(n, f);
            }
            if (!b32 && !get_act(b, ox)) {
                return false;
            }
            const uint64_t addr = dctx->dequant_weight(ar, 1);
            const int lw = new_lt(syn_type_bf16, { (uint64_t) (E * nrb * 128 * K) }, LT_IN, addr);
            p.lts[lw].region = (ar->buffer && ggml_backend_buffer_get_usage(ar->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ?
                                   dctx->pool_region(addr) : 0;
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(T * nu * nrb, 1)).i(K).i(M).i(nrb).i(nu).i(T).i(b->ne[1] == 1)
              .i(ox.off).i(ox.nb[1]).i(ox.nb[2]).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
            add_node("syn_mmid_f32", { lw, ox.lt, oi.lt }, { out }, pr);
            return end_out(n, out);
        }

        case GGML_OP_MUL_MAT:
            if (s0->ne[0] == 0) {
                return zero_product(n);
            }
            return lower_mul_mat(n);
        case GGML_OP_LIGHTNING_INDEXER:
            return lower_lid(n);
        case GGML_OP_TOP_K_MASK: {
            // DSA form (no scores / blocks / extra cells): the DSA kernel; with any of
            // them the general kernel (-DGEN; QSA in Qwen3.8-Flash-Next)
            opnd om, ok;
            if (!get(s0, om) || !get(s1, ok)) {
                return false;
            }
            if (om.nb[0] != 1 || ok.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
            const ggml_tensor * sc = n->src[2], * bl = n->src[3], * ex = n->src[4];
            if (sc || bl || ex) {
                opnd os, ob, oe;
                if ((sc && !get(sc, os)) || (bl && !get(bl, ob)) || (ex && !get(ex, oe))) {
                    return false;
                }
                const int64_t n_pick = bl ? bl->ne[1] : sc ? sc->ne[0] : n->ne[0];
                const int out = begin_out(n);
                params_t pr;
                pr.i(nmem_for(rows, 1)).i(n->ne[0]).i(s1->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
                  .i(om.off).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
                  .i(ok.off).i(ok.nb[1])
                  .i(sc ? os.off : 0).i(sc ? os.nb[1] : 0).i(n_pick)
                  .i(bl ? ob.off : 0).i(bl ? ob.nb[1] : 0).i(bl ? bl->ne[0] : 0)
                  .i(ex ? oe.off : 0).i(ex ? oe.nb[1] : 0).i(ex ? ex->ne[0] : 0)
                  .i((sc ? 1 : 0) | (bl ? 2 : 0) | (ex ? 4 : 0));
                add_node(n->type == GGML_TYPE_F16 ? "syn_topk_mask_g_f16_f32" : "syn_topk_mask_g_f32_f32",
                         { om.lt, ok.lt, sc ? os.lt : ok.lt, bl ? ob.lt : ok.lt, ex ? oe.lt : ok.lt }, { out }, pr);
                return end_out(n, out);
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(rows, 1)).i(n->ne[0]).i(s1->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3])
              .i(om.off).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]).i(s0->ne[1]).i(s0->ne[2]).i(s0->ne[3])
              .i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ok.nb[3]);
            add_node(n->type == GGML_TYPE_F16 ? "syn_topk_mask_f16_f32" : "syn_topk_mask_f32_f32", { om.lt, ok.lt }, { out }, pr);
            return end_out(n, out);
        }
        case GGML_OP_DSV4_HC_COMB: {
            // the 4x4 stream combination of each token (softmax rows + Sinkhorn normalization)
            opnd om, os, ob;
            if (!get(s0, om) || !get(s1, os) || !get(n->src[2], ob)) {
                return false;
            }
            if (om.nb[0] != 1 || ob.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t nt = s0->ne[1];
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(nt, 1)).i(nt).i(om.off).i(om.nb[1]).i(os.off).i(os.nb[0]).i(ob.off)
              .f(ggml_get_op_params_f32(n, 0)).i(ggml_get_op_params_i32(n, 1));
            add_node("syn_hc_comb_f32", { om.lt, os.lt, ob.lt }, { out }, pr);
            return end_out(n, out);
        }
        case GGML_OP_DSV4_HC_PRE: {
            // hyper-connection streams -> one input: scale * sum_h x[:, h, t] * (sigmoid(gate[:, h, t]) or w[h, t])
            opnd ox, ow;
            if (!get(s0, ox) || !get(s1, ow)) {
                return false;
            }
            const bool gated = ggml_get_op_params_i32(n, 1) != 0;
            if (ox.nb[0] != 1 || (gated && ow.nb[0] != 1)) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t ne0 = s0->ne[0], hc = s0->ne[1], T = s0->ne[2];
            const int out = begin_out(n);
            const int64_t chunk = hc_chunk(ne0, T);
            params_t pr;
            pr.i(nmem_for(T * ((ne0 + chunk - 1) / chunk), 1)).i(ne0).i(hc).i(T).i(ox.off).i(ox.nb[1]).i(ox.nb[2])
              .i(ow.off).i(ow.nb[0]).i(ow.nb[1]).i(ow.nb[2]).f(ggml_get_op_params_f32(n, 0)).i(gated).i(chunk);
            add_node("syn_hc_pre_f32", { ox.lt, ow.lt }, { out }, pr);
            return end_out(n, out);
        }
        case GGML_OP_DSV4_HC_POST: {
            // out[:, d, t] = x[:, t] post[d, t] + sum_s residual[:, s, t] comb[d, s, t] (comb absent: + residual[:, d, t])
            auto hp = hc_plans.find(n);
            const hc_plan * pl = hp != hc_plans.end() ? &hp->second : nullptr;
            const ggml_tensor * pt = pl && pl->p ? pl->p : n->src[2], * cb = n->src[3];
            opnd ox, orr, op, oc;
            if (!get(s0, ox) || !get(s1, orr) || !get(pt, op) || (cb && !get(cb, oc))) {
                return false;
            }
            if (ox.nb[0] != 1 || orr.nb[0] != 1) {
                return fail("strided dim 0: %s", n);
            }
            const int64_t ne0 = s0->ne[0], T = s0->ne[1], hc = s1->ne[1];
            const bool sig = pl && pl->p;
            params_t pr;
            if (pl && pl->norm) {
                opnd og;
                if (!get(pl->mul->src[1], og)) {
                    return false;
                }
                const int out = fresh_out(n);
                const int xn = fresh_out(pl->mul);
                std::vector<int> outs = { out, xn };
                if (pl->bf) {
                    const int xb = new_lt(syn_type_bf16, { (uint64_t) ggml_nelements(pl->mul) });
                    roots[pl->mul].lt_bf = xb;
                    outs.push_back(xb);
                }
                pr.i(nmem_for(T * hc, 1)).i(ne0).i(hc).i(T).i(ox.off).i(ox.nb[1])
                  .i(orr.off).i(orr.nb[1]).i(orr.nb[2]).i(op.off).i(op.nb[0]).i(op.nb[1])
                  .i(cb ? oc.off : 0).i(cb ? oc.nb[0] : 0).i(cb ? oc.nb[1] : 0).i(cb ? oc.nb[2] : 0).i(cb != nullptr)
                  .i(sig).f(pl->pa).f(pl->pb).i(og.off).i(og.nb[1]).f(ggml_get_op_params_f32(pl->norm, 0));
                add_node(pl->bf ? "syn_hc_postn_bf_f32" : "syn_hc_postn_f32",
                         { ox.lt, orr.lt, op.lt, cb ? oc.lt : op.lt, og.lt }, outs, pr);
                return true;
            }
            const int out = begin_out(n);
            const int64_t chunk = hc_chunk(ne0, T);
            pr.i(nmem_for(T * ((ne0 + chunk - 1) / chunk), 1)).i(ne0).i(hc).i(T).i(ox.off).i(ox.nb[1])
              .i(orr.off).i(orr.nb[1]).i(orr.nb[2]).i(op.off).i(op.nb[0]).i(op.nb[1])
              .i(cb ? oc.off : 0).i(cb ? oc.nb[0] : 0).i(cb ? oc.nb[1] : 0).i(cb ? oc.nb[2] : 0).i(cb != nullptr)
              .i(sig).f(sig ? pl->pa : 1.0f).f(sig ? pl->pb : 1.0f).i(chunk);
            add_node("syn_hc_post_f32", { ox.lt, orr.lt, op.lt, cb ? oc.lt : op.lt }, { out }, pr);
            return end_out(n, out);
        }
        case GGML_OP_FLASH_ATTN_EXT:
            return lower_flash_attn(n);
        case GGML_OP_ROPE:
            return lower_rope(n);

        case GGML_OP_OUT_PROD:
            return lower_out_prod(n);
        case GGML_OP_SET:
        case GGML_OP_ACC:
            return lower_set_acc(n);
        case GGML_OP_ADD1:
            return lower_add1(n);
        case GGML_OP_COUNT_EQUAL:
            return lower_count_equal(n);
        case GGML_OP_GROUP_NORM:
            return lower_group_norm(n);
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
        case GGML_OP_DIAG_MASK_INF:
            return lower_diag_tri(n);
        case GGML_OP_SOLVE_TRI:
            return lower_solve_tri(n);
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
            return lower_im2col(n);
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
            return lower_conv(n);
        case GGML_OP_CONV_2D_DW:
            return lower_conv_2d_dw(n);
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return lower_conv_transpose(n);
        case GGML_OP_COL2IM_1D:
            return lower_col2im(n);
        case GGML_OP_UPSCALE:
            return lower_upscale(n);
        case GGML_OP_PAD_REFLECT_1D:
            return lower_reflect(n);
        case GGML_OP_ROLL:
            return lower_roll(n);
        case GGML_OP_TIMESTEP_EMBEDDING:
            return lower_timestep(n);
        case GGML_OP_SSM_SCAN:
            {
                // the fast kernel's shapes, else the general path
                const ggml_tensor * x = n->src[1], * A = n->src[3], * B = n->src[4];
                const int64_t nc = n->src[0]->ne[0], nr = x->ne[0], nh = x->ne[1], ng = B->ne[1];
                const bool a1 = A->ne[0] != 1;
                const bool fast = n->type == GGML_TYPE_F32 && nc <= (a1 ? 64 : 256) && ng > 0 && nh % ng == 0 &&
                                  ((nr % 4 == 0) || (nr == 1 && (nh / ng) % 4 == 0)) && n->src[6]->type == GGML_TYPE_I32 &&
                                  ggml_is_contiguous(n->src[6]);
                if (!fast) {
                    return ssm_scan_general_ok(n) ? lower_ssm_scan_general(n) : fail("ssm_scan layout: %s", n);
                }
            }
            {
                opnd os, ox, od, oa, ob, oc, oi;
                if (!get(n->src[0], os) || !get(n->src[1], ox) || !get(n->src[2], od) ||
                    !get(n->src[3], oa) || !get(n->src[4], ob) || !get(n->src[5], oc) ||
                    !get(n->src[6], oi)) {
                    return false;
                }
                const int64_t nc = n->src[0]->ne[0], nr = n->src[1]->ne[0], nh = n->src[1]->ne[1];
                const int64_t nt = n->src[1]->ne[2], ns = n->src[1]->ne[3], ng = n->src[4]->ne[1];
                if (ox.nb[0] != 1 || ox.nb[1] != nr || ob.nb[0] != 1 || ob.nb[1] != nc ||
                    oc.nb[0] != 1 || oc.nb[1] != nc) {
                    return ssm_scan_general_ok(n) ? lower_ssm_scan_general(n) : fail("ssm_scan layout: %s", n);
                }
                const int out = begin_out(n);
                params_t pr;
                pr.i(nmem_for(ns * nr * nh / 4, 4)).i(nc).i(nr).i(nh).i(ng).i(nt).i(ns)
                  .i(ggml_get_op_params_i32(n, 0)).i(n->src[3]->ne[0] != 1)
                  .i(os.off).i(ox.off).i(ox.nb[2]).i(ox.nb[3]).i(od.off).i(oa.off)
                  .i(ob.off).i(ob.nb[2]).i(ob.nb[3]).i(oc.off).i(oc.nb[2]).i(oc.nb[3]).i(oi.off);
                add_node("syn_ssm_scan_f32", { os.lt, ox.lt, od.lt, oa.lt, ob.lt, oc.lt, oi.lt }, { out }, pr);
                return end_out(n, out);
            }
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7: {
            const int nin = n->op == GGML_OP_RWKV_WKV7 ? 7 : n->op == GGML_OP_RWKV_WKV6 ? 6 : 5;
            const int64_t S = n->src[1]->ne[0], T = n->src[1]->ne[2], ns = n->src[nin - 1]->ne[1];
            const bool f32_ok = n->type == GGML_TYPE_F32 && S > 0 && ns > 0 && T % ns == 0;
            bool contig = f32_ok;
            for (int k = 0; contig && k < nin; k++) {
                contig = n->src[k]->type == GGML_TYPE_F32 && ggml_is_contiguous(n->src[k]);
            }
            // bounded kernel: S % 64 == 0 within its per-op ceiling
            if (f32_ok && contig && S % 64 == 0 &&
                S <= (n->op == GGML_OP_RWKV_WKV7 ? 256 : 128)) {
                std::vector<int> ins;
                for (int k = 0; k < nin; k++) {
                    const int lt = flat_f32_lt(n->src[k]);
                    if (lt < 0) {
                        return false;
                    }
                    ins.push_back(lt);
                }
                const int out = begin_out(n);
                params_t pr;
                if (n->op == GGML_OP_RWKV_WKV7) {
                    pr.i(nmem_for(ns * n->src[1]->ne[1] * S / 4, 4)).i(S).i(n->src[1]->ne[1]).i(T).i(ns);
                    add_node("syn_wkv7_f32", ins, { out }, pr);
                } else {
                    pr.i(nmem_for(ns * n->src[1]->ne[1] * S / 64, 1)).i(S).i(n->src[1]->ne[1]).i(T).i(ns);
                    if (n->op == GGML_OP_GATED_LINEAR_ATTN) {
                        pr.f(ggml_get_op_params_f32(n, 0));
                    }
                    add_node(n->op == GGML_OP_RWKV_WKV6 ? "syn_wkv6_f32" : "syn_gla_f32", ins, { out }, pr);
                }
                return end_out(n, out);
            }
            // general path: arbitrary positive S, masked tails
            if (!f32_ok || !contig) {
                return fail("wkv config: %s", n);
            }
            return lower_wkv_general(n);
        }
        case GGML_OP_SSM_CONV: {
            opnd ox, ow;
            if (!get(n->src[0], ox) || !get(n->src[1], ow)) {
                return false;
            }
            const int64_t nc = n->src[1]->ne[0];
            if (n->type != GGML_TYPE_F32 || n->src[0]->type != GGML_TYPE_F32 || n->src[1]->type != GGML_TYPE_F32 ||
                ox.nb[0] != 1 || ow.nb[0] != 1) {
                return fail("ssm_conv layout: %s", n);
            }
            const int64_t C = n->ne[0], nt = n->ne[1], ns = n->ne[2], nx0 = n->src[0]->ne[0];
            auto dens = [&](const opnd & o, int64_t ne0, int64_t ne1, int64_t ne2) {
                const uint64_t ne = (uint64_t) (ne0 * ne1 * ne2);
                if (o.off == 0 && o.nb[1] == ne0 && (ne2 == 1 || o.nb[2] == ne0 * ne1) &&
                    p.lts[o.lt].dims == 1 && p.lts[o.lt].sz[0] == ne) {
                    return o.lt;
                }
                const int lt = new_lt(syn_type_single, { ne });
                params_t pr;
                pr.i(nmem_for(ne1 * ne2 * ((ne0 + 127) / 128), 8)).i(ne0).i(ne1).i(ne2).i(1)
                  .i(o.off).i(1).i(o.nb[1]).i(o.nb[2]).i(0);
                add_node("syn_cpy_f32_f32_f32", { o.lt }, { lt }, pr);
                return lt;
            };
            const int xt = transpose2d(dens(ox, nx0, C, ns), nx0, C * ns);
            const int wt = transpose2d(dens(ow, nc, C, 1), nc, C);
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(ns * ((C + 63) / 64) * ((nt + 63) / 64), 1)).i(nc).i(C).i(nt).i(ns).i(C * ns);
            add_node("syn_ssm_conv_c_f32", { xt, wt }, { out }, pr);
            return end_out(n, out);
        }

        default:
            return fail("unsupported op %s", n);
    }
}

static bool full_root_weight(const ggml_tensor * w) {
    const ggml_tensor * wr = root_of(w);
    const bool full = wr == w || ((const char *) w->data == (const char *) wr->data && ggml_nbytes(w) == ggml_nbytes(wr));
    // the device formats follow the root's shape: a [K, 1, M, 1] root is M matrices, not one
    return full && ggml_is_contiguous(w) && w->ne[2] == 1 && w->ne[3] == 1 && wr->ne[2] == 1 && wr->ne[3] == 1;
}

static uint64_t cat_weights(ggml_backend_synapse_device_context * dctx, const std::vector<const ggml_tensor *> & ws) {
    std::string key;
    uint64_t total = 0;
    for (const ggml_tensor * w : ws) {
        key += std::to_string((uint64_t) (uintptr_t) w->data) + ":" + std::to_string(ggml_nbytes(w)) + ";";
        total += ggml_nbytes(w);
    }
    auto it = dctx->wcat.find(key);
    if (it != dctx->wcat.end()) {
        return it->second.first;
    }
    const uint64_t addr = dctx->pool_alloc(total);
    uint64_t off = 0;
    std::vector<std::pair<uint64_t, uint64_t>> srcs;
    dctx->flush_uploads();
    for (const ggml_tensor * w : ws) {
        const uint64_t src = (uint64_t) (uintptr_t) w->data;
        SYNAPSE_CHECK(synMemCopyAsync(dctx->stream, src, ggml_nbytes(w), addr + off, DRAM_TO_DRAM));
        srcs.push_back({ src, ggml_nbytes(w) });
        off += ggml_nbytes(w);
    }
    dctx->sync();
    dctx->wcat[key] = { addr, srcs };
    return addr;
}

/* columns per weight matrix and number of matrices when the broadcast of w's batches over a's
 * is folded into the columns (a's batches of one weight matrix are adjacent); false: they are not */
static bool mm_batches(const ggml_tensor * w, const ggml_tensor * a, int64_t & ncol, int64_t & nb) {
    const int64_t ne02 = w->ne[2], ne03 = w->ne[3], ne12 = a->ne[2], ne13 = a->ne[3];
    if (ne02 == 0 || ne03 == 0 || ne12 % ne02 || ne13 % ne03) {
        return false;
    }
    const int64_t r2 = ne12 / ne02, r3 = ne13 / ne03;
    if (ne02 * ne03 == 1) {
        ncol = a->ne[1] * ne12 * ne13;
        nb = 1;
    } else if (r3 == 1) {
        ncol = a->ne[1] * r2;
        nb = ne02 * ne03;
    } else if (ne02 == 1) {
        ncol = a->ne[1] * ne12 * r3;
        nb = ne03;
    } else {
        return false;
    }
    return true;
}

/* activation a as a dense (K, cols) tensor of type adt: a bf16 producer output is used as is,
 * else converted by kernel cvt (walking a's own strides). cvt names the f32 source's kernel; an
 * f16 / bf16 a takes the syn_cpy kernel from its own type */
int lowerer::mm_activation(const ggml_tensor * a, synDataType adt, const char * cvt) {
    if (a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16) {
        const bool h = a->type == GGML_TYPE_F16;
        cvt = adt == syn_type_bf16 ? (h ? "syn_cpy_f16_bf16_f32" : "syn_cpy_bf16_bf16_f32") :
              adt == syn_type_fp16 ? (h ? "syn_cpy_f16_f16_f32"  : "syn_cpy_bf16_f16_f32") :
                                     (h ? "syn_cpy_f16_f32_f32"  : "syn_cpy_bf16_f32_f32");
    } else if (a->type != GGML_TYPE_F32) {
        fail("mul_mat activation type: %s", a);
        return -1;
    }
    const int64_t K = a->ne[0];
    const int64_t N = a->ne[1] * a->ne[2] * a->ne[3];
    const ggml_tensor * ar = root_of(a);
    auto rit = roots.find(ar);
    if (adt == syn_type_bf16 && rit != roots.end() && (rit->second.bf16 || rit->second.lt_bf >= 0)) {
        if ((const char *) a->data != (const char *) ar->data || ggml_nelements(a) != ggml_nelements(ar) || !ggml_is_contiguous(a)) {
            fail("bf16 activation view: %s", a);
            return -1;
        }
        return reshape(rit->second.bf16 ? rit->second.lt : rit->second.lt_bf, syn_type_bf16, { (uint64_t) K, (uint64_t) N });
    }
    // F16 / BF16 activations: the copy kernel converts from their own type
    std::string kn;
    if (a->type != GGML_TYPE_F32) {
        const char * st = a->type == GGML_TYPE_F16 ? "f16" : a->type == GGML_TYPE_BF16 ? "bf16" : nullptr;
        const char * dt = adt == syn_type_single ? "f32" : adt == syn_type_fp16 ? "f16" : adt == syn_type_bf16 ? "bf16" : nullptr;
        if (!st || !dt) {
            fail("activation type: %s", a);
            return -1;
        }
        kn = std::string("syn_cpy_") + st + "_" + dt + "_f32";
        cvt = kn.c_str();
    }
    opnd oa;
    if (!get(a, oa)) {
        return -1;
    }
    const int la = new_lt(adt, { (uint64_t) (K * N) });
    params_t pr;
    pr.i(nmem_for(N * ((K + 127) / 128), 8)).i(a->ne[0]).i(a->ne[1]).i(a->ne[2]).i(a->ne[3])
      .i(oa.off).i(a->ne[0] > 1 ? oa.nb[0] : 1).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]);
    add_node(cvt, { oa.lt }, { la }, pr);
    auto rf = roots.find(ar);
    if (adt == syn_type_bf16 && a->type == GGML_TYPE_F32 && rf != roots.end() && produced.count(ar) && ar->view_src == nullptr &&
        (const char *) a->data == (const char *) ar->data && ggml_nelements(a) == ggml_nelements(ar) &&
        ggml_is_contiguous(a) && rf->second.mlen == 0 && rf->second.boff < 0) {
        rf->second.lt_bf = la;
    }
    return reshape(la, adt, { (uint64_t) K, (uint64_t) N });
}

/* the converted weight at addr as a persistent input of the given shape */
int lowerer::wfmt_lt(const ggml_tensor * w, const synapse_wfmt * f, uint64_t addr, synDataType dt,
                     std::initializer_list<uint64_t> sz) {
    const int lt = new_lt(dt, sz, LT_IN, addr);
    if (f->in_place) {
        p.lts[lt].region = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(w->buffer);
    } else {
        p.lts[lt].region = f->own ? 0 : dctx->pool_region(addr);
    }
    return lt;
}

/* activation a (f32) as columns: dim 0 contiguous, column c at o.off + c * sx1 (a dense copy if needed) */
bool lowerer::x_columns(const ggml_tensor * a, opnd & o, int64_t & sx1) {
    if (!get_f32(a, o)) {
        return false;
    }
    const bool cols = (a->ne[0] <= 1 || o.nb[0] == 1) && (a->ne[2] <= 1 || o.nb[2] == o.nb[1] * a->ne[1]) &&
                      (a->ne[3] <= 1 || o.nb[3] == o.nb[2] * a->ne[2]);
    if (!cols) {
        dense_opnd(a, cpy_dense(a, o, "syn_cpy_f32_f32_f32", syn_type_single), o);
    }
    sx1 = a->ne[1] > 1 ? o.nb[1] : (a->ne[2] > 1 ? o.nb[2] : (a->ne[3] > 1 ? o.nb[3] : a->ne[0]));
    return true;
}

static int64_t qmv_units(const synapse_qt * qt, int64_t K) {
    return K / std::max(qt->g, 32);
}

static int64_t qmv_splits(int64_t items, int64_t nkc) {
    static const int64_t target = getenv("GGML_SYNAPSE_QMV_ITEMS") ? atoll(getenv("GGML_SYNAPSE_QMV_ITEMS")) : 72;
    return std::max<int64_t>(1, std::min<int64_t>(nkc, (target + items - 1) / items));
}

/* GEMV node g into out (M x ncol f32); nks: K partial sums reduced by syn_ksum, rb: row blocks
 * per item, xs: x inputs (x view, or int8 x + group data) */
static void qmv_node(lowerer & L, const char * g, int lw, std::vector<int> xs, int lids, int out, const params_t & base,
                     int64_t M, int64_t ncol, int64_t nrb, int64_t nks, int rb) {
    params_t pr = base;
    pr.v[0] = nmem_for(nks * ncol * ((nrb + rb - 1) / rb), 1);
    pr.i(nks);
    std::vector<int> ins = { lw };
    ins.insert(ins.end(), xs.begin(), xs.end());
    ins.push_back(lids);
    if (nks == 1) {
        L.add_node(g, ins, { out }, pr);
        return;
    }
    const int part = L.new_lt(syn_type_single, { (uint64_t) (nks * ncol * M) });
    L.add_node(g, ins, { part }, pr);
    params_t ps;
    ps.i(nmem_for((M * ncol + 63) / 64, 4)).i(M * ncol).i(nks);
    L.add_node("syn_ksum_f32", { part }, { out }, ps);
}

/* int8 GEMV (syn_qmv8_*, syn_qmm.c -DI8): row blocks per item (RB8) */
static int qmv8_rb(const synapse_qt * qt) {
    return qt->p2 || qt->p1 ? 2 : 4;
}

/* x of the int8 GEMV: syn_xq8c layout for some codes (syn_qmm.c XQ), else each k-pair as 256 B */
static bool qmv8_compact(const synapse_qt * qt) {
    return qt->type == GGML_TYPE_Q4_0 || qt->type == GGML_TYPE_Q4_1 || qt->type == GGML_TYPE_Q8_0;
}

static int64_t qmv8_cyc(ggml_type t, int64_t & cpar) {
    cpar = t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ? 100 : 0;
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: return 300;
        case GGML_TYPE_Q8_0: return 385;
        case GGML_TYPE_Q4_K: return 300;
        case GGML_TYPE_MXFP4: return 550;
        case GGML_TYPE_Q5_K: return 700;
        case GGML_TYPE_Q6_K: return 1160;
        default: return 700;
    }
}

/* int8 GEMV node into out (M x ncol f32) */
static void qmv8_node(lowerer & L, const synapse_qt * qt, int lw, int x8, int xg, int lids, int out, const params_t & base,
                      int64_t K, int64_t M, int64_t ncol, int64_t nu, bool has_ids, bool bcast) {
    const int64_t ku = std::max(qt->g, 32), nrb = (M + 127) / 128, rb = qmv8_rb(qt), nkc = K / ku;
    const bool flat = has_ids && bcast;
    const int64_t R = flat ? nu * nrb : nrb, nbx = (R + rb - 1) / rb, nxg = flat ? ncol / nu : ncol;
    const int64_t items = nxg * nbx;
    static const char * env = getenv("GGML_SYNAPSE_QMV_ITEMS");
    int64_t nks = 1;
    if (env) {
        nks = std::max<int64_t>(1, std::min<int64_t>(nkc, (atoll(env) + items - 1) / items));
    } else {
        int64_t cpar;
        const int64_t csb = qmv8_cyc(qt->type, cpar), citem = 600, cksum = 2000;
        int64_t best = INT64_MAX;
        for (int64_t n = 1; n <= std::min<int64_t>(nkc, 64); n++) {
            const int64_t per = (nkc + n - 1) / n;
            if (n > 1 && (n - 1) * per >= nkc) {
                continue;   // an empty split
            }
            int64_t nsb = 0;
            for (int64_t k = 0; k < n; k++) {
                const int64_t a = k * per * ku, b = std::min(nkc, (k + 1) * per) * ku;
                nsb = std::max(nsb, (b - 1) / qt->sb - a / qt->sb + 1);
            }
            const int64_t t = (items * n + 23) / 24 * (rb * (per * ku * csb / 256 + nsb * cpar) + citem) + (n > 1 ? cksum : 0);
            if (t < best) {
                best = t;
                nks = n;
            }
        }
    }
    params_t pr = base;
    pr.v[0] = nmem_for(nks * items, 1);
    pr.i(nks).f(1.0f / nbx).f(1.0f / nxg).f(1.0f / nrb).f(1.0f / nu);
    const std::string g = qt_kname("syn_qmv8_", qt->type);
    if (nks == 1) {
        L.add_node(g.c_str(), { lw, x8, xg, lids }, { out }, pr);
        return;
    }
    const int part = L.new_lt(syn_type_single, { (uint64_t) (nks * ncol * M) });
    L.add_node(g.c_str(), { lw, x8, xg, lids }, { part }, pr);
    params_t ps;
    ps.i(nmem_for((M * ncol + 63) / 64, 4)).i(M * ncol).i(nks);
    L.add_node("syn_ksum_f32", { part }, { out }, ps);
}

/* int8 x of the int8 GEMVs: each k-pair as 256 B (lookup-table codes), or 8 steps per 256 B */
bool lowerer::x_prep(const ggml_tensor * a, int & x8, int & xg, bool compact) {
    opnd ox;
    int64_t sx1;
    if (!x_columns(a, ox, sx1)) {
        return false;
    }
    const int64_t K = a->ne[0], N = a->ne[1] * a->ne[2] * a->ne[3];
    const auto key = std::make_tuple(ox.lt, ox.off, sx1, K, compact);
    auto it = xpreps.find(key);
    if (it != xpreps.end()) {
        x8 = it->second.first;
        xg = it->second.second;
        return true;
    }
    x8 = new_lt(syn_type_int8, { (uint64_t) ((compact ? K / 8 : K / 2) * 256 * N) });
    xg = new_lt(syn_type_single, { (uint64_t) (K / 32 * 256 * N) });
    params_t pr;
    pr.i(nmem_for(N * ((K + 63) / 64), 2)).i(K).i(N).i(ox.off).i(sx1);
    add_node(compact ? "syn_xq8c_f32" : "syn_xq8_f32", { ox.lt }, { x8, xg }, pr);
    xpreps[key] = { x8, xg };
    return true;
}

/* int8 GEMV column limit (x quantized per 32 steps); IQ4 codes stay bf16 */
static bool qmv_int8(const synapse_qt * qt, int64_t ncol) {
    static const bool off = getenv("GGML_SYNAPSE_NO_I8") != nullptr;
    return !off && qt && ncol <= 16 && qt->type != GGML_TYPE_IQ4_NL && qt->type != GGML_TYPE_IQ4_XS;
}

/* MUL_MAT with a quantized weight in a device format */
bool lowerer::lower_qmm(const ggml_tensor * n, const synapse_wfmt * f) {
    const ggml_tensor * w = n->src[0], * a = n->src[1];
    const ggml_tensor * wr = root_of(w);
    const int64_t K = w->ne[0], M = w->ne[1];
    const int64_t N = a->ne[1] * a->ne[2] * a->ne[3];
    const bool out_direct = n->view_src == nullptr && is_output[n];
    const uint64_t oaddr = (uint64_t) (uintptr_t) n->data;
    int ly;
    if (f->kind == SYN_W_BF16) {
        static const bool no_mv = getenv("GGML_SYNAPSE_NO_BF16_MV") != nullptr;
        const auto rit = roots.find(root_of(a));
        if (!no_mv && N <= 4 && M <= 64 && K % 128 == 0 && (rit == roots.end() || !rit->second.bf16)) {
            opnd ox;
            int64_t sx1;
            if (!x_columns(a, ox, sx1)) {
                return false;
            }
            const int lw = wfmt_lt(wr, f, f->addr, syn_type_bf16, { (uint64_t) (K * M) });
            ly = out_direct ? new_lt(syn_type_single, { (uint64_t) (M * N) }, LT_OUT, oaddr) :
                              new_lt(syn_type_single, { (uint64_t) (M * N) });
            params_t pr;
            pr.i(nmem_for(N * ((M + 3) / 4), 1)).i(K).i(M).i(N).i(0).i(K).i(ox.off).i(sx1);
            add_node("syn_mv_bf_f32", { lw, ox.lt }, { ly }, pr);
            if (n->view_src != nullptr) {
                return write_view(n, ly, GGML_TYPE_F32);
            }
            roots[n].lt = ly;
            return true;
        }
        // bf16 copy of a q8_0 weight: bf16 GEMM
        const int la = mm_activation(a, syn_type_bf16, "syn_cpy_f32_bf16_f32");
        if (la < 0) {
            return false;
        }
        const int lw = wfmt_lt(wr, f, f->addr, syn_type_bf16, { (uint64_t) K, (uint64_t) M });
        const int y2 = out_direct ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N }, LT_OUT, oaddr) :
                                    new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N });
        params_t pr;
        synGEMMParams gp(false, true);
        pr.v.resize((sizeof(gp) + 3) / 4);
        memcpy(pr.v.data(), &gp, sizeof(gp));
        pr.bytes = sizeof(gp);
        add_node("gemm", { la, lw }, { y2 }, pr);
        ly = reshape(y2, syn_type_single, { (uint64_t) (M * N) });
    } else if (f->kind == SYN_W_FP8_ROWS && fp8_fuse()) {
        // fp8 activation with a scale per token (the GEMM's [1, N] activation scale); fp8 GEMM with
        // the per-row weight scales
        const ggml_tensor * ar = root_of(a);
        const bool whole = (const char *) a->data == (const char *) ar->data && ggml_nelements(a) == ggml_nelements(ar) &&
                           ggml_is_contiguous(a);
        auto rit = roots.find(ar);
        int q, qs;
        if (rit != roots.end() && rit->second.lt_f8 >= 0 && whole) {
            q = rit->second.lt_f8;
            qs = rit->second.lt_f8s;
        } else if (rit != roots.end() && rit->second.fp8) {
            return fail("fp8 activation view: %s", a);
        } else {
            opnd ox;
            int64_t sx1;
            if (!x_columns(a, ox, sx1)) {
                return false;
            }
            q = new_lt(syn_type_fp8_143, { (uint64_t) (K * N) });
            qs = new_lt(syn_type_single, { (uint64_t) N });
            params_t pq;
            pq.i(nmem_for(N, 1)).i(K).i(N).i(ox.off).i(sx1).i(fp8_pow2());
            add_node("syn_fp8qr_f32", { ox.lt }, { q, qs }, pq);
            rit = roots.find(ar);
            if (whole && rit != roots.end() && produced.count(ar) && ar->view_src == nullptr && rit->second.mlen == 0 &&
                rit->second.boff < 0) {
                rit->second.lt_f8 = q;
                rit->second.lt_f8s = qs;
            }
        }
        const int lw = wfmt_lt(wr, f, f->addr, syn_type_fp8_143, { (uint64_t) K, (uint64_t) M });
        const int ls = wfmt_lt(wr, f, f->scales, syn_type_single, { (uint64_t) M, 1 });
        const int y2 = out_direct ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N }, LT_OUT, oaddr) :
                                    new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N });
        params_t pr;
        synGEMMParams gp(false, true);
        pr.v.resize((sizeof(gp) + 3) / 4);
        memcpy(pr.v.data(), &gp, sizeof(gp));
        pr.bytes = sizeof(gp);
        add_node("fp8_gemm_f32", { reshape(q, syn_type_fp8_143, { (uint64_t) K, (uint64_t) N }), lw,
                                   reshape(qs, syn_type_single, { 1, (uint64_t) N }), ls }, { y2 }, pr);
        ly = reshape(y2, syn_type_single, { (uint64_t) (M * N) });
    } else if (f->kind == SYN_W_FP8_ROWS) {
        // (GGML_SYNAPSE_NO_FP8_FUSE=1) per-tensor fp8 activation, fp8 GEMM with the per-row weight scales
        opnd ox;
        int64_t sx1;
        if (!x_columns(a, ox, sx1)) {
            return false;
        }
        const int q = new_lt(syn_type_fp8_143, { (uint64_t) K, (uint64_t) N });
        const int sc = new_lt(syn_type_single, { 1 });
        // GGML_SYNAPSE_FP8_TOKEN_SCALE: per-token activation scales at prefill (Qwen3.8-27B FP8: prefill logits
        // max |d| 0.93 -> 0.69)
        static const bool fp8_tok = getenv("GGML_SYNAPSE_FP8_TOKEN_SCALE") != nullptr;
        int st = -1;
        if (K * N <= 65536) {
            params_t p1;
            p1.i(1).i(K).i(N).i(ox.off).i(sx1);
            add_node("syn_fp8q1_f32", { ox.lt }, { q, sc }, p1);
        } else {
            const int am = new_lt(syn_type_single, { (uint64_t) N });
            params_t pa;
            pa.i(nmem_for(N, 1)).i(K).i(N).i(ox.off).i(sx1);
            add_node("syn_fp8amax_f32", { ox.lt }, { am }, pa);
            if (fp8_tok) {
                // a scale per token (prefill: the tokens' ranges differ), applied to the output columns
                st = new_lt(syn_type_single, { (uint64_t) N });
                add_node("syn_fp8qt_f32", { ox.lt, am }, { q, sc, st }, pa);
            } else {
                add_node("syn_fp8q_f32", { ox.lt, am }, { q, sc }, pa);
            }
        }
        const int q2 = q;
        const int lw = wfmt_lt(wr, f, f->addr, syn_type_fp8_143, { (uint64_t) K, (uint64_t) M });
        // per-row scales as [M, 1]: the fp8_gemm extractor broadcasts them over the output's columns
        const int ls = wfmt_lt(wr, f, f->scales, syn_type_single, { (uint64_t) M, 1 });
        const int y2 = out_direct && st < 0 ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N }, LT_OUT, oaddr) :
                                              new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N });
        params_t pr;
        synGEMMParams gp(false, true);
        pr.v.resize((sizeof(gp) + 3) / 4);
        memcpy(pr.v.data(), &gp, sizeof(gp));
        pr.bytes = sizeof(gp);
        add_node("fp8_gemm_f32", { q2, lw, sc, ls }, { y2 }, pr);
        ly = reshape(y2, syn_type_single, { (uint64_t) (M * N) });
        if (st >= 0) {
            const int yc = out_direct ? new_lt(syn_type_single, { (uint64_t) (M * N) }, LT_OUT, oaddr) :
                                        new_lt(syn_type_single, { (uint64_t) (M * N) });
            params_t pc;
            pc.i(nmem_for(N * ((M + 1023) / 1024), 1)).i(M).i(N);
            add_node("syn_fp8cols_f32", { ly, st }, { yc }, pc);
            ly = yc;
        }
    } else {
        const synapse_qt * qt = synapse_qt_get(f->type);
        const int64_t nrb = (M + 127) / 128, nsb = K / qt->sb;
        const uint64_t wbytes = synapse_qt_planar_size(qt, K, M, 1);
        const int lw = wfmt_lt(wr, f, f->addr, syn_type_int8, { wbytes, 1 });
        static const int64_t gemv_max = getenv("GGML_SYNAPSE_GEMV_MAX") ? atoll(getenv("GGML_SYNAPSE_GEMV_MAX")) : 4;
        if (N <= gemv_max) {
            // GEMV over the native codes
            ly = out_direct ? new_lt(syn_type_single, { (uint64_t) (M * N) }, LT_OUT, oaddr) :
                              new_lt(syn_type_single, { (uint64_t) (M * N) });
            const int rb = qt->p2 || qt->p1 ? 2 : 4;
            params_t pr;
            if (qmv_int8(qt, N)) {
                int x8, xg;
                if (!x_prep(a, x8, xg, qmv8_compact(qt))) {
                    return false;
                }
                pr.i(0).i(K).i(M).i(nrb).i(N).i(1).i(0).i(0).i(1).i(0).i(0).i(0);
                qmv8_node(*this, qt, lw, x8, xg, x8, ly, pr, K, M, N, 1, false, false);
            } else {
                opnd ox;
                int64_t sx1;
                if (!x_columns(a, ox, sx1)) {
                    return false;
                }
                pr.i(0).i(K).i(M).i(nrb).i(N).i(1).i(0).i(0).i(ox.off).i(sx1).i(0).i(0).i(0).i(0);
                qmv_node(*this, qt_kname("syn_qmv_", f->type).c_str(), lw, { ox.lt }, ox.lt, ly, pr, M, N, nrb,
                         qmv_splits(N * ((nrb + rb - 1) / rb), qmv_units(qt, K)), rb);
            }
        } else {
            // dequantize to bf16 W^T, then bf16 GEMM
            const int wt = new_lt(syn_type_bf16, { (uint64_t) (M * K) });
            params_t pd;
            pd.i(nmem_for(nrb * nsb, 2)).i(K).i(M).i(nrb).i(1).i(M).i(M * K);
            add_node(qt_kname("syn_qdq_", f->type).c_str(), { lw }, { wt }, pd);
            const int lwt = reshape(wt, syn_type_bf16, { (uint64_t) M, (uint64_t) K });
            const int la = mm_activation(a, syn_type_bf16, "syn_cpy_f32_bf16_f32");
            if (la < 0) {
                return false;
            }
            const int y2 = out_direct ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N }, LT_OUT, oaddr) :
                                        new_lt(syn_type_single, { (uint64_t) M, (uint64_t) N });
            params_t pr;
            synGEMMParams gp(false, false);
            pr.v.resize((sizeof(gp) + 3) / 4);
            memcpy(pr.v.data(), &gp, sizeof(gp));
            pr.bytes = sizeof(gp);
            add_node("gemm", { la, lwt }, { y2 }, pr);
            ly = reshape(y2, syn_type_single, { (uint64_t) (M * N) });
        }
    }
    if (n->view_src != nullptr) {
        return write_view(n, ly, GGML_TYPE_F32);
    }
    roots[n].lt = ly;
    return true;
}

/* MUL_MAT_ID: all experts on all tokens, then each (token, slot)
   selects its expert's column. Quantized experts are dequantized to bf16 in the graph. Shared
   activations ([K, 1, T]) multiply all experts as one M E x K operand; per-slot activations are
   gathered per expert (zeros where the expert is unused) for a batch GEMM over experts.
   done: false when not applicable */
/* MUL_MAT_ID, paged: the token-expert pairs grouped by expert into pages of PS rows
 * (syn_moe_route), the page rows gathered as bf16 [K, PS, P] (syn_moe_gx), each page's
 * expert dequantized to bf16 [M, K, P] (syn_qdqp_*), a batch_gemm over the pages, then each pair's row scattered to [M, nu, T] (syn_moe_scat).
 * P = ceil(pairs / PS) + min(E, pairs) bounds the pages any routing needs. GGML_SYNAPSE_MOE_PAGED=0: off;
 * GGML_SYNAPSE_MOE_PSH: log2 PS. */
static bool qdqp_type(ggml_type t) {
    return t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K || t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K ||
           t == GGML_TYPE_Q8_0;
}

// whether MUL_MAT_ID n runs paged, and its page size (2^psh rows) and page count
bool lowerer::paged_shape(const ggml_tensor * n, const synapse_wfmt * f, int & psh, int64_t & P) {
    const ggml_tensor * as = n->src[0], * b = n->src[1], * ids = n->src[2];
    const int64_t K = as->ne[0], M = as->ne[1], E = as->ne[2], nu = ids->ne[0], T = ids->ne[1];
    static const bool on = !getenv("GGML_SYNAPSE_MOE_PAGED") || atoi(getenv("GGML_SYNAPSE_MOE_PAGED")) != 0;
    static const int psh_env = getenv("GGML_SYNAPSE_MOE_PSH") ? atoi(getenv("GGML_SYNAPSE_MOE_PSH")) : 0;
    const int64_t pairs = T * nu;
    if (!on || !f || f->kind != SYN_W_PLANAR || !qdqp_type(f->type) || T < 2 || pairs < E || E > 512) {
        return false;   // (E: the routing kernels keep per-expert counters in local arrays of 512)
    }
    const synapse_qt * qt = synapse_qt_get(f->type);
    if (K % qt->sb != 0 || b->ne[2] != T || (b->ne[1] != 1 && b->ne[1] != nu) || b->ne[3] != 1) {
        return false;
    }
    psh = psh_env;
    if (psh <= 0) {
        psh = 6;
        while (psh < 7 && (1ll << psh) < 2 * pairs / E) {
            psh++;
        }
    }
    const int64_t PS = 1ll << psh;
    P = (pairs + PS - 1) / PS + std::min<int64_t>(E, pairs);
    // operands within one tensor's reach (2^31 bytes)
    return (uint64_t) M * PS * P * 4 < (1ull << 31) && (uint64_t) K * PS * P * 2 < (1ull << 31) &&
           (uint64_t) M * K * P * 2 < (1ull << 31);
}

// MUL_MAT_ID m runs paged on the routing ids with these pages
bool lowerer::paged_same(const ggml_tensor * m, const ggml_tensor * ids, int psh, int64_t P) {
    if (m->op != GGML_OP_MUL_MAT_ID || m->src[2] != ids || !dq_type(m->src[0]->type)) {
        return false;
    }
    const synapse_wfmt * f = dctx->weight_format(root_of(m->src[0]));
    int psh2;
    int64_t P2;
    return f && paged_shape(m, f, psh2, P2) && psh2 == psh && P2 == P;
}

bool lowerer::paged_glu_tail(const ggml_tensor * g, const ggml_tensor * ids, int64_t ff, int psh, int64_t P) {
    if (g->op != GGML_OP_GLU || ggml_get_glu_op(g) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(g, 1) != 0 ||
        !g->src[0] || !g->src[1] || g->view_src || is_output[g] || cons[g].size() != 1 || !ggml_is_contiguous(g) ||
        g->ne[0] != ff || (uint64_t) ff * (1ull << psh) * P * 2 >= (1ull << 31)) {
        return false;
    }
    const ggml_tensor * d = cons[g][0];
    return d->src[1] == g && d->src[0]->ne[0] == ff && paged_same(d, ids, psh, P);
}

// the SwiGLU node reading paged MUL_MAT_ID n as page rows, else nullptr; side: n is its gate (0), up (1), or both
// halves of a merged gate_up (2).
const ggml_tensor * lowerer::paged_glu_next(const ggml_tensor * n, int psh, int64_t P, int & side) {
    static const bool off = getenv("GGML_SYNAPSE_NO_PAGED_GLU") != nullptr;
    const ggml_tensor * ids = n->src[2];
    if (off || is_output[n]) {
        return nullptr;
    }
    if (cons[n].size() == 1) {
        const ggml_tensor * g = cons[n][0];
        const int64_t ff = n->ne[0];
        if (g->op != GGML_OP_GLU || !g->src[0] || !g->src[1] || (g->src[0] != n && g->src[1] != n) || g->src[0] == g->src[1]) {
            return nullptr;
        }
        side = g->src[0] == n ? 0 : 1;
        const ggml_tensor * o = g->src[1 - side];
        if (is_output[o] || cons[o].size() != 1 || o->ne[0] != ff || !ggml_are_same_shape(o, n) ||
            !ggml_is_contiguous(n) || !ggml_is_contiguous(o) || !paged_same(o, ids, psh, P) ||
            !paged_glu_tail(g, ids, ff, psh, P)) {
            return nullptr;
        }
        return g;
    }
    const int64_t M = n->ne[0], ff = M / 2;
    if (M % 2 != 0 || cons[n].size() != 2) {
        return nullptr;
    }
    const ggml_tensor * v0 = cons[n][0];
    const ggml_tensor * g = v0->op == GGML_OP_VIEW && cons[v0].size() == 1 ? cons[v0][0] : nullptr;
    if (!g || g->op != GGML_OP_GLU || !g->src[0] || !g->src[1]) {
        return nullptr;
    }
    for (int k = 0; k < 2; k++) {
        const ggml_tensor * v = g->src[k];
        if (v->op != GGML_OP_VIEW || v->view_src != n || is_output[v] || cons[v].size() != 1 || v->ne[0] != ff ||
            v->ne[1] != n->ne[1] || v->ne[2] != n->ne[2] || v->nb[1] != n->nb[1] || v->nb[2] != n->nb[2] ||
            v->view_offs != (size_t) k * ff * sizeof(float)) {
            return nullptr;
        }
    }
    if (!paged_glu_tail(g, ids, ff, psh, P)) {
        return nullptr;
    }
    side = 2;
    return g;
}

// a product over an empty inner dimension (K = 0: a tensor-parallel split that gives this card none of the
// columns, e.g. 768 expert rows in blocks of 256 over 4 or 8 cards): zeros (the native constant kernel)
bool lowerer::zero_product(const ggml_tensor * n) {
    const int out = begin_out32(n);
    params_t pc;   // ns_ConstantKernel::Params
    pc.f(0.0f);
    add_node("constant_f32", {}, { out }, pc);
    return end_out32(n, out);
}

bool lowerer::lower_moe_glu(const ggml_tensor * g, const paged_glu_t & q) {
    if (q.ya < 0 || q.yb < 0) {
        return fail("paged SwiGLU without both operands: %s", g);
    }
    const int xp = new_lt(syn_type_bf16, { (uint64_t) q.ff, 1ull << q.psh, (uint64_t) q.P });
    params_t pm;
    pm.i(nmem_for(q.P << q.psh, 4)).i(q.ff).i(q.psh).i(q.P).i(q.offa).i(q.sa).i(q.offb).i(q.sb);
    add_node("syn_moe_glu_f32", { q.ya, q.yb, q.pe }, { xp }, pm);
    if (getenv("GGML_SYNAPSE_MOE_PAGED_LOG")) {
        GGML_LOG_INFO("%s: %s ff %lld PS %lld P %lld\n", __func__, g->name, (long long) q.ff, 1ll << q.psh, (long long) q.P);
    }
    paged_x[g] = xp;
    return true;
}

bool lowerer::lower_moe_paged(const ggml_tensor * n, const synapse_wfmt * f, bool & done) {
    const ggml_tensor * as = n->src[0], * b = n->src[1], * ids = n->src[2];
    const ggml_tensor * ar = root_of(as);
    const int64_t K = as->ne[0], M = as->ne[1], E = as->ne[2], nu = ids->ne[0], T = ids->ne[1];
    done = false;
    const int64_t pairs = T * nu;
    int psh;
    int64_t P;
    if (!paged_shape(n, f, psh, P)) {
        return false;
    }
    const synapse_qt * qt = synapse_qt_get(f->type);
    const int64_t PS = 1ll << psh;
    const auto px = paged_x.find(b);   // the input already in this routing's page layout
    opnd ox, oi;
    if ((px == paged_x.end() && !get_act(b, ox)) || !get(ids, oi)) {
        return false;
    }
    if (px == paged_x.end() && b->ne[0] > 1 && ox.nb[0] != 1) {
        return false;
    }
    done = true;
    if (getenv("GGML_SYNAPSE_MOE_PAGED_LOG")) {
        GGML_LOG_INFO("%s: %s K %lld M %lld E %lld nu %lld T %lld: PS %lld P %lld\n", __func__, ggml_type_name(f->type),
                      (long long) K, (long long) M, (long long) E, (long long) nu, (long long) T, (long long) PS, (long long) P);
    }
    const std::string key = std::to_string(oi.lt) + ":" + std::to_string(oi.off) + ":" + std::to_string(oi.nb[0]) + ":" +
                            std::to_string(oi.nb[1]) + ":" + std::to_string(nu) + ":" + std::to_string(T) + ":" +
                            std::to_string(E) + ":" + std::to_string(psh);
    auto it = moe_routes.find(key);
    if (it == moe_routes.end()) {
        // chunks of tc tokens with tc nu a multiple of 32 and rows of Ep ints (whole vectors): 128-byte aligned
        // member regions
        const int64_t Ep = (E + 63) & ~63ll;
        int64_t tc = 32;
        while ((T + tc - 1) / tc > 64) {
            tc *= 2;
        }
        const int64_t NC = (T + tc - 1) / tc;
        moe_route r;
        r.pe = new_lt(syn_type_int32, { (uint64_t) P });
        r.pairs = new_lt(syn_type_int32, { (uint64_t) ((pairs + 31) & ~31ll) });
        const int cnt = new_lt(syn_type_int32, { (uint64_t) (NC * Ep) });
        const int tot = new_lt(syn_type_int32, { (uint64_t) Ep });
        const int pgb = new_lt(syn_type_int32, { (uint64_t) Ep });
        const int base = new_lt(syn_type_int32, { (uint64_t) (NC * Ep) });
        const int cur = new_lt(syn_type_int32, { (uint64_t) (NC * Ep) });
        params_t pc;
        pc.i(NC).i(nu).i(T).i(E).i(Ep).i(tc).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
        add_node("syn_moe_cnt_f32", { oi.lt }, { cnt }, pc);
        params_t pt;
        pt.i(Ep / 64).i(Ep).i(NC);
        add_node("syn_moe_tot_f32", { cnt }, { tot }, pt);
        params_t pp;
        pp.i(1).i(E).i(psh).i(P);
        add_node("syn_moe_plan_f32", { tot }, { r.pe, pgb }, pp);
        add_node("syn_moe_base_f32", { cnt, pgb }, { base }, pt);
        params_t pa;
        pa.i(NC).i(nu).i(T).i(E).i(Ep).i(tc).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
        add_node("syn_moe_asg_f32", { oi.lt, base }, { r.pairs, cur }, pa);
        it = moe_routes.emplace(key, r).first;
    }
    const moe_route & r = it->second;
    int xp;
    if (px != paged_x.end()) {
        xp = px->second;
    } else {
        xp = new_lt(syn_type_bf16, { (uint64_t) K, (uint64_t) PS, (uint64_t) P });
        params_t pg;
        const int bcast = b->ne[1] != 1 ? 0 : nu <= 32 ? 2 : 1;   // (2: token-major, a token's rows in arrays of 32)
        pg.i(nmem_for(pairs * ((K + 127) / 128), 8)).i(K).i(nu).i(T).i(psh).i(bcast).i(ox.off).i(ox.nb[1]).i(ox.nb[2]);
        add_node("syn_moe_gx_f32", { ox.lt, r.pairs }, { xp }, pg);
    }
    const uint64_t wbytes = synapse_qt_planar_size(qt, K, M, E);
    const int lw = wfmt_lt(ar, f, f->addr, syn_type_int8, { wbytes / (uint64_t) E, (uint64_t) E });
    const int wd = new_lt(syn_type_bf16, { (uint64_t) M, (uint64_t) K, (uint64_t) P });
    params_t pd;
    pd.i(0).i(K).i(M).i(qt->sb);
    add_node(qt_kname("syn_qdqp_", f->type).c_str(), { lw, r.pe }, { wd }, pd);
    const int y = new_lt(syn_type_single, { (uint64_t) M, (uint64_t) PS, (uint64_t) P });
    add_node("batch_gemm", { xp, wd }, { y }, gemm_params(false, false));
    int side = 0;
    if (const ggml_tensor * g = paged_glu_next(n, psh, P, side)) {
        paged_glu_t & q = paged_glu[g];
        const int yf = reshape(y, syn_type_single, { (uint64_t) (M * PS * P) });
        q.pe = r.pe;
        q.psh = psh;
        q.P = P;
        q.ff = side == 2 ? M / 2 : M;
        if (side != 1) {
            q.ya = yf;
            q.offa = 0;
            q.sa = M;
        }
        if (side != 0) {
            q.yb = yf;
            q.offb = side == 2 ? M / 2 : 0;
            q.sb = M;
        }
        return true;
    }
    if (!is_output[n] && cons[n].size() == 1) {
        auto rows_ok = [&](const norm_plan & pl) {
            return pl.wts && pl.ebase == n && pl.slice0->view_offs == 0 && pl.nslices == nu && pl.slice_stride == M &&
                   pl.slice0->nb[1] == (size_t) (M * nu) * sizeof(float);
        };
        bool take = false;
        for (const auto & ms : moe_sum_plans) {
            take = take || rows_ok(ms.second);
        }
        for (const auto & np : norm_plans) {
            take = take || (rows_ok(np.second) && nu <= 16);
        }
        if (take) {
            paged_rows[n] = { reshape(y, syn_type_single, { (uint64_t) (M * PS * P) }), r.pairs };
            return true;
        }
    }
    const int out = begin_out(n);
    params_t ps;
    ps.i(nmem_for(pairs * ((M + 63) / 64), 8)).i(M).i(nu).i(T).i(psh);
    add_node("syn_moe_scat_f32", { y, r.pairs }, { out }, ps);
    return end_out(n, out);
}

bool lowerer::lower_moe_dense(const ggml_tensor * n, const synapse_wfmt * f, bool & done) {
    const ggml_tensor * as = n->src[0], * b = n->src[1], * ids = n->src[2];
    const ggml_tensor * ar = root_of(as);
    const int64_t K = as->ne[0], M = as->ne[1], E = as->ne[2], nu = ids->ne[0], T = ids->ne[1];
    static const double ratio = getenv("GGML_SYNAPSE_MOE_DENSE") ? atof(getenv("GGML_SYNAPSE_MOE_DENSE")) : 1.0;
    const bool bcast = b->ne[1] == 1;
    done = false;
    if (ratio <= 0 || (double) (T * nu) < ratio * (double) E || T < 2) {
        return false;
    }
    if (!f && as->type != GGML_TYPE_BF16 && !dq_type(as->type)) {
        return false;
    }
    // all experts as one bf16 operand: within one tensor's reach (2^30 elements); per-slot activations also
    // produce every expert's output for every token ([M, T, E] f32, same reach)
    if (M * K * E > (1ll << 30) || (!bcast && M * T * E > (1ll << 30))) {
        return false;
    }
    done = true;
    const uint64_t ME = (uint64_t) (M * E);
    int lw;
    bool tb;
    if (f) {
        // in-graph dequantization to bf16 W^T: [M, E, K] (one operand) or [M, K, E] (batch)
        const synapse_qt * qt = f->kind == SYN_W_PLANAR ? synapse_qt_get(f->type) : nullptr;
        const int64_t nrb = (M + 127) / 128;
        const uint64_t wbytes = qt ? synapse_qt_planar_size(qt, K, M, E) : synapse_fp8_planar_size(K, M, E);
        const int lq = wfmt_lt(ar, f, f->addr, syn_type_int8, { wbytes / (uint64_t) E, (uint64_t) E });
        const int wt = new_lt(syn_type_bf16, { ME * (uint64_t) K });
        params_t pd;
        pd.i(nmem_for(E * nrb * (qt ? K / qt->sb : (K + 63) / 64), 2)).i(K).i(M).i(nrb).i(E);
        if (bcast) {
            pd.i(M * E).i(M);
        } else {
            pd.i(M).i(M * K);
        }
        add_node(qt ? qt_kname("syn_qdq_", f->type).c_str() : "syn_qdq_fp8_f32", { lq }, { wt }, pd);
        lw = bcast ? reshape(wt, syn_type_bf16, { ME, (uint64_t) K }) :
                     reshape(wt, syn_type_bf16, { (uint64_t) M, (uint64_t) K, (uint64_t) E });
        tb = false;
    } else {
        // bf16 rows [K, M, E] as stored (other quantized types: a bf16 copy)
        int lwf;
        if (as->type == GGML_TYPE_BF16) {
            lwf = root_lt(ar, false);
        } else {
            const uint64_t addr = dctx->dequant_weight(ar);
            lwf = new_lt(syn_type_bf16, { ME * (uint64_t) K }, LT_IN, addr);
            p.lts[lwf].region = (ar->buffer && ggml_backend_buffer_get_usage(ar->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ?
                                    dctx->pool_region(addr) : 0;
        }
        if (lwf < 0) {
            return false;
        }
        lw = bcast ? reshape(lwf, syn_type_bf16, { (uint64_t) K, ME }) :
                     reshape(lwf, syn_type_bf16, { (uint64_t) K, (uint64_t) M, (uint64_t) E });
        tb = true;
    }
    opnd oi;
    if (!get(ids, oi)) {
        return false;
    }
    int y;
    int64_t se, st;
    if (bcast) {
        const int la = mm_activation(b, syn_type_bf16, "syn_cpy_f32_bf16_f32");
        if (la < 0) {
            return false;
        }
        y = new_lt(syn_type_single, { ME, (uint64_t) T });
        add_node("gemm", { la, lw }, { y }, gemm_params(false, tb));
        se = M;
        st = M * E;
    } else {
        opnd ox;
        if (!get_act(b, ox)) {
            return false;
        }
        if (b->ne[0] > 1 && ox.nb[0] != 1) {
            return fail("strided dim 0: %s", n);
        }
        const int xe = new_lt(syn_type_bf16, { (uint64_t) (K * T * E) });
        params_t pg;
        pg.i(nmem_for(T * E, 4)).i(K).i(nu).i(T).i(E).i(ox.off).i(ox.nb[1]).i(ox.nb[2]).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
        add_node("syn_moe_gather_f32", { ox.lt, oi.lt }, { xe }, pg);
        y = new_lt(syn_type_single, { (uint64_t) M, (uint64_t) T, (uint64_t) E });
        add_node("batch_gemm", { reshape(xe, syn_type_bf16, { (uint64_t) K, (uint64_t) T, (uint64_t) E }), lw }, { y },
                 gemm_params(false, tb));
        se = M * T;
        st = M;
    }
    const int ly = reshape(y, syn_type_single, { ME * (uint64_t) T });
    const int out = begin_out(n);
    params_t ps;
    ps.i(nmem_for(T * nu, 4)).i(M).i(nu).i(T).i(se).i(st).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
    add_node("syn_moe_sel_f32", { ly, oi.lt }, { out }, ps);
    return end_out(n, out);
}

/* MUL_MAT_ID with quantized experts in a device format: a GEMV per (token, slot) */
bool lowerer::lower_qmmid(const ggml_tensor * n, const synapse_wfmt * f) {
    const ggml_tensor * as = n->src[0], * b = n->src[1], * ids = n->src[2];
    const ggml_tensor * ar = root_of(as);
    const int64_t K = as->ne[0], M = as->ne[1], E = as->ne[2], nu = ids->ne[0], T = ids->ne[1];
    const int64_t nrb = (M + 127) / 128;
    opnd ox, oi;
    if (!get_act(b, ox) || !get(ids, oi)) {
        return false;
    }
    if (b->ne[0] > 1 && ox.nb[0] != 1) {
        return fail("strided dim 0: %s", n);
    }
    uint64_t wbytes;
    std::string g;
    int64_t nkc;
    if (f->kind == SYN_W_FP8_PLANAR) {
        wbytes = synapse_fp8_planar_size(K, M, E);
        g = "syn_qmv_fp8_f32";
        nkc = K / 64;   // 64-step blocks
    } else {
        const synapse_qt * qt = synapse_qt_get(f->type);
        wbytes = synapse_qt_planar_size(qt, K, M, E);
        g = qt_kname("syn_qmv_", f->type);
        nkc = qmv_units(qt, K);
    }
    const int lw = wfmt_lt(ar, f, f->addr, syn_type_int8, { wbytes / (uint64_t) E, (uint64_t) E });
    const int out = begin_out(n);
    const synapse_qt * qt = f->kind == SYN_W_PLANAR ? synapse_qt_get(f->type) : nullptr;
    const int rb = qt && (qt->p2 || qt->p1) ? 2 : 4;
    params_t pr;
    // each (token, slot) pair is its own column of its own matrix
    if (b->ne[1] * T <= 64 && qmv_int8(qt, 1)) {
        int x8, xg;
        if (!x_prep(b, x8, xg, qmv8_compact(qt))) {
            return false;
        }
        pr.i(0).i(K).i(M).i(nrb).i(T * nu).i(nu).i(b->ne[1] == 1).i(1).i(b->ne[1]).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
        qmv8_node(*this, qt, lw, x8, xg, oi.lt, out, pr, K, M, T * nu, nu, true, b->ne[1] == 1);
    } else {
        pr.i(0).i(K).i(M).i(nrb).i(T * nu).i(nu).i(b->ne[1] == 1).i(1)
          .i(ox.off).i(ox.nb[1]).i(ox.nb[2]).i(oi.off).i(oi.nb[0]).i(oi.nb[1]);
        qmv_node(*this, g.c_str(), lw, { ox.lt }, oi.lt, out, pr, M, T * nu, nrb,
                 qmv_splits(T * nu * ((nrb + rb - 1) / rb), nkc), rb);
    }
    return end_out(n, out);
}

/* batched (3-D/4-D) or strided weights: dense weight in the compute type, a batch GEMM */
bool lowerer::lower_mul_mat_batched(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0];
    const ggml_tensor * a = n->src[1];
    const int64_t K = w->ne[0], M = w->ne[1];
    int64_t ncol, nb;
    if (!mm_batches(w, a, ncol, nb)) {
        return fail("mul_mat broadcast: %s", n);
    }
    const ggml_tensor * wr = root_of(w);
    const bool full = (const char *) w->data == (const char *) wr->data && ggml_nbytes(w) == ggml_nbytes(wr) &&
                      ggml_is_contiguous(w);
    // q8_0 per-head matrices (MLA k_b / v_b): fp8 rows like the 2-D q8_0 weights. The activation gets a scale per
    // column (token x head), the batched fp8 GEMM runs unscaled, then both scales are applied to its output
    if (w->type == GGML_TYPE_Q8_0 && full && w->ne[3] == 1 && nb == w->ne[2] && !produced.count(wr)) {
        const synapse_wfmt * f = dctx->weight_format_rows(wr);
        if (f && f->kind == SYN_W_FP8_ROWS) {
            opnd ox;
            int64_t sx1;
            if (!x_columns(a, ox, sx1)) {
                return false;
            }
            const int64_t N = ncol * nb;
            params_t pa;
            pa.i(nmem_for(N, 1)).i(K).i(N).i(ox.off).i(sx1);
            const int st = new_lt(syn_type_single, { (uint64_t) N });
            int q;
            if (fp8_fuse()) {
                q = new_lt(syn_type_fp8_143, { (uint64_t) (K * N) });
                params_t pq = pa;
                add_node("syn_fp8qr_f32", { ox.lt }, { q, st }, pq.i(fp8_pow2()));
            } else {
                const int am = new_lt(syn_type_single, { (uint64_t) N });
                add_node("syn_fp8amax_f32", { ox.lt }, { am }, pa);
                q = new_lt(syn_type_fp8_143, { (uint64_t) K, (uint64_t) N });
                add_node("syn_fp8qt_f32", { ox.lt, am }, { q, new_lt(syn_type_single, { 1 }), st }, pa);
            }
            const int lw = wfmt_lt(wr, f, f->addr, syn_type_fp8_143, { (uint64_t) K, (uint64_t) M, (uint64_t) nb });
            const int ls = wfmt_lt(wr, f, f->scales, syn_type_single, { (uint64_t) (M * nb) });
            const int g = new_lt(syn_type_single, { (uint64_t) M, (uint64_t) ncol, (uint64_t) nb });
            add_node("batch_gemm", { reshape(q, syn_type_fp8_143, { (uint64_t) K, (uint64_t) ncol, (uint64_t) nb }), lw }, { g },
                     gemm_params(false, true));
            const bool out_direct = n->view_src == nullptr && is_output[n];
            const int ly = out_direct ? new_lt(syn_type_single, { (uint64_t) (M * N) }, LT_OUT, (uint64_t) (uintptr_t) n->data) :
                                        new_lt(syn_type_single, { (uint64_t) (M * N) });
            params_t pc;
            pc.i(nmem_for(N * ((M + 1023) / 1024), 1)).i(M).i(N).i(ncol);
            add_node("syn_fp8rc_f32", { reshape(g, syn_type_single, { (uint64_t) (M * N) }), st, ls }, { ly }, pc);
            if (n->view_src != nullptr) {
                return write_view(n, ly, GGML_TYPE_F32);
            }
            roots[n].lt = ly;
            return true;
        }
    }
    synDataType dt;
    const char * cvt;
    int lwf = -1;
    if (dq_type(w->type)) {
        if (!full) {
            return fail("quantized weight view: %s", n);
        }
        const uint64_t addr = dctx->dequant_weight(wr);
        lwf = new_lt(syn_type_bf16, { (uint64_t) (K * M * nb) }, LT_IN, addr);
        p.lts[lwf].region = (wr->buffer && ggml_backend_buffer_get_usage(wr->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) ?
                                dctx->pool_region(addr) : 0;
        dt = syn_type_bf16;
        cvt = "syn_cpy_f32_bf16_f32";
    } else {
        const char * wc;
        switch (w->type) {
            case GGML_TYPE_BF16: dt = syn_type_bf16;   cvt = "syn_cpy_f32_bf16_f32"; wc = "syn_cpy_bf16_bf16_f32"; break;
            case GGML_TYPE_F16:  dt = syn_type_fp16;   cvt = "syn_cpy_f32_f16_f32";  wc = "syn_cpy_f16_f16_f32";   break;
            case GGML_TYPE_F32:  dt = syn_type_single; cvt = "syn_cpy_f32_f32_f32";  wc = "syn_cpy_f32_f32_f32";   break;
            default: return fail("mul_mat weight type: %s", n);
        }
        if (full && roots.count(wr) == 0 && !produced.count(wr)) {
            lwf = root_lt(wr, false);
        } else {
            // strided view or a graph value: a dense copy in its own type
            opnd ow;
            if (!get(w, ow)) {
                return false;
            }
            lwf = new_lt(dt, { (uint64_t) (K * M * nb) });
            const int64_t rows = M * w->ne[2] * w->ne[3];
            params_t pr;
            pr.i(nmem_for(ow.nb[0] == 1 ? rows * ((K + 127) / 128) : ggml_nelements(w), ow.nb[0] == 1 ? 8 : 256))
              .i(w->ne[0]).i(w->ne[1]).i(w->ne[2]).i(w->ne[3])
              .i(ow.off).i(K > 1 ? ow.nb[0] : 1).i(ow.nb[1]).i(ow.nb[2]).i(ow.nb[3]);
            add_node(wc, { ow.lt }, { lwf }, pr);
        }
    }
    if (lwf < 0) {
        return false;
    }
    const int la = mm_activation(a, dt, cvt);
    if (la < 0) {
        return false;
    }
    int lw, la3, ly2;
    const bool out_direct = n->view_src == nullptr && is_output[n];
    const uint64_t oaddr = (uint64_t) (uintptr_t) n->data;
    const char * g;
    if (nb == 1) {
        lw = reshape(lwf, dt, { (uint64_t) K, (uint64_t) M });
        la3 = la;
        ly2 = out_direct ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) ncol }, LT_OUT, oaddr) :
                           new_lt(syn_type_single, { (uint64_t) M, (uint64_t) ncol });
        g = "gemm";
    } else {
        lw = reshape(lwf, dt, { (uint64_t) K, (uint64_t) M, (uint64_t) nb });
        la3 = reshape(la, dt, { (uint64_t) K, (uint64_t) ncol, (uint64_t) nb });
        ly2 = out_direct ? new_lt(syn_type_single, { (uint64_t) M, (uint64_t) ncol, (uint64_t) nb }, LT_OUT, oaddr) :
                           new_lt(syn_type_single, { (uint64_t) M, (uint64_t) ncol, (uint64_t) nb });
        g = "batch_gemm";
    }
    const int ly = reshape(ly2, syn_type_single, { (uint64_t) (M * ncol * nb) });
    params_t pr;
    synGEMMParams gp(false, true);
    pr.v.resize((sizeof(gp) + 3) / 4);
    memcpy(pr.v.data(), &gp, sizeof(gp));
    pr.bytes = sizeof(gp);
    add_node(g, { la3, lw }, { ly2 }, pr);
    if (n->view_src != nullptr) {
        return write_view(n, ly, GGML_TYPE_F32);
    }
    roots[n].lt = ly;
    return true;
}

bool lowerer::lower_mul_mat(const ggml_tensor * n) {
    const ggml_tensor * w = n->src[0];
    const ggml_tensor * a = n->src[1];
    const int64_t K = w->ne[0];
    const int64_t N = a->ne[1] * a->ne[2] * a->ne[3];
    // f16 / bf16 activations: converted by mm_activation or x_columns
    const bool a_flt = a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16;
    if (a->ne[0] != K || !a_flt || n->type != GGML_TYPE_F32) {
        return fail("mul_mat shape: %s", n);
    }
    // a cache written in this graph (a whole-cache K / V view, one KV head) is read through a copy that waits for the write
    if (w->ne[2] != 1 || w->ne[3] != 1 || !full_root_weight(w) || produced.count(root_of(w)) || written_root(root_of(w))) {
        return lower_mul_mat_batched(n);
    }
    if (dq_type(w->type)) {
        // quantized weights in a device format: fp8 rows, or < 8-bit codes
        const synapse_wfmt * f = dctx->weight_format(root_of(w));
        if (f && f->kind != SYN_W_FP8_PLANAR) {
            return lower_qmm(n, f);
        }
    }
    const ggml_tensor * wr = root_of(w);

    if (w->type == GGML_TYPE_F32 && N <= 4 && K % 64 == 0 && w->nb[0] == sizeof(float) && ggml_nbytes(w) <= (16u << 20)) {
        opnd ox;
        int64_t sx1;
        if (x_columns(a, ox, sx1)) {
            const int64_t M = w->ne[1];
            const int lwf = root_shaped(wr, syn_type_single, { (uint64_t) ggml_nelements(wr) });
            if (lwf < 0) {
                return false;
            }
            const int out = begin_out(n);
            params_t pr;
            pr.i(nmem_for(N * ((M + 3) / 4), 1)).i(K).i(M).i(N)
              .i((int64_t) (((const char *) w->data - (const char *) wr->data) / sizeof(float))).i((int64_t) (w->nb[1] / sizeof(float)))
              .i(ox.off).i(sx1);
            add_node("syn_mv_f32", { lwf, ox.lt }, { out }, pr);
            return end_out(n, out);
        }
    }

    static const bool no_merge = getenv("GGML_SYNAPSE_NO_MERGE") != nullptr;
    std::vector<const ggml_tensor *> grp = { n };
    if (!no_merge && w->type == GGML_TYPE_BF16 && n->view_src == nullptr && !is_output[n]) {
        for (const ggml_tensor * c : cons[a]) {
            if (c != n && c->op == GGML_OP_MUL_MAT && c->src[1] == a && !skip[c] && c->view_src == nullptr &&
                !is_output[c] && c->type == GGML_TYPE_F32 && c->src[0]->type == GGML_TYPE_BF16 &&
                c->src[0]->ne[0] == K && full_root_weight(c->src[0]) && order[c] > order[n]) {
                grp.push_back(c);
            }
        }
    }
    int64_t Mtot = 0;
    for (const ggml_tensor * c : grp) {
        Mtot += c->src[0]->ne[1];
    }

    int lw;
    synDataType adt;
    const char * cvt;
    if (grp.size() > 1) {
        std::vector<const ggml_tensor *> ws;
        for (const ggml_tensor * c : grp) {
            ws.push_back(c->src[0]);
            skip[c] = c != n;
        }
        const int lwf = new_lt(syn_type_bf16, { (uint64_t) (K * Mtot) }, LT_IN, cat_weights(dctx, ws));
        p.lts[lwf].region = dctx->pool_region(p.lts[lwf].addr);
        lw = reshape(lwf, syn_type_bf16, { (uint64_t) K, (uint64_t) Mtot });
        adt = syn_type_bf16;
        cvt = "syn_cpy_f32_bf16_f32";
    } else if (w->type == GGML_TYPE_BF16) {
        lw = root_shaped(wr, syn_type_bf16, { (uint64_t) K, (uint64_t) Mtot });
        adt = syn_type_bf16;
        cvt = "syn_cpy_f32_bf16_f32";
    } else if (w->type == GGML_TYPE_F16) {
        lw = root_shaped(wr, syn_type_fp16, { (uint64_t) K, (uint64_t) Mtot });
        adt = syn_type_fp16;
        cvt = "syn_cpy_f32_f16_f32";
    } else if (w->type == GGML_TYPE_F32) {
        lw = root_shaped(wr, syn_type_single, { (uint64_t) K, (uint64_t) Mtot });
        adt = syn_type_single;
        cvt = "syn_cpy_f32_f32_f32";
    } else if (dq_type(w->type)) {
        const uint64_t addr = dctx->dequant_weight(wr);
        lw = new_lt(syn_type_bf16, { (uint64_t) K, (uint64_t) Mtot }, LT_IN, addr);
        p.lts[lw].region = dctx->pool_region(addr);
        adt = syn_type_bf16;
        cvt = "syn_cpy_f32_bf16_f32";
    } else {
        return fail("mul_mat weight type: %s", n);
    }
    if (lw < 0) {
        return false;
    }

    // activation (K, N) in the compute type
    const int la2 = mm_activation(a, adt, cvt);
    if (la2 < 0) {
        return false;
    }

    // y (Mtot, N) f32
    int ly2;
    if (grp.size() == 1 && n->view_src == nullptr && is_output[n]) {
        ly2 = new_lt(syn_type_single, { (uint64_t) Mtot, (uint64_t) N }, LT_OUT, (uint64_t) (uintptr_t) n->data);
    } else {
        ly2 = new_lt(syn_type_single, { (uint64_t) Mtot, (uint64_t) N });
    }
    const int ly = reshape(ly2, syn_type_single, { (uint64_t) (Mtot * N) });
    params_t pr;
    synGEMMParams gp(false, true);
    pr.v.resize((sizeof(gp) + 3) / 4);
    memcpy(pr.v.data(), &gp, sizeof(gp));
    pr.bytes = sizeof(gp);
    add_node("gemm", { la2, lw }, { ly2 }, pr);
    if (grp.size() > 1) {
        int64_t off = 0;
        for (const ggml_tensor * c : grp) {
            root_state & rs = roots[c];
            rs.lt = ly;
            rs.roff = off;
            rs.mlen = c->src[0]->ne[1];
            rs.mstride = Mtot;
            off += c->src[0]->ne[1];
        }
        return true;
    }
    if (n->view_src != nullptr) {
        return write_view(n, ly, GGML_TYPE_F32);
    }
    roots[n].lt = ly;
    return true;
}

/* flash attention, modes 1 and 2 (see fa_mme): scores, softmax with mask / alibi / sinks (P in dt),
   P V, then the [Dv, H, nq, B] output. K and V are f16 or bf16 (dt) views */
bool lowerer::lower_fa_mme(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov, const opnd & om,
                           synDataType dt, int mode) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = mask_of(n), * snk = n->src[4];
    const float scale = ggml_get_op_params_f32(n, 0);
    const float max_bias = ggml_get_op_params_f32(n, 1);
    const int64_t D = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    const int64_t nkv = k->ne[1], Hkv = k->ne[2], hpg = H / Hkv;
    const bool bf = dt == syn_type_bf16;
    if (fa_q_bf.count(n) && !(mode == 1 && fa_sdpa(n))) {
        return fail("rope-q plan without sdpa: %s", n);
    }
    const int lsl0 = max_bias > 0.0f ? alibi_table_lt(H, max_bias) : -1;
    const int lsk0 = snk ? flat_f32_lt(snk) : -1;
    if (snk && lsk0 < 0) {
        return false;
    }
    // softmax over the scores s [nkv, nqb, H, B] of queries i0 .. i0 + nqb - 1. With the mask shared by the
    // heads and no ALiBi: P unnormalized (syn_softmax_u), lsum = its 1 / row sums [H, nqb, B], applied to
    // P V by syn_fa_norm; else lsum = -1
    static const bool no_u = getenv("GGML_SYNAPSE_NO_SOFTMAX_U") != nullptr;
    const bool use_u = !no_u && max_bias == 0.0f && (!mk || (mk->type == GGML_TYPE_F16 && mk->ne[2] == 1));
    int lsum = -1;
    auto softmax = [&](int s, int64_t i0, int64_t nqb) {
        const int ls = reshape(s, syn_type_single, { (uint64_t) (nkv * nqb * H * B) });
        const int pm = new_lt(dt, { (uint64_t) (nkv * nqb * H * B) });
        if (use_u) {
            lsum = new_lt(syn_type_single, { (uint64_t) (nqb * H * B) });
            int64_t hg = std::min<int64_t>(H, 16);
            while (hg > 1 && nqb * B * ((H + hg - 1) / hg) < 96) {
                hg = (hg + 1) / 2;
            }
            const int64_t items = nqb * B * ((H + hg - 1) / hg);
            params_t pr;
            pr.i(nmem_for(items, 1)).i(nkv).i(nqb).i(H).i(B).i(hg);
            if (mk) {
                pr.i(om.off + i0 * om.nb[1]).i(om.nb[1]).i(om.nb[3]).i(mk->ne[1] - i0).i(mk->ne[3]).i(1);
            } else {
                pr.i(0).i(0).i(0).i(1).i(1).i(0);
            }
            pr.f(scale).i(snk != nullptr);
            add_node(bf ? "syn_softmax_u_b_f32" : "syn_softmax_u_h_f32", { ls, mk ? om.lt : ls, lsk0 >= 0 ? lsk0 : ls },
                     { pm, lsum }, pr);
            return pm;
        }
        params_t pr;
        pr.i(nmem_for(nqb * H * B, 2)).i(nkv).i(nqb).i(H).i(B).i(0).i(nkv).i(nkv * nqb).i(nkv * nqb * H);
        if (mk) {
            pr.i(om.off + i0 * om.nb[1]).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]).i(mk->ne[1] - i0).i(mk->ne[2]).i(mk->ne[3]).i(1);
        } else {
            pr.i(0).i(0).i(0).i(0).i(1).i(1).i(1).i(0);
        }
        pr.f(scale).i(max_bias > 0.0f).i(snk != nullptr);
        add_node(bf ? "syn_softmax_f16_b_f32" : "syn_softmax_f16_h_f32",
                 { ls, mk ? om.lt : ls, lsl0 >= 0 ? lsl0 : ls, lsk0 >= 0 ? lsk0 : ls }, { pm }, pr);
        return pm;
    };
    // K / V rows the GEMMs read in place: persistent inputs aliasing the cache windows (in the caches' section). A native
    // GEMM takes no ordering token, so the producer of the scores GEMM's other operand takes the caches as inputs: it and
    // all after it wait for this graph's cache writes. (Not a no-op node "producing" the windows: the GC puts such an
    // output in SRAM or behind a copy and copies that garbage back over the cache.)
    // GGML_SYNAPSE_FA_KVCOPY=1, a window not 128-byte aligned or K / V not cache inputs: the rows copied first.
    // Also a small window of n elements (under 2 MiB): the GC can load it before this graph's cache write (stale rows
    // seen with 256 cells and >= 768 query rows per KV head)
    static const bool kvcopy = getenv("GGML_SYNAPSE_FA_KVCOPY") != nullptr;
    auto window_addr = [&](const opnd & o, uint64_t n, uint64_t & addr) {
        const lt_t & t = p.lts[base_of(o.lt)];
        addr = t.addr + (uint64_t) o.off * syn_type_size(dt);
        return !kvcopy && !no_kv_view && n * syn_type_size(dt) >= (2u << 20) && t.kind == LT_IN && t.region != 0 && t.dt == dt &&
               addr % 128 == 0;
    };
    auto window = [&](const opnd & o, std::initializer_list<uint64_t> sz, uint64_t addr) {
        const int lt = new_lt(dt, sz, LT_IN, addr);
        p.lts[lt].region = p.lts[base_of(o.lt)].region;
        kv_window[lt] = base_of(o.lt);
        return lt;
    };
    if (mode == 2) {
        // mode 2: rows of all KV heads [D Hkv, nkv]; queries block-diagonal [D Hkv, nq H]
        // (column iq + nq h nonzero in its KV head's rows): S = K^T Qbd, O = V P [Dv Hkv, nq H].
        // Rows in place: Qbd takes the caches as inputs
        const uint64_t R = (uint64_t) (D * Hkv), Rv = (uint64_t) (Dv * Hkv), C = (uint64_t) (nq * H);
        uint64_t ak = 0, av = 0;
        const bool in_place = window_addr(ok, R * nkv, ak) && window_addr(ov, Rv * nkv, av);
        int lk, lv;
        if (in_place) {
            lk = window(ok, { R, (uint64_t) nkv }, ak);
            lv = window(ov, { Rv, (uint64_t) nkv }, av);
        } else {
            auto flat = [&](const opnd & o, uint64_t rows) {
                const int lt = new_lt(dt, { rows * (uint64_t) nkv });
                params_t pr;
                pr.i(nmem_for((int64_t) (rows * nkv / 512), 4)).i((int64_t) (rows * nkv)).i(o.off);
                add_node(bf ? "syn_memcpy_b_f32" : "syn_memcpy_h_f32", { o.lt }, { lt }, pr);
                return reshape(lt, dt, { rows, (uint64_t) nkv });
            };
            lk = flat(ok, R);
            lv = flat(ov, Rv);
        }
        const int qbd = new_lt(dt, { R * C });
        params_t pq;
        pq.i(nmem_for((int64_t) (C * ((R + 127) / 128)), 4)).i(D).i(Hkv).i(H).i(nq).i(hpg).i(oq.off).i(oq.nb[1]).i(oq.nb[2]);
        add_node(bf ? "syn_fa_qbd_b_f32" : "syn_fa_qbd_h_f32", { oq.lt, ok.lt, ov.lt }, { qbd }, pq);
        const int s = new_lt(syn_type_single, { (uint64_t) nkv, C });
        add_node("gemm", { reshape(qbd, dt, { R, C }), lk }, { s }, gemm_params(false, true));
        const int pm = softmax(s, 0, nq);
        const int o = new_lt(syn_type_single, { Rv, C });
        add_node("gemm", { reshape(pm, dt, { (uint64_t) nkv, C }), lv }, { o }, gemm_params(false, false));
        if (in_place) {
            // the GEMMs are the real readers: a later write of the caches waits for them
            reader_outs[base_of(ok.lt)].push_back(s);
            reader_outs[base_of(ov.lt)].push_back(o);
        }
        // out (d, h = hk hpg + g, iq) = O[hk Dv + d, iq + nq h]: [Dv, hpg, Hkv, nq] strided
        const int lo = reshape(o, syn_type_single, { Rv * C });
        const int out = begin_out(n);
        params_t pc;
        if (lsum >= 0) {
            // 1 / row sum of out (d, g, hk, iq): lsum[hk hpg + g + H iq]
            pc.i(nmem_for(H * nq * ((Dv + 127) / 128), 8)).i(Dv).i(hpg).i(Hkv).i(nq).i(0)
              .i(nq * (int64_t) Rv).i(Dv + nq * hpg * (int64_t) Rv).i((int64_t) Rv).i(1).i(hpg).i(H);
            add_node("syn_fa_norm_f32", { lo, lsum }, { out }, pc);
            return end_out(n, out);
        }
        pc.i(nmem_for(H * nq * ((Dv + 127) / 128), 8)).i(Dv).i(hpg).i(Hkv).i(nq).i(0).i(1)
          .i(nq * (int64_t) Rv).i(Dv + nq * hpg * (int64_t) Rv).i((int64_t) Rv);
        add_node("syn_cpy_f32_f32_f32", { lo }, { out }, pc);
        return end_out(n, out);
    }
    // native fused SDPA (sdpa_recomp_fwd_bf16). GQA: each KV head a batch
    // entry, q [D, nq, hpg, Hkv] against k / v [D, nkv, 1, Hkv]; the mask additive bf16 [nkv, nq]; bf16 output
    // (Synapse's slicer rejects an f32 SDPA output at T >= 2048) permuted into [Dv, H, nq] f32 by a copy.
    // D == Dv in {64, 128, 256} (not MLA: wrong results and a device error at D 576), one sequence, no sinks /
    // ALiBi / softcap. bf16 throughout (PPL +0.05-0.14% vs the exact path).
    // GGML_SYNAPSE_FA_SDPA=0: the GEMM + softmax path
    if (mode == 1 && fa_sdpa(n)) {
        auto to_bf16 = [&](const opnd & o, ggml_type st, int64_t ne0, int64_t ne1, int64_t ne2, int64_t off) {
            const int lt = new_lt(syn_type_bf16, { (uint64_t) (ne0 * ne1 * ne2) });
            params_t pr;
            pr.i(nmem_for(ne1 * ne2 * ((ne0 + 127) / 128), 8)).i(ne0).i(ne1).i(ne2).i(1)
              .i(off).i(ne0 > 1 ? o.nb[0] : 1).i(o.nb[1]).i(o.nb[2]).i(o.nb[3]);
            const char * g = st == GGML_TYPE_F32 ? "syn_cpy_f32_bf16_f32" : st == GGML_TYPE_F16 ? "syn_cpy_f16_bf16_f32" :
                                                                                               "syn_cpy_bf16_bf16_f32";
            add_node(g, { o.lt }, { lt }, pr);
            return lt;
        };
        const uint64_t uD = (uint64_t) D, unq = (uint64_t) nq, unkv = (uint64_t) nkv, uhpg = (uint64_t) hpg, uHkv = (uint64_t) Hkv;
        auto fq = fa_q_bf.find(n);
        const int lq = reshape(fq != fa_q_bf.end() ? fq->second : to_bf16(oq, q->type, D, nq, H, oq.off), syn_type_bf16,
                               { uD, unq, uhpg, uHkv });
        const int lk = reshape(to_bf16(ok, k->type, D, nkv, Hkv, ok.off), syn_type_bf16, { uD, unkv, 1, uHkv });
        const int lv = reshape(to_bf16(ov, v->type, D, nkv, Hkv, ov.off), syn_type_bf16, { uD, unkv, 1, uHkv });
        const int lm = reshape(to_bf16(om, mk->type, nkv, nq, 1, om.off), syn_type_bf16, { unkv, unq, 1, 1 });
        // f32 output up to 1024 queries (one bf16 rounding less), bf16 above (the slicer limit)
        const bool o32 = nq <= 1024;
        const synDataType odt = o32 ? syn_type_single : syn_type_bf16;
        const int lo = new_lt(odt, { uD, unq, uhpg, uHkv });
        // ns_Sdpa::Params: scale, is_causal, dropout {ratio, seed, disable mask out}, is_inference, softmax mode
        // (GGML_SYNAPSE_FA_SDPA_SOFTMAX: 0 default, 4 f32)
        static const int sdpa_sm = getenv("GGML_SYNAPSE_FA_SDPA_SOFTMAX") ? atoi(getenv("GGML_SYNAPSE_FA_SDPA_SOFTMAX")) : 0;
        params_t ps;
        ps.f(scale).i(0).f(0.0f).i(0).i(0).i(1).i(sdpa_sm);
        add_node("sdpa_recomp_fwd_bf16", { lq, lk, lv, lm }, { lo }, ps);
        // [D, nq, H] -> [D, H, nq] f32
        const int out = begin_out(n);
        params_t pc;
        pc.i(nmem_for(H * nq * ((D + 127) / 128), 8)).i(D).i(H).i(nq).i(1).i(0).i(1).i(D * nq).i(D).i(0);
        add_node(o32 ? "syn_cpy_f32_f32_f32" : "syn_cpy_bf16_f32_f32", { reshape(lo, odt, { (uint64_t) (D * nq * H) }) }, { out }, pc);
        return end_out(n, out);
    }
    // dense copies [ne0, ne1, ne2, ne3] in dt of a view at offset off. kv_dep: the copy also takes the K / V caches (not
    // read), so it and the GEMMs after it wait for this graph's cache writes
    auto dense = [&](const opnd & o, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int64_t off, bool from_f32,
                     bool kv_dep = false) {
        const uint64_t ne = (uint64_t) (ne0 * ne1 * ne2 * ne3);
        if (!from_f32 && off == 0 && o.nb[0] == 1 && o.nb[1] == ne0 && o.nb[2] == ne0 * ne1 &&
            o.nb[3] == ne0 * ne1 * ne2 && p.lts[o.lt].dims == 1 && p.lts[o.lt].sz[0] == ne) {
            return o.lt;
        }
        const int lt = new_lt(dt, { ne });
        params_t pr;
        pr.i(nmem_for(ne1 * ne2 * ne3 * ((ne0 + 127) / 128), 8)).i(ne0).i(ne1).i(ne2).i(ne3)
          .i(off).i(ne0 > 1 ? o.nb[0] : 1).i(o.nb[1]).i(o.nb[2]).i(o.nb[3]);
        const char * g = from_f32 ? (kv_dep ? (bf ? "syn_cpy_f32_bf16_kv_f32" : "syn_cpy_f32_f16_kv_f32") :
                                              (bf ? "syn_cpy_f32_bf16_f32" : "syn_cpy_f32_f16_f32")) :
                                    (bf ? "syn_cpy_bf16_bf16_f32" : "syn_cpy_f16_f16_f32");
        add_node(g, kv_dep ? std::vector<int>{ o.lt, ok.lt, ov.lt } : std::vector<int>{ o.lt }, { lt }, pr);
        return lt;
    };
    // V the leading Dv columns of K's rows (MLA): P V from K's rows, the extra columns dropped at the end
    const bool v_in_k = Dv < D && v->type == k->type && v->view_src && v->view_src == k->view_src && v->view_offs == k->view_offs &&
                        v->nb[1] == k->nb[1] && v->nb[2] == k->nb[2] && v->nb[3] == k->nb[3];
    const int64_t Dw = v_in_k ? D : Dv;   // rows of the P V product
    const uint64_t nb = (uint64_t) (Hkv * B);
    // K / V [ne0, nkv, Hkv, B] one dense block of a cache (one KV head: MLA, MQA): read in place, the queries' copy of
    // each block takes the caches. Else copied dense (GQA caches interleave the heads per cell)
    auto in_cache = [&](const opnd & o, int64_t ne0, uint64_t & addr) {
        return window_addr(o, (uint64_t) (ne0 * nkv) * nb, addr) && o.nb[0] == 1 && (nkv == 1 || o.nb[1] == ne0) && (Hkv == 1 || o.nb[2] == ne0 * nkv) &&
               (B == 1 || o.nb[3] == ne0 * nkv * Hkv);
    };
    uint64_t ak = 0, av = 0;
    const bool k_win = in_cache(ok, D, ak);
    const bool v_win = v_in_k ? k_win : in_cache(ov, Dv, av);
    const uint64_t uD = (uint64_t) D, uDv = (uint64_t) Dv, unkv = (uint64_t) nkv;
    const int lk = k_win ? window(ok, { uD, unkv, nb }, ak) : reshape(dense(ok, D, nkv, Hkv, B, ok.off, false), dt, { uD, unkv, nb });
    const int lv = v_in_k ? lk :
                   v_win  ? window(ov, { uDv, unkv, nb }, av) :
                            reshape(dense(ov, Dv, nkv, Hkv, B, ov.off, false), dt, { uDv, unkv, nb });
    // query blocks: scores within the budget
    int64_t qb = std::max<int64_t>(1, fa_mme_budget() / (nkv * H * B * 4));
    qb = std::min(qb, nq);
    const int64_t nblk = (nq + qb - 1) / qb;
    qb = (nq + nblk - 1) / nblk;
    std::vector<int> parts;
    for (int64_t i0 = 0; i0 < nq; i0 += qb) {
        const int64_t nqb = std::min(qb, nq - i0);
        const uint64_t nc = (uint64_t) (nqb * hpg);
        const int lq = dense(oq, D, nqb, H, B, oq.off + i0 * oq.nb[1], true, k_win || v_win);
        const int s = new_lt(syn_type_single, { (uint64_t) nkv, nc, nb });
        add_node("batch_gemm", { reshape(lq, dt, { (uint64_t) D, nc, nb }), lk }, { s }, gemm_params(false, true));
        const int pm = softmax(s, i0, nqb);
        const int o = new_lt(syn_type_single, { (uint64_t) Dw, nc, nb });
        add_node("batch_gemm", { reshape(pm, dt, { (uint64_t) nkv, nc, nb }), lv }, { o }, gemm_params(false, false));
        // the GEMMs are the real readers of windows: a later write of the caches waits for them
        if (k_win) {
            reader_outs[base_of(ok.lt)].push_back(s);
        }
        if (v_win) {
            reader_outs[base_of(ov.lt)].push_back(o);
        }
        // [Dw, nqb, H, B] -> [Dv, H, nqb, B] (the first Dv of each row)
        const int lo = reshape(o, syn_type_single, { (uint64_t) Dw * nc * nb });
        const bool last = nblk == 1;
        const int pt = last ? begin_out(n) : new_lt(syn_type_single, { (uint64_t) (Dv * H * nqb * B) });
        params_t pc;
        if (lsum >= 0) {
            // 1 / row sum of out (d, h, i, b): lsum[h + H (i + nqb b)]
            pc.i(nmem_for(H * nqb * B * ((Dv + 127) / 128), 8)).i(Dv).i(H).i(nqb).i(B).i(0).i(Dw * nqb).i(Dw).i(Dw * nqb * H)
              .i(1).i(H).i(nqb * H);
            add_node("syn_fa_norm_f32", { lo, lsum }, { pt }, pc);
        } else {
            pc.i(nmem_for(H * nqb * B * ((Dv + 127) / 128), 8)).i(Dv).i(H).i(nqb).i(B).i(0).i(1).i(Dw * nqb).i(Dw).i(Dw * nqb * H);
            add_node("syn_cpy_f32_f32_f32", { lo }, { pt }, pc);
        }
        if (last) {
            return end_out(n, pt);
        }
        parts.push_back(reshape(pt, syn_type_single, { (uint64_t) Dv, (uint64_t) H, (uint64_t) nqb, (uint64_t) B }));
    }
    // blocks joined along the query dim
    const bool out_direct = n->view_src == nullptr && is_output[n];
    const int out4 = out_direct ? new_lt(syn_type_single, { (uint64_t) Dv, (uint64_t) H, (uint64_t) nq, (uint64_t) B }, LT_OUT,
                                         (uint64_t) (uintptr_t) n->data) :
                                  new_lt(syn_type_single, { (uint64_t) Dv, (uint64_t) H, (uint64_t) nq, (uint64_t) B });
    params_t pj;
    pj.i(2);
    add_node("concat", parts, { out4 }, pj);
    const int ly = reshape(out4, syn_type_single, { (uint64_t) (Dv * H * nq * B) });
    if (n->view_src != nullptr) {
        return write_view(n, ly, GGML_TYPE_F32);
    }
    roots[n].lt = ly;
    return true;
}

/* GATED_DELTA_NET of a prefill chunk in the chunked (WY) form:
   per 64-token chunk the intra-chunk work as batched products over all (head, chunk) pairs; the
   chunk-to-chunk state step is sequential.
   Head size 128, one sequence, scalar decay. false (no error set): not applicable */
bool lowerer::lower_gdn_chunked(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov,
                                const opnd & og, const opnd & ob, const opnd & os) {
    static const int64_t tmin = getenv("GGML_SYNAPSE_GDN_CK") ? atoll(getenv("GGML_SYNAPSE_GDN_CK")) : 8;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * g = n->src[3], * st = n->src[5];
    const int64_t S = v->ne[0], nv = v->ne[1], T = v->ne[2], ns = v->ne[3], nk = k->ne[1];
    if (tmin <= 0 || T < tmin || S != 128 || ns != 1 || g->ne[0] != 1 || q->ne[1] != nk || nv % nk != 0) {
        return false;
    }
    constexpr int64_t CH = 64;
    const int64_t NC = (T + CH - 1) / CH;
    const uint64_t uS = (uint64_t) S, uC = CH, unv = (uint64_t) nv, unk = (uint64_t) nk, uNC = (uint64_t) NC;
    auto bg = [&](int x, int w, bool ta, bool tb, std::initializer_list<uint64_t> shape) {
        const int y = new_lt(syn_type_single, shape);
        add_node("batch_gemm", { x, w }, { y }, gemm_params(ta, tb));
        return y;
    };
    // stage 1: chunk blocks
    const int Qk = new_lt(syn_type_single, { uS, uC, unk, uNC }), Kk = new_lt(syn_type_single, { uS, uC, unk, uNC });
    const int Vn = new_lt(syn_type_single, { uS, uC, unv, uNC }), GB = new_lt(syn_type_single, { uC, 2, unv, uNC });
    {
        // gates as [T, nv] (a row of tokens per head)
        auto rows = [&](const opnd & o) {
            int src = o.lt;
            if (!(o.off == 0 && o.nb[1] == 1 && o.nb[2] == nv && p.lts[o.lt].dims == 1 && p.lts[o.lt].sz[0] == (uint64_t) (nv * T))) {
                src = new_lt(syn_type_single, { (uint64_t) (nv * T) });
                params_t pr;
                pr.i(nmem_for(T, 8)).i(nv).i(T).i(1).i(1).i(o.off).i(o.nb[1]).i(o.nb[2]).i(0).i(0);
                add_node("syn_cpy_f32_f32_f32", { o.lt }, { src }, pr);
            }
            return transpose2d(src, nv, T);
        };
        const int gT = rows(og), bT = rows(ob);
        params_t pr;
        pr.i(nmem_for(nv * NC, 1)).i(T).i(NC).i(nk).i(nv).i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(ok.off).i(ok.nb[1]).i(ok.nb[2])
          .i(ov.off).i(ov.nb[1]).i(ov.nb[2]).f(1.0f / sqrtf((float) S));
        add_node("syn_gdn_ck_prep_f32", { oq.lt, ok.lt, ov.lt, gT, bT }, { Qk, Kk, Vn, GB }, pr);
    }
    int Qn = Qk, Kn = Kk;
    if (nv != nk) {
        // value head h reads q/k head h % nk
        Qn = new_lt(syn_type_single, { uS, uC, unv, uNC });
        Kn = new_lt(syn_type_single, { uS, uC, unv, uNC });
        params_t pc;
        pc.i(2);
        add_node("concat", std::vector<int>(nv / nk, Qk), { Qn }, pc);
        add_node("concat", std::vector<int>(nv / nk, Kk), { Kn }, pc);
    }
    // stage 2: Gram matrices -> P, T = (I + A)^-1 with the gate scalings
    const int KK = bg(Kk, Kk, false, true, { uC, uC, unk, uNC });
    const int QK = bg(Qk, Kk, false, true, { uC, uC, unk, uNC });
    int Pm, Tb, Tbg, Tkb, Tkbg;
    Pm = new_lt(syn_type_single, { uC, uC, unv, uNC });
    Tb = new_lt(syn_type_single, { uC, uC, unv, uNC });
    Tbg = new_lt(syn_type_single, { uC, uC, unv, uNC });
    Tkb = new_lt(syn_type_single, { uC, uC, unv, uNC });
    Tkbg = new_lt(syn_type_single, { uC, uC, unv, uNC });
    {
        params_t pr;
        pr.i(nmem_for(nv * NC, 1)).i(nk).i(nv).i(NC);
        add_node("syn_gdn_ck_solve_f32", { KK, QK, GB }, { Pm, Tb, Tbg, Tkb, Tkbg }, pr);
    }
    // stage 3: chunk-local solutions and state-transition terms
    const int U = bg(Tb, Vn, false, false, { uS, uC, unv, uNC });
    const int W = bg(Tbg, Kn, false, false, { uS, uC, unv, uNC });
    const int B = bg(Kn, bg(Tkb, Vn, false, false, { uS, uC, unv, uNC }), true, false, { uS, uS, unv, uNC });
    const int KW = bg(Kn, bg(Tkbg, Kn, false, false, { uS, uC, unv, uNC }), true, false, { uS, uS, unv, uNC });
    // stage 4: the state chain (states row-major [i][j]; ggml keeps them transposed)
    int S0;
    {
        const uint64_t ne = uS * uS * unv;
        int src = os.lt;
        if (!(os.off == 0 && p.lts[os.lt].dims == 1 && p.lts[os.lt].sz[0] == ne)) {
            src = new_lt(syn_type_single, { ne });
            params_t pr;
            pr.i(nmem_for(S * nv * ((S + 127) / 128), 8)).i(S).i(S * nv).i(1).i(1).i(os.off).i(1).i(S).i(0).i(0);
            add_node("syn_cpy_f32_f32_f32", { os.lt }, { src }, pr);
        }
        S0 = reshape(transpose2d(src, S, S, nv), syn_type_single, { uS, uS, unv, 1 });
    }
    auto chunks = [&](int x, std::initializer_list<uint64_t> one) {
        std::vector<int> parts;
        if (NC == 1) {
            parts.push_back(x);
            return parts;
        }
        for (int64_t c = 0; c < NC; c++) {
            parts.push_back(new_lt(syn_type_single, one));
        }
        params_t pc;
        pc.i(3);
        add_node("split", { x }, parts, pc);
        return parts;
    };
    const std::vector<int> KWc = chunks(KW, { uS, uS, unv, 1 }), Bc = chunks(B, { uS, uS, unv, 1 });
    const std::vector<int> GBc = chunks(GB, { uC, 2, unv, 1 });
    std::vector<int> Ss = { S0 };
    int Sc = S0;
    for (int64_t c = 0; c < NC; c++) {
        const int X = bg(KWc[c], Sc, false, false, { uS, uS, unv, 1 });
        const int Sn = new_lt(syn_type_single, { uS, uS, unv, 1 });
        params_t pr;
        pr.i(nmem_for(nv * 16, 4)).i(nv);
        add_node("syn_gdn_ck_step_f32", { Sc, X, Bc[c], GBc[c] }, { Sn }, pr);
        Sc = Sn;
        if (c < NC - 1) {
            Ss.push_back(Sn);
        }
    }
    int Sall = S0;
    if (NC > 1) {
        Sall = new_lt(syn_type_single, { uS, uS, unv, uNC });
        params_t pc;
        pc.i(3);
        add_node("concat", Ss, { Sall }, pc);
    }
    // stage 5: o = gamma (Q S0) + P (U - W S0)
    const int QS = bg(Qn, Sall, false, false, { uS, uC, unv, uNC });
    const int WS = bg(W, Sall, false, false, { uS, uC, unv, uNC });
    const int D = new_lt(syn_type_single, { uS, uC, unv, uNC });
    add_node("sub_fwd_f32", { U, WS }, { D }, params_t());
    const int PD = bg(Pm, D, false, false, { uS, uC, unv, uNC });
    const int St = transpose2d(reshape(Sc, syn_type_single, { uS * uS * unv }), S, S, nv);
    const int out = begin_out(n);
    const int64_t K = ggml_get_op_params_i32(n, 0), snap = S * S * nv;
    // K > 1 (rollback snapshots): slots 1..K-1, the states after tokens T-K..T-2, from a token-loop replay
    // that starts at the chunk boundary at or before token T-K; joined after [outputs | slot 0]
    const int ck = K > 1 ? new_lt(syn_type_single, { (uint64_t) (S * nv * T + snap) }) : out;
    params_t pr;
    pr.i(nmem_for(nv * NC + nv * S / 16, 2)).i(T).i(nv).i(NC);
    add_node("syn_gdn_ck_out_f32", { QS, PD, GB, St }, { ck }, pr);
    if (K > 1) {
        const int64_t c0 = (T - K) / CH;
        const int Sr = reshape(transpose2d(reshape(Ss[c0], syn_type_single, { uS * uS * unv }), S, S, nv),
                               syn_type_single, { uS * uS * unv });
        const int rep = new_lt(syn_type_single, { (uint64_t) ((K - 1) * snap) });
        params_t pt;
        pt.i(nmem_for(nv * S / 4, 4)).i(S).i(nv).i(T).i(1).i(q->ne[1]).i(nk)
          .i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(oq.nb[3])
          .i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ok.nb[3])
          .i(ov.off).i(ov.nb[1]).i(ov.nb[2]).i(ov.nb[3])
          .i(og.off).i(ob.off).i(0).i(0).i(0).f(1.0f / sqrtf((float) S))
          .i(K).i(c0 * CH).i(T - 1).i(0).i(-snap);
        add_node("syn_gdn_f32", { oq.lt, ok.lt, ov.lt, og.lt, ob.lt, Sr }, { rep }, pt);
        params_t pc;
        pc.i(0);
        add_node("concat", { ck, rep }, { out }, pc);
    }
    return end_out(n, out);
}

/* LIGHTNING_INDEXER: per stream an f32 product of the keys with all heads' queries, then
   relu, the per-head weights, the head sum and the mask. Token blocks keep the
   scores [nkv, H, qb, ns] within the attention budget (blocks joined by a native concat) */
bool lowerer::lower_lid(const ggml_tensor * n) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * w = n->src[2], * mk = n->src[3];
    const int64_t D = q->ne[0], H = q->ne[1], T = q->ne[2], ns = q->ne[3], nkv = k->ne[2];
    opnd oq, okk, ow, om;
    if (!get(q, oq) || !get(k, okk) || !get(w, ow) || !get(mk, om)) {
        return false;
    }
    if (oq.nb[0] != 1 || ow.nb[0] != 1 || om.nb[0] != 1 || (k->type != GGML_TYPE_Q8_0 && k->type != GGML_TYPE_Q4_0 &&
        k->type != GGML_TYPE_Q4_1 && k->type != GGML_TYPE_Q5_0 && k->type != GGML_TYPE_Q5_1 && okk.nb[0] != 1)) {
        return fail("lightning_indexer layout: %s", n);
    }
    // keys as dense f32 [D, nkv, ns]
    const uint64_t nk = (uint64_t) (D * nkv * ns);
    int lk = new_lt(syn_type_single, { nk });
    {
        params_t pr;
        if (ggml_is_quantized(k->type)) {
            pr.i(nmem_for(nkv * ns * ((D + 63) / 64), 8)).i(D).i(nkv).i(ns).i(1).i(okk.off).i(okk.nb[2]).i(okk.nb[3]).i(0);
            add_node(qt_kname("syn_dqrows32_", k->type).c_str(), { okk.lt }, { lk }, pr);
        } else {
            pr.i(nmem_for(nkv * ns * ((D + 127) / 128), 8)).i(D).i(nkv).i(ns).i(1).i(okk.off).i(1).i(okk.nb[2]).i(okk.nb[3]).i(0);
            add_node(k->type == GGML_TYPE_F16 ? "syn_cpy_f16_f32_f32" : k->type == GGML_TYPE_BF16 ? "syn_cpy_bf16_f32_f32" :
                     "syn_cpy_f32_f32_f32", { okk.lt }, { lk }, pr);
        }
    }
    const int lk3 = reshape(lk, syn_type_single, { (uint64_t) D, (uint64_t) nkv, (uint64_t) ns });
    int64_t qb = std::max<int64_t>(1, fa_mme_budget() / (nkv * H * ns * 4));
    qb = std::min(qb, T);
    const int64_t nblk = (T + qb - 1) / qb;
    qb = (T + nblk - 1) / nblk;
    std::vector<int> parts;
    for (int64_t t0 = 0; t0 < T; t0 += qb) {
        const int64_t nqb = std::min(qb, T - t0);
        const uint64_t nc = (uint64_t) (H * nqb);
        // queries of the block as dense f32 [D, H nqb, ns]
        const int lq = new_lt(syn_type_single, { (uint64_t) D * nc * (uint64_t) ns });
        {
            params_t pr;
            pr.i(nmem_for(H * nqb * ns * ((D + 127) / 128), 8)).i(D).i(H).i(nqb).i(ns)
              .i(oq.off + t0 * oq.nb[2]).i(1).i(oq.nb[1]).i(oq.nb[2]).i(oq.nb[3]);
            add_node("syn_cpy_f32_f32_f32", { oq.lt }, { lq }, pr);
        }
        const int sc = new_lt(syn_type_single, { (uint64_t) nkv, nc, (uint64_t) ns });
        add_node("batch_gemm", { reshape(lq, syn_type_single, { (uint64_t) D, nc, (uint64_t) ns }), lk3 }, { sc },
                 gemm_params(false, true));
        const bool last = nblk == 1;
        const int pt = last ? begin_out(n) : new_lt(syn_type_single, { (uint64_t) (nkv * nqb * ns) });
        params_t pr;
        pr.i(nmem_for(ns * nqb * ((nkv + 63) / 64), 4)).i(nkv).i(H).i(nqb).i(ns).i(t0)
          .i(ow.off).i(ow.nb[1]).i(ow.nb[3]).i(om.off).i(om.nb[1]).i(om.nb[3]).i(mk->ne[3]);
        add_node("syn_lid_f32", { reshape(sc, syn_type_single, { (uint64_t) nkv * nc * (uint64_t) ns }), ow.lt, om.lt }, { pt }, pr);
        if (last) {
            return end_out(n, pt);
        }
        parts.push_back(reshape(pt, syn_type_single, { (uint64_t) nkv, (uint64_t) nqb, 1, (uint64_t) ns }));
    }
    const bool out_direct = n->view_src == nullptr && is_output[n];
    const int out4 = out_direct ? new_lt(syn_type_single, { (uint64_t) nkv, (uint64_t) T, 1, (uint64_t) ns }, LT_OUT,
                                         (uint64_t) (uintptr_t) n->data) :
                                  new_lt(syn_type_single, { (uint64_t) nkv, (uint64_t) T, 1, (uint64_t) ns });
    params_t pj;
    pj.i(1);
    add_node("concat", parts, { out4 }, pj);
    const int ly = reshape(out4, syn_type_single, { (uint64_t) (nkv * T * ns) });
    if (n->view_src != nullptr) {
        return write_view(n, ly, GGML_TYPE_F32);
    }
    roots[n].lt = ly;
    return true;
}

bool lowerer::lower_flash_attn(const ggml_tensor * n) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = mask_of(n), * snk = n->src[4];
    const float scale = ggml_get_op_params_f32(n, 0);
    const float max_bias = ggml_get_op_params_f32(n, 1);
    const float softcap = ggml_get_op_params_f32(n, 2);
    const int64_t D = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    const int64_t nkv = k->ne[1], Hkv = k->ne[2];
    if (!fa_config_ok(n)) {
        return fail("flash_attn config: %s", n);
    }
    // mixed floating K/V pairs (e.g. F32 K with BF16 V) are not blindly narrowed to F16: the
    // general fixed-memory kernel takes dense F32 operands
    auto is_flt = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; };
    const bool mixed_float = k->type != v->type && is_flt(k->type) && is_flt(v->type);
    opnd oq, ok, ov, om;
    if ((!fa_q_bf.count(n) && !get(q, oq)) || !get(k, ok) || !get(v, ov)) {
        return false;
    }
    if (!mixed_float) {
        const int tr = lower_fa_tile(n, oq, ok, ov);
        if (tr != 0) {
            return tr > 0;
        }
    }
    if (mk && !get(mk, om)) {
        return false;
    }
    const int mme = fa_mme(n);
    ggml_type kt = k->type;
    if (mixed_float) {
        // both dense F32 via the existing helpers; the general kernel reads F32 only
        if (!get_f32(k, ok) || !get_f32(v, ov)) {
            return false;
        }
        kt = GGML_TYPE_F32;
        if (oq.nb[0] != 1 || ok.nb[0] != 1 || ov.nb[0] != 1) {
            return fail("strided dim 0: %s", n);
        }
        return lower_fattn_general(n);
    }
    if (k->type != v->type || (kt != GGML_TYPE_F16 && kt != GGML_TYPE_BF16 && (mme || (kt != GGML_TYPE_F32 && kt != GGML_TYPE_Q8_0)))) {
        // dequantize (convert) to contiguous f16
        auto to_f16 = [&](const ggml_tensor * t, opnd & o) {
            if (t->type == GGML_TYPE_F16) {
                return true;
            }
            const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
            const int lt = new_lt(syn_type_fp16, { (uint64_t) ggml_nelements(t) });
            params_t pr;
            if (t->type == GGML_TYPE_F32) {
                pr.i(nmem_for(ne1 * ne2 * ne3 * ((ne0 + 127) / 128), 8)).i(ne0).i(ne1).i(ne2).i(ne3)
                  .i(o.off).i(o.nb[0]).i(o.nb[1]).i(o.nb[2]).i(o.nb[3]);
                add_node("syn_cpy_f32_f16_f32", { o.lt }, { lt }, pr);
            } else {
                pr.i(nmem_for(ne1 * ne2 * ne3 * ((ne0 + 63) / 64), 8)).i(ne0).i(ne1).i(ne2).i(ne3)
                  .i(o.off).i(o.nb[1]).i(o.nb[2]).i(o.nb[3]);
                add_node(qt_kname("syn_dqrows_", t->type).c_str(), { o.lt }, { lt }, pr);
            }
            o.lt = lt;
            o.off = 0;
            o.nb[0] = 1;
            o.nb[1] = ne0;
            o.nb[2] = ne0 * ne1;
            o.nb[3] = ne0 * ne1 * ne2;
            return true;
        };
        if (!to_f16(k, ok) || !to_f16(v, ov)) {
            return false;
        }
        kt = GGML_TYPE_F16;
    }
    if (ov.lt == ok.lt) {
        // v is a view of k (MLA): a distinct alias, the node must not take one tensor twice
        const lt_t & t = p.lts[ok.lt];
        ov.lt = reshape(ok.lt, t.dt, { t.sz[0] });
    }
    if (mme) {
        return lower_fa_mme(n, oq, ok, ov, om, kt == GGML_TYPE_BF16 ? syn_type_bf16 : syn_type_fp16, mme);
    }
    const bool q8 = kt == GGML_TYPE_Q8_0;
    if (!fa_tpc_config_ok(n)) {
        // mixed floating pairs / unusual widths: the general fixed-memory kernel
        return lower_fattn_general(n);
    }
    if (oq.nb[0] != 1 || (!q8 && (ok.nb[0] != 1 || ov.nb[0] != 1)) || (mk && om.nb[0] != 1)) {
        return fail("flash_attn layout: %s", n);
    }
    const bool wide = D > 256 || Dv > 256;
    const char * g;
    switch (kt) {
        case GGML_TYPE_F16:  g = wide ? "syn_fattn_w_f16_f32" : "syn_fattn_f16_f32";   break;
        case GGML_TYPE_BF16: g = wide ? "syn_fattn_w_bf16_f32" : "syn_fattn_bf16_f32"; break;
        case GGML_TYPE_F32:  g = wide ? "syn_fattn_w_f32_f32" : "syn_fattn_f32_f32";   break;
        default:             g = wide ? "syn_fattn_w_q8_f32" : "syn_fattn_q8_f32";     break;
    }
    // sinks: one extra logit per head; alibi: per-head mask slopes
    const int lsl = max_bias > 0.0f ? alibi_table_lt(H, max_bias) : oq.lt;
    const int64_t hpg = H / Hkv;
    int64_t gsz = hpg % 4 == 0 ? 4 : (hpg % 2 == 0 ? 2 : 1);
    if (B * nq * Hkv < 24) {
        gsz = 1;
    }
    const int64_t base_items = B * (H / gsz) * nq;
    static const bool no_dec = getenv("GGML_SYNAPSE_NO_FA_DEC") != nullptr;
    const bool dec = !no_dec && D == Dv && (D == 64 || D == 128 || D == 256) && B == 1 && gsz == 1 && softcap == 0.0f && kt == GGML_TYPE_F16 &&
                     !snk && max_bias == 0.0f && (!mk || (mk->ne[2] == 1 && mk->ne[3] == 1));
    static const int64_t fin_max = getenv("GGML_SYNAPSE_FA_ONE_CHUNK") ? atoll(getenv("GGML_SYNAPSE_FA_ONE_CHUNK")) : 1024;
    auto gpf = gate_plans.find(n);
    if (dec && D == 256 && gpf != gate_plans.end() && nkv <= fin_max && want_bf16(gpf->second.mul)) {
        opnd og;
        if (!get(gpf->second.gate, og)) {
            return false;
        }
        const int kvl = kvlen_table_lt(mk, nq, nkv);
        const int out = fresh_out(gpf->second.mul);
        params_t pr;
        pr.split(base_items, 1).i(nq).i(H).i(hpg).i(nkv).i(nkv).i(1)
          .f(1.0f).f(1.0f / (float) nq).f(1.0f / (float) hpg)
          .i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ov.off).i(ov.nb[1]).i(ov.nb[2])
          .i(mk ? om.off : 0).i(mk ? om.nb[1] : 0).i(mk ? 1 : 0).f(scale).i(og.off).i(og.nb[1]).i(og.nb[2]);
        add_node("syn_fattn_decg_f32", { oq.lt, ok.lt, ov.lt, mk ? om.lt : oq.lt, kvl, og.lt }, { out }, pr);
        return true;
    }
    static const int64_t fa_items = getenv("GGML_SYNAPSE_FA_ITEMS") ? atoll(getenv("GGML_SYNAPSE_FA_ITEMS")) : 96;
    int64_t chunk = 32;
    while (chunk < 8192 && base_items * ((nkv + chunk - 1) / chunk) > fa_items) {
        chunk *= 2;
    }
    const int64_t nch = (nkv + chunk - 1) / chunk;
    const int64_t nvv = (Dv + 63) / 64;
    const int part = new_lt(syn_type_single, { (uint64_t) (B * H * nq * nch * (nvv * 64 + 128)) });
    if (dec) {
        params_t pr;
        pr.split(base_items * nch, 1).i(nq).i(H).i(hpg).i(nkv).i(chunk).i(nch)
          .f(1.0f / (float) nch).f(1.0f / (float) nq).f(1.0f / (float) hpg)
          .i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ov.off).i(ov.nb[1]).i(ov.nb[2])
          .i(mk ? om.off : 0).i(mk ? om.nb[1] : 0).i(mk ? 1 : 0).f(scale).i(0).i(0).i(0);
        add_node(D == 256 ? "syn_fattn_dec_f32" : D == 128 ? "syn_fattn_dec128_f32" : "syn_fattn_dec64_f32",
                 { oq.lt, ok.lt, ov.lt, mk ? om.lt : oq.lt, kvlen_table_lt(mk, nq, nkv) }, { part }, pr);
    } else {
        params_t pr;
        pr.i(nmem_for(base_items * nch, 1)).i(D | (Dv << 16)).i(nq).i(H).i(B).i(hpg).i(nkv).i(chunk)
          .i(oq.off).i(oq.nb[1]).i(oq.nb[2]).i(oq.nb[3])
          .i(ok.off).i(ok.nb[1]).i(ok.nb[2]).i(ok.nb[3])
          .i(ov.off).i(ov.nb[1]).i(ov.nb[2]).i(ov.nb[3]);
        if (mk) {
            pr.i(om.off).i(om.nb[1]).i(om.nb[2]).i(om.nb[3]).i(mk->ne[2]).i(mk->ne[3]);
        } else {
            pr.i(0).i(0).i(0).i(0).i(1).i(1);
        }
        float sc = scale;
        if (softcap != 0.0f) {
            sc /= softcap;
        }
        static const bool no_kvl = getenv("GGML_SYNAPSE_NO_FA_KVL") != nullptr;
        const bool use_kvl = !no_kvl && mk && nq <= 64 && mk->ne[2] == 1 && mk->ne[3] == 1;
        pr.f(sc).f(softcap).i(gsz | ((mk != nullptr) << 8) | ((max_bias > 0.0f) << 9) | (use_kvl << 10));
        add_node(g, { oq.lt, ok.lt, ov.lt, mk ? om.lt : oq.lt, lsl, use_kvl ? kvlen_table_lt(mk, nq, nkv) : oq.lt }, { part }, pr);
    }
    // sinks[h] read from a tensor starting at sinks[0]
    const int sinks_lt = snk ? flat_f32_lt(snk) : oq.lt;
    if (sinks_lt < 0) {
        return false;
    }
    auto gpit = gate_plans.find(n);
    if (gpit != gate_plans.end()) {
        const ggml_tensor * g = gpit->second.gate;
        opnd og;
        if (!get(g, og)) {
            return false;
        }
        const int out = fresh_out(gpit->second.mul);
        if (D == 256 && Dv == 256 && B == 1 && !snk && roots[gpit->second.mul].bf16 && !getenv("GGML_SYNAPSE_NO_FA_DEC")) {
            params_t pc;
            pc.split(H * nq, 1).i(nq).i(H).i(nch).i(og.off).i(og.nb[1]).i(og.nb[2]).f(1.0f / (float) H);
            add_node("syn_fattn_combg_c_f32", { part, og.lt }, { out }, pc);
            return true;
        }
        params_t pr;
        pr.i(nmem_for(B * H * nq, 1)).i(Dv).i(nq).i(H).i(B).i(nch).i(snk != nullptr).i(og.off).i(og.nb[1]).i(og.nb[2]).i(og.nb[3]);
        add_node(roots[gpit->second.mul].bf16 ? "syn_fattn_combg_bf_f32" : "syn_fattn_combg_f32", { part, og.lt, sinks_lt }, { out }, pr);
        return true;
    }
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(B * H * nq, 1)).i(Dv).i(nq).i(H).i(B).i(nch).i(snk != nullptr);
    add_node(Dv > 256 ? "syn_fattn_comb_w_f32" : "syn_fattn_comb_f32", { part, sinks_lt }, { out }, pr);
    return end_out(n, out);
}

/* tiled decode attention: shapes and op params the window recipe handles (one sequence of <= 31 queries, f16 / bf16
 * caches holding all KV heads of a cell in one row as lower_fa_mme mode 2, an f16 mask shared by the heads, optional
 * sinks; no ALiBi or softcap). The caches' placement and the mask's content are checked later. */
bool lowerer::fa_tile_shape_ok(const ggml_tensor * n) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = n->src[3], * snk = n->src[4];
    if (fa_tile_w() == 0 || !mk || !k || !v) {
        return false;
    }
    const int64_t D = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2], B = q->ne[3];
    const int64_t Hkv = k->ne[2];
    const size_t es = ggml_type_size(k->type);
    // batches of > 31 queries: the native SDPA per window (D == Dv in 64 / 128 / 256, no sinks), else the MME form as
    // decode (up to 8192 query rows x heads; GGML_SYNAPSE_FA_TILE_MME_BATCH=0: untiled)
    static const bool mme_batch = !getenv("GGML_SYNAPSE_FA_TILE_MME_BATCH") || atoi(getenv("GGML_SYNAPSE_FA_TILE_MME_BATCH")) != 0;
    if (nq > 1024 || (nq > 31 && !fa_tile_sdpa(n) && (!mme_batch || nq * H > 8192))) {
        return false;
    }
    return ggml_get_op_params_f32(n, 1) == 0.0f && ggml_get_op_params_f32(n, 2) == 0.0f && B == 1 &&
           Hkv > 0 && H % Hkv == 0 && q->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 && k->ne[0] == D &&
           k->type == v->type && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && Dv % 64 == 0 &&
           v->ne[2] == Hkv && k->nb[0] == es && k->nb[2] == D * es && k->nb[1] == D * Hkv * es && v->nb[0] == es &&
           v->nb[2] == Dv * es && v->nb[1] == Dv * Hkv * es && (D * Hkv) % 128 == 0 && (Dv * Hkv) % 128 == 0 &&
           mk->type == GGML_TYPE_F16 && mk->ne[2] == 1 && mk->ne[3] == 1 && mk->ne[1] >= nq && mk->ne[0] == k->ne[1] &&
           (!snk || (snk->type == GGML_TYPE_F32 && ggml_nelements(snk) >= H && ggml_is_contiguous(snk)));
}

/* tiled decode attention: 1 lowered, 0 not applicable, -1 error */
int lowerer::lower_fa_tile(const ggml_tensor * n, const opnd & oq, const opnd & ok, const opnd & ov) {
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = mask_of(n), * snk = n->src[4];
    const bool pre = fa_tile_sdpa(n);
    // the window tables come from the host's copy of the mask: a mask computed in this graph (e.g. a DSA selection) has none
    if (no_tile || !fa_tile_shape_ok(n) || mk != n->src[3] || produced.count(root_of(mk)) || (fa_q_bf.count(n) && !pre) ||
        gate_plans.count(n) ||
        (!fa_q_bf.count(n) && oq.nb[0] != 1)) {
        return 0;
    }
    const int64_t D = q->ne[0], Dv = v->ne[0], nq = q->ne[1], H = q->ne[2];
    const int64_t Hkv = k->ne[2], hpg = H / Hkv;
    const size_t es = ggml_type_size(k->type);
    const synDataType dt = k->type == GGML_TYPE_BF16 ? syn_type_bf16 : syn_type_fp16;
    const lt_t & kc = p.lts[base_of(ok.lt)], & vc = p.lts[base_of(ov.lt)];
    const uint64_t kaddr = kc.addr + (uint64_t) ok.off * es, vaddr = vc.addr + (uint64_t) ov.off * es;
    const int64_t R = D * Hkv, Rv = Dv * Hkv, C = nq * H;
    if (kc.kind != LT_IN || vc.kind != LT_IN || kc.region == 0 || vc.region == 0 || kc.dt != dt || vc.dt != dt ||
        kaddr % 128 != 0 || vaddr % 128 != 0) {
        return 0;
    }
    // cells from the view's first cell to the cache end: the windows stay inside the cache
    const int64_t cells = std::min<int64_t>((int64_t) ((kc.addr + kc.nbytes - kaddr) / (R * es)),
                                            (int64_t) ((vc.addr + vc.nbytes - vaddr) / (Rv * es)));
    // a small cache (sliding window) takes windows of its own size
    const int64_t W = std::min<int64_t>(fa_tile_w(), cells / 256 * 256);
    if (W < 256 || cells < mk->ne[0]) {
        return 0;
    }
    int g = -1;
    for (size_t i = 0; i < tile_groups.size(); i++) {
        if (tile_groups[i].mask == mk && tile_groups[i].W == W && tile_groups[i].cells == cells) {
            g = (int) i;
        }
    }
    if (g < 0) {
        tile_group_t tg;
        tg.mask = mk;
        tg.W = W;
        tg.cells = cells;
        // window sizes: decode W, 4 W, 16 W; batches W / 4, W, 4 W
        const bool batch = nq > 31;
        for (int64_t m = batch ? 1 : 4; m <= (batch ? 16 : 64) && W * m / 4 <= cells; m *= 4) {
            if (W * m / 4 >= 256) {
                tg.sizes.push_back(W * m / 4);
            }
        }
        tg.tmax = (cells + tg.sizes[0] - 1) / tg.sizes[0];
        tg.entry = ((uint64_t) (2 + 4 * nq) * 4 + 255) / 256 * 256;
        tg.tab_off = arena_alloc((uint64_t) tg.tmax * tg.entry);
        host_table ht;
        ht.arena_off = tg.tab_off;
        ht.nbytes = (uint64_t) tg.tmax * tg.entry;
        ggml_backend_synapse_device_context * dc = dctx;
        const uint64_t maddr = (uint64_t) (uintptr_t) mk->data, mrow = mk->nb[1];
        const int64_t nkv = mk->ne[0], tmax = tg.tmax, ent = (int64_t) tg.entry / 4;
        const std::vector<int64_t> sizes = tg.sizes;
        auto win = tg.win;
        auto bad = tile_bad;
        // per query row the valid keys must be at most two ranges of zeros (a ring-buffer cache that wrapped), -inf
        // elsewhere (else: the untiled program). The keys [bot, top) any row reads are covered by the fewest windows: the
        // largest size not past top rounded up to the smallest size
        ht.fill = [dc, maddr, mrow, nq, nkv, W, cells, tmax, ent, sizes, win, bad](void * dst) {
            static const bool dbg = getenv("GGML_SYNAPSE_FA_TILE_DEBUG") != nullptr;
            int32_t * T = (int32_t *) dst;
            memset(T, 0, (size_t) (tmax * ent * 4));
            std::vector<uint16_t> row(nkv);
            std::vector<int64_t> rr(4 * nq, 0);   // per row lo1 hi1 lo2 hi2
            int64_t bot = nkv, top = 0;
            for (int64_t i = 0; i < nq; i++) {
                if (!dc->shadow_get(maddr + i * mrow, row.data(), nkv * sizeof(uint16_t))) {
                    *bad = 1;   // the mask is not host-visible: the untiled program
                    return true;
                }
                // f16 bits: -inf 0xFC00, +-0 0x0000 / 0x8000; runs of zeros between -inf
                int nrun = 0;
                bool other = false;
                for (int64_t j = 0; j < nkv;) {
                    if (row[j] == 0xFC00) {
                        j++;
                        continue;
                    }
                    const int64_t l = j;
                    while (j < nkv && row[j] != 0xFC00) {
                        other |= (row[j] & 0x7FFF) != 0;
                        j++;
                    }
                    if (nrun < 2) {
                        rr[4 * i + 2 * nrun] = l;
                        rr[4 * i + 2 * nrun + 1] = j;
                    }
                    nrun++;
                    bot = std::min(bot, l);
                    top = std::max(top, j);
                }
                if (nrun > 2 || other) {
                    if (dbg && !*bad) {
                        fprintf(stderr, "synapse: tiled attention: mask row %lld has %d ranges%s: untiled\n", (long long) i,
                                nrun, other ? " and non-zero values" : "");
                    }
                    *bad = 1;   // holes or a bias: the mask is needed
                }
            }
            if (top == 0) {
                bot = 0;
                top = 1;
            }
            win->clear();
            const int64_t G = sizes[0];
            for (int64_t a = bot / G * G; a < top && (int64_t) win->size() / 2 < tmax;) {
                const int64_t need = (top - a + G - 1) / G * G;
                size_t si = 0;
                while (si + 1 < sizes.size() && sizes[si + 1] <= need) {
                    si++;
                }
                const int64_t sz = sizes[si], start = std::min(a, cells - sz);
                int32_t * e = T + (win->size() / 2) * ent;
                e[0] = win->empty();
                for (int64_t i = 0; i < nq; i++) {
                    for (int r = 0; r < 2; r++) {
                        const int64_t x = std::max(rr[4 * i + 2 * r], a), y = std::min(rr[4 * i + 2 * r + 1], a + sz);
                        e[2 + 4 * i + 2 * r] = (int32_t) (std::min(x, y) - start);
                        e[3 + 4 * i + 2 * r] = (int32_t) (std::max(x, y) - start);
                    }
                }
                win->push_back(start);
                win->push_back((int64_t) si);
                a += sz;
            }
            return true;
        };
        p.tables.push_back(ht);
        tile_groups.push_back(tg);
        g = (int) tile_groups.size() - 1;
    }
    const bool bf = dt == syn_type_bf16;
    fa_tile_t t;
    t.group = g;
    t.W = W;
    if (snk) {
        // the sink logits copied to a tensor the window recipe reads (made persistent by the split)
        opnd os;
        if (!get(snk, os)) {
            return -1;
        }
        t.sinks = new_lt(syn_type_single, { (uint64_t) H });
        params_t pc;
        pc.i(1).i(H).i(1).i(1).i(1).i(os.off).i(1).i(0).i(0).i(0);
        add_node("syn_cpy_f32_f32_f32", { os.lt }, { t.sinks }, pc);
    }
    t.pre = pre;
    t.D = D;
    t.Hkv = Hkv;
    // per window: decode keeps acc [Rv, C] + ml [4 C]; prefill acc [D, nq H] + ml [2 nq H]
    const uint64_t acc_n = pre ? (uint64_t) (D * nq * H) : (uint64_t) (Rv * C);
    const uint64_t ml_n = pre ? (uint64_t) (2 * nq * H) : (uint64_t) (4 * C);
    if (pre) {
        // bf16 Q [D, nq, H] for the native SDPA: the rope's bf16 output, or a copy
        auto fq = fa_q_bf.find(n);
        if (fq != fa_q_bf.end()) {
            t.qbd = fq->second;
        } else {
            t.qbd = new_lt(syn_type_bf16, { (uint64_t) (D * nq * H) });
            params_t pr;
            pr.i(nmem_for(nq * H * ((D + 127) / 128), 8)).i(D).i(nq).i(H).i(1)
              .i(oq.off).i(oq.nb[0]).i(oq.nb[1]).i(oq.nb[2]).i(oq.nb[3]);
            add_node("syn_cpy_f32_bf16_f32", { oq.lt }, { t.qbd }, pr);
        }
    } else {
        t.qbd = new_lt(dt, { (uint64_t) (R * C) });
        params_t pq;
        pq.i(nmem_for(C * ((R + 127) / 128), 4)).i(D).i(Hkv).i(H).i(nq).i(hpg).i(oq.off).i(oq.nb[1]).i(oq.nb[2]);
        add_node(bf ? "syn_fa_qbd_b_f32" : "syn_fa_qbd_h_f32", { oq.lt, ok.lt, ov.lt }, { t.qbd }, pq);
    }
    t.cut = p.nodes.size();
    for (int i = 0; i < 2; i++) {
        t.acc_ab[i] = arena_alloc(acc_n * 4);
        t.ml_ab[i] = arena_alloc(ml_n * 4);
    }
    t.acc_f = new_lt(syn_type_single, { acc_n }, LT_ARENA);
    p.lts[t.acc_f].addr = arena_alloc(acc_n * 4);
    t.ml_f = new_lt(syn_type_single, { ml_n }, LT_ARENA);
    p.lts[t.ml_f].addr = arena_alloc(ml_n * 4);
    t.kaddr = kaddr;
    t.vaddr = vaddr;
    t.R = R;
    t.Rv = Rv;
    t.C = C;
    t.nq = nq;
    t.H = H;
    t.hpg = hpg;
    t.Dv = Dv;
    t.dt = dt;
    t.scale = ggml_get_op_params_f32(n, 0);
    fa_tiles.push_back(t);
    if (pre) {
        // acc is the normalized output [D, nq, H]: -> [D, H, nq] (as lower_fa_mme's SDPA path)
        const int out = begin_out(n);
        params_t pc;
        pc.i(nmem_for(H * nq * ((D + 127) / 128), 8)).i(D).i(H).i(nq).i(1).i(0).i(1).i(D * nq).i(D).i(0);
        add_node("syn_cpy_f32_f32_f32", { t.acc_f }, { out }, pc);
        return end_out(n, out) ? 1 : -1;
    }
    // out (d, h = hk hpg + g, iq) = acc[hk Dv + d, iq + nq h] / l: as lower_fa_mme mode 2
    const int out = begin_out(n);
    params_t pc;
    pc.i(nmem_for(H * nq * ((Dv + 127) / 128), 8)).i(Dv).i(hpg).i(Hkv).i(nq).i(0)
      .i(nq * Rv).i(Dv + nq * hpg * Rv).i(Rv).i(1).i(hpg).i(H);
    add_node("syn_fa_norm_f32", { t.acc_f, t.ml_f }, { out }, pc);
    return end_out(n, out) ? 1 : -1;
}

/* the program cut at the tiled FAs: segment s = nodes [cut s-1, cut s). Internal values read in a later segment than
 * the one making them become persistent arena tensors; reshape aliases are made again where they are read */
bool lowerer::split_segments(std::vector<program> & segs) {
    const size_t nn = p.nodes.size();
    const int nseg = (int) fa_tiles.size() + 1;
    std::vector<int> seg(nn);
    for (size_t i = 0, s = 0; i < nn; i++) {
        while (s < fa_tiles.size() && i >= fa_tiles[s].cut) {
            s++;
        }
        seg[i] = (int) s;
    }
    std::vector<int> prod(p.lts.size(), -1);
    for (size_t i = 0; i < nn; i++) {
        for (int o : p.nodes[i].outs) {
            if (prod[o] < 0) {
                prod[o] = seg[i];
            }
        }
    }
    std::vector<char> persist(p.lts.size(), 0);
    for (const auto & t : fa_tiles) {
        persist[t.qbd] = 1;
        if (t.sinks >= 0) {
            persist[t.sinks] = 1;
        }
    }
    for (size_t i = 0; i < nn; i++) {
        for (int x : p.nodes[i].ins) {
            if (p.lts[x].kind != LT_INTERNAL || prod[x] == seg[i] || prod[x] < 0) {
                continue;
            }
            if (prod[x] > seg[i]) {
                err = "tile split: a value read before the segment making it";
                return false;
            }
            const int b = base_of(x);
            if (p.lts[b].kind == LT_INTERNAL) {
                persist[b] = 1;
            }
        }
    }
    for (size_t i = 0; i < p.lts.size(); i++) {
        if (persist[i]) {
            lt_t & t = p.lts[i];
            t.kind = LT_ARENA;
            t.addr = arena_alloc(t.nbytes);
            t.region = 0;
            t.sec = -1;
            t.sec_off = 0;
        }
    }
    segs.assign(nseg, program());
    for (int s = 0; s < nseg; s++) {
        program & q = segs[s];
        std::vector<int> map(p.lts.size(), -1);
        std::vector<ln_t> pre;
        std::function<int(int)> use = [&](int x) -> int {
            if (map[x] >= 0) {
                return map[x];
            }
            lt_t c = p.lts[x];
            c.base = -1;
            if (c.kind == LT_INTERNAL && prod[x] >= 0 && prod[x] != s) {
                const int lb = use(base_of(x));
                q.lts.push_back(c);
                map[x] = (int) q.lts.size() - 1;
                ln_t r;
                r.guid = "reshape";
                r.ins = { lb };
                r.outs = { map[x] };
                pre.push_back(r);
                return map[x];
            }
            q.lts.push_back(c);
            map[x] = (int) q.lts.size() - 1;
            return map[x];
        };
        std::vector<ln_t> body;
        std::vector<int> local(nn, -1);
        for (size_t i = 0; i < nn; i++) {
            if (seg[i] != s) {
                continue;
            }
            ln_t m = p.nodes[i];
            for (int & x : m.ins) {
                x = use(x);
            }
            for (int & x : m.outs) {
                x = use(x);
            }
            local[i] = (int) body.size();
            body.push_back(std::move(m));
        }
        q.nodes = pre;
        for (auto & m : body) {
            std::vector<int> deps;
            for (int d : m.deps) {
                if (local[d] >= 0) {
                    deps.push_back(local[d] + (int) pre.size());
                }
            }
            m.deps = deps;
            q.nodes.push_back(std::move(m));
        }
        q.arena_size = p.arena_size;
    }
    return true;
}

/* the tile recipe: one W-key window of the K / V caches against the queries, merged into the running state */
static program fa_tile_program(int64_t R, int64_t Rv, int64_t C, int64_t nq, int64_t H, int64_t hpg, int64_t Dv, int64_t W,
                               synDataType dt, float scale, bool sinks, uint64_t tab_ints) {
    program q;
    auto lt = [&](synDataType d, std::initializer_list<uint64_t> sz, int kind) {
        lt_t t;
        t.dt = d;
        t.dims = (unsigned) sz.size();
        int k = 0;
        uint64_t nb = syn_type_size(d);
        for (uint64_t s : sz) {
            t.sz[k++] = s;
            nb *= s;
        }
        t.kind = kind;
        t.nbytes = nb;
        q.lts.push_back(t);
        return (int) q.lts.size() - 1;
    };
    auto node = [&](const char * g, std::vector<int> ins, std::vector<int> outs, const params_t & pr) {
        ln_t n;
        n.guid = g;
        n.ins = ins;
        n.outs = outs;
        n.params.resize(pr.bytes ? pr.bytes : pr.v.size() * 4);
        memcpy(n.params.data(), pr.v.data(), n.params.size());
        q.nodes.push_back(n);
    };
    const uint64_t uR = R, uRv = Rv, uC = C, uW = W;
    const int qbd = lt(dt, { uR, uC }, LT_ARENA);
    const int kw = lt(dt, { uR, uW }, LT_IN);
    const int vw = lt(dt, { uRv, uW }, LT_IN);
    const int tab = lt(syn_type_int32, { tab_ints }, LT_ARENA);
    const int acc_in = lt(syn_type_single, { uRv * uC }, LT_ARENA);
    const int ml_in = lt(syn_type_single, { 4 * uC }, LT_ARENA);
    const int ml_out = lt(syn_type_single, { 4 * uC }, LT_ARENA);
    const int acc_out = lt(syn_type_single, { uRv * uC }, LT_ARENA);
    const int lsk = sinks ? lt(syn_type_single, { (uint64_t) H }, LT_ARENA) : -1;   // lt 8
    const int s = lt(syn_type_single, { uW, uC }, LT_INTERNAL);
    const int s1 = lt(syn_type_single, { uW * uC }, LT_INTERNAL);
    const int pm = lt(dt, { uW * uC }, LT_INTERNAL);
    const int pm2 = lt(dt, { uW, uC }, LT_INTERNAL);
    const int o = lt(syn_type_single, { uRv, uC }, LT_INTERNAL);
    const int o1 = lt(syn_type_single, { uRv * uC }, LT_INTERNAL);
    const int dep1 = lt(syn_type_int32, { tab_ints }, LT_INTERNAL);
    const int dep2 = lt(syn_type_int32, { tab_ints }, LT_INTERNAL);
    const params_t none;
    node("reshape", { tab }, { dep1 }, none);
    node("reshape", { tab }, { dep2 }, none);
    node("gemm", { qbd, kw }, { s }, gemm_params(false, true));
    node("reshape", { s }, { s1 }, none);
    params_t ps;
    ps.i(nmem_for(C, 1)).i(W).i(C).i(nq).i(H).f(scale);
    if (sinks) {
        node(dt == syn_type_bf16 ? "syn_fat_sm_bs_f32" : "syn_fat_sm_hs_f32", { s1, tab, ml_in, lsk, dep1 }, { pm, ml_out }, ps);
    } else {
        node(dt == syn_type_bf16 ? "syn_fat_sm_b_f32" : "syn_fat_sm_h_f32", { s1, tab, ml_in, dep1 }, { pm, ml_out }, ps);
    }
    node("reshape", { pm }, { pm2 }, none);
    node("gemm", { pm2, vw }, { o }, gemm_params(false, false));
    node("reshape", { o }, { o1 }, none);
    params_t pr;
    pr.i(nmem_for(C, 1)).i(Rv).i(C).i(nq).i(hpg).i(Dv);
    node("syn_fat_mrg_f32", { o1, ml_out, acc_in, tab, dep2 }, { acc_out }, pr);
    return q;
}

/* the prefill window recipe: the window's K / V rows as bf16 per KV head, its mask from the table, the native SDPA with
 * softmax statistics, merged into the running normalized output */
static program fa_tile_program_pre(int64_t D, int64_t nq, int64_t hpg, int64_t Hkv, int64_t W, synDataType dt, float scale,
                                   uint64_t tab_ints) {
    program q;
    auto lt = [&](synDataType d, std::initializer_list<uint64_t> sz, int kind) {
        lt_t t;
        t.dt = d;
        t.dims = (unsigned) sz.size();
        int k = 0;
        uint64_t nb = syn_type_size(d);
        for (uint64_t s : sz) {
            t.sz[k++] = s;
            nb *= s;
        }
        t.kind = kind;
        t.nbytes = nb;
        q.lts.push_back(t);
        return (int) q.lts.size() - 1;
    };
    auto node = [&](const char * g, std::vector<int> ins, std::vector<int> outs, const params_t & pr) {
        ln_t n;
        n.guid = g;
        n.ins = ins;
        n.outs = outs;
        n.params.resize(pr.bytes ? pr.bytes : pr.v.size() * 4);
        memcpy(n.params.data(), pr.v.data(), n.params.size());
        q.nodes.push_back(n);
    };
    const uint64_t uD = D, unq = nq, uhpg = hpg, uHkv = Hkv, uW = W, R = D * Hkv, rows = nq * hpg * Hkv;
    const int qb = lt(syn_type_bf16, { uD, unq, uhpg, uHkv }, LT_ARENA);
    const int kw = lt(dt, { R * uW }, LT_IN);
    const int vw = lt(dt, { R * uW }, LT_IN);
    const int tab = lt(syn_type_int32, { tab_ints }, LT_ARENA);
    const int acc_in = lt(syn_type_single, { uD * rows }, LT_ARENA);
    const int ml_in = lt(syn_type_single, { 2 * rows }, LT_ARENA);
    const int ml_out = lt(syn_type_single, { 2 * rows }, LT_ARENA);
    const int acc_out = lt(syn_type_single, { uD * rows }, LT_ARENA);
    const int kb = lt(syn_type_bf16, { uD * uW * uHkv }, LT_INTERNAL);
    const int vb = lt(syn_type_bf16, { uD * uW * uHkv }, LT_INTERNAL);
    const int kb4 = lt(syn_type_bf16, { uD, uW, 1, uHkv }, LT_INTERNAL);
    const int vb4 = lt(syn_type_bf16, { uD, uW, 1, uHkv }, LT_INTERNAL);
    const int mk = lt(syn_type_bf16, { uW * unq }, LT_INTERNAL);
    const int mk4 = lt(syn_type_bf16, { uW, unq, 1, 1 }, LT_INTERNAL);
    const int o = lt(syn_type_single, { uD, unq, uhpg, uHkv }, LT_INTERNAL);
    const int mx = lt(syn_type_single, { 1, unq, uhpg, uHkv }, LT_INTERNAL);
    const int li = lt(syn_type_single, { 1, unq, uhpg, uHkv }, LT_INTERNAL);
    const int o1 = lt(syn_type_single, { uD * rows }, LT_INTERNAL);
    const int mx1 = lt(syn_type_single, { rows }, LT_INTERNAL);
    const int li1 = lt(syn_type_single, { rows }, LT_INTERNAL);
    int dep[4];
    const params_t none;
    for (int i = 0; i < 4; i++) {
        dep[i] = lt(syn_type_int32, { tab_ints }, LT_INTERNAL);
        node("reshape", { tab }, { dep[i] }, none);
    }
    // the window's rows [D Hkv] per cell -> [D, W, Hkv] bf16
    const char * cpy = dt == syn_type_bf16 ? "syn_cpy_bf16_bf16_f32" : "syn_cpy_f16_bf16_f32";
    params_t pk;
    pk.i(nmem_for(W * Hkv * ((D + 127) / 128), 8)).i(D).i(W).i(Hkv).i(1).i(0).i(1).i(R).i(D).i(R * W);
    node(cpy, { kw, dep[0] }, { kb }, pk);
    node(cpy, { vw, dep[1] }, { vb }, pk);
    node("reshape", { kb }, { kb4 }, none);
    node("reshape", { vb }, { vb4 }, none);
    params_t pm;
    pm.i(nmem_for(nq, 4)).i(W).i(nq);
    node("syn_fat_mask_f32", { tab, dep[2] }, { mk }, pm);
    node("reshape", { mk }, { mk4 }, none);
    params_t ps;   // ns_Sdpa::Params: scale, is_causal, dropout {ratio, seed, disable mask out}, is_inference, softmax mode
    ps.f(scale).i(0).f(0.0f).i(0).i(1).i(0).i(0);
    node("sdpa_recomp_fwd_bf16", { qb, kb4, vb4, mk4 }, { o, mx, li }, ps);
    node("reshape", { o }, { o1 }, none);
    node("reshape", { mx }, { mx1 }, none);
    node("reshape", { li }, { li1 }, none);
    params_t pr;
    pr.i(nmem_for((int64_t) rows, 32)).i(D).i((int64_t) rows);
    node("syn_fat_pmrg_f32", { o1, mx1, li1, acc_in, ml_in, tab, dep[3] }, { acc_out, ml_out }, pr);
    return q;
}

/* rope: rotation on device, cos/sin table computed on the host from the position input
 * (bit-identical to the CPU backend's cache) and uploaded before each launch */
static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1 - std::min(1.0f, std::max(0.0f, y));
}

static void rope_yarn(float theta_extrap, float freq_scale, const float corr_dims[2], int64_t i0, float ext_factor,
                      float mscale, float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

struct rope_cfg {
    int n_dims, mode, n_ctx_orig;
    int64_t ne0;               // row length (vision ropes the whole row)
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    bool truncate;
    int sections[4];
    int64_t ntok;
    std::vector<float> ff;     // freq factors (empty: none)
};

/* table row t from positions pos (ntok, or 4*ntok for mrope):
 *   NEOX / MROPE / IMROPE: [cos(i) | sin(i)], i < n_dims/2 (row n_dims)
 *   VISION:                [cos(i) | sin(i)], i < n_dims (row 2*n_dims = ne0; sections restart theta)
 *   NORMAL:                [cos(i) twice per pair | -sin(i), sin(i)] (row 2*n_dims) */
static int64_t rope_row(const rope_cfg & c) {
    return c.mode == GGML_ROPE_TYPE_NEOX || c.mode == GGML_ROPE_TYPE_MROPE || c.mode == GGML_ROPE_TYPE_IMROPE ? c.n_dims : 2 * c.n_dims;
}

static void rope_table(const rope_cfg & c, const int32_t * pos, float * out) {
    const float theta_scale = powf(c.freq_base, -2.0f / c.n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(c.n_dims, c.n_ctx_orig, c.freq_base, c.beta_fast, c.beta_slow, c.truncate, corr_dims);
    const bool mrope = c.mode & GGML_ROPE_TYPE_MROPE;
    const bool imrope = c.mode == GGML_ROPE_TYPE_IMROPE;
    const bool vision = c.mode == GGML_ROPE_TYPE_VISION;
    const bool normal = c.mode == GGML_ROPE_TYPE_NORMAL;
    const int64_t rl = rope_row(c);
    const int np = vision ? c.n_dims : c.n_dims / 2;   // rotated pairs
    for (int64_t t = 0; t < c.ntok; t++) {
        float * row = out + t * rl;
        const float b_t = (float) pos[t];
        const float b_h = mrope ? (float) pos[t + c.ntok] : 0.0f;
        const float b_w = mrope ? (float) pos[t + 2 * c.ntok] : 0.0f;
        const float b_e = mrope ? (float) pos[t + 3 * c.ntok] : 0.0f;
        float th_t = b_t, th_h = b_h, th_w = b_w, th_e = b_e;
        const int sect_dims = c.sections[0] + c.sections[1] + c.sections[2] + c.sections[3];
        const int sec_w = c.sections[1] + c.sections[0];
        const int sec_e = c.sections[2] + sec_w;
        for (int i = 0; i < np; i++) {
            const int i0 = 2 * i;
            const float ff = c.ff.empty() ? 1.0f : c.ff[i];
            float theta = th_t;
            if (mrope) {
                const int sector = i % sect_dims;
                if (vision) {
                    // sections restart their theta (indep_sects)
                    if (sector == 0) {
                        th_t = b_t;
                    } else if (sector == c.sections[0]) {
                        th_h = b_h;
                    } else if (sector == sec_w) {
                        th_w = b_w;
                    } else if (sector == sec_e) {
                        th_e = b_e;
                    }
                    theta = th_t;
                }
                if (imrope) {
                    if (sector % 3 == 1 && sector < 3 * c.sections[1]) {
                        theta = th_h;
                    } else if (sector % 3 == 2 && sector < 3 * c.sections[2]) {
                        theta = th_w;
                    } else if (sector % 3 == 0 && sector < 3 * c.sections[0]) {
                        theta = th_t;
                    } else {
                        theta = th_e;
                    }
                } else {
                    if (sector >= c.sections[0] && sector < sec_w) {
                        theta = th_h;
                    } else if (sector >= sec_w && sector < sec_w + c.sections[2]) {
                        theta = th_w;
                    } else if (sector >= sec_w + c.sections[2]) {
                        theta = th_e;
                    }
                }
            }
            float cs, sn;
            rope_yarn(theta / ff, c.freq_scale, corr_dims, i0, c.ext_factor, c.attn_factor, &cs, &sn);
            if (normal) {
                row[i0] = cs;
                row[i0 + 1] = cs;
                row[c.n_dims + i0] = -sn;
                row[c.n_dims + i0 + 1] = sn;
            } else {
                row[i] = cs;
                row[np + i] = sn;
            }
            th_t *= theta_scale;
            th_h *= theta_scale;
            th_w *= theta_scale;
            th_e *= theta_scale;
        }
    }
}

static bool rope_config(const ggml_tensor * n, rope_cfg & c, int & n_offs) {
    const int32_t * op = (const int32_t *) n->op_params;
    c.n_dims = op[1];
    c.mode = op[2];
    c.n_ctx_orig = op[4];
    memcpy(&c.freq_base, op + 5, 4);
    memcpy(&c.freq_scale, op + 6, 4);
    memcpy(&c.ext_factor, op + 7, 4);
    memcpy(&c.attn_factor, op + 8, 4);
    memcpy(&c.beta_fast, op + 9, 4);
    memcpy(&c.beta_slow, op + 10, 4);
    memcpy(c.sections, op + 11, 16);
    n_offs = op[15];
    c.truncate = ggml_rope_get_truncate(n);
    c.ntok = n->src[0]->ne[2];
    c.ne0 = n->src[0]->ne[0];
    const bool mode_ok = c.mode == GGML_ROPE_TYPE_NORMAL || c.mode == GGML_ROPE_TYPE_NEOX || c.mode == GGML_ROPE_TYPE_MROPE ||
                         c.mode == GGML_ROPE_TYPE_IMROPE || (c.mode == GGML_ROPE_TYPE_VISION && 2 * c.n_dims == c.ne0 && n_offs == 0);
    // the multi-section modes need a section (ggml asserts it; the tables take i % (sum of sections))
    const bool sect_ok = !(c.mode & GGML_ROPE_TYPE_MROPE) || c.sections[0] > 0 || c.sections[1] > 0 || c.sections[2] > 0;
    const ggml_type t = n->src[0]->type;
    return mode_ok && sect_ok && (t == GGML_TYPE_F32 || t == GGML_TYPE_F16) && n->type == t && n->src[1]->type == GGML_TYPE_I32 &&
           (!n->src[2] || n->src[2]->type == GGML_TYPE_F32);
}

/* the NEOX-ordered f32 rope */
static bool rope_neox_f32(const ggml_tensor * n, const rope_cfg & c) {
    return (c.mode == GGML_ROPE_TYPE_NEOX || c.mode == GGML_ROPE_TYPE_MROPE || c.mode == GGML_ROPE_TYPE_IMROPE) &&
           n->type == GGML_TYPE_F32;
}

/* the per-launch host table (cos | sin rows) of a rope node */
int lowerer::rope_table_lt(const ggml_tensor * n) {
    rope_cfg c;
    int n_offs;
    if (!rope_config(n, c, n_offs)) {
        fail("rope config: %s", n);
        return -1;
    }
    const ggml_tensor * pos = n->src[1], * ffac = n->src[2];
    std::string tkey((const char *) n->op_params, sizeof(n->op_params));
    tkey += std::to_string((uint64_t) (uintptr_t) pos->data) + ":" + std::to_string(c.ntok);
    if (ffac) {
        tkey += ":" + std::to_string((uint64_t) (uintptr_t) ffac->data);
    }
    auto tit = rope_tables.find(tkey);
    if (tit != rope_tables.end()) {
        return tit->second;
    }
    if (ffac) {
        // freq factors are a model weight: read once at lowering
        c.ff.resize(c.mode == GGML_ROPE_TYPE_VISION ? c.n_dims : c.n_dims / 2);
        dctx->copy_d2h(c.ff.data(), (uint64_t) (uintptr_t) ffac->data, c.ff.size() * sizeof(float));
    }
    const uint64_t tbytes = (uint64_t) c.ntok * rope_row(c) * sizeof(float);
    const int tab = new_lt(syn_type_single, { (uint64_t) (c.ntok * rope_row(c)) }, LT_ARENA);
    p.lts[tab].addr = arena_alloc(tbytes);
    host_table ht;
    ht.arena_off = p.lts[tab].addr;
    ht.nbytes = tbytes;
    const uint64_t pos_addr = (uint64_t) (uintptr_t) pos->data;
    const size_t pos_bytes = ggml_nbytes(pos);
    ggml_backend_synapse_device_context * dc = dctx;
    ht.fill = [c, pos_addr, pos_bytes, dc](void * dst) {
        std::vector<int32_t> pv(pos_bytes / 4);
        if (!dc->shadow_get(pos_addr, pv.data(), pos_bytes)) {
            return false;
        }
        rope_table(c, pv.data(), (float *) dst);
        return true;
    };
    p.tables.push_back(ht);
    rope_tables[tkey] = tab;
    return tab;
}

/* computed-position ROPE: the cos/sin table computed by a device node (syn_rope_tab) from the
 * positions tensor (an actual device input read at graph execution, never shadow-read) */
int lowerer::rope_table_dev_lt(const ggml_tensor * n) {
    rope_cfg c;
    int n_offs;
    if (!rope_config(n, c, n_offs)) {
        fail("rope config: %s", n);
        return -1;
    }
    const ggml_tensor * pos = n->src[1], * ffac = n->src[2];
    if (pos->op == GGML_OP_NONE && pos->view_src == nullptr) {
        return rope_table_lt(n);   // leaf: the host table path
    }
    if (pos->type != GGML_TYPE_I32 && pos->type != GGML_TYPE_I64) {
        fail("rope dev positions type: %s", pos);
        return -1;
    }
    std::string tkey("dev:");
    tkey += std::to_string((uint64_t) (uintptr_t) pos) + ":" +
            std::string((const char *) n->op_params, sizeof(n->op_params)) + ":" +
            std::to_string(c.ntok);
    auto tit = rope_tables.find(tkey);
    if (tit != rope_tables.end()) {
        return tit->second;
    }
    const int64_t rl = rope_row(c);
    const int tab = new_lt(syn_type_single, { (uint64_t) (c.ntok * rl) });
    opnd op;
    if (!get(pos, op)) {
        return -1;
    }
    int ff = -1;
    int64_t ff_stride = 0;
    if (ffac) {
        opnd of;
        if (!get_f32(ffac, of)) {
            return -1;
        }
        if (of.off != 0) {
            dense_opnd(ffac, cpy_dense(ffac, of, "syn_cpy_f32_f32_f32", syn_type_single), of);
        }
        ff = of.lt;
        ff_stride = ffac->ne[0] > 1 ? of.nb[0] : 1;
    }
    if (pos->ne[0] > 1 && op.nb[0] != 1) {
        fail("rope dev positions layout: %s", pos);
        return -1;
    }
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(c.n_dims, c.n_ctx_orig, c.freq_base, c.beta_fast, c.beta_slow, c.truncate, corr_dims);
    params_t pr;
    pr.i(nmem_for(c.ntok, 1)).i(c.ntok).i(c.n_dims).i(c.mode).i(rl).i(op.off)
      .i(c.sections[0]).i(c.sections[1]).i(c.sections[2]).i(c.sections[3])
      .i(ff >= 0 ? 1 : 0).i(ff_stride)
      .f(powf(c.freq_base, -2.0f / c.n_dims)).f(c.freq_scale).f(c.ext_factor).f(c.attn_factor)
      .f(corr_dims[0]).f(corr_dims[1]);
    add_node("syn_rope_tab_f32", { op.lt, ff >= 0 ? ff : op.lt }, { tab }, pr);
    rope_tables[tkey] = tab;
    return tab;
}

/* per query row: index of the last KV row the mask does not hide, + 1; host table filled from the
 * mask's host shadow before each launch */
int lowerer::kvlen_table_lt(const ggml_tensor * mk, int64_t nq, int64_t nkv) {
    const std::string key = "kvlen:" + std::to_string((uint64_t) (uintptr_t) (mk ? mk->data : nullptr)) + ":" +
                            std::to_string(nq) + ":" + std::to_string(nkv);
    auto it = rope_tables.find(key);
    if (it != rope_tables.end()) {
        return it->second;
    }
    const int tab = new_lt(syn_type_int32, { 64 }, LT_ARENA);
    p.lts[tab].addr = arena_alloc(256);
    host_table ht;
    ht.arena_off = p.lts[tab].addr;
    ht.nbytes = 256;
    // a mask computed in this graph has no host copy (one at its address would be a stale input's): all keys
    const bool host_mask = mk && !produced.count(root_of(mk));
    const uint64_t maddr = host_mask ? (uint64_t) (uintptr_t) mk->data : 0;
    const uint64_t mrow = host_mask ? mk->nb[1] : 0;
    ggml_backend_synapse_device_context * dc = dctx;
    ht.fill = [dc, maddr, mrow, nq, nkv](void * dst) {
        int32_t * len = (int32_t *) dst;
        std::vector<ggml_fp16_t> row(nkv);
        for (int64_t i = 0; i < 64; i++) {
            int32_t n = (int32_t) nkv;
            if (maddr && i < nq && dc->shadow_get(maddr + i * mrow, row.data(), nkv * sizeof(ggml_fp16_t))) {
                while (n > 0 && ggml_fp16_to_fp32(row[n - 1]) == -INFINITY) {
                    n--;
                }
            }
            len[i] = n;
        }
        return true;
    };
    p.tables.push_back(ht);
    rope_tables[key] = tab;
    return tab;
}

/* ALiBi slope per head (as the CPU backend), a constant host table */
int lowerer::alibi_table_lt(int64_t H, float max_bias) {
    const std::string key = "alibi:" + std::to_string(H) + ":" + std::to_string(max_bias);
    auto it = rope_tables.find(key);
    if (it != rope_tables.end()) {
        return it->second;
    }
    const int tab = new_lt(syn_type_single, { (uint64_t) std::max<int64_t>(H, 64) }, LT_ARENA);
    p.lts[tab].addr = arena_alloc(std::max<int64_t>(H, 64) * 4);
    host_table ht;
    ht.arena_off = p.lts[tab].addr;
    ht.nbytes = std::max<int64_t>(H, 64) * 4;
    ht.fill = [H, max_bias](void * dst) {
        float * sl = (float *) dst;
        const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2((double) H));
        const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
        const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
        for (int64_t h = 0; h < std::max<int64_t>(H, 64); h++) {
            sl[h] = h >= H ? 1.0f : (uint32_t) h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1);
        }
        return true;
    };
    p.tables.push_back(ht);
    rope_tables[key] = tab;
    return tab;
}

/* t (contiguous f32) as a tensor that starts at its first element */
int lowerer::flat_f32_lt(const ggml_tensor * t) {
    opnd o;
    if (!get(t, o)) {
        return -1;
    }
    if (o.off == 0) {
        return o.lt;
    }
    const int64_t ne = ggml_nelements(t);
    const int lt = new_lt(syn_type_single, { (uint64_t) ne });
    params_t pc;
    pc.i(1).i(ne).i(1).i(1).i(1).i(o.off).i(1).i(0).i(0).i(0);
    add_node("syn_cpy_f32_f32_f32", { o.lt }, { lt }, pc);
    return lt;
}

bool lowerer::lower_rope(const ggml_tensor * n) {
    rope_cfg c;
    int n_offs;
    if (!rope_config(n, c, n_offs)) {
        return fail("rope config: %s", n);
    }
    opnd oa;
    if (!get(n->src[0], oa)) {
        return false;
    }
    if (oa.nb[0] != 1) {
        return fail("rope layout: %s", n);
    }
    // computed positions (ARANGE, casts, anything not a graph leaf): the cos/sin table on the
    // device (syn_rope_tab), the positions an actual device input - never shadow-read before
    // graph execution. Leaf positions keep the host table.
    const ggml_tensor * pos = n->src[1];
    const bool pos_leaf = pos->op == GGML_OP_NONE && pos->view_src == nullptr;
    const int tab = pos_leaf ? rope_table_lt(n) : rope_table_dev_lt(n);
    if (tab < 0) {
        return false;
    }
    // the SDPA's Q: bf16 [D, nq, H]
    auto rf = rope_fa.find(n);
    const bool qbf = rf != rope_fa.end();
    const int out = qbf ? new_lt(syn_type_bf16, { (uint64_t) ggml_nelements(n) }) : begin_out(n);
    const int64_t ost1 = qbf ? n->ne[0] * n->ne[2] : n->ne[0], ost2 = qbf ? n->ne[0] : n->ne[0] * n->ne[1];
    const bool normal = c.mode == GGML_ROPE_TYPE_NORMAL, h = n->type == GGML_TYPE_F16;
    // vision: NEOX over the whole row (pairs i, i + n_dims)
    const int kdims = c.mode == GGML_ROPE_TYPE_VISION ? 2 * c.n_dims : c.n_dims;
    const int64_t ntok = n->ne[2] * n->ne[3];
    const int64_t nhb0 = std::min<int64_t>(n->ne[1], (48 + ntok - 1) / ntok);
    const int64_t hb = (n->ne[1] + nhb0 - 1) / nhb0, nhb = (n->ne[1] + hb - 1) / hb;
    const int nm = nmem_for(ntok * nhb, 1);
    params_t pr;
    pr.i(nm).i(n->ne[0]).i(n->ne[1]).i(n->ne[2]).i(n->ne[3]).i(kdims).i(n_offs)
      .i(oa.off).i(oa.nb[1]).i(oa.nb[2]).i(oa.nb[3]).i(hb).i(nhb).i((ntok * nhb + nm - 1) / nm).i(ost1).i(ost2);
    if (qbf) {
        add_node(normal ? "syn_ropen_bf_f32" : "syn_rope_bf_f32", { oa.lt, tab }, { out }, pr);
        fa_q_bf[rf->second] = out;
        return true;
    }
    add_node(normal ? (h ? "syn_ropen_h_f32" : "syn_ropen_f32") : (h ? "syn_rope_h_f32" : "syn_rope_f32"), { oa.lt, tab }, { out }, pr);
    return end_out(n, out);
}

/* the FLASH_ATTN_EXT goes to the native SDPA (lower_fa_mme mode 1, see there) */
bool lowerer::fa_sdpa(const ggml_tensor * n) {
    static const bool use_sdpa = !getenv("GGML_SYNAPSE_FA_SDPA") || atoi(getenv("GGML_SYNAPSE_FA_SDPA")) != 0;
    const ggml_tensor * q = n->src[0], * k = n->src[1], * v = n->src[2], * mk = mask_of(n), * snk = n->src[4];
    const int64_t D = q->ne[0];
    return use_sdpa && fa_config_ok(n) && fa_mme(n) == 1 && q->ne[3] == 1 && D == v->ne[0] && (D == 64 || D == 128 || D == 256) &&
           mk && mk->type == GGML_TYPE_F16 && mk->ne[2] == 1 && mk->ne[3] == 1 && !snk && ggml_get_op_params_f32(n, 1) == 0.0f &&
           ggml_get_op_params_f32(n, 2) == 0.0f && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) && v->type == k->type;
}

bool lowerer::run() {
    // roots computed here, and which of them are read outside this graph
    std::unordered_map<const ggml_tensor *, int> uses;
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (!is_view_op(t) && t->view_src == nullptr) {
            produced[t] = true;
        }
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (t->src[k]) {
                uses[t->src[k]]++;
            }
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        bool ext = (t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
        if (!ext && cg->use_counts) {
            const size_t h = ggml_hash_find(&cg->visited_hash_set, t);
            if (h != GGML_HASHSET_FULL && ggml_bitset_get(cg->visited_hash_set.used, h)) {
                ext = cg->use_counts[h] == 0 || cg->use_counts[h] > uses[t];
            } else {
                ext = uses[t] == 0;
            }
        } else if (!ext) {
            ext = uses[t] == 0;
        }
        if (ext) {
            const ggml_tensor * r = root_of(t);
            if (produced.count(r)) {
                is_output[r] = true;
            }
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        order[t] = i;
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (t->src[k]) {
                auto & v = cons[t->src[k]];
                if (std::find(v.begin(), v.end(), t) == v.end()) {
                    v.push_back(t);
                }
            }
        }
    }
    plan_fusions();
    // a value held in a non-compute buffer (user-allocated, as in test-backend-ops) is written to its
    // memory even when only this graph reads it
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || is_view_op(t) || t->view_src || skip[t] || !produced.count(t) ||
            norm_plans.count(t) || gate_plans.count(t) || router_plans.count(t) || t->op == GGML_OP_GATED_DELTA_NET || !t->buffer ||
            ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
            continue;
        }
        is_output[t] = true;
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || is_view_op(t) || ggml_nelements(t) == 0 || skip[t]) {
            continue;
        }
        cur_src = std::string(ggml_op_desc(t)) + " " + t->name;
        if (t->src[0] && t->src[0]->op == GGML_OP_NONE && t->src[0]->name[0] && (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID)) {
            cur_src += std::string(" w=") + t->src[0]->name;   // the weight (GGML_SYNAPSE_DUMP_PROG, profiles)
        }
        if (!lower_node(t) || edge_lost) {
            if (err.empty()) {
                err = std::string("lowering failed at ") + ggml_op_desc(t);
            }
            err += std::string(" [lowering ") + ggml_op_desc(t) + " " + t->name;
            for (int k = 0; k < GGML_MAX_SRC && t->src[k]; k++) {
                err += std::string(k ? ", " : " <- ") + t->src[k]->name + " " + ggml_type_name(t->src[k]->type);
            }
            err += "]";
            return false;
        }
    }
    // a produced output that was never lowered (pure view of an external root, etc.)
    for (auto & kv : is_output) {
        if (!kv.second || ggml_nelements(kv.first) == 0) {
            continue;
        }
        auto it = roots.find(kv.first);
        if (it == roots.end() || it->second.lt < 0) {
            err = std::string("output not produced: ") + kv.first->name + " op=" + ggml_op_desc(kv.first) +
                  " skip=" + std::to_string((int) skip[kv.first]) + " planned=" + std::to_string((int) norm_plans.count(kv.first));
            for (int i = 0; i < cg->n_nodes; i++) {
                fprintf(stderr, "  node %d %s %s flags=%d\n", i, ggml_op_desc(cg->nodes[i]), cg->nodes[i]->name, cg->nodes[i]->flags);
            }
            return false;
        }
    }
    // outputs overlapping a persistent input or an earlier output (memory ggml reused within the
    // graph: two launch tensors may not share an address) go through the arena + a copy after the
    // launch; the copies run in order, so the last writer's value stays
    std::vector<std::pair<uint64_t, uint64_t>> ins;
    std::vector<int> ins_lt;   // lt of each range; -1: an earlier output
    for (size_t i = 0; i < p.lts.size(); i++) {
        if (p.lts[i].kind == LT_IN) {
            ins.push_back({ p.lts[i].addr, p.lts[i].addr + p.lts[i].nbytes });
            ins_lt.push_back((int) i);
        }
    }
    std::vector<std::pair<int, post_copy>> staged;
    for (size_t i = 0; i < p.lts.size(); i++) {
        lt_t & t = p.lts[i];
        if (t.kind != LT_OUT) {
            continue;
        }
        std::vector<int> over;
        for (size_t k = 0; k < ins.size(); k++) {
            if (t.addr < ins[k].second && ins[k].first < t.addr + t.nbytes) {
                over.push_back(ins_lt[k]);
            }
        }
        if ((t.addr % 128) == 0 && !over.empty() && out_alias_ok((int) i, over)) {
            // inputs the graph has finished reading when it writes this output
            t.region = p.lts[over[0]].region;
            alias_out[(int) i] = over;
        } else if (!over.empty() || (t.addr % 128) != 0) {
            const uint64_t dst = t.addr;
            arena_out[(int) i] = true;
            t.kind = LT_ARENA;
            t.addr = arena_alloc(t.nbytes);
            staged.push_back({ (int) i, { t.addr, dst, t.nbytes } });
            continue;
        }
        ins.push_back({ t.addr, t.addr + t.nbytes });
        ins_lt.push_back(-1);
    }
    for (const auto & t : p.lts) {
        if (t.kind == LT_IN && (t.addr % 128) != 0) {
            err = "persistent input not 128-byte aligned";
            return false;
        }
    }
    // keep side-effect writes alive: every sink joins into one persistent token. A sink consumed
    // only as an ordering token is not enough: when nothing downstream reaches a graph output (a
    // ubatch requesting no logits) the compiler drops the whole chain, cache writes included
    {
        std::vector<int> dangling = sinks;
        if (!dangling.empty()) {
            int t = dangling[0];
            if (dangling.size() > 1 || true) {
                ln_t n;
                n.guid = "syn_join_f32";
                for (size_t k = 0; k < 4; k++) {
                    n.ins.push_back(dangling[std::min(k, dangling.size() - 1)]);
                }
                t = new_lt(syn_type_single, { 64 }, LT_ARENA);
                p.lts[t].addr = arena_alloc(256);
                arena_out[t] = true;
                n.outs = { t };
                const int32_t one = 1;
                n.params.resize(4);
                memcpy(n.params.data(), &one, 4);
                p.nodes.push_back(n);
                // more than four: chain the rest through internal joins first
                if (dangling.size() > 4) {
                    p.nodes.pop_back();
                    const int j = join(dangling);
                    ln_t m;
                    m.guid = "syn_join_f32";
                    m.ins = { j, j, j, j };
                    m.outs = { t };
                    m.params = n.params;
                    p.nodes.push_back(m);
                }
            }
        }
    }
    std::unordered_map<uint64_t, int> groups;
    const uint64_t arena_key = ~0ull;
    for (size_t i = 0; i < p.lts.size(); i++) {
        lt_t & t = p.lts[i];
        if ((t.kind == LT_IN || alias_out.count((int) i)) && t.region != 0) {
            auto it = groups.emplace(t.region, (int) groups.size()).first;
            t.sec = it->second;
            t.sec_off = t.addr - t.region;
        } else if (t.kind == LT_ARENA && !is_arena_out(t)) {
            auto it = groups.emplace(arena_key, (int) groups.size()).first;
            t.sec = it->second;
            t.sec_off = t.addr;
        }
    }
    // a section spans its tensors: a group whose span covers any other persistent tensor
    // (e.g. a graph output in the compute buffer) is split back into one section per tensor
    std::unordered_map<int, std::pair<uint64_t, uint64_t>> span;
    for (const auto & t : p.lts) {
        if (t.sec >= 0) {
            const uint64_t a0 = t.kind == LT_ARENA ? ~0ull : t.addr;
            if (t.kind == LT_ARENA) {
                continue;   // the arena is one private allocation
            }
            auto it = span.find(t.sec);
            if (it == span.end()) {
                span[t.sec] = { a0, t.addr + t.nbytes };
            } else {
                it->second.first = std::min(it->second.first, a0);
                it->second.second = std::max(it->second.second, t.addr + t.nbytes);
            }
        }
    }
    std::unordered_map<int, bool> split;
    for (const auto & t : p.lts) {
        if (t.kind == LT_INTERNAL || t.kind == LT_ARENA) {
            continue;
        }
        for (const auto & sp : span) {
            if (t.sec != sp.first && t.addr < sp.second.second && sp.second.first < t.addr + t.nbytes) {
                split[sp.first] = true;
            }
        }
    }
    for (auto & t : p.lts) {
        if (t.sec >= 0 && split.count(t.sec)) {
            t.sec = -1;
            t.sec_off = 0;
        } else if (t.sec >= 0 && t.kind != LT_ARENA) {
            // the runtime places a section at (tensor address - offset): anchor it at its lowest tensor
            t.sec_off = t.addr - span[t.sec].first;
        }
    }
    // an in-place output whose group was split: a section of its own with the inputs it overwrites,
    // when no other persistent tensor lies in between; else back to the arena + copy
    int n_sec = (int) groups.size();
    for (auto & kv : alias_out) {
        const int o = kv.first;
        lt_t & t = p.lts[o];
        if (t.sec >= 0) {
            continue;
        }
        uint64_t a0 = t.addr, a1 = t.addr + t.nbytes;
        bool ok = true;
        for (int k : kv.second) {
            ok = ok && p.lts[k].sec < 0;
            a0 = std::min(a0, p.lts[k].addr);
            a1 = std::max(a1, p.lts[k].addr + p.lts[k].nbytes);
        }
        for (size_t j = 0; ok && j < p.lts.size(); j++) {
            const lt_t & u = p.lts[j];
            ok = u.kind == LT_INTERNAL || u.kind == LT_ARENA || (int) j == o ||
                 std::find(kv.second.begin(), kv.second.end(), (int) j) != kv.second.end() || u.addr >= a1 || u.addr + u.nbytes <= a0;
        }
        if (ok) {
            t.sec = n_sec;
            t.sec_off = t.addr - a0;
            for (int k : kv.second) {
                p.lts[k].sec = n_sec;
                p.lts[k].sec_off = p.lts[k].addr - a0;
            }
            n_sec++;
            continue;
        }
        const uint64_t dst = t.addr;
        arena_out[o] = true;
        t.kind = LT_ARENA;
        t.addr = arena_alloc(t.nbytes);
        staged.push_back({ o, { t.addr, dst, t.nbytes } });
    }
    // K / V windows read in place whose group was split: a section of their own with their cache, when no other
    // persistent tensor overlaps the cache (overlapping sections fail at launch); else the caller lowers again
    // with copies
    std::map<int, std::vector<int>> win_split;   // cache -> its windows
    for (const auto & kv : kv_window) {
        if (p.lts[kv.first].sec < 0) {
            win_split[kv.second].push_back(kv.first);
        }
    }
    for (const auto & ws : win_split) {
        lt_t & c = p.lts[ws.first];
        bool ok = c.sec < 0;
        for (size_t j = 0; ok && j < p.lts.size(); j++) {
            const lt_t & u = p.lts[j];
            ok = u.kind == LT_INTERNAL || u.kind == LT_ARENA || (int) j == ws.first ||
                 std::find(ws.second.begin(), ws.second.end(), (int) j) != ws.second.end() || u.addr >= c.addr + c.nbytes ||
                 u.addr + u.nbytes <= c.addr;
        }
        if (!ok) {
            err = "kv window: cache section shared";
            kv_view_failed = true;
            return false;
        }
        c.sec = n_sec;
        c.sec_off = 0;
        for (int w : ws.second) {
            p.lts[w].sec = n_sec;
            p.lts[w].sec_off = p.lts[w].addr - c.addr;
        }
        n_sec++;
    }
    for (const auto & s : staged) {
        alias_out.erase(s.first);
    }
    std::sort(staged.begin(), staged.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    for (const auto & s : staged) {
        p.posts.push_back(s.second);
    }
    return true;
}

/* ---------------- graph_compute ---------------- */

static uint64_t graph_fingerprint(const ggml_cgraph * cg) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t x) {
        h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };
    mix((uint64_t) cg->n_nodes);
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        mix((uint64_t) (uintptr_t) t);
        mix((uint64_t) (uintptr_t) t->data);
        mix(((uint64_t) t->op << 32) | (uint64_t) t->type | ((uint64_t) (t->flags & GGML_TENSOR_FLAG_COMPUTE) << 16));
        for (int d = 0; d < 4; d++) {
            mix((uint64_t) t->ne[d]);
            mix((uint64_t) t->nb[d]);
        }
        h = fnv1a(t->op_params, sizeof(t->op_params), h);
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (t->src[k]) {
                mix((uint64_t) (uintptr_t) t->src[k]);
                mix((uint64_t) (uintptr_t) t->src[k]->data);
            }
        }
    }
    return h;
}

void ggml_synapse_prepared_free(ggml_backend_synapse_device_context * dctx) {
    dctx->sync();
    dctx->prepared.clear();
    dctx->graph_memos.clear();
}

static void dump_program(const program & p) {
    // GGML_SYNAPSE_DUMP_PROG=1: to stderr; =<prefix>: one file per program, <prefix>.<pid>.<n>
    if (const char * dp = getenv("GGML_SYNAPSE_DUMP_PROG")) {
        static std::atomic<int> n_dumped{0};
        FILE * f = stderr;
        if (strcmp(dp, "1") != 0) {
            const std::string fn = std::string(dp) + "." + std::to_string(getpid()) + "." + std::to_string(n_dumped++);
            f = fopen(fn.c_str(), "w");
        }
        if (f) {
            // the recipe key: profiler node names are n<index>_<key[0:6]>_<guid>
            fprintf(f, "key %s\n", hex128(program_signature(p)).c_str());
            // the persistent tensors' sections (part of the key)
            for (size_t i = 0; i < p.lts.size(); i++) {
                const lt_t & t = p.lts[i];
                if (t.kind != LT_INTERNAL) {
                    fprintf(f, "lt %zu kind %d sec %d off %llu size %llu\n", i, t.kind, t.sec, (unsigned long long) t.sec_off,
                            (unsigned long long) t.nbytes);
                }
            }
        }
        for (size_t i = 0; f && i < p.nodes.size(); i++) {
            const ln_t & nd = p.nodes[i];
            fprintf(f, "node %zu %s ins=", i, nd.guid.c_str());
            for (int k : nd.ins) {
                fprintf(f, "%d(%s%llu) ", k, p.lts[k].kind == LT_INTERNAL ? "" : "P", (unsigned long long) p.lts[k].sz[0]);
            }
            fprintf(f, "outs=");
            for (int k : nd.outs) {
                fprintf(f, "%d ", k);
            }
            fprintf(f, "params=");
            for (size_t k = 0; k + 4 <= nd.params.size(); k += 4) {
                int32_t v;
                memcpy(&v, nd.params.data() + k, 4);
                fprintf(f, "%d ", v);
            }
            fprintf(f, "| %s\n", nd.src.c_str());
        }
        if (f && f != stderr) {
            fclose(f);
        }
    }
}

/* a recipe for program p, from the in-process table, the disk cache or compiled; logs new ones */
static std::shared_ptr<synapse_recipe> synapse_recipe_for(ggml_backend_synapse_device_context * dctx, const program & p,
                                                          const ggml_cgraph * cg, const char * what) {
    const std::string sig = program_signature(p);
    {
        std::lock_guard<std::mutex> lock(dctx->recipes_mtx);
        auto it = dctx->recipes.find(sig);
        if (it != dctx->recipes.end()) {
            return it->second;
        }
    }
    dump_program(p);
    auto rec = synapse_compile(p, sig);
    if (!rec) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(dctx->recipes_mtx);
        dctx->recipes[sig] = rec;
    }
    int64_t tokens, kv;
    graph_shape(cg, tokens, kv);
    ggml_synapse_log_info("synapse: %s %s %.6s: %lld tokens, kv %lld, %zu nodes, %.0f ms", rec->from_disk ? "loaded" : "compiled",
                          what, hex128(sig).c_str(), (long long) tokens, (long long) kv, p.nodes.size(), rec->build_ms);
    return rec;
}

/* launch entries of recipe rec for program p (the persistent tensors' addresses; arena ones relative) */
static void launch_entries(const synapse_recipe & rec, const program & p, std::vector<synLaunchTensorInfoExt> & launch,
                           std::vector<std::pair<size_t, uint64_t>> & launch_arena) {
    launch.resize(rec.persist.size());
    for (size_t i = 0; i < rec.persist.size(); i++) {
        const lt_t & t = p.lts[rec.persist[i]];
        synLaunchTensorInfoExt & li = launch[i];
        memset(&li, 0, sizeof(li));
        li.tensorName = rec.names[i].c_str();
        li.tensorId = rec.ids[i];
        li.tensorType = DATA_TENSOR;
        li.pTensorAddress = t.addr;
        if (t.kind == LT_ARENA) {
            launch_arena.emplace_back(i, t.addr);
        }
    }
}

static std::shared_ptr<synapse_prepared> synapse_prepare(ggml_backend_synapse_device_context * dctx, const ggml_cgraph * cg,
                                                          bool no_out_alias = false, bool no_kv_view = false,
                                                          bool no_tile = false) {
    lowerer L;
    L.dctx = dctx;
    L.cg = cg;
    L.no_tile = no_tile;
    // outputs in place over inputs rely on the order inside one graph: off when the program is cut
    L.no_out_alias = no_out_alias || (!no_tile && lowerer::fa_tile_w() > 0);
    L.no_kv_view = no_kv_view;
    if (!L.run()) {
        if (L.kv_view_failed && !no_kv_view) {
            return synapse_prepare(dctx, cg, no_out_alias, true, no_tile);
        }
        if (!L.fa_tiles.empty() && !no_tile) {
            GGML_LOG_WARN("%s: %s; lowering without tiled attention\n", __func__, L.err.c_str());
            return synapse_prepare(dctx, cg, no_out_alias, no_kv_view, true);
        }
        GGML_LOG_ERROR("%s: %s\n", __func__, L.err.c_str());
        return nullptr;
    }
    if (!L.fa_tiles.empty()) {
        std::vector<program> segs;
        if (!L.split_segments(segs)) {
            GGML_LOG_WARN("%s: %s; lowering without tiled attention\n", __func__, L.err.c_str());
            return synapse_prepare(dctx, cg, no_out_alias, no_kv_view, true);
        }
        auto prep = std::make_shared<synapse_prepared>();
        prep->dctx = dctx;
        prep->arena_size = L.p.arena_size;
        prep->tables = std::move(L.p.tables);
        if (!prep->tables.empty()) {
            prep->host_size = L.p.arena_size;
            prep->host = malloc(prep->host_size);
        }
        for (const auto & pc : L.p.posts) {
            prep->post_off.push_back(pc.arena_off);
            prep->post_src.push_back(pc.arena_off);
            prep->post_dst.push_back(pc.dst);
            prep->post_size.push_back(pc.nbytes);
        }
        prep->tile_bad = L.tile_bad;
        for (const auto & g : L.tile_groups) {
            synapse_tile_group pg;
            pg.tab_off = g.tab_off;
            pg.entry = g.entry;
            pg.win = g.win;
            prep->tile_groups.push_back(pg);
        }
        for (size_t s = 0; s < segs.size(); s++) {
            char what[64];
            snprintf(what, sizeof(what), "segment %zu/%zu", s, segs.size());
            auto rec = synapse_recipe_for(dctx, segs[s], cg, what);
            if (!rec) {
                GGML_LOG_WARN("%s: segment %zu failed to compile; lowering without tiled attention\n", __func__, s);
                return synapse_prepare(dctx, cg, no_out_alias, no_kv_view, true);
            }
            synapse_step st;
            st.rec = rec;
            launch_entries(*rec, segs[s], st.launch, st.launch_arena);
            prep->ws_max = std::max(prep->ws_max, rec->ws_size);
            prep->steps.push_back(std::move(st));
            if (s + 1 == segs.size()) {
                break;
            }
            const auto & t = L.fa_tiles[s];
            synapse_step tw;
            tw.tile = (int) s;
            synapse_tile ti;
            for (int64_t sz : L.tile_groups[t.group].sizes) {
                const program tp = t.pre ? fa_tile_program_pre(t.D, t.nq, t.hpg, t.Hkv, sz, t.dt, t.scale,
                                                               L.tile_groups[t.group].entry / 4) :
                                           fa_tile_program(t.R, t.Rv, t.C, t.nq, t.H, t.hpg, t.Dv, sz, t.dt, t.scale, t.sinks >= 0,
                                                           L.tile_groups[t.group].entry / 4);
                auto trec = synapse_recipe_for(dctx, tp, cg, "attention window");
                if (!trec) {
                    GGML_LOG_WARN("%s: the attention window recipe failed to compile; lowering without tiled attention\n", __func__);
                    return synapse_prepare(dctx, cg, no_out_alias, no_kv_view, true);
                }
                std::vector<synLaunchTensorInfoExt> li;
                std::vector<std::pair<size_t, uint64_t>> unused;
                launch_entries(*trec, tp, li, unused);
                prep->ws_max = std::max(prep->ws_max, trec->ws_size);
                // the window programs' persistent lts are 0..8 in idx order, the same for every size
                for (size_t i = 0; i < trec->persist.size(); i++) {
                    ti.idx[trec->persist[i]] = (int) i;
                }
                tw.wrec.push_back(trec);
                tw.wlaunch.push_back(std::move(li));
            }
            const uint64_t es = syn_type_size(t.dt);
            ti.group = t.group;
            ti.qbd_off = L.p.lts[t.qbd].addr;
            ti.sinks_off = t.sinks >= 0 ? L.p.lts[t.sinks].addr : 0;
            ti.acc_off[0] = t.acc_ab[0];
            ti.acc_off[1] = t.acc_ab[1];
            ti.acc_off[2] = L.p.lts[t.acc_f].addr;
            ti.ml_off[0] = t.ml_ab[0];
            ti.ml_off[1] = t.ml_ab[1];
            ti.ml_off[2] = L.p.lts[t.ml_f].addr;
            ti.kaddr = t.kaddr;
            ti.vaddr = t.vaddr;
            ti.krow = (uint64_t) t.R * es;
            ti.vrow = (uint64_t) t.Rv * es;
            prep->tiles.push_back(ti);
            prep->steps.push_back(std::move(tw));
        }
        if (getenv("GGML_SYNAPSE_VERBOSE")) {
            fprintf(stderr, "%s: %d ggml nodes -> %zu segments + %zu tiled attentions in %zu window groups, arena %" PRIu64 " B\n",
                    __func__, cg->n_nodes, segs.size(), L.fa_tiles.size(), L.tile_groups.size(), L.p.arena_size);
        }
        return prep;
    }
    dump_program(L.p);
    const std::string sig = program_signature(L.p);
    std::shared_ptr<synapse_recipe> rec;
    {
        std::lock_guard<std::mutex> lock(dctx->recipes_mtx);
        auto it = dctx->recipes.find(sig);
        if (it != dctx->recipes.end()) {
            rec = it->second;
        }
    }
    if (!rec) {
        rec = synapse_compile(L.p, sig);
        if (!rec && (!L.alias_out.empty() || !L.kv_window.empty())) {
            GGML_LOG_WARN("%s: compile failed with in-place outputs / K V reads, retrying with copies\n", __func__);
            return synapse_prepare(dctx, cg, true, true);
        }
        if (!rec) {
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(dctx->recipes_mtx);
            dctx->recipes[sig] = rec;
        }
        // one line per new recipe (llama.cpp -lv 4)
        int64_t tokens, kv;
        graph_shape(cg, tokens, kv);
        ggml_synapse_log_info("synapse: %s graph %.6s: %lld tokens, kv %lld, %d ggml -> %zu nodes, %.0f ms (%s .. %s)",
                              rec->from_disk ? "loaded" : "compiled", hex128(sig).c_str(), (long long) tokens, (long long) kv,
                              cg->n_nodes, L.p.nodes.size(), rec->build_ms, cg->nodes[0]->name, cg->nodes[cg->n_nodes - 1]->name);
    }
    auto prep = std::make_shared<synapse_prepared>();
    prep->dctx = dctx;
    prep->rec = rec;
    prep->arena_size = L.p.arena_size;
    prep->tables = std::move(L.p.tables);
    if (!prep->tables.empty()) {
        prep->host_size = L.p.arena_size;
        prep->host = malloc(prep->host_size);
    }
    prep->launch.resize(rec->persist.size());
    for (size_t i = 0; i < rec->persist.size(); i++) {
        const lt_t & t = L.p.lts[rec->persist[i]];
        synLaunchTensorInfoExt & li = prep->launch[i];
        memset(&li, 0, sizeof(li));
        li.tensorName = rec->names[i].c_str();
        li.tensorId = rec->ids[i];
        li.tensorType = DATA_TENSOR;
        li.pTensorAddress = t.addr;
        prep->launch_info.push_back(t);
        if (t.kind == LT_ARENA) {
            prep->launch_arena.emplace_back(i, t.addr);
        }
    }
    for (const auto & pc : L.p.posts) {
        prep->post_off.push_back(pc.arena_off);
        prep->post_src.push_back(pc.arena_off);
        prep->post_dst.push_back(pc.dst);
        prep->post_size.push_back(pc.nbytes);
    }
    if (getenv("GGML_SYNAPSE_VERBOSE")) {
        fprintf(stderr, "%s: %d ggml nodes -> %zu synapse nodes, %zu tensors (%zu persistent, %zu launch), arena %" PRIu64 " B, %zu copies after, %zu outputs in place, %zu K / V windows read in place\n",
                      __func__, cg->n_nodes, L.p.nodes.size(), L.p.lts.size(), rec->persist.size(), prep->launch.size(), L.p.arena_size,
                      L.p.posts.size(), L.alias_out.size(), L.kv_window.size());
        if (atoi(getenv("GGML_SYNAPSE_VERBOSE")) >= 2) {
            std::map<std::tuple<int, int, uint64_t>, int> ph;
            for (int pi : rec->persist) {
                const lt_t & t = L.p.lts[pi];
                ph[{ t.kind, t.sec, t.nbytes }]++;
            }
            for (const auto & kv : ph) {
                fprintf(stderr, "    persist kind %d sec %3d nbytes %10" PRIu64 " x%d\n", std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), kv.second);
            }
        }
        std::map<std::string, int> hist;
        for (const auto & nd : L.p.nodes) {
            hist[nd.guid]++;
        }
        for (const auto & kv : hist) {
            fprintf(stderr, "    %-28s %d\n", kv.first.c_str(), kv.second);
        }
        fflush(stderr);
    }
    return prep;
}

enum ggml_status ggml_synapse_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) backend->context;

    bool any = false;
    for (int i = 0; i < cgraph->n_nodes && !any; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        any = (t->flags & GGML_TENSOR_FLAG_COMPUTE) && !is_view_op(t) && ggml_nelements(t) > 0;
    }
    if (!any) {
        return GGML_STATUS_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(dctx->compute_mtx);
    static const bool timing = getenv("GGML_SYNAPSE_TIMING") != nullptr;
    static const bool trace_ev = getenv("GGML_SYNAPSE_TRACE") && atoi(getenv("GGML_SYNAPSE_TRACE")) != 2;
    if (trace_ev) {
        char w[200];
        snprintf(w, sizeof(w), "graph_compute begin n %d %s .. %s", cgraph->n_nodes, cgraph->nodes[0]->name,
                 cgraph->nodes[cgraph->n_nodes - 1]->name);
        ggml_synapse_trace(w);
    } else {
        ggml_synapse_trace("graph_compute begin");
    }
    const int64_t t_start = timing ? ggml_time_us() : 0;
    static const bool full_check = getenv("GGML_SYNAPSE_FULL_CHECK") != nullptr;
    std::shared_ptr<synapse_prepared> prep;
    if (dctx->graph_memos.size() >= 4096 && dctx->graph_memos.find(cgraph) == dctx->graph_memos.end()) {
        dctx->graph_memos.clear();
    }
    auto & last = dctx->graph_memos[cgraph];
    {
        bool same = last.prep && (int) last.nodes.size() == cgraph->n_nodes;
        bool marked = same && !full_check && cgraph->n_nodes > 0;
        for (int i = 0; marked && i < cgraph->n_nodes; i += 32) {
            marked = cgraph->nodes[i]->extra == last.prep.get() && cgraph->nodes[i] == last.nodes[i].t;
        }
        marked = marked && cgraph->nodes[cgraph->n_nodes - 1]->extra == last.prep.get();
        for (int i = 0; same && !marked && i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            same = last.nodes[i].t == t && last.nodes[i].data == t->data && last.nodes[i].src0 == t->src[0] &&
                   last.nodes[i].op == t->op && last.nodes[i].flags == t->flags &&
                   memcmp(last.nodes[i].ne, t->ne, sizeof(t->ne)) == 0 && memcmp(last.nodes[i].nb, t->nb, sizeof(t->nb)) == 0;
        }
        if (same) {
            prep = last.prep;
        }
    }
    const uint64_t fp = prep ? 0 : graph_fingerprint(cgraph);
    auto it = prep ? dctx->prepared.end() : dctx->prepared.find(fp);
    if (prep) {
    } else if (it != dctx->prepared.end()) {
        prep = it->second;
    } else {
        if (timing) {
            fprintf(stderr, "synapse: prepare miss (n_nodes=%d)\n", cgraph->n_nodes);
        }
        ggml_synapse_trace("prepare miss");
        prep = synapse_prepare(dctx, cgraph);
        if (!prep) {
            return GGML_STATUS_FAILED;
        }
        if (dctx->prepared.size() >= 4096) {
            ggml_synapse_prepared_free(dctx);
        }
        dctx->prepared[fp] = prep;
    }

    {
        auto & last = dctx->graph_memos[cgraph];   // again: a prepare may have dropped the memos
        if (last.prep != prep || cgraph->nodes[cgraph->n_nodes - 1]->extra != prep.get()) {
            last.prep = prep;
            last.nodes.resize(cgraph->n_nodes);
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * t = cgraph->nodes[i];
                t->extra = prep.get();
                auto & e = last.nodes[i];
                e.t = t;
                e.data = t->data;
                e.src0 = t->src[0];
                e.op = t->op;
                e.flags = t->flags;
                memcpy(e.ne, t->ne, sizeof(t->ne));
                memcpy(e.nb, t->nb, sizeof(t->nb));
            }
        }
    }
    const int64_t t_fp = timing ? ggml_time_us() : 0;
    ggml_synapse_trace("fp");
    if (!prep->steps.empty()) {
        // tiled attention: the tables first (they give the windows, and whether every mask is one range per row)
        *prep->tile_bad = 0;
        for (const auto & t : prep->tables) {
            if (!t.fill((char *) prep->host + t.arena_off)) {
                GGML_LOG_ERROR("%s: graph input not host-visible (rope positions / mask)\n", __func__);
                return GGML_STATUS_FAILED;
            }
        }
        if (*prep->tile_bad) {
            if (!prep->fallback) {
                ggml_synapse_log_info("synapse: attention mask not a plain prefix: the untiled program");
                prep->fallback = synapse_prepare(dctx, cgraph, false, false, true);
                if (!prep->fallback) {
                    return GGML_STATUS_FAILED;
                }
            }
            prep = prep->fallback;
        }
    }
    if (!prep->steps.empty()) {
        if (prep->ws_max > dctx->ws_cap) {
            dctx->sync();
            if (dctx->ws_addr) {
                dctx->dev_free(dctx->ws_addr);
            }
            dctx->ws_addr = dctx->dev_malloc(prep->ws_max);
            dctx->ws_cap = prep->ws_max;
        }
        if (prep->arena_size > dctx->arena_cap) {
            dctx->sync();
            if (dctx->arena_addr) {
                dctx->dev_free(dctx->arena_addr);
            }
            dctx->arena_cap = (prep->arena_size + prep->arena_size / 4 + (1u << 20) - 1) & ~(uint64_t) ((1u << 20) - 1);
            dctx->arena_addr = dctx->dev_malloc(dctx->arena_cap);
        }
        if (prep->arena != dctx->arena_addr) {
            prep->rebase(dctx->arena_addr);
        }
        for (const auto & t : prep->tables) {
            dctx->copy_h2d_async_small(prep->arena + t.arena_off, (char *) prep->host + t.arena_off, t.nbytes);
        }
        dctx->flush_uploads();
        const uint64_t A = prep->arena;
        int nlaunch = 0;
        auto launch = [&](std::vector<synLaunchTensorInfoExt> & li, const synapse_recipe & rec) {
            const synStatus st = synLaunchWithExternalEventsExt(dctx->stream, li.data(), (uint32_t) li.size(), dctx->ws_addr, rec.r,
                                                                nullptr, 0, 0);
            if (st != synSuccess) {
                for (const auto & e : li) {
                    GGML_LOG_ERROR("%s: launch tensor %s at 0x%" PRIx64 "\n", __func__, e.tensorName, e.pTensorAddress);
                }
            }
            SYNAPSE_CHECK(st);
        };
        for (auto & st : prep->steps) {
            if (st.tile < 0) {
                launch(st.launch, *st.rec);
                continue;
            }
            const synapse_tile & ti = prep->tiles[st.tile];
            const synapse_tile_group & tg = prep->tile_groups[ti.group];
            const std::vector<int64_t> & win = *tg.win;
            const int nt = (int) win.size() / 2;
            int cur = 1;   // the first window reads B (ignored) and writes A
            for (int w = 0; w < nt; w++) {
                const int out = w + 1 == nt ? 2 : (w % 2 == 0 ? 0 : 1);
                const int64_t start = win[2 * w];
                auto & li = st.wlaunch[win[2 * w + 1]];
                li[ti.idx[0]].pTensorAddress = A + ti.qbd_off;
                li[ti.idx[1]].pTensorAddress = ti.kaddr + (uint64_t) start * ti.krow;
                li[ti.idx[2]].pTensorAddress = ti.vaddr + (uint64_t) start * ti.vrow;
                li[ti.idx[3]].pTensorAddress = A + tg.tab_off + (uint64_t) w * tg.entry;
                li[ti.idx[4]].pTensorAddress = A + ti.acc_off[cur];
                li[ti.idx[5]].pTensorAddress = A + ti.ml_off[cur];
                li[ti.idx[6]].pTensorAddress = A + ti.ml_off[out];
                li[ti.idx[7]].pTensorAddress = A + ti.acc_off[out];
                if (ti.idx[8] >= 0) {
                    li[ti.idx[8]].pTensorAddress = A + ti.sinks_off;
                }
                launch(li, *st.wrec[win[2 * w + 1]]);
                nlaunch++;
                cur = out;
            }
        }
        dctx->set_pending();
        if (!prep->post_src.empty()) {
            SYNAPSE_CHECK(synMemCopyAsyncMultiple(dctx->stream, prep->post_src.data(), prep->post_size.data(), prep->post_dst.data(),
                                                  DRAM_TO_DRAM, prep->post_src.size()));
        }
        if (timing) {
            const int64_t t_launch = ggml_time_us();
            dctx->sync();
            fprintf(stderr, "synapse: tiled graph n_nodes=%d steps %zu window launches %d | launch %.3f | device %.3f ms\n", cgraph->n_nodes,
                    prep->steps.size(), nlaunch, (t_launch - t_fp) / 1000.0, (ggml_time_us() - t_launch) / 1000.0);
        } else if (getenv("GGML_SYNAPSE_SYNC")) {
            dctx->sync();
        }
        return GGML_STATUS_SUCCESS;
    }
    if (prep->rec->ws_size > dctx->ws_cap) {
        dctx->sync();
        if (dctx->ws_addr) {
            dctx->dev_free(dctx->ws_addr);
        }
        dctx->ws_addr = dctx->dev_malloc(prep->rec->ws_size);
        dctx->ws_cap = prep->rec->ws_size;
    }
    if (prep->arena_size > dctx->arena_cap) {
        dctx->sync();
        if (dctx->arena_addr) {
            dctx->dev_free(dctx->arena_addr);
        }
        dctx->arena_cap = (prep->arena_size + prep->arena_size / 4 + (1u << 20) - 1) & ~(uint64_t) ((1u << 20) - 1);
        dctx->arena_addr = dctx->dev_malloc(dctx->arena_cap);
    }
    if (prep->arena != dctx->arena_addr) {
        prep->rebase(dctx->arena_addr);
    }

    for (const auto & t : prep->tables) {
        char * h = (char *) prep->host + t.arena_off;
        if (!t.fill(h)) {
            GGML_LOG_ERROR("%s: graph input not host-visible (rope positions)\n", __func__);
            return GGML_STATUS_FAILED;
        }
        dctx->copy_h2d_async_small(prep->arena + t.arena_off, h, t.nbytes);
    }

    const int64_t t_tab = timing ? ggml_time_us() : 0;
    ggml_synapse_trace("tab");
    dctx->flush_uploads();
    ggml_synapse_trace("flushed");
    {
        const synStatus st = synLaunchWithExternalEventsExt(dctx->stream, prep->launch.data(), (uint32_t) prep->launch.size(),
                                                            dctx->ws_addr, prep->rec->r, nullptr, 0, 0);
        if (st != synSuccess) {
            for (size_t i = 0; i < prep->launch.size(); i++) {
                GGML_LOG_ERROR("%s: launch tensor %s at 0x%" PRIx64 " (%" PRIu64 " B, section %d +%" PRIu64 ")\n", __func__,
                               prep->launch[i].tensorName, prep->launch[i].pTensorAddress, prep->launch_info[i].nbytes,
                               prep->launch_info[i].sec, prep->launch_info[i].sec_off);
            }
            for (int i = 0; i < cgraph->n_nodes; i++) {
                const ggml_tensor * t = cgraph->nodes[i];
                GGML_LOG_ERROR("%s:   node %d %s %s data %p %zu B\n", __func__, i, ggml_op_desc(t), t->name, t->data, ggml_nbytes(t));
            }
        }
        SYNAPSE_CHECK(st);
    }
    dctx->set_pending();
    ggml_synapse_trace("launched");
    const int64_t t_lau = timing ? ggml_time_us() : 0;
    if (!prep->post_src.empty()) {
        SYNAPSE_CHECK(synMemCopyAsyncMultiple(dctx->stream, prep->post_src.data(), prep->post_size.data(),
                                              prep->post_dst.data(), DRAM_TO_DRAM, prep->post_src.size()));
    }
    if (timing) {
        const int64_t t_launch = ggml_time_us();
        dctx->sync();
        static int64_t t_prev_end = 0;
        const int64_t t_end = ggml_time_us();
        fprintf(stderr, "synapse: prep=%p nsyn=%zu ", (void *) prep.get(), prep->launch.size());
        fprintf(stderr, "synapse: graph n_nodes=%d gap %.3f | fp %.3f tab %.3f launch %.3f post %.3f | device %.3f ms\n",
                cgraph->n_nodes, t_prev_end ? (t_start - t_prev_end) / 1000.0 : 0.0, (t_fp - t_start) / 1000.0,
                (t_tab - t_fp) / 1000.0, (t_lau - t_tab) / 1000.0, (t_launch - t_lau) / 1000.0, (t_end - t_launch) / 1000.0);
        t_prev_end = t_end;
    } else if (getenv("GGML_SYNAPSE_SYNC")) {
        dctx->sync();
    }
    return GGML_STATUS_SUCCESS;
}

/* ---------------- supports_op ---------------- */

static bool rows_contig(const ggml_tensor * t) {
    return t->ne[0] <= 1 || t->nb[0] == ggml_type_size(t->type);
}

static bool small_enough(const ggml_tensor * t) {
    return ggml_nelements(t) < INT32_MAX && ggml_nbytes(root_of(t)) / 2 < (size_t) INT32_MAX;
}

// expert weights of MUL_MAT_ID beyond 2^31 elements / bytes: fine in a device format (read as
// [bytes per matrix, matrices]) when one matrix stays within 2^31 bytes
static bool experts_ok(const ggml_tensor * op, int k) {
    const ggml_tensor * w = op->src[k];
    if (op->op != GGML_OP_MUL_MAT_ID || k != 0 || w->view_src != nullptr) {
        return false;
    }
    bool in_place = false;
    const int kind = ggml_backend_synapse_device_context::weight_kind(w, in_place);
    return (kind == SYN_W_PLANAR || kind == SYN_W_FP8_PLANAR) && ggml_nbytes(w) / w->ne[2] < (size_t) INT32_MAX;
}

static bool supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op);

// GGML_SYNAPSE_LOG_UNSUPPORTED=1: each distinct rejected op (op, types, shapes, contiguity) once on stderr
bool ggml_synapse_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool ok = supports_op_impl(dev, op);
    static const bool log = getenv("GGML_SYNAPSE_LOG_UNSUPPORTED") && atoi(getenv("GGML_SYNAPSE_LOG_UNSUPPORTED")) != 0;
    if (!ok && log) {
        std::string k = ggml_op_desc(op);
        const auto add = [&k](const char * tag, const ggml_tensor * t) {
            char b[192];
            snprintf(b, sizeof(b), " %s %s [%lld,%lld,%lld,%lld]%s", tag, ggml_type_name(t->type), (long long) t->ne[0],
                     (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_is_contiguous(t) ? "" : " nc");
            k += b;
        };
        add("->", op);
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            char tag[8];
            snprintf(tag, sizeof(tag), "s%d", i);
            add(tag, op->src[i]);
        }
        static std::mutex mtx;
        static std::set<std::string> seen;
        std::lock_guard<std::mutex> lock(mtx);
        if (seen.insert(k).second) {
            fprintf(stderr, "synapse: unsupported %s (%s)\n", k.c_str(), op->name);
        }
    }
    return ok;
}

// src1 precision (ggml_prec_set_src): the GEMM converts src1 to the weight type, or down to Q8 rank for
// quantized weights (int8 GEMV, fp8 rows). TODO: honor F32 / BF16 for F16 and quantized weights (see TODO.md)
static bool mm_src1_prec_ok(const ggml_tensor * op) {
    const ggml_prec req = ggml_prec_get_src(op, 1);
    if (req == GGML_PREC_UNDEFINED) {
        return true;
    }
    const ggml_type wt = op->src[0]->type;
    const ggml_prec cvt = wt == GGML_TYPE_F32  ? GGML_PREC_F32 :
                          wt == GGML_TYPE_BF16 ? GGML_PREC_BF16 :
                          wt == GGML_TYPE_F16  ? GGML_PREC_F16 : GGML_PREC_Q8;
    return cvt <= req;
}

static bool supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    static const char * deny = getenv("GGML_SYNAPSE_NOT");
    if (deny && *deny) {
        const char * name = ggml_op_desc(op);
        const char * p = strcasestr(deny, name);
        if (p && (p[strlen(name)] == 0 || p[strlen(name)] == ',')) {
            return false;
        }
    }
    if (op->op == GGML_OP_NONE) {
        return true;   // a leaf (weights of any size): nothing to run
    }
    if (!small_enough(op)) {
        return false;
    }
    for (int k = 0; k < GGML_MAX_SRC; k++) {
        if (op->src[k] && !small_enough(op->src[k]) && !experts_ok(op, k)) {
            return false;
        }
    }
    const ggml_tensor * s0 = op->src[0];
    const ggml_tensor * s1 = op->src[1];
    auto f32 = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32; };
    // f16 / bf16 operands of elementwise ops are converted through f32
    auto flt = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    };
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_VIEW:
        case GGML_OP_RESHAPE:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return flt(op) && flt(s0) && flt(s1) && rows_contig(s0) && rows_contig(s1) && ggml_can_repeat(s1, op);

        case GGML_OP_UNARY:
            return (unary_code(op) >= 0 || unaryx_code(op) >= 0) && flt(op) && flt(s0) && rows_contig(s0);
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_FILL:
            return flt(op) && flt(s0) && rows_contig(s0);

        case GGML_OP_GLU: {
            const ggml_glu_op g = ggml_get_glu_op(op);
            if (g != GGML_GLU_OP_SWIGLU && g != GGML_GLU_OP_GEGLU && g != GGML_GLU_OP_REGLU && glux_code(op) < 0) {
                return false;
            }
            return flt(op) && flt(s0) && rows_contig(s0) && (!s1 || (flt(s1) && rows_contig(s1)));
        }

        case GGML_OP_REPEAT:
            return flt(op) && flt(s0) && rows_contig(s0);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_SUM:
            // SUM_ROWS inherits the source type; MEAN / SUM stay F32
            if (op->op == GGML_OP_SUM_ROWS && flt(op) && flt(s0) && op->type == s0->type && rows_contig(s0)) {
                return true;
            }
            return f32(op) && f32(s0) && rows_contig(s0);
        case GGML_OP_ARGMAX:
            return op->type == GGML_TYPE_I32 && f32(s0) && rows_contig(s0);
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D: {
            const int32_t * pp = (const int32_t *) op->op_params;
            if (pp[0] != GGML_OP_POOL_MAX && pp[0] != GGML_OP_POOL_AVG) {
                return false;   // POOL_COUNT is a sentinel
            }
            // the general path (syn_pool: independent k and stride, padding) covers everything else
            if (f32(op) && f32(s0) && rows_contig(s0)) {
                return true;
            }
            // non-F32 sources: no path (the general kernel reads F32; the row path is F32 too)
            return false;
        }
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            // long rows: indices exact in f32; the sort's padded rows indexable in i32
            return op->type == GGML_TYPE_I32 && f32(s0) && rows_contig(s0) && s0->ne[0] <= (1 << 24) &&
                   (s0->ne[0] <= 4096 || op->op == GGML_OP_TOP_K ||
                    ggml_nrows(s0) * (int64_t) std::max<uint64_t>(4096, 1ull << (64 - __builtin_clzll((uint64_t) s0->ne[0] - 1))) < (1ll << 31));
        case GGML_OP_ADD_ID:
            return f32(op) && f32(s0) && f32(s1) && op->src[2]->type == GGML_TYPE_I32 && rows_contig(s0) && rows_contig(s1);
        case GGML_OP_MUL_MAT_ID: {
            const ggml_tensor * as = s0, * ids = op->src[2];
            const ggml_tensor * ar = root_of(as);
            const bool full = (const char *) as->data == (const char *) ar->data && ggml_nbytes(as) == ggml_nbytes(ar) &&
                              ggml_is_contiguous(as);
            // widening: F16 / BF16 activations (converted to the GEMM's compute type)
            return full && mm_src1_prec_ok(op) && (dq_type(as->type) || as->type == GGML_TYPE_BF16 || as->type == GGML_TYPE_F16 || as->type == GGML_TYPE_F32) &&
                   flt(s1) && f32(op) && rows_contig(s1) && ids->type == GGML_TYPE_I32 && s1->ne[3] == 1 &&
                   (s1->ne[1] == 1 || s1->ne[1] == ids->ne[0]) && s1->ne[2] == ids->ne[1];
        }

        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
            // widening: the norms inherit the source type (F32/F16/BF16), MEAN stays F32
            return flt(op) && flt(s0) && op->type == s0->type && rows_contig(s0);

        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT: {
            const ggml_type st = s0->type, dt = op->type;
            const bool ok = (st == GGML_TYPE_F32 && (dt == GGML_TYPE_F32 || dt == GGML_TYPE_F16 || dt == GGML_TYPE_BF16)) ||
                            ((st == GGML_TYPE_F16 || st == GGML_TYPE_BF16) && (dt == GGML_TYPE_F32 || dt == st)) ||
                            (st == GGML_TYPE_F16 && dt == GGML_TYPE_BF16) || (st == GGML_TYPE_BF16 && dt == GGML_TYPE_F16) ||
                            (st == GGML_TYPE_I32 && dt == GGML_TYPE_I32) ||   // (top-k indices)
                            (st == GGML_TYPE_F32 && dt == GGML_TYPE_I32) || (st == GGML_TYPE_I32 && dt == GGML_TYPE_F32) ||   // (casts)
                            // quantized destinations: a CPY variant of the setrows kernels
                            (st == GGML_TYPE_F32 && (dt == GGML_TYPE_Q8_0 || dt == GGML_TYPE_Q4_0 || dt == GGML_TYPE_Q4_1 ||
                                                     dt == GGML_TYPE_Q5_0 || dt == GGML_TYPE_Q5_1 || dt == GGML_TYPE_IQ4_NL)) ||
                            // quantized sources to F32 (the dqrows32 family: F32 output, not the f16 one)
                            ((st == GGML_TYPE_Q4_0 || st == GGML_TYPE_Q4_1 || st == GGML_TYPE_Q5_0 || st == GGML_TYPE_Q5_1 ||
                              st == GGML_TYPE_Q8_0) && dt == GGML_TYPE_F32);
            if (!ok) {
                return false;
            }
            // quantized destination: a contiguous view (written in place, see lower_cpy_quant), source rows block
            // aligned (rows quantized in the source's shape); a ggml_cast to a quantized type has no view
            if (ggml_is_quantized(dt) && (op->ne[0] % ggml_blck_size(dt) != 0 || s0->ne[0] % ggml_blck_size(dt) != 0 ||
                                          !ggml_is_contiguous(op) || op->view_src == nullptr)) {
                return false;
            }
            if (ggml_is_quantized(st) && (op->ne[0] % ggml_blck_size(st) != 0 || s0->ne[0] % ggml_blck_size(st) != 0 ||
                                          s0->nb[0] != ggml_type_size(st))) {
                return false;
            }
            // a strided destination view is written row by row: its dim 0 must be contiguous
            return rows_contig(op);
        }

        case GGML_OP_CUMSUM:
            return f32(op) && f32(s0) && rows_contig(s0);

        case GGML_OP_ARANGE:
            return f32(op);

        case GGML_OP_PAD: {
            // zero (pp[8] == 0, the sampler's padding row) or circular padding (pp[8] == 1,
            // positive modulo per dimension; the syn_pad_circ kernel)
            const int32_t * pp = (const int32_t *) op->op_params;
            bool ok = f32(op) && f32(s0) && rows_contig(s0) && (pp[8] == 0 || pp[8] == 1);
            for (int i = 0; i < 8; i++) {
                ok = ok && pp[i] >= 0;
            }
            return ok;
        }

        case GGML_OP_GET_ROWS: {
            if (s1->type != GGML_TYPE_I32 && s1->type != GGML_TYPE_I64) {
                return false;
            }
            if (s0->type == GGML_TYPE_I32) {
                // (the backend top-k sampler's candidate ids)
                return op->type == GGML_TYPE_I32 && rows_contig(s0);
            }
            if (!f32(op)) {
                return false;
            }
            if (dq_type(s0->type)) {
                // gathered from the dequantized copy of the whole (weight) tensor
                const ggml_tensor * r = root_of(s0);
                return (const char *) s0->data == (const char *) r->data && ggml_nbytes(s0) == ggml_nbytes(r) &&
                       ggml_is_contiguous(s0);
            }
            return (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16 || s0->type == GGML_TYPE_BF16) && rows_contig(s0);
        }

        case GGML_OP_SET_ROWS:
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                    ((op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 ||
                      op->type == GGML_TYPE_Q5_0 || op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_IQ4_NL) &&
                     s0->ne[0] % 32 == 0)) &&
                   flt(s0) && (s1->type == GGML_TYPE_I32 || s1->type == GGML_TYPE_I64) && rows_contig(s0) && rows_contig(op);

        case GGML_OP_CONCAT:
            // widening: same declared source / result types F32/F16/BF16; F32 takes strided (transposed) sources
            return flt(op) && op->type == s0->type && s0->type == s1->type &&
                   (op->type == GGML_TYPE_F32 || (rows_contig(s0) && rows_contig(s1)));

        case GGML_OP_SSM_CONV:
            // no tap-count ceiling: the general kernel loops k < nc with bounded scratch
            return f32(op) && f32(s0) && f32(s1) && rows_contig(s0) && rows_contig(s1);

        case GGML_OP_TOP_K_MASK: {
            // rows of top_k / score / extra as one flat dim (the general kernel takes one row stride)
            auto flat_rows = [](const ggml_tensor * t) {
                return t->nb[0] == ggml_type_size(t->type) && t->nb[2] == t->nb[1] * t->ne[1] && t->nb[3] == t->nb[2] * t->ne[2];
            };
            const ggml_tensor * sc = op->src[2], * bl = op->src[3], * ex = op->src[4];
            const bool gen = sc || bl || ex;
            return (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) && s0->type == op->type && s1->type == GGML_TYPE_I32 &&
                   rows_contig(s0) && rows_contig(s1) && op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3] < (1ll << 31) &&
                   (!gen || flat_rows(s1)) && (!sc || (sc->type == GGML_TYPE_F32 && flat_rows(sc))) &&
                   (!bl || (bl->type == GGML_TYPE_I32 && bl->nb[0] == 4)) && (!ex || (ex->type == GGML_TYPE_I32 && flat_rows(ex)));
        }
        case GGML_OP_DSV4_HC_COMB:
            return f32(op) && f32(s0) && f32(s1) && f32(op->src[2]) && s0->ne[0] == 24 && rows_contig(s0) && rows_contig(op->src[2]);
        case GGML_OP_DSV4_HC_PRE: {
            const bool gated = ggml_get_op_params_i32(op, 1) != 0;
            return f32(op) && f32(s0) && f32(s1) && s0->ne[1] <= 8 && rows_contig(s0) && (!gated || rows_contig(s1));
        }
        case GGML_OP_DSV4_HC_POST:
            return f32(op) && f32(s0) && f32(s1) && f32(op->src[2]) && (!op->src[3] || f32(op->src[3])) && s1->ne[1] <= 8 &&
                   rows_contig(s0) && rows_contig(s1);

        case GGML_OP_LIGHTNING_INDEXER: {
            const ggml_tensor * k = s1;
            const bool kq = k->type == GGML_TYPE_Q8_0 || k->type == GGML_TYPE_Q4_0 || k->type == GGML_TYPE_Q4_1 ||
                            k->type == GGML_TYPE_Q5_0 || k->type == GGML_TYPE_Q5_1;
            return f32(op) && f32(s0) && f32(op->src[2]) && op->src[3]->type == GGML_TYPE_F16 &&
                   (kq || k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_BF16) &&
                   s0->ne[1] <= 64 && rows_contig(s0) && rows_contig(k) && rows_contig(op->src[2]) && rows_contig(op->src[3]);
        }

        case GGML_OP_SSM_SCAN: {
            const ggml_tensor * x = op->src[1], * A = op->src[3], * B = op->src[4];
            const int64_t nh = x->ne[1], ng = B->ne[1];
            // no state-size cap: the general path covers the rest
            bool ok = f32(op) && op->src[6]->type == GGML_TYPE_I32 && ng > 0 && nh % ng == 0 &&
                      ggml_is_contiguous(op->src[6]);
            for (int k = 0; k < 6; k++) {
                ok = ok && f32(op->src[k]);
            }
            return ok && ssm_scan_general_ok(op);
        }

        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7: {
            const int nin = op->op == GGML_OP_RWKV_WKV7 ? 7 : op->op == GGML_OP_RWKV_WKV6 ? 6 : 5;
            // WKV6: the head size of the time_faaaa weight [S, H] (the loader's placement probe builds the
            // other operands with a dummy size)
            const int64_t S = op->src[op->op == GGML_OP_RWKV_WKV6 ? 3 : 1]->ne[0], T = op->src[1]->ne[2], ns = op->src[nin - 1]->ne[1];
            // no S%64: the general path covers the rest (WKV7 holds a state row in local memory: S <= 8192)
            bool ok = f32(op) && S > 0 && ns > 0 && T % ns == 0 && (op->op != GGML_OP_RWKV_WKV7 || S <= 64 * 128);
            for (int k = 0; k < nin; k++) {
                ok = ok && f32(op->src[k]) && ggml_is_contiguous(op->src[k]);
            }
            return ok;
        }

        case GGML_OP_GATED_DELTA_NET: {
            const ggml_tensor * v = op->src[2];
            const int64_t S = v->ne[0];
            return ggml_get_op_params_i32(op, 0) >= 1 && S % 64 == 0 && S <= 256 &&
                   op->src[0]->ne[3] == v->ne[3] && op->src[1]->ne[3] == v->ne[3] &&
                   rows_contig(op->src[0]) && rows_contig(op->src[1]) && rows_contig(v);
        }

        case GGML_OP_SOFT_MAX:
            return f32(op) && f32(s0) && rows_contig(s0) &&
                   (!op->src[2] || (f32(op->src[2]) && ggml_is_contiguous(op->src[2]) && ggml_nelements(op->src[2]) >= s0->ne[2])) &&
                   (!s1 || ((s1->type == GGML_TYPE_F16 || f32(s1)) && rows_contig(s1)));

        case GGML_OP_MUL_MAT: {
            const ggml_tensor * w = s0;
            const ggml_tensor * wr = root_of(w);
            const bool full = (const char *) w->data == (const char *) wr->data && ggml_nbytes(w) == ggml_nbytes(wr) &&
                              ggml_is_contiguous(w);
            int64_t ncol, nb;
            // widening: F16 / BF16 activations admitted (the GEMM inputs stay same-type: the weight
            // must be F32/F16/BF16 or dequantized-to-BF16; the activation converted to the GEMM's
            // compute type by mm_activation / x_columns' get_f32)
            if (!flt(s1) || !f32(op) || !mm_batches(w, s1, ncol, nb) || !mm_src1_prec_ok(op)) {
                return false;
            }
            if (dq_type(w->type)) {
                return full;
            }
            return (w->type == GGML_TYPE_BF16 || w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_F32) && rows_contig(w);
        }

        case GGML_OP_FLASH_ATTN_EXT: {
            const ggml_tensor * q = s0, * k = s1, * v = op->src[2], * mk = op->src[3];
            // fa_config_ok is the shared valid-operation predicate (support and lowering); the
            // width/multiple and type-pair restrictions live in fa_tpc_config_ok only
            return fa_config_ok(op) && rows_contig(q) && rows_contig(k) && rows_contig(v) && (!mk || rows_contig(mk));
        }

        case GGML_OP_ROPE: {
            rope_cfg c;
            int n_offs;
            // leaf positions use the host table; computed positions the device table node
            return rope_config(op, c, n_offs) && rows_contig(s0) &&
                   (!op->src[2] || ggml_nelements(op->src[2]) >= (c.mode == GGML_ROPE_TYPE_VISION ? c.n_dims : c.n_dims / 2));
        }

        case GGML_OP_OUT_PROD: {
            if (op->type != GGML_TYPE_F32 || !flt(s0) || !flt(s1)) {
                return false;
            }
            if (s0->ne[0] > 1 && s0->nb[0] != ggml_type_size(s0->type)) {
                return false;   // non-transposed src0 only
            }
            if (s1->ne[1] != s0->ne[1]) {
                return false;
            }
            if (s1->ne[2] % s0->ne[2] != 0 || s1->ne[3] % s0->ne[3] != 0) {
                return false;
            }
            const int64_t r2 = s1->ne[2] / s0->ne[2], r3 = s1->ne[3] / s0->ne[3];
            return s0->ne[0] > 0 && s0->ne[1] > 0 && s1->ne[0] > 0 && r2 >= 1 && r3 >= 1 &&
                   op->ne[0] == s0->ne[0] && op->ne[1] == s1->ne[0] && op->ne[2] == s1->ne[2] &&
                   op->ne[3] == s1->ne[3];
        }

        case GGML_OP_SET:
        case GGML_OP_ACC: {
            // same-type F32 or I32 SET; ACC F32. Params {nb1, nb2, nb3, offset, inplace} in bytes.
            if (op->op == GGML_OP_ACC) {
                return op->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32 &&
                       rows_contig(s0) && rows_contig(s1);
            }
            if (!((op->type == GGML_TYPE_F32 && s0->type == GGML_TYPE_F32 && s1->type == GGML_TYPE_F32) ||
                  (op->type == GGML_TYPE_I32 && s0->type == GGML_TYPE_I32 && s1->type == GGML_TYPE_I32))) {
                return false;
            }
            const int32_t * pp = (const int32_t *) op->op_params;
            const int64_t es = ggml_type_size(op->type);
            const int64_t last = pp[3] + (s1->ne[1] > 0 ? (int64_t) (s1->ne[1] - 1) * pp[0] : 0) +
                                 (s1->ne[2] > 0 ? (int64_t) (s1->ne[2] - 1) * pp[1] : 0) +
                                 (s1->ne[3] > 0 ? (int64_t) (s1->ne[3] - 1) * pp[2] : 0) + s1->ne[0] * es;
            return pp[0] >= 0 && pp[1] >= 0 && pp[2] >= 0 && pp[3] >= 0 && pp[0] % es == 0 && pp[1] % es == 0 &&
                   pp[2] % es == 0 && pp[3] % es == 0 && last <= (int64_t) ggml_nbytes(s0) &&
                   rows_contig(s1) && ggml_nelements(s1) <= ggml_nelements(op);
        }

        case GGML_OP_ADD1:
            // shares the ADD lowering with a scalar src1 (broadcast over dim 0); same types
            return flt(op) && op->type == s0->type && s1->type == op->type && ggml_nelements(s1) == 1 &&
                   rows_contig(s0) && rows_contig(s1);

        case GGML_OP_COUNT_EQUAL:
            return op->type == GGML_TYPE_I64 && s0->type == GGML_TYPE_I32 && s1->type == GGML_TYPE_I32 &&
                   ggml_are_same_shape(s0, s1);

        case GGML_OP_GROUP_NORM:
            return f32(op) && f32(s0) && rows_contig(s0) && ggml_get_op_params_i32(op, 0) >= 1 &&
                   s0->ne[2] % ggml_get_op_params_i32(op, 0) == 0;

        case GGML_OP_DIAG:
            return f32(op) && f32(s0) && s0->ne[1] == 1 && op->ne[0] == s0->ne[0] && op->ne[1] == s0->ne[0] &&
                   op->ne[2] == s0->ne[2] && op->ne[3] == s0->ne[3];

        case GGML_OP_TRI:
            return f32(op) && f32(s0) && s0->ne[0] == s0->ne[1] && op->ne[0] == s0->ne[0] &&
                   ggml_get_op_params_i32(op, 0) >= 0 && ggml_get_op_params_i32(op, 0) <= 3;

        case GGML_OP_DIAG_MASK_INF:
            return f32(op) && f32(s0) && ggml_are_same_shape(s0, op) && ggml_get_op_params_i32(op, 0) >= 0;

        case GGML_OP_SOLVE_TRI:
            return f32(op) && f32(s0) && f32(s1) && s0->ne[0] == s0->ne[1] && s0->ne[1] == s1->ne[1] &&
                   s0->ne[2] == s1->ne[2] && s0->ne[3] == s1->ne[3] && ggml_is_contiguous(s0) &&
                   ggml_is_contiguous(s1) && op->ne[0] == s1->ne[0] && s0->ne[0] <= 256;   // (the kernel's local rows)

        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
            // the kernel tensor (src0) supplies geometry only: any type
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && f32(s1) && rows_contig(s1);

        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
            return op->type == GGML_TYPE_F32 && f32(s1) &&
                   (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16 || s0->type == GGML_TYPE_BF16);

        case GGML_OP_CONV_2D_DW:
            // CWHN: the CPU reads the kernel as (ky KW + kx) C + c, so its channels must be contiguous
            return op->type == GGML_TYPE_F32 && f32(s1) &&
                   (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16) &&
                   (ggml_is_contiguous(s1) || (ggml_is_contiguous_channels(s1) && s0->nb[3] == ggml_type_size(s0->type)));

        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            return op->type == GGML_TYPE_F32 && f32(s1) &&
                   (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16);

        case GGML_OP_COL2IM_1D:
            return op->type == s0->type && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 ||
                                            op->type == GGML_TYPE_BF16) && ggml_is_contiguous(s0);

        case GGML_OP_UPSCALE:
            return f32(op) && f32(s0);

        case GGML_OP_PAD_REFLECT_1D: {
            const int32_t * pp = (const int32_t *) op->op_params;
            return f32(op) && f32(s0) && rows_contig(s0) && pp[0] >= 0 && pp[1] >= 0 &&
                   pp[0] < s0->ne[0] && pp[1] < s0->ne[0];
        }

        case GGML_OP_ROLL: {
            const int32_t * pp = (const int32_t *) op->op_params;
            for (int d = 0; d < 4; d++) {
                if (pp[d] >= s0->ne[d] || -pp[d] >= s0->ne[d]) {
                    return false;
                }
            }
            return f32(op) && f32(s0);
        }

        case GGML_OP_TIMESTEP_EMBEDDING:
            return f32(op) && f32(s0) && rows_contig(s0);

        default:
            return false;
    }
}

/* ---------------- fusion planning ---------------- */

static bool is_unary(const ggml_tensor * t, ggml_unary_op u) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == u;
}

static bool is_gamma_of(const ggml_tensor * w, const ggml_tensor * x) {
    // a row vector broadcast over x's rows
    return w->type == GGML_TYPE_F32 && w->ne[0] == x->ne[0] && ggml_nrows(w) == 1 && w->nb[0] == sizeof(float);
}

void lowerer::plan_router(const ggml_tensor * sg) {
    const int64_t E = sg->ne[0];
    if (sg->type != GGML_TYPE_F32 || sg->view_src || is_output[sg] || E > 512 || sg->ne[2] != 1 || sg->ne[3] != 1 ||
        sg->src[0]->type != GGML_TYPE_F32 || sg->src[0]->nb[0] != sizeof(float)) {
        return;
    }
    // a plain softmax (no mask / sinks, scale 1, no ALiBi)
    const bool smax = sg->op == GGML_OP_SOFT_MAX;
    static const bool no_smax = getenv("GGML_SYNAPSE_NO_ROUTER_SMAX") != nullptr;
    if (smax && (no_smax || sg->src[1] || sg->src[2] || ggml_get_op_params_f32(sg, 0) != 1.0f || ggml_get_op_params_f32(sg, 1) != 0.0f)) {
        return;
    }
    // the only one consumer of t (through view ops, which have no other consumers), or null
    auto single = [&](const ggml_tensor * t, std::vector<const ggml_tensor *> & seen) -> const ggml_tensor * {
        while (true) {
            if (is_output[t] || cons[t].size() != 1) {
                return nullptr;
            }
            const ggml_tensor * c = cons[t][0];
            if (!is_view_op(c)) {
                return c;
            }
            seen.push_back(c);
            t = c;
        }
    };
    router_plan pl;
    pl.smax = smax;
    std::vector<const ggml_tensor *> fused, views;
    const ggml_tensor * add = nullptr, * gr = nullptr;
    for (const ggml_tensor * c : cons[sg]) {
        if (!smax && c->op == GGML_OP_ADD && c->src[0] == sg && !add) {
            add = c;
        } else if (smax && c->op == GGML_OP_ARGSORT && c->src[0] == sg && !add) {
            add = c;
        } else if (is_view_op(c) && cons[c].size() == 1 && !is_output[c] && cons[c][0]->op == GGML_OP_GET_ROWS &&
                   cons[c][0]->src[0] == c && !gr) {
            views.push_back(c);
            gr = cons[c][0];
        } else {
            return;
        }
    }
    if (!add || !gr) {
        return;
    }
    const ggml_tensor * b = smax ? nullptr : add->src[1];
    if (!smax && (b->type != GGML_TYPE_F32 || b->ne[0] != E || ggml_nelements(b) != E || !ggml_is_contiguous(b) ||
                  is_output[add] || cons[add].size() != 1)) {
        return;
    }
    const ggml_tensor * as = smax ? add : cons[add][0];
    if (as->op != GGML_OP_ARGSORT || as->src[0] != (smax ? sg : add) || ggml_get_op_params_i32(as, 0) != GGML_SORT_ORDER_DESC || is_output[as] ||
        as->ne[1] != sg->ne[1]) {
        return;
    }
    int k = 0;
    for (const ggml_tensor * c : cons[as]) {
        const size_t vo = c->op == GGML_OP_VIEW ? *(const size_t *) c->op_params : 1;
        if (c->op != GGML_OP_VIEW || vo != 0 || c->nb[1] != as->nb[1] || (k && c->ne[0] != k)) {
            return;
        }
        k = (int) c->ne[0];
    }
    if (k < 1 || k > 64 || gr->src[1]->view_src != as || gr->ne[1] != k || gr->type != GGML_TYPE_F32) {
        return;
    }
    fused = { add, as, gr };
    if (smax) {
        fused = { as, gr };
    }
    const ggml_tensor * w = gr;
    std::vector<const ggml_tensor *> seen;
    if (!is_output[gr]) {
        const ggml_tensor * sum = nullptr, * div = nullptr;
        bool other = false;
        std::vector<const ggml_tensor *> gv;
        std::function<void(const ggml_tensor *)> walk = [&](const ggml_tensor * t) {
            for (const ggml_tensor * c : cons[t]) {
                if (is_view_op(c) && !is_output[c]) {
                    gv.push_back(c);
                    walk(c);
                } else if (c->op == GGML_OP_SUM_ROWS && !sum) {
                    sum = c;
                } else if (c->op == GGML_OP_DIV && c->src[0] == t && !div) {
                    div = c;
                } else {
                    other = true;
                }
            }
        };
        walk(gr);
        if (!other && sum && div) {
            const ggml_tensor * cl = cons[sum].size() == 1 && !is_output[sum] ? cons[sum][0] : nullptr;
            if (cl && cl->op == GGML_OP_CLAMP && cons[cl].size() == 1 && cons[cl][0] == div && div->src[1] == cl && !is_output[cl] &&
                ggml_get_op_params_f32(cl, 1) == INFINITY) {
                pl.norm = true;
                pl.cmin = ggml_get_op_params_f32(cl, 0);
                fused.insert(fused.end(), { sum, cl, div });
                views.insert(views.end(), gv.begin(), gv.end());
                w = div;
            } else {
                return;
            }
        } else if (other && (sum || div)) {
            return;
        }
    }
    {
        std::vector<const ggml_tensor *> sv;
        const ggml_tensor * c = single(w, sv);
        if (c && c->op == GGML_OP_SCALE && ggml_get_op_params_f32(c, 1) == 0.0f) {
            pl.scale = ggml_get_op_params_f32(c, 0);
            fused.push_back(c);
            views.insert(views.end(), sv.begin(), sv.end());
            w = c;
        }
    }
    if (w->type != GGML_TYPE_F32 || ggml_nelements(w) != (int64_t) k * sg->ne[1] || !ggml_is_contiguous(w) || w->view_src) {
        return;
    }
    for (const ggml_tensor * f : fused) {
        if (skip[f] || f->view_src) {
            return;
        }
    }
    pl.bias = b;
    pl.argsort = as;
    pl.weights = w;
    pl.k = k;
    for (const ggml_tensor * f : fused) {
        skip[f] = true;
    }
    router_plans[sg] = pl;
}

bool lowerer::lower_router(const ggml_tensor * sg, const router_plan & pl) {
    opnd ol, ob;
    if (!get(sg->src[0], ol) || (pl.bias && !get(pl.bias, ob))) {
        return false;
    }
    if (!pl.bias) {
        ob = ol;   // (not read)
    }
    const int64_t E = sg->ne[0], T = sg->ne[1];
    // the ids land in the argsort's memory when that is user-allocated (read back as a value, as by
    // test-backend-ops), as for any value in a non-compute buffer
    if (pl.argsort->buffer && ggml_backend_buffer_get_usage(pl.argsort->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        is_output[pl.argsort] = true;
    }
    const int ids = fresh_out(pl.argsort);
    const ggml_tensor * w = pl.weights;
    const int lw = is_output[w] ? new_lt(syn_type_single, { (uint64_t) ggml_nelements(w) }, LT_OUT, (uint64_t) (uintptr_t) w->data) :
                                  new_lt(syn_type_single, { (uint64_t) ggml_nelements(w) });
    roots[w].lt = lw;
    params_t pr;
    pr.i(nmem_for(T, 1)).i(E).i(T).i(pl.k).i(ol.off).i(ol.nb[1]).i(ob.off).f(pl.cmin).f(pl.scale).i(pl.norm).i(pl.smax);
    add_node("syn_router_f32", { ol.lt, ob.lt }, { ids, lw }, pr);
    return true;
}

/* DSA attention mask: ADD(view(SET_ROWS(view(FILL(kq, -inf)), FILL(0), top_k)), kq). With top-k selecting
   every key (k == n_kv: the indices are a permutation) it equals kq */
void lowerer::plan_dsa_mask(const ggml_tensor * add) {
    const ggml_tensor * kq = add->src[1], * v2 = add->src[0];
    if (add->type != kq->type || !ggml_are_same_shape(add, kq) || is_output[add] || add->view_src || !is_view_op(v2) ||
        cons[add].empty()) {
        return;
    }
    for (const ggml_tensor * c : cons[add]) {
        if (c->op != GGML_OP_FLASH_ATTN_EXT || c->src[3] != add) {
            return;
        }
    }
    const ggml_tensor * sr = v2->src[0];
    if (!sr || sr->op != GGML_OP_SET_ROWS || is_output[sr] || cons[sr].size() != 1 || cons[v2].size() != 1) {
        return;
    }
    const ggml_tensor * f1 = sr->view_src, * z = sr->src[0], * idx = sr->src[1];   // (SET_ROWS: values, indices, dst)
    if (!f1 || f1->op != GGML_OP_FILL || f1->src[0] != kq || ggml_get_op_params_f32(f1, 0) != -INFINITY || is_output[f1] ||
        !z || z->op != GGML_OP_FILL || ggml_get_op_params_f32(z, 0) != 0.0f || is_output[z] || !idx ||
        idx->ne[0] != kq->ne[0] || sr->ne[1] != kq->ne[0]) {
        return;
    }
    mask_alias[add] = kq;
    for (const ggml_tensor * t : { add, sr, f1, z }) {
        skip[t] = true;
    }
}

bool lowerer::lower_moe_sum(const ggml_tensor * n, const norm_plan & pl) {
    opnd oe, oc, ov;
    const auto pg = paged_rows.find(pl.ebase);
    if (pg != paged_rows.end()) {
        // the expert outputs as page rows (lower_moe_paged)
        if ((pl.extra && !get(pl.extra, oc)) || !get(pl.wts, ov)) {
            return false;
        }
        if (!pl.extra) {
            oc = ov;   // (not read: factor 0)
            oc.off = 0;
            oc.nb[1] = 0;
        }

        const int64_t ne0 = n->ne[0], T = n->ne[1];
        const int out = begin_out(n);
        params_t pm;
        pm.i(nmem_for(T * ((ne0 + 63) / 64), 4)).i(ne0).i(T).i(pl.nslices).i(0).i(0).i(0).i(oc.off).i(oc.nb[1])
          .f(pl.extra ? 1.0f : 0.0f).i(ov.off).i(ov.nb[1]).i(ov.nb[2]).i(0);
        add_node("syn_moe_sumi_f32", { pg->second.y, oc.lt, ov.lt, pg->second.pairs }, { out }, pm);
        return end_out(n, out);
    }
    if (!get(pl.ebase, oe) || (pl.extra && !get(pl.extra, oc)) || (pl.wts && !get(pl.wts, ov))) {
        return false;
    }
    if (!pl.wts) {
        ov = oe;   // (not read: unit weights)
    }
    if (!pl.extra) {
        oc = ov;   // (not read: factor 0)
        oc.off = 0;
        oc.nb[1] = 0;
    }
    oe.off += (int64_t) (pl.slice0->view_offs / sizeof(float));
    const int64_t ne0 = n->ne[0], T = n->ne[1];
    const int out = begin_out(n);
    params_t pr;
    pr.i(nmem_for(T * ((ne0 + 63) / 64), 4)).i(ne0).i(T).i(pl.nslices).i((int32_t) pl.slice_stride)
      .i(oe.off).i((int64_t) (pl.slice0->nb[1] / sizeof(float))).i(oc.off).i(oc.nb[1]).f(pl.extra ? 1.0f : 0.0f)
      .i(ov.off).i(ov.nb[1]).i(ov.nb[2]).i(pl.wts == nullptr);
    add_node("syn_moe_sum_f32", { oe.lt, oc.lt, ov.lt }, { out }, pr);
    return end_out(n, out);
}

void lowerer::plan_fusions() {
    static const bool off = getenv("GGML_SYNAPSE_NO_FUSE") != nullptr;
    if (off) {
        return;
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) && n->op == GGML_OP_ADD) {
            plan_dsa_mask(n);
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) && ((n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SIGMOID) ||
                                                       n->op == GGML_OP_SOFT_MAX)) {
            plan_router(n);
        }
    }
    static const bool no_gdn = getenv("GGML_SYNAPSE_NO_FUSE_GDN") != nullptr;
    for (int i = 0; i < cg->n_nodes && !no_gdn; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->op != GGML_OP_GATED_DELTA_NET) {
            continue;
        }
        gdn_plan pl;
        std::vector<const ggml_tensor *> members;
        const ggml_tensor * anchor = nullptr;
        if (!match_gdn(n, pl, members, anchor)) {
            if (getenv("GGML_SYNAPSE_VERBOSE")) {
                fprintf(stderr, "synapse: gdn fusion not matched at %s: %s\n", n->name, err.c_str());
            }
            err.clear();
            continue;
        }
        for (const ggml_tensor * m : members) {
            if (m != anchor) {
                skip[m] = true;
            }
        }
        gdn_plans[anchor] = pl;
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) && n->op == GGML_OP_CONCAT && !skip[n]) {
            plan_convx(n);
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->op != GGML_OP_FLASH_ATTN_EXT || skip[n] || !fa_sdpa(n)) {
            continue;
        }
        const ggml_tensor * q = n->src[0];
        const ggml_tensor * r = q->src[0];
        // (through the per-stream view of one stream: the rope's own layout)
        if (r && r->op == GGML_OP_VIEW && r->src[0] && r->data == r->src[0]->data && ggml_are_same_shape(r, r->src[0]) &&
            r->nb[1] == r->src[0]->nb[1] && r->nb[2] == r->src[0]->nb[2] && cons[r].size() == 1) {
            r = r->src[0];
        }
        // the rope must be computed in this graph (a node of another split is an input here, in a card worker without srcs)
        if (q->op != GGML_OP_PERMUTE || !r || !order.count(q) || !order.count(r) || r->op != GGML_OP_ROPE ||
            r->type != GGML_TYPE_F32 || !r->src[0] || r->src[0]->type != GGML_TYPE_F32 ||
            r->view_src || is_output[r] || skip[r] || cons[r].size() != 1 || cons[q].size() != 1 || rope_fa.count(r)) {
            continue;
        }
        // q = permute(r, 0, 2, 1, 3): [D, nq, H] over r's [D, H, nq]
        if (q->ne[0] != r->ne[0] || q->ne[1] != r->ne[2] || q->ne[2] != r->ne[1] || r->ne[3] != 1 || q->nb[1] != r->nb[2] ||
            q->nb[2] != r->nb[1] || !ggml_is_contiguous(r)) {
            continue;
        }
        rope_fa[r] = n;
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        // F32 only: half-typed norms take the typed path
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->op != GGML_OP_RMS_NORM || n->view_src || is_output[n] ||
            n->type != GGML_TYPE_F32 || n->src[0]->type != GGML_TYPE_F32) {
            continue;
        }
        const auto & cn = cons[n];
        if (cn.size() != 1) {
            continue;
        }
        const ggml_tensor * mul = cn[0];
        if (mul->op != GGML_OP_MUL || mul->view_src || mul->src[0] != n || !is_gamma_of(mul->src[1], n) ||
            !ggml_are_same_shape(mul, n) || skip[mul]) {
            continue;
        }
        norm_plan pl;
        pl.mul = mul;
        pl.w = mul->src[1];
        // the add's other users must come after the norm
        const ggml_tensor * a = n->src[0];
        if (a->op == GGML_OP_ADD && a->view_src == nullptr && ggml_are_same_shape(a, n) && order.count(a) &&
            !skip[a] && a->type == GGML_TYPE_F32 && a->src[0]->type == GGML_TYPE_F32 && a->src[1]->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(a->src[0], a) && ggml_are_same_shape(a->src[1], a) &&
            a->src[0]->nb[0] == sizeof(float) && a->src[1]->nb[0] == sizeof(float)) {
            bool ok = true;
            for (const ggml_tensor * c : cons[a]) {
                if (c != n && order[c] < order[n]) {
                    ok = false;
                }
            }
            if (ok) {
                pl.add = a;
                skip[a] = true;
                match_slices(pl, n);
            }
        }
        const ggml_tensor * rp0 = cons[mul].size() == 1 ? cons[mul][0] : nullptr;
        const bool vlm_rows = n->ne[0] % 256 == 0 && n->ne[0] <= 4096;
        const bool c_rows = rp0 && n->op == GGML_OP_RMS_NORM && n->ne[0] % 64 == 0 && n->ne[0] <= 1024 && n->ne[3] == 1 &&
                            !want_bf16(rp0);
        if (!pl.add && rp0 && rp0->op == GGML_OP_ROPE && rp0->src[0] == mul && !is_output[mul] && rp0->view_src == nullptr &&
            (vlm_rows || c_rows) && n->src[0]->nb[0] == 4) {
            const ggml_tensor * rp = cons[mul][0];
            rope_cfg c;
            int n_offs;
            if (rope_config(rp, c, n_offs) && rope_neox_f32(rp, c) && n_offs == 0 && (c.n_dims == 64 || c.n_dims % 128 == 0) &&
                c.n_dims <= n->ne[0] && rp->src[1]->op == GGML_OP_NONE) {
                pl.rope = rp;
                skip[rp] = true;
            }
        }
        skip[mul] = true;
        norm_plans[n] = pl;
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * f = cg->nodes[i];
        if ((f->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || f->op != GGML_OP_ADD || skip[f] || f->view_src ||
            f->type != GGML_TYPE_F32 || !ggml_is_contiguous(f) || f->ne[2] != 1 || f->ne[3] != 1) {
            continue;
        }
        for (int side = 0; side < 2; side++) {
            const ggml_tensor * x = f->src[1 - side];
            if (x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, f) || x->nb[0] != sizeof(float)) {
                continue;
            }
            norm_plan pl;
            pl.add = f;
            if (match_chain(pl, f->src[side], x, nullptr, side, false, true) && pl.ebase) {
                moe_sum_plans[f] = pl;
                break;
            }
        }
        static const bool no_top = getenv("GGML_SYNAPSE_NO_MOE_SUM_TOP") != nullptr;
        if (!no_top && !moe_sum_plans.count(f) && !skip[f] &&
            (is_output[f] || cons[f].size() != 1 || cons[f][0]->op != GGML_OP_ADD)) {
            norm_plan pl;
            pl.add = f;
            if (match_chain(pl, f, nullptr, nullptr, 0, true, true)) {
                skip[f] = false;
                moe_sum_plans[f] = pl;
            }
        }
    }
    {
        auto simple = [&](const ggml_tensor * t) {
            return t->op == GGML_OP_SET_ROWS && (t->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_nelements(t) > 0 &&
                   (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) &&
                   t->src[0]->type == GGML_TYPE_F32 && t->src[0]->ne[2] == 1 && t->src[0]->ne[3] == 1 &&
                   t->src[0]->nb[0] == 4 && t->src[1]->ne[1] == 1 && t->src[1]->ne[2] == 1 &&
                   ggml_is_contiguous(t->src[1]) && t->nb[0] == ggml_type_size(t->type) && !skip[t];
        };
        const ggml_tensor * prev = nullptr;
        bool prev_read = false;   // the first write's destination is read before the second write
        for (int i = 0; i < cg->n_nodes; i++) {
            const ggml_tensor * t = cg->nodes[i];
            if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }
            if (t->op != GGML_OP_SET_ROWS) {
                for (int k = 0; prev && k < GGML_MAX_SRC; k++) {
                    if (t->src[k] && root_of(t->src[k]) == root_of(prev)) {
                        prev_read = true;
                    }
                }
                continue;
            }
            if (prev && !prev_read && simple(prev) && simple(t) && prev->type == t->type && prev->src[0]->ne[0] == t->src[0]->ne[0] &&
                prev->src[0]->ne[1] == t->src[0]->ne[1] && prev->src[1]->type == t->src[1]->type &&
                root_of(prev) != root_of(t)) {
                setrows_pair[t] = prev;
                skip[prev] = true;
                prev = nullptr;
            } else {
                prev = t;
                prev_read = false;
            }
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * fa = cg->nodes[i];
        if ((fa->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fa->op != GGML_OP_FLASH_ATTN_EXT || fa->view_src ||
            is_output[fa] || cons[fa].size() != 1 || fa_mme(fa) != 0) {
            continue;
        }
        // a tiled FA's output is finalized after its windows: the gate stays a separate op
        if (!no_tile && fa_tile_shape_ok(fa)) {
            continue;
        }
        if (!fa_tpc_config_ok(fa)) {
            continue;
        }
        const ggml_tensor * rs = cons[fa][0];
        if (!(is_view_op(rs) && rs->data == fa->data && ggml_is_contiguous(rs) && ggml_nelements(rs) == ggml_nelements(fa) &&
              cons[rs].size() == 1 && !is_output[rs])) {
            continue;
        }
        const ggml_tensor * m = cons[rs][0];
        if (m->op != GGML_OP_MUL || m->view_src || !ggml_are_same_shape(m->src[0], m->src[1]) || is_output[m]) {
            continue;
        }
        const ggml_tensor * sg = m->src[0] == rs ? m->src[1] : m->src[0];
        if (!is_unary(sg, GGML_UNARY_OP_SIGMOID) || sg->view_src || cons[sg].size() != 1 || is_output[sg]) {
            continue;
        }
        const ggml_tensor * g = sg->src[0];
        const ggml_tensor * cont = nullptr;
        if (g->op == GGML_OP_CONT && g->view_src == nullptr && cons[g].size() == 1 && !is_output[g]) {
            cont = g;
            g = g->src[0];
        }
        // the gate as [D, H, nq, B] like the attention output
        const int64_t D = fa->ne[0], H = fa->ne[1], nq = fa->ne[2], B = fa->ne[3];
        if (g->type != GGML_TYPE_F32 || g->nb[0] != 4 || ggml_nelements(g) != D * H * nq * B || g->ne[0] != D ||
            g->ne[1] != H) {
            continue;
        }
        gate_plan gp;
        gp.mul = m;
        gp.gate = g;
        gate_plans[fa] = gp;
        skip[m] = true;
        skip[sg] = true;
        if (cont) {
            skip[cont] = true;
        }
    }
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || !is_unary(n, GGML_UNARY_OP_SILU) || n->view_src || skip[n]) {
            continue;
        }
        const ggml_tensor * sc = n->src[0];
        if (sc->op == GGML_OP_SCALE && ggml_get_op_params_f32(sc, 1) == 0.0f && sc->view_src == nullptr && !is_output[sc] &&
            !skip[sc] && cons[sc].size() == 1 && sc->type == GGML_TYPE_F32 && sc->src[0]->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(sc, sc->src[0])) {
            silu_scale[n] = sc;
            skip[sc] = true;
        }
    }
    static const bool no_hc = getenv("GGML_SYNAPSE_NO_FUSE_HC") != nullptr;
    for (int i = 0; i < cg->n_nodes && !no_hc; i++) {
        const ggml_tensor * n = cg->nodes[i];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) && n->op == GGML_OP_DSV4_HC_POST && n->view_src == nullptr) {
            plan_hc(n);
        }
    }
}

void lowerer::plan_hc(const ggml_tensor * n) {
    const int64_t ne0 = n->ne[0], hc = n->ne[1];
    hc_plan pl;
    auto one_use = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->view_src == nullptr && !is_output[t] && !skip[t] && cons[t].size() == 1;
    };
    auto plain_scale = [&](const ggml_tensor * t) {
        return t->op == GGML_OP_SCALE && ggml_get_op_params_f32(t, 1) == 0.0f && one_use(t);
    };
    // post = scale(sigmoid(scale(p)))
    const ggml_tensor * w = n->src[2];
    if (plain_scale(w) && is_unary(w->src[0], GGML_UNARY_OP_SIGMOID) && one_use(w->src[0]) &&
        plain_scale(w->src[0]->src[0])) {
        const ggml_tensor * a = w->src[0]->src[0];
        const ggml_tensor * p0 = a->src[0];
        if (p0->type == GGML_TYPE_F32 && ggml_are_same_shape(p0, w)) {
            pl.p = p0;
            pl.pa = ggml_get_op_params_f32(a, 0);
            pl.pb = ggml_get_op_params_f32(w, 0);
            skip[w] = skip[w->src[0]] = skip[a] = true;
        }
    }
    // rms_norm(out) * gamma[ne0, hc]
    if (ne0 % 128 == 0 && ne0 <= 8192 && hc <= 8 && n->ne[3] == 1) {
        for (const ggml_tensor * r : cons[n]) {
            if (r->op != GGML_OP_RMS_NORM || r->src[0] != n || !ggml_are_same_shape(r, n) || !one_use(r) ||
                norm_plans.count(r)) {
                continue;
            }
            const ggml_tensor * m = cons[r][0];
            const ggml_tensor * g = m->src[1];
            // the fused node runs at the HC_POST: a gamma computed later in the graph is not ready there
            // (f.ex. Engram's rms_norm(key), [ne0, hc] with one token)
            const ggml_tensor * gr = root_of(g);
            if (m->op != GGML_OP_MUL || m->src[0] != r || m->view_src || skip[m] || !ggml_are_same_shape(m, r) ||
                g->type != GGML_TYPE_F32 || g->ne[0] != ne0 || g->ne[1] != hc || ggml_nelements(g) != ne0 * hc ||
                !ggml_is_contiguous(g) || (!is_output[m] && want_bf16(m)) || (produced.count(gr) && order[gr] > order[n])) {
                continue;
            }
            pl.norm = r;
            pl.mul = m;
            // a bf16 copy for GEMM readers of the norm (directly or through whole-tensor reshapes)
            std::function<bool(const ggml_tensor *, int)> gemm_reader = [&](const ggml_tensor * t, int depth) {
                for (const ggml_tensor * c : cons[t]) {
                    if (c->op == GGML_OP_MUL_MAT && c->src[1] == t && c->src[0] != t && bf16_gemm_weight(c->src[0])) {
                        return true;
                    }
                    if (depth < 4 && (c->op == GGML_OP_RESHAPE || c->op == GGML_OP_VIEW) && c->data == t->data &&
                        ggml_nelements(c) == ggml_nelements(t) && ggml_is_contiguous(c) && gemm_reader(c, depth + 1)) {
                        return true;
                    }
                }
                return false;
            };
            pl.bf = gemm_reader(m, 0);
            skip[r] = skip[m] = true;
            break;
        }
    }
    if (pl.p || pl.norm) {
        hc_plans[n] = pl;
    }
}

/* ---------------- gated delta net block ---------------- */


bool lowerer::match_gdn(const ggml_tensor * G, gdn_plan & pl, std::vector<const ggml_tensor *> & mem,
                        const ggml_tensor *& anchor) {
#define REQ(c) do { if (!(c)) { err = "gdn: " #c; return false; } } while (0)
    const ggml_tensor * q = G->src[0], * k = G->src[1], * v = G->src[2];
    const ggml_tensor * g = G->src[3], * b = G->src[4], * st = G->src[5];
    const int64_t S = v->ne[0], H = v->ne[1], T = v->ne[2], ns = v->ne[3];
    // T > 1 (MTP verify batches): the multi-token kernels
    REQ(ggml_get_op_params_i32(G, 0) >= 1 && S == 128 && T >= 1 && T <= 8 && g->ne[0] == 1);
    REQ(G->view_src == nullptr && !is_output[G]);
    pl.G = G;
    // q, k: scale(rms_norm(view of Y))
    const ggml_tensor * qs = q, * ks = k;
    // the fused producers are computed in this graph (a node of another split is an input here, in a card worker
    // without srcs)
    auto in_graph = [&](const ggml_tensor * t) { return order.count(t) > 0; };
    REQ(in_graph(qs) && in_graph(ks));
    REQ(qs->op == GGML_OP_SCALE && ks->op == GGML_OP_SCALE && qs->view_src == nullptr && ks->view_src == nullptr);
    REQ(ggml_get_op_params_f32(qs, 1) == 0.0f && ggml_get_op_params_f32(ks, 1) == 0.0f);
    REQ(ggml_get_op_params_f32(qs, 0) == ggml_get_op_params_f32(ks, 0));
    const ggml_tensor * qr = qs->src[0], * kr = ks->src[0];
    REQ(in_graph(qr) && in_graph(kr));
    REQ(qr->op == GGML_OP_RMS_NORM && kr->op == GGML_OP_RMS_NORM && qr->view_src == nullptr && kr->view_src == nullptr);
    REQ(ggml_get_op_params_f32(qr, 0) == ggml_get_op_params_f32(kr, 0));
    pl.eps_qk = ggml_get_op_params_f32(qr, 0);
    pl.qk_scale = ggml_get_op_params_f32(qs, 0);
    pl.qv = qr->src[0];
    pl.kv = kr->src[0];
    pl.vv = v;
    REQ(pl.qv->ne[0] == S && pl.kv->ne[0] == S && pl.qv->nb[1] == S * 4 && pl.kv->nb[1] == S * 4 && v->nb[1] == S * 4);
    REQ(pl.qv->nb[0] == 4 && pl.kv->nb[0] == 4 && v->nb[0] == 4);
    const ggml_tensor * Y = root_of(pl.qv);
    REQ(root_of(pl.kv) == Y && root_of(v) == Y && is_unary(Y, GGML_UNARY_OP_SILU) && Y->view_src == nullptr);
    pl.Y = Y;
    REQ(in_graph(Y));
    const ggml_tensor * conv = Y->src[0];
    REQ(in_graph(conv) && conv->op == GGML_OP_SSM_CONV && conv->view_src == nullptr);
    const ggml_tensor * cat = conv->src[0];
    pl.wconv = conv->src[1];
    const int64_t nc = pl.wconv->ne[0], C = conv->ne[0];
    REQ(in_graph(cat) && cat->op == GGML_OP_CONCAT && ggml_get_op_params_i32(cat, 0) == 0 && cat->view_src == nullptr);
    REQ(cat->ne[0] == nc - 1 + T && cat->ne[1] == C && cat->ne[2] == ns && pl.wconv->nb[0] == 4 && ggml_is_contiguous(pl.wconv));
    const ggml_tensor * cs = cat->src[0], * tr = cat->src[1];
    REQ(in_graph(tr) && tr->op == GGML_OP_TRANSPOSE && tr->ne[0] == T);
    pl.xsrc = tr->src[0];
    REQ(pl.xsrc->ne[0] == C && pl.xsrc->ne[1] == T && pl.xsrc->nb[0] == 4);
    pl.gr_r = root_of(cs);
    REQ(in_graph(pl.gr_r) && pl.gr_r->op == GGML_OP_GET_ROWS && cs->data == pl.gr_r->data && ggml_is_contiguous(cs) && cs->ne[0] == nc - 1);
    pl.Rr = root_of(pl.gr_r->src[0]);
    pl.idx = pl.gr_r->src[1];
    REQ(pl.idx->type == GGML_TYPE_I32 && pl.gr_r->src[0]->data == pl.Rr->data && pl.gr_r->src[0]->type == GGML_TYPE_F32);
    REQ(produced.count(pl.Rr) == 0 && pl.gr_r->ne[1] == ns && pl.gr_r->ne[0] == (nc - 1) * C);
    // state
    pl.gr_s = root_of(st);
    REQ(in_graph(pl.gr_s) && pl.gr_s->op == GGML_OP_GET_ROWS && st->data == pl.gr_s->data && pl.gr_s->src[1] == pl.idx);
    pl.Rs = root_of(pl.gr_s->src[0]);
    REQ(pl.gr_s->src[0]->data == pl.Rs->data && pl.gr_s->src[0]->type == GGML_TYPE_F32 && produced.count(pl.Rs) == 0);
    REQ(pl.gr_s->ne[0] == S * S * H && pl.gr_s->ne[1] == ns);
    // gates
    const ggml_tensor * gm = root_of(g);
    REQ(in_graph(gm) && gm->op == GGML_OP_MUL && g->data == gm->data);
    const ggml_tensor * sp = gm->src[0];
    pl.A = gm->src[1];
    REQ(in_graph(sp) && is_unary(sp, GGML_UNARY_OP_SOFTPLUS) && sp->view_src == nullptr && pl.A->ne[0] == H && ggml_nelements(pl.A) == H);
    const ggml_tensor * ad = sp->src[0];
    REQ(in_graph(ad) && ad->op == GGML_OP_ADD && ad->view_src == nullptr);
    pl.alpha = ad->src[0];
    pl.dt = ad->src[1];
    REQ(pl.alpha->ne[0] == H && pl.alpha->nb[0] == 4 && ggml_nelements(pl.dt) == H && pl.dt->nb[0] == 4);
    const ggml_tensor * bs = b;
    REQ(in_graph(bs) && is_unary(bs, GGML_UNARY_OP_SIGMOID) && bs->view_src == nullptr);
    pl.beta = bs->src[0];
    REQ(ggml_nelements(pl.beta) == H * T * ns && pl.beta->nb[0] == 4 && pl.beta->ne[0] == 1 && pl.beta->nb[1] == 4);
    // cache writes: conv state (last nc-1 columns of the concat) and the new recurrent state;
    // the conv-state view may be copied through a CONT of its own (qwen4exp)
    auto cpy_src = [&](const ggml_tensor * c) -> const ggml_tensor * {
        const ggml_tensor * s0 = c->src[0];
        if (s0->op == GGML_OP_CONT && in_graph(s0) && s0->view_src == nullptr && !is_output[s0] && cons[s0].size() == 1) {
            return s0->src[0];
        }
        return s0;
    };
    auto find_cpy = [&](const ggml_tensor * root, const ggml_tensor *& out) {
        for (const auto & kv2 : order) {
            const ggml_tensor * c = kv2.first;
            if (c->op == GGML_OP_CPY && ggml_nelements(c) > 0 && root_of(cpy_src(c)) == root) {
                if (out) {
                    return false;
                }
                out = c;
            }
        }
        return out != nullptr;
    };
    // conv state: slot 0 gets the shifted state (concat columns T..), rollback slots 1..K-1 the old one (columns 0..)
    for (const auto & kv2 : order) {
        const ggml_tensor * c = kv2.first;
        if (c->op != GGML_OP_CPY || ggml_nelements(c) == 0 || root_of(cpy_src(c)) != cat) {
            continue;
        }
        const ggml_tensor * sv = cpy_src(c);
        if ((const char *) sv->data - (const char *) cat->data == 4 * T) {
            REQ(pl.cpy_r == nullptr);
            pl.cpy_r = c;
        } else {
            const ptrdiff_t so = (const char *) sv->data - (const char *) cat->data;
            REQ(so >= 0 && so < 4 * T && so % 4 == 0 && sv->ne[0] == nc - 1 && sv->ne[1] == C && sv->nb[1] == cat->nb[1] &&
                root_of(c) == pl.Rr && ggml_is_contiguous(c) && ggml_nelements(c) == (nc - 1) * C * ns &&
                cpy_src(c) == c->src[0]);
            pl.cpy_rx.push_back(c);
        }
    }
    REQ(pl.cpy_r != nullptr && find_cpy(G, pl.cpy_s));
    REQ(cpy_src(pl.cpy_s) == pl.cpy_s->src[0]);
    if (cpy_src(pl.cpy_r) != pl.cpy_r->src[0]) {
        pl.cont_r = pl.cpy_r->src[0];
    }
    {
        const ggml_tensor * sv = cpy_src(pl.cpy_r);
        REQ((const char *) sv->data - (const char *) cat->data == 4 * T && sv->ne[0] == nc - 1 && sv->ne[1] == C &&
            sv->nb[1] == cat->nb[1] && root_of(pl.cpy_r) == pl.Rr && ggml_is_contiguous(pl.cpy_r));
        const ggml_tensor * sv2 = pl.cpy_s->src[0];
        const int64_t nw = std::min<int64_t>(T, ggml_get_op_params_i32(G, 0));
        REQ((const char *) sv2->data - (const char *) G->data == (ptrdiff_t) (4 * S * H * T * ns) &&
            ggml_nelements(sv2) == S * S * H * ns * nw && root_of(pl.cpy_s) == pl.Rs && ggml_is_contiguous(sv2) &&
            pl.cpy_s->nb[0] == 4 && pl.cpy_s->ne[0] == S * S * H && (nw == 1 ? ggml_is_contiguous(pl.cpy_s) : pl.cpy_s->ne[2] == nw));
    }
    // output: rms_norm(attn view) * w * silu(z)
    const ggml_tensor * rn = nullptr;
    for (const auto & kv2 : order) {
        const ggml_tensor * c = kv2.first;
        if (c->op == GGML_OP_RMS_NORM && root_of(c->src[0]) == G) {
            REQ(rn == nullptr);
            rn = c;
        }
    }
    REQ(rn != nullptr && rn->src[0]->data == G->data && rn->ne[0] == S && rn->ne[1] == H && rn->ne[2] * rn->ne[3] == T * ns &&
        rn->src[0]->nb[1] == S * 4);
    pl.eps_o = ggml_get_op_params_f32(rn, 0);
    REQ(cons[rn].size() == 1);
    const ggml_tensor * mw = cons[rn][0];
    REQ(mw->op == GGML_OP_MUL && mw->src[0] == rn && is_gamma_of(mw->src[1], rn) && mw->view_src == nullptr);
    pl.wn = mw->src[1];
    REQ(cons[mw].size() == 1);
    const ggml_tensor * fm = cons[mw][0];
    REQ(fm->op == GGML_OP_MUL && fm->view_src == nullptr && ggml_are_same_shape(fm, mw));
    const ggml_tensor * zs = fm->src[0] == mw ? fm->src[1] : fm->src[0];
    pl.zsig = is_unary(zs, GGML_UNARY_OP_SIGMOID);
    REQ((is_unary(zs, GGML_UNARY_OP_SILU) || pl.zsig) && zs->view_src == nullptr && cons[zs].size() == 1);
    pl.z = zs->src[0];
    REQ(pl.z->ne[0] == S && pl.z->nb[0] == 4 && pl.z->nb[1] == S * 4);
    anchor = fm;
    mem = { pl.gr_r, cat, pl.cpy_r, conv, Y, qr, qs, kr, ks, pl.gr_s, G, pl.cpy_s, ad, sp, gm, bs, rn, mw, zs, fm };
    if (pl.cont_r) {
        mem.push_back(pl.cont_r);
    }
    mem.insert(mem.end(), pl.cpy_rx.begin(), pl.cpy_rx.end());
    std::unordered_map<const ggml_tensor *, bool> in;
    for (const ggml_tensor * m : mem) {
        in[m] = true;
    }
    // no member value may be read outside the group
    std::function<bool(const ggml_tensor *)> inside = [&](const ggml_tensor * t) -> bool {
        for (const ggml_tensor * c : cons[t]) {
            if (in.count(c) || ggml_nelements(c) == 0) {
                continue;
            }
            if (is_view_op(c) && inside(c)) {
                continue;
            }
            return false;
        }
        return true;
    };
    for (const ggml_tensor * m : mem) {
        REQ(m == fm || (!is_output[m] && !skip[m] && inside(m)));
    }
    REQ(order[fm] > order[G]);
    return true;
#undef REQ
}

/* conv weights [nc, C] (tap fastest) as [nc][C] (channel fastest) for syn_ssm_convx */
static uint64_t conv_weights_t(ggml_backend_synapse_device_context * dctx, const ggml_tensor * w) {
    const std::string key = "convt:" + std::to_string((uint64_t) (uintptr_t) w->data) + ":" + std::to_string(ggml_nbytes(w));
    auto it = dctx->wcat.find(key);
    if (it != dctx->wcat.end()) {
        return it->second.first;
    }
    const int64_t nc = w->ne[0], C = w->ne[1];
    std::vector<float> src(nc * C), dst(nc * C);
    dctx->copy_d2h(src.data(), (uint64_t) (uintptr_t) w->data, src.size() * 4);
    for (int64_t c = 0; c < C; c++) {
        for (int64_t k = 0; k < nc; k++) {
            dst[k * C + c] = src[nc * c + k];
        }
    }
    const uint64_t addr = dctx->pool_alloc(dst.size() * 4);
    dctx->copy_h2d(addr, dst.data(), dst.size() * 4);
    dctx->wcat[key] = { addr, { { (uint64_t) (uintptr_t) w->data, ggml_nbytes(w) } } };
    return addr;
}

void lowerer::plan_convx(const ggml_tensor * cat) {
    if (ggml_get_op_params_i32(cat, 0) != 0 || cat->view_src || is_output[cat] || cat->type != GGML_TYPE_F32) {
        return;
    }
    const ggml_tensor * cs = cat->src[0], * tr = cat->src[1];
    if (tr->op != GGML_OP_TRANSPOSE || cs->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) {
        return;
    }
    // readers: the conv and (through a view of the last nc - 1 columns) one conv-state copy
    convx_plan pl;
    const ggml_tensor * tv = nullptr;
    for (const ggml_tensor * c : cons[cat]) {
        if (c->op == GGML_OP_SSM_CONV && c->src[0] == cat && !pl.conv) {
            pl.conv = c;
        } else if (c->op == GGML_OP_VIEW && !tv && cons[c].size() == 1 && cons[c][0]->op == GGML_OP_CPY && cons[c][0]->src[0] == c) {
            tv = c;
            pl.cpy = cons[c][0];
        } else {
            return;
        }
    }
    const ggml_tensor * conv = pl.conv;
    if (!conv || conv->view_src || is_output[conv] || cons[conv].size() != 1) {
        return;
    }
    const ggml_tensor * Y = cons[conv][0];
    const ggml_tensor * w = conv->src[1];
    const int64_t nc = w->ne[0], C = conv->ne[0], T = conv->ne[1], ns = conv->ne[2];
    if (!is_unary(Y, GGML_UNARY_OP_SILU) || Y->view_src || Y->src[0] != conv || want_bf16(Y) || nc != 4 || C % 64 != 0 ||
        cat->ne[0] != nc - 1 + T || cat->ne[1] != C || cat->ne[2] != ns || cat->ne[3] != 1 || !ggml_is_contiguous(cat)) {
        return;
    }
    // static weights, a dense state [nc - 1, C, ns], x [C, T, ns] with contiguous channels
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[1] != C || produced.count(root_of(w)) || !w->buffer ||
        ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return;
    }
    const ggml_tensor * x = tr->src[0];
    if (cs->ne[0] != nc - 1 || cs->ne[1] != C || cs->ne[2] != ns || !ggml_is_contiguous(cs) || x->ne[0] != C || x->ne[1] != T ||
        x->ne[2] != ns || x->nb[0] != 4) {
        return;
    }
    if (pl.cpy) {
        const ggml_tensor * cp = pl.cpy;
        if ((size_t) ((const char *) tv->data - (const char *) cat->data) != (size_t) T * 4 || tv->ne[0] != nc - 1 || tv->ne[1] != C ||
            tv->ne[2] != ns || tv->nb[1] != cat->nb[1] || tv->nb[2] != cat->nb[2] || !cp->view_src || cp->type != GGML_TYPE_F32 ||
            ggml_nelements(cp) != (nc - 1) * C * ns || cp->nb[0] != 4 || skip[cp]) {
            return;
        }
    }
    skip[conv] = true;
    skip[Y] = true;
    if (pl.cpy) {
        skip[pl.cpy] = true;
    }
    pl.Y = Y;
    convx_plans[cat] = pl;
}

bool lowerer::lower_convx(const ggml_tensor * cat, const convx_plan & pl) {
    const ggml_tensor * w = pl.conv->src[1];
    const int64_t C = pl.conv->ne[0], T = pl.conv->ne[1], ns = pl.conv->ne[2];
    opnd ox, os;
    if (!get(cat->src[1]->src[0], ox) || !get(cat->src[0], os)) {
        return false;
    }
    if (ox.nb[0] != 1 || os.nb[0] != 1 || os.nb[1] != 3) {
        return fail("conv layout: %s", cat);
    }
    const int lw = new_lt(syn_type_single, { (uint64_t) (4 * C) }, LT_IN, conv_weights_t(dctx, w));
    p.lts[lw].region = dctx->pool_region(p.lts[lw].addr);
    const int out = fresh_out(pl.Y);
    const int tail = new_lt(syn_type_single, { (uint64_t) (3 * C * ns) });
    const int64_t tb = 64, ntb = (T + tb - 1) / tb, nit = ns * (C / 64) * ntb;
    const int nm = nmem_for(nit, 1);
    params_t pr;
    pr.i(nm).i(C).i(T).i(ns).i(ox.off).i(ox.nb[1]).i(ox.nb[2]).i(os.off).i(os.nb[2]).i(tb).i(ntb).i((nit + nm - 1) / nm);
    add_node("syn_ssm_convx_f32", { ox.lt, os.lt, lw }, { out, tail }, pr);
    return !pl.cpy || write_view(pl.cpy, tail, GGML_TYPE_F32);
}

/* conv weights [4, C] (tap fastest) re-arranged for syn_gdn_conv: [taps 0..2 interleaved (3C) | tap 3 (C)] */
static uint64_t conv_weights(ggml_backend_synapse_device_context * dctx, const ggml_tensor * w) {
    const std::string key = "conv:" + std::to_string((uint64_t) (uintptr_t) w->data) + ":" + std::to_string(ggml_nbytes(w));
    auto it = dctx->wcat.find(key);
    if (it != dctx->wcat.end()) {
        return it->second.first;
    }
    const int64_t C = w->ne[1];
    // [taps 0..2 interleaved (3C) | tap 3 (C) | lane table (192)]: lane e of the 192-slot state
    // block holds (e / 3) % 16 for the tap-2 slots (e % 3 == 2), else -1
    std::vector<float> src(4 * C), dst(4 * C + 192);
    dctx->copy_d2h(src.data(), (uint64_t) (uintptr_t) w->data, src.size() * 4);
    for (int64_t c = 0; c < C; c++) {
        for (int k = 0; k < 3; k++) {
            dst[3 * c + k] = src[4 * c + k];
        }
        dst[3 * C + c] = src[4 * c + 3];
    }
    for (int e = 0; e < 192; e++) {
        dst[4 * C + e] = e % 3 == 2 ? (float) ((e / 3) % 16) : -1.0f;
    }
    const uint64_t addr = dctx->pool_alloc(dst.size() * 4);
    dctx->copy_h2d(addr, dst.data(), dst.size() * 4);
    dctx->wcat[key] = { addr, { { (uint64_t) (uintptr_t) w->data, ggml_nbytes(w) } } };
    return addr;
}

bool lowerer::lower_gdn(const ggml_tensor * n, const gdn_plan & pl) {
    const int64_t S = 128, H = pl.vv->ne[1], ns = pl.vv->ne[3];
    const int64_t nc = pl.wconv->ne[0], C = pl.Y->ne[0];
    // conv + silu, conv state shifted in place
    opnd ox, ow, oi;
    if (!get(pl.xsrc, ox) || !get(pl.wconv, ow) || !get(pl.idx, oi)) {
        return false;
    }
    const int lr = root_lt(pl.Rr);
    const int ls = root_lt(pl.Rs);
    if (lr < 0 || ls < 0) {
        return false;
    }
    const int64_t srow_r = (int64_t) (pl.gr_r->src[0]->nb[1] / 4);
    const int64_t doff_r = ((const char *) pl.cpy_r->data - (const char *) pl.Rr->data) / 4;
    if (srow_r <= 0 || doff_r % srow_r) {
        return fail("gdn conv state row: %s", n);
    }
    if (nc != 4 || C % 64 != 0 || !ggml_is_contiguous(pl.wconv)) {
        return fail("gdn conv shape: %s", n);
    }
    const int64_t T = pl.vv->ne[2];
    if (T > 1) {
        return lower_gdn_mt(n, pl, ox, oi, lr, ls, srow_r, doff_r / srow_r);
    }
    if (!pl.cpy_rx.empty()) {
        // rollback slots 1..K-1: the old conv state rows, gathered before the kernels below shift it in place
        opnd osrc;
        if (!get(pl.gr_r->src[0], osrc)) {
            return false;
        }
        const ggml_tensor * gr = pl.gr_r;
        const int lo = new_lt(syn_type_single, { (uint64_t) ggml_nelements(gr) });
        params_t pg;
        pg.i(nmem_for(gr->ne[1] * ((gr->ne[0] + 127) / 128), 4)).i(gr->ne[0]).i(gr->ne[1]).i(1).i(1)
          .i(osrc.off).i(osrc.nb[1]).i(osrc.nb[2]).i(osrc.nb[3])
          .i(oi.off).i(oi.nb[0]).i(oi.nb[1]).i(oi.nb[2]).i(pl.idx->type == GGML_TYPE_I64);
        add_node("syn_get_rows_f32_f32", { osrc.lt, oi.lt }, { lo }, pg);
        for (const ggml_tensor * c : pl.cpy_rx) {
            opnd d;
            if (!get(c, d) || d.nb[0] != 1) {
                return fail("gdn conv rollback slot: %s", c);
            }
            const int64_t rows = c->ne[1] * c->ne[2] * c->ne[3];
            params_t pw;
            pw.i(nmem_for(rows * ((c->ne[0] + 127) / 128), 8)).i(c->ne[0]).i(c->ne[1]).i(c->ne[2]).i(c->ne[3])
              .i(d.off).i(d.nb[1]).i(d.nb[2]).i(d.nb[3])
              .i(0).i(c->ne[0]).i(c->ne[0] * c->ne[1]).i(c->ne[0] * c->ne[1] * c->ne[2]);
            add_node("syn_wr_f32_f32", { d.lt, lo }, { sink() }, pw, true);
        }
    }
    static const bool full = getenv("GGML_SYNAPSE_GDN_SPLIT") == nullptr;
    if (full && pl.qv->ne[1] == H && C == 3 * H * S && !pl.zsig) {
        opnd oa, ob, odt, oA, oz, own;
        if (!get(pl.alpha, oa) || !get(pl.beta, ob) || !get(pl.dt, odt) || !get(pl.A, oA) || !get(pl.z, oz) ||
            !get(pl.wn, own)) {
            return false;
        }
        const int64_t srow_s = (int64_t) (pl.gr_s->src[0]->nb[1] / 4);
        const int64_t doff_s = ((const char *) pl.cpy_s->data - (const char *) pl.Rs->data) / 4;
        if (srow_s <= 0 || doff_s % srow_s) {
            return fail("gdn state row: %s", n);
        }
        const int lwc = new_lt(syn_type_single, { (uint64_t) (4 * C + 192) }, LT_IN, conv_weights(dctx, pl.wconv));
        p.lts[lwc].region = dctx->pool_region(p.lts[lwc].addr);
        const int out = fresh_out(n);
        const bool bf = roots[n].bf16;
        params_t pr;
        pr.split(ns * H, 1).i(H).i(ns).i(C).i(srow_r).i(doff_r / srow_r).i(srow_s).i(doff_s / srow_s).i(oi.off)
          .i(ox.off).i(ox.nb[2]).i(0)
          .i(oa.off).i(oa.nb[2]).i(ob.off).i(ob.nb[3]).i(odt.off).i(oA.off)
          .i(oz.off).i(oz.nb[3]).i(own.off)
          .f(pl.eps_qk).f(pl.qk_scale).f(pl.eps_o).f(1.0f / (float) H);
        add_node(bf ? "syn_gdn_full_bf_f32" : "syn_gdn_full_f32",
                 { lr, ls, oi.lt, ox.lt, lwc, oa.lt, ob.lt, odt.lt, oA.lt, oz.lt, own.lt }, { out }, pr, 0x3);
        GGML_UNUSED(ow);
        return true;
    }
    const int ly = new_lt(syn_type_single, { (uint64_t) (C * ns) });
    roots[pl.Y].lt = ly;
    {
        const int lwc = new_lt(syn_type_single, { (uint64_t) (4 * C) }, LT_IN, conv_weights(dctx, pl.wconv));
        p.lts[lwc].region = dctx->pool_region(p.lts[lwc].addr);
        params_t pr;
        pr.i(nmem_for((C / 64) * ns, 4)).i(C).i(ns).i(srow_r).i(oi.off).i(doff_r / srow_r)
          .i(ox.off).i(ox.nb[2]).i(0).i(3 * C);
        add_node("syn_gdn_conv_f32", { lr, oi.lt, ox.lt, lwc, lwc }, { ly }, pr, true);
    }
    GGML_UNUSED(ow);
    opnd oq, ok, ov, oa, ob, odt, oA, oz, own;
    if (!get(pl.qv, oq) || !get(pl.kv, ok) || !get(pl.vv, ov) || !get(pl.alpha, oa) || !get(pl.beta, ob) ||
        !get(pl.dt, odt) || !get(pl.A, oA) || !get(pl.z, oz) || !get(pl.wn, own)) {
        return false;
    }
    const int64_t srow_s = (int64_t) (pl.gr_s->src[0]->nb[1] / 4);
    const int64_t doff_s = ((const char *) pl.cpy_s->data - (const char *) pl.Rs->data) / 4;
    if (srow_s <= 0 || doff_s % srow_s) {
        return fail("gdn state row: %s", n);
    }
    const int out = fresh_out(n);
    const bool bf = roots[n].bf16;
    params_t pr;
    pr.i(nmem_for(ns * H, 1)).i(H).i(ns).i(pl.qv->ne[1]).i(srow_s).i(oi.off).i(doff_s / srow_s)
      .i(oq.off).i(ok.off).i(ov.off).i(oq.nb[3])
      .i(oa.off).i(oa.nb[2]).i(ob.off).i(ob.nb[3]).i(odt.off).i(oA.off)
      .i(oz.off).i(oz.nb[3]).i(own.off)
      .f(pl.eps_qk).f(pl.qk_scale).f(pl.eps_o).i(pl.zsig);
    GGML_UNUSED(S);
    add_node(bf ? "syn_gdn_dec_bf_f32" : "syn_gdn_dec_f32",
             { ls, oi.lt, ly, oa.lt, ob.lt, odt.lt, oA.lt, oz.lt, own.lt }, { out }, pr, true);
    return true;
}

/* the gdn block for T = 2..8 tokens per sequence (MTP verify): the conv over the batch with all K rollback
   slots of the conv state, the delta rule over the tokens (K state snapshots), then the gated output norm */
bool lowerer::lower_gdn_mt(const ggml_tensor * n, const gdn_plan & pl, const opnd & ox, const opnd & oi, int lr,
                           int ls, int64_t srow_r, int64_t drow_r) {
    const int64_t S = 128, H = pl.vv->ne[1], T = pl.vv->ne[2], ns = pl.vv->ne[3], C = pl.Y->ne[0];
    const int64_t K = ggml_get_op_params_i32(pl.G, 0), nw = std::min<int64_t>(T, K);
    const ggml_tensor * cat = pl.Y->src[0]->src[0];
    if ((int64_t) pl.cpy_rx.size() != K - 1 || (ns > 1 && pl.cpy_r->nb[1] != (size_t) (4 * srow_r))) {
        return fail("gdn conv slots: %s", n);
    }
    // conv rollback slot r at row drow_r + r * srs_r, taps from concat column max(0, T - r)
    int64_t srs_r = 0;
    {
        std::vector<std::pair<int64_t, int64_t>> sl;
        for (const ggml_tensor * c : pl.cpy_rx) {
            const int64_t off = ((const char *) c->data - (const char *) pl.Rr->data) / 4;
            if (off % srow_r) {
                return fail("gdn conv slot row: %s", c);
            }
            sl.push_back({ off / srow_r, ((const char *) c->src[0]->data - (const char *) cat->data) / 4 });
        }
        std::sort(sl.begin(), sl.end());
        for (size_t i = 0; i < sl.size(); i++) {
            const int64_t r = (int64_t) i + 1;
            if (i == 0) {
                srs_r = sl[0].first - drow_r;
            }
            if (srs_r <= 0 || sl[i].first != drow_r + r * srs_r || sl[i].second != std::max<int64_t>(0, T - r)) {
                return fail("gdn conv slot layout: %s", n);
            }
        }
    }
    const int64_t srow_s = (int64_t) (pl.gr_s->src[0]->nb[1] / 4);
    const int64_t doff_s = ((const char *) pl.cpy_s->data - (const char *) pl.Rs->data) / 4;
    if (srow_s <= 0 || doff_s % srow_s || (ns > 1 && pl.cpy_s->nb[1] != (size_t) (4 * srow_s)) ||
        (nw > 1 && pl.cpy_s->nb[2] % (4 * srow_s))) {
        return fail("gdn state row: %s", n);
    }
    const int64_t srs_s = nw > 1 ? (int64_t) (pl.cpy_s->nb[2] / (4 * srow_s)) : 0;
    const int ly = new_lt(syn_type_single, { (uint64_t) (C * T * ns) });
    roots[pl.Y].lt = ly;
    opnd oq, ok, ov, oa, ob, odt, oA, oz, own;
    if (!get(pl.qv, oq) || !get(pl.kv, ok) || !get(pl.vv, ov) || !get(pl.alpha, oa) || !get(pl.beta, ob) ||
        !get(pl.dt, odt) || !get(pl.A, oA) || !get(pl.z, oz) || !get(pl.wn, own)) {
        return false;
    }
    // (z: heads contiguous; its token / sequence strides are parameters, a merged projection's row too)
    if (oq.nb[2] != C || ok.nb[2] != C || ov.nb[2] != C || oq.nb[3] != C * T || ok.nb[3] != C * T || ov.nb[3] != C * T ||
        oz.nb[1] != S) {
        GGML_LOG_DEBUG("gdn mt: q %lld %lld k %lld %lld v %lld %lld a %lld b %lld z %lld %lld (C %lld H %lld)\n",
                       (long long) oq.nb[2], (long long) oq.nb[3], (long long) ok.nb[2], (long long) ok.nb[3],
                       (long long) ov.nb[2], (long long) ov.nb[3], (long long) oa.nb[1], (long long) ob.nb[2],
                       (long long) oz.nb[1], (long long) oz.nb[2], (long long) C, (long long) H);
        return fail("gdn mt layout: %s", n);
    }
    {
        const int lwc = new_lt(syn_type_single, { (uint64_t) (4 * C) }, LT_IN, conv_weights(dctx, pl.wconv));
        p.lts[lwc].region = dctx->pool_region(p.lts[lwc].addr);
        params_t pr;
        pr.i(nmem_for((C / 64) * ns, 1)).i(C).i(ns).i(srow_r).i(oi.off).i(drow_r)
          .i(ox.off).i(ox.nb[2]).i(0).i(3 * C).i(T).i(K).i(ox.nb[1]).i(srs_r);
        add_node("syn_gdn_conv_mt_f32", { lr, oi.lt, ox.lt, lwc, lwc }, { ly }, pr, true);
    }
    const int lo = new_lt(syn_type_single, { (uint64_t) (S * H * T * ns) });
    {
        params_t pr;
        pr.i(nmem_for(ns * H * 4, 1)).i(H).i(ns).i(pl.qv->ne[1]).i(srow_s).i(oi.off).i(doff_s / srow_s)
          .i(oq.off).i(ok.off).i(ov.off).i(oq.nb[3]).i(oq.nb[2])
          .i(oa.off).i(oa.nb[2]).i(oa.nb[1]).i(ob.off).i(ob.nb[3]).i(ob.nb[2]).i(odt.off).i(oA.off)
          .i(T).i(K).i(srs_s).f(pl.eps_qk).f(pl.qk_scale);
        add_node("syn_gdn_mt_f32", { ls, oi.lt, ly, oa.lt, ob.lt, odt.lt, oA.lt }, { lo }, pr, true);
    }
    const int out = fresh_out(n);
    const bool bf = roots[n].bf16;
    params_t pr;
    pr.i(nmem_for(ns * T * H, 2)).i(H).i(T).i(ns).i(oz.off).i(oz.nb[3]).i(oz.nb[2]).i(own.off).f(pl.eps_o).i(pl.zsig);
    add_node(bf ? "syn_gdn_onorm_bf_f32" : "syn_gdn_onorm_f32", { lo, oz.lt, own.lt }, { out }, pr);
    return true;
}

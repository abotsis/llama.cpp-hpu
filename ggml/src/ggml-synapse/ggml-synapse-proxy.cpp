// Multi-card operation: one worker process per card (Synapse acquires a single card per process).
//
// The parent exposes a proxy device per card. Every operation is a command to the card's worker
// (ggml-synapse-worker, which runs the single-card backend) through one shared-memory segment:
//   [header: a command ring per worker][per worker: payload area | staging area][reduce slots]
// Commands carry serialized tensors inline (graphs in the payload area); bulk data moves through
// the staging areas. Copies between cards go device -> worker A staging -> device on worker B.
// Used when several cards are visible (ggml-synapse.cpp); a card's worker starts when the card is first used.

#include "ggml-synapse-impl.h"

#include <hccl.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <list>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

extern char ** environ;

#define PROXY_MAGIC 0x53594e50u   // "SYNP"
#define PROXY_MAX_WORKERS 16

enum proxy_cmd : uint32_t {
    CMD_HELLO = 1,   // -> r0 total, r1 free device memory
    CMD_ALLOC,       // a0 size -> r0 handle, r1 base address
    CMD_FREE,        // a0 handle
    CMD_CLEAR,       // a0 handle, a1 value
    CMD_SET,         // payload: tensor; a0 offset, a1 size, a2 segment offset of the data
    CMD_GET,         // payload: tensor; a0 offset, a1 size, a2 segment offset of the destination, a3 host buffer id
                     //   (not 0: a2 is the offset in that buffer)
    CMD_MEMSET,      // payload: tensor; a0 offset, a1 size, a2 value
    CMD_SET2D,       // payload: tensor; a0 offset, a1 row size, a2 segment offset of the rows (packed), a3 rows,
                     //   a4 row stride in the tensor
    CMD_GET2D,       // as CMD_SET2D, rows packed into the segment; a5 host buffer id (not 0: rows to a2 in that buffer,
                     //   a6 apart)
    CMD_COMPUTE,     // a0 graph id, a1 graph id to drop (0: none); payload: the graph, or none for a graph
                     //   already sent under a0 -> r0 status
    CMD_SYNC,
    CMD_EXIT,
    CMD_ALLREDUCE,   // payload: tensor (f32); a0 generation, a1 bytes, a2 rank, a3 ranks: the sum over the
                     //   ranks' tensors, through the reduce slots
    CMD_CCL_ID,      // a new HCCL unique id into the header
    CMD_CCL_INIT,    // a0 communicator key, a1 rank, a2 ranks: join (blocks until every rank joined) -> r0 status
    CMD_CCL_REDUCE,  // a0 communicator key, a1 device address, a2 elements, a3 bf16 (else f32): in-place hcclAllReduce on the card's stream -> r0 status
    CMD_CCL_FREE,    // a0 communicator key
    CMD_HOST_MAP,    // a0 host buffer id, a1 its file descriptor in the parent, a2 size, a3 parent pid -> r0 0: mapped
    CMD_HOST_UNMAP,  // a0 host buffer id
    CMD_SET_RAW,     // a0 buffer handle, a1 device address, a2 size, a3 segment offset of the data, a4 buffer usage:
                     //   bytes to a tensor of a non-weight buffer (no tensor payload)
};

#define PROXY_RING   64
#define PROXY_INLINE 4096
// small writes (graph inputs) go through a ring of chunks shared by all workers (proxy_state::small)
#define PROXY_SMALL_N     64
#define PROXY_SMALL_CHUNK (64 << 10)

// one command of a worker's ring
struct proxy_slot {
    uint32_t cmd;
    uint32_t big;             // payload in the worker's payload area, else inline
    uint64_t a[8];
    uint64_t r[2];
    uint64_t payload_len;
    uint8_t  inl[PROXY_INLINE];
};

// worker -> parent log lines (ggml_synapse_log_info): the parent's logger applies llama.cpp's verbosity, a
// worker's own stderr would show every level. The parent drains them whenever it waits on the worker.
#define PROXY_LOG_RING 512   // (a multi-card start compiles ~2 recipes per layer before the first wait)
#define PROXY_LOG_LEN  256
struct proxy_logring {
    std::atomic<uint32_t> head;   // lines written by the worker
    char line[PROXY_LOG_RING][PROXY_LOG_LEN];
};

static proxy_logring * g_worker_log = nullptr;   // set in a card worker process

void ggml_synapse_log_info(const char * fmt, ...) {
    char b[PROXY_LOG_LEN];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof(b), fmt, ap);
    va_end(ap);
    if (g_worker_log) {
        const uint32_t h = g_worker_log->head.load(std::memory_order_relaxed);
        memcpy(g_worker_log->line[h % PROXY_LOG_RING], b, sizeof(b));
        g_worker_log->head.store(h + 1, std::memory_order_release);
    } else {
        GGML_LOG_INFO("%s\n", b);
    }
}

// a worker's command ring: command s (1, 2, ...) in slot[s % PROXY_RING]
struct alignas(64) proxy_mailbox {
    std::atomic<uint32_t> req;    // last command submitted by the parent (futex word)
    std::atomic<uint32_t> done;   // last command the worker finished (futex word)
    std::atomic<uint32_t> ready;  // worker initialized: 1 ok, 2 failed
    std::atomic<int32_t>  err;    // status of the first failed command nobody waited for
    // threads asleep on req / done
    std::atomic<uint32_t> req_sleepers;
    std::atomic<uint32_t> done_sleepers;
    proxy_slot slot[PROXY_RING];
    proxy_logring log;
};

struct proxy_header {
    uint32_t magic;
    uint32_t n_workers;
    uint64_t payload_size;   // per worker
    uint64_t staging_size;   // per worker
    uint64_t area_off;       // offset of worker 0's area; each area is payload_size + staging_size
    uint64_t reduce_off;     // reduce slots: two per worker (by generation parity), reduce_slot bytes each
    uint64_t reduce_slot;
    uint64_t small_off;      // small writes: PROXY_SMALL_N chunks of PROXY_SMALL_CHUNK bytes
    proxy_mailbox mb[PROXY_MAX_WORKERS];
    alignas(64) std::atomic<uint32_t> arrived[PROXY_MAX_WORKERS];   // last generation each rank's slot holds
    hcclUniqueId ccl_id;     // HCCL rendezvous: created by rank 0's worker, read by every rank
};

// serialized tensor (index-linked within one message)
struct proxy_tensor {
    int32_t  type;
    int32_t  op;
    int32_t  flags;
    int32_t  usage;           // buffer usage
    uint64_t buffer;          // worker buffer handle, 0: none
    int64_t  ne[GGML_MAX_DIMS];
    uint64_t nb[GGML_MAX_DIMS];
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  src[GGML_MAX_SRC];
    int32_t  view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];
};

static long futex(std::atomic<uint32_t> * addr, int op, uint32_t val) {
    return syscall(SYS_futex, (uint32_t *) addr, op, val, nullptr, nullptr, 0);
}

static void futex_wait_ne(std::atomic<uint32_t> * addr, uint32_t not_val, pid_t peer, std::atomic<uint32_t> * sleepers = nullptr,
                          int64_t spin_us = 0) {
    // wait until *addr != not_val; peer: a process whose death aborts the wait (0: none); sleepers: counts the
    // waiters asleep (see futex_store)
    if (spin_us > 0) {
        const int64_t t0 = ggml_time_us();
        for (int spin = 1;; spin++) {
            if (addr->load(std::memory_order_acquire) != not_val) {
                return;
            }
            __builtin_ia32_pause();
            if (spin % 64 == 0 && ggml_time_us() - t0 > spin_us) {
                break;
            }
        }
    }
    for (int spin = 0; spin_us == 0 && spin < 20000; spin++) {
        if (addr->load(std::memory_order_acquire) != not_val) {
            return;
        }
        __builtin_ia32_pause();
    }
    while (addr->load(std::memory_order_acquire) == not_val) {
        struct timespec ts = { 1, 0 };
        if (sleepers) {
            // (seq_cst: either the writer sees the count or this load sees its store)
            sleepers->fetch_add(1);
            if (addr->load() == not_val) {
                syscall(SYS_futex, (uint32_t *) addr, FUTEX_WAIT, not_val, &ts, nullptr, 0);
            }
            sleepers->fetch_sub(1);
        } else {
            syscall(SYS_futex, (uint32_t *) addr, FUTEX_WAIT, not_val, &ts, nullptr, 0);
        }
        if (peer > 0 && addr->load(std::memory_order_acquire) == not_val) {
            int st = 0;
            if (waitpid(peer, &st, WNOHANG) == peer) {
                GGML_ABORT("ggml-synapse: card worker %d exited (status %d)", (int) peer, st);
            }
        }
    }
}

// *addr = val, waking the waiters asleep on it
static void futex_store(std::atomic<uint32_t> * addr, uint32_t val, std::atomic<uint32_t> * sleepers) {
    addr->store(val);
    if (sleepers->load() != 0) {
        futex(addr, FUTEX_WAKE, INT_MAX);
    }
}

// ---------------------------------------------------------------------------
// serialization
// ---------------------------------------------------------------------------

struct proxy_buffer_context;
static bool proxy_buffer_is(ggml_backend_buffer_t buffer);

struct proxy_ser {
    std::vector<proxy_tensor> ts;
    std::unordered_map<const ggml_tensor *, int> idx;
    bool deep = true;   // sources too (graphs); a tensor read or written alone needs only its view chain
    const struct proxy_worker * w = nullptr;   // graphs: the worker that runs them
    const std::unordered_set<const ggml_tensor *> * nodes = nullptr;   // graphs: sources of these only

    int add(const ggml_tensor * t);
    int add_one(const ggml_tensor * t);
};

// ---------------------------------------------------------------------------
// parent side
// ---------------------------------------------------------------------------

struct proxy_worker {
    int index;
    int card;             // visible card (SYNAPSE<card>)
    int module;           // its Synapse module id (the worker sees only this card)
    std::string description;
    bool failed = false;  // the worker did not start (card in use or missing): not retried
    pid_t pid = 0;
    proxy_mailbox * mb;
    uint8_t * payload;
    uint8_t * staging;
    uint64_t staging_off;   // segment offset of the staging area
    std::mutex mtx;
    uint32_t seq = 0;     // last command submitted
    int half = 0;         // staging half of the next unwaited piece
    uint32_t half_seq[2] = { 0, 0 };   // last command reading each staging half
    bool dirty = false;   // device work issued since the last sync
    uint32_t sync_seq = 0;   // last command synchronizing the card (GET, GET2D, SYNC)
    // reads queued by get_tensor_async: finished (waited for, copied out of the staging area) by the next
    // synchronize, or before the staging area is used again
    struct pending_read {
        uint8_t * dst;   // nullptr: written in place (a shared host buffer)
        uint64_t  off;   // staging offset of the data (rows packed)
        size_t    row_size, rows, dst_stride;
    };
    std::vector<pending_read> reads;
    uint32_t reads_seq = 0;
    uint64_t reads_off = 0;   // staging bytes the reads hold
    std::vector<proxy_worker *> peers;   // the workers of its tensor-parallel group
    std::atomic<uint32_t> log_tail{0};   // worker log lines already printed
    uint64_t mem_total = 0, mem_free = 0;
    std::string name;
    // graphs the worker holds: node signatures by id, most recent first
    struct graph_sig {
        uint32_t id;
        const ggml_cgraph * cg;
        int n_nodes;
        std::vector<uint8_t> sig;    // node identities
        std::vector<uint8_t> blob;   // the serialized graph
    };
    std::list<graph_sig> graphs;
    // graphs by a hash of their blob
    std::unordered_multimap<size_t, std::list<graph_sig>::iterator> by_blob;
    uint32_t next_graph = 1;
};

struct proxy_state {
    int fd = -1;
    uint8_t * base = nullptr;
    size_t size = 0;
    proxy_header * hdr = nullptr;
    std::vector<proxy_worker *> workers;
    // small writes: chunks in the segment shared by all workers
    struct {
        std::mutex mtx;
        int next = 0;
        int last = -1;
        size_t last_size = 0;
        uint32_t seq[PROXY_SMALL_N][PROXY_MAX_WORKERS] = {};   // last command of each worker reading each chunk
    } small;
    std::string exe;                     // the worker binary
    std::vector<std::string> env;        // the workers' environment (without HABANA_VISIBLE_MODULES)
    std::mutex start_mtx;
};

static proxy_state g_proxy;

static bool proxy_start(proxy_worker * w);

struct proxy_buffer_context {
    proxy_worker * w;
    uint64_t handle;
    uint64_t base;
    size_t size;
};

// a tensor after its sources and view source (post-order, the order add_one's own recursion takes), walked with
// an explicit stack (no recursion depth limit on long graphs). A source that is the tensor itself (ggml_cast's
// src[1]) is serialized as its own index.
int proxy_ser::add(const ggml_tensor * t) {
    auto it = idx.find(t);
    if (it != idx.end()) {
        return it->second;
    }
    std::vector<std::pair<const ggml_tensor *, int>> st;   // tensor, next dependency to visit
    st.emplace_back(t, 0);
    while (!st.empty()) {
        const ggml_tensor * x = st.back().first;
        int & k = st.back().second;
        if (idx.count(x)) {
            st.pop_back();
            continue;
        }
        const bool d = deep && (!nodes || nodes->count(x));
        const ggml_tensor * dep = nullptr;
        while (!dep && k <= GGML_MAX_SRC) {
            const ggml_tensor * c = k < GGML_MAX_SRC ? (d ? x->src[k] : nullptr) : x->view_src;
            // (a view source on another worker: add_one substitutes the local copy)
            if (k == GGML_MAX_SRC && c && w && c->buffer && proxy_buffer_is(c->buffer) &&
                ((proxy_buffer_context *) c->buffer->context)->w != w) {
                c = nullptr;
            }
            k++;
            if (c && c != x && !idx.count(c)) {
                dep = c;
            }
        }
        if (dep) {
            for (const auto & e : st) {
                if (e.first == dep) {
                    std::string chain;
                    for (const auto & f : st) {
                        chain += std::string(" ") + f.first->name + "(" + ggml_op_desc(f.first) + ")";
                    }
                    GGML_ABORT("ggml-synapse: cycle in the graph at %s:%s", dep->name, chain.substr(chain.size() > 2000 ? chain.size() - 2000 : 0).c_str());
                }
            }
            st.emplace_back(dep, 0);
        } else {
            add_one(x);
            st.pop_back();
        }
    }
    return idx.at(t);
}

int proxy_ser::add_one(const ggml_tensor * t) {
    auto it = idx.find(t);
    if (it != idx.end()) {
        return it->second;
    }
    proxy_tensor p;
    memset(&p, 0, sizeof(p));
    // a graph's inputs (tensors computed elsewhere) without their history
    const bool d = deep && (!nodes || nodes->count(t));
    for (int k = 0; k < GGML_MAX_SRC; k++) {
        p.src[k] = d && t->src[k] ? (t->src[k] == t ? -2 : add(t->src[k])) : -1;   // -2: itself (set below)
    }
    // a view placed with its consumer on another worker than its source (the scheduler keeps views
    // with their source and copies only srcs): it reads the copy of the source made here
    const ggml_tensor * vs = t->view_src;
    const void * data = t->data;
    ggml_backend_buffer_t buf = t->buffer;
    if (w && vs && vs->buffer && proxy_buffer_is(vs->buffer) && ((proxy_buffer_context *) vs->buffer->context)->w != w) {
        const ggml_tensor * c = t->src[0];
        while (c && c->view_src == vs) {
            c = c->src[0];
        }
        if (!c || c->view_src || !c->buffer || !proxy_buffer_is(c->buffer) ||
            ((proxy_buffer_context *) c->buffer->context)->w != w || ggml_nbytes(c) != ggml_nbytes(vs)) {
            const ggml_tensor * s0 = t->src[0];
            GGML_ABORT("ggml-synapse: graph for %s reads %s (%s), a view of %s on another card (src0 %s %s view_src %s buf %s local %d nb %zu/%zu)",
                       w->name.c_str(), t->name, ggml_op_desc(t), vs->name, s0 ? s0->name : "-", s0 ? ggml_op_desc(s0) : "-",
                       s0 && s0->view_src ? s0->view_src->name : "-", s0 && s0->buffer ? ggml_backend_buffer_name(s0->buffer) : "-",
                       s0 && s0->buffer && proxy_buffer_is(s0->buffer) ? (int) (((proxy_buffer_context *) s0->buffer->context)->w == w) : -1,
                       s0 ? ggml_nbytes(s0) : 0, ggml_nbytes(vs));
        }
        vs = c;
        data = (const char *) c->data + t->view_offs;
        buf = c->buffer;
        if (d && t->src[0]) {
            p.src[0] = add(c);
        }
    }
    p.view_src = vs ? add(vs) : -1;
    p.type = t->type;
    p.op = t->op;
    p.flags = t->flags;
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        p.ne[d] = t->ne[d];
        p.nb[d] = t->nb[d];
    }
    memcpy(p.op_params, t->op_params, sizeof(p.op_params));
    p.view_offs = t->view_offs;
    p.data = (uint64_t) (uintptr_t) data;
    if (buf && proxy_buffer_is(buf)) {
        p.buffer = ((proxy_buffer_context *) buf->context)->handle;
        p.usage = ggml_backend_buffer_get_usage(buf);
    }
    memcpy(p.name, t->name, sizeof(p.name));
    const int i = (int) ts.size();
    for (int k = 0; k < GGML_MAX_SRC; k++) {
        if (p.src[k] == -2) {
            p.src[k] = i;
        }
    }
    ts.push_back(p);
    idx[t] = i;
    return i;
}

// the worker's new log lines, printed by this (the parent) process
static void proxy_drain_log(proxy_worker * w) {
    const uint32_t h = w->mb->log.head.load(std::memory_order_acquire);
    uint32_t t = w->log_tail.exchange(h);
    if (h - t > PROXY_LOG_RING) {
        t = h - PROXY_LOG_RING;   // (overwritten)
    }
    for (; t != h; t++) {
        GGML_LOG_INFO("%s: %.*s\n", w->name.c_str(), PROXY_LOG_LEN, w->mb->log.line[t % PROXY_LOG_RING]);
    }
}

// wait until the worker finished command s; results: a wait for the device (a synchronize)
static void proxy_wait_seq(proxy_worker * w, uint32_t s, bool results = false) {
    static const int64_t spin_us = getenv("GGML_SYNAPSE_PROXY_SPIN_US") ? atoll(getenv("GGML_SYNAPSE_PROXY_SPIN_US")) : 5000;
    for (uint32_t d; (int32_t) ((d = w->mb->done.load(std::memory_order_acquire)) - s) < 0;) {
        futex_wait_ne(&w->mb->done, d, w->pid, &w->mb->done_sleepers, results ? std::max<int64_t>(spin_us, 1) : 0);
    }
    proxy_drain_log(w);
    const int32_t err = w->mb->err.load(std::memory_order_acquire);
    if (err != 0) {
        GGML_ABORT("ggml-synapse: %s: a queued command failed (status %d)", w->name.c_str(), err);
    }
}

// wait for every command submitted so far
static void proxy_wait(proxy_worker * w) {
    proxy_wait_seq(w, w->seq);
}

// the next command's slot (waits while the ring is full)
static proxy_slot * proxy_begin(proxy_worker * w, uint32_t cmd) {
    GGML_ASSERT(w->pid > 0 && "card worker not started");
    const uint32_t s = w->seq + 1;
    if (s > PROXY_RING && (int32_t) (w->mb->done.load(std::memory_order_acquire) - (s - PROXY_RING)) < 0) {
        ggml_synapse_trace("p ring full");
        proxy_wait_seq(w, s - PROXY_RING);
        ggml_synapse_trace("p ring free");
    }
    proxy_slot * sl = &w->mb->slot[s % PROXY_RING];
    sl->cmd = cmd;
    sl->big = 0;
    memset(sl->a, 0, sizeof(sl->a));
    sl->payload_len = 0;
    return sl;
}

// a command's payload: inline, or (large) in the payload area once every earlier command finished
static void proxy_payload(proxy_worker * w, proxy_slot * sl, const void * data, size_t len) {
    if (len <= PROXY_INLINE) {
        memcpy(sl->inl, data, len);
    } else {
        GGML_ASSERT(len <= g_proxy.hdr->payload_size);
        proxy_wait(w);
        memcpy(w->payload, data, len);
        sl->big = 1;
    }
    sl->payload_len = len;
}

static void proxy_issue(proxy_worker * w) {
    // device work pending until a sync (a read synchronizes the worker first)
    const uint32_t c = w->mb->slot[(w->seq + 1) % PROXY_RING].cmd;
    if (c == CMD_MEMSET || c == CMD_CLEAR || c == CMD_COMPUTE || c == CMD_ALLREDUCE || c == CMD_CCL_REDUCE) {
        w->dirty = true;
    } else if (c == CMD_GET || c == CMD_GET2D || c == CMD_SYNC) {
        w->dirty = false;
        w->sync_seq = w->seq + 1;
    }
    w->seq++;
    futex_store(&w->mb->req, w->seq, &w->mb->req_sleepers);
}

// GGML_SYNAPSE_PROXY_STATS=1: per command count, wall time (issue to completion) and payload bytes at exit
struct proxy_stat {
    std::atomic<uint64_t> n{0}, ns{0}, bytes{0};
};
static proxy_stat g_stats[32];
static bool proxy_stats_on() {
    static const bool on = getenv("GGML_SYNAPSE_PROXY_STATS") != nullptr;
    return on;
}

// GGML_SYNAPSE_PROXY_STATS: tensor uploads (weights at load)
static struct {
    uint64_t n, bytes, in_ns, wait_ns, copy_ns, between_ns, t_last;
} g_set_stat;

static uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

// one command; wait: until it finished (its results in the returned slot)
static proxy_slot * proxy_call(proxy_worker * w, uint32_t cmd, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                               uint64_t a3 = 0, const void * payload = nullptr, size_t payload_len = 0, uint64_t a4 = 0,
                               bool wait = true) {
    proxy_slot * mb = proxy_begin(w, cmd);
    mb->a[0] = a0;
    mb->a[1] = a1;
    mb->a[2] = a2;
    mb->a[3] = a3;
    mb->a[4] = a4;
    if (payload_len) {
        proxy_payload(w, mb, payload, payload_len);
    }
    const uint64_t t0 = proxy_stats_on() ? now_ns() : 0;
    proxy_issue(w);
    if (wait) {
        proxy_wait(w);
    }
    if (t0) {
        const uint64_t t1 = now_ns();
        g_stats[cmd].n++;
        g_stats[cmd].ns += t1 - t0;
        g_stats[cmd].bytes += payload_len;
        // GGML_SYNAPSE_PROXY_STATS=2: every call (start relative to the previous call's end, duration)
        static const bool trace = atoi(getenv("GGML_SYNAPSE_PROXY_STATS")) >= 2;
        static uint64_t t_prev = 0;
        if (trace) {
            fprintf(stderr, "proxy-trace w%d cmd %2u a0 %8lu a1 %8lu idle %8.1f us dur %8.1f us\n", w->index, cmd,
                    (unsigned long) a0, (unsigned long) a1, t_prev ? (t0 - t_prev) / 1e3 : 0.0, (t1 - t0) / 1e3);
        }
        t_prev = t1;
    }
    return mb;
}

static std::vector<uint8_t> proxy_ser_tensor(const ggml_tensor * t) {
    proxy_ser s;
    s.deep = false;
    const int i = s.add(t);
    std::vector<uint8_t> out(sizeof(int32_t) * 2 + s.ts.size() * sizeof(proxy_tensor));
    const int32_t n = (int32_t) s.ts.size();
    memcpy(out.data(), &n, 4);
    memcpy(out.data() + 4, &i, 4);
    memcpy(out.data() + 8, s.ts.data(), s.ts.size() * sizeof(proxy_tensor));
    return out;
}

static void proxy_finish_reads(proxy_worker * w);

// data in the segment at seg_off -> tensor bytes [offset, offset + size) on worker w
static void proxy_set_from(proxy_worker * w, const ggml_tensor * t, size_t offset, size_t size, uint64_t seg_off) {
    const std::vector<uint8_t> p = proxy_ser_tensor(t);
    proxy_call(w, CMD_SET, offset, size, seg_off, 0, p.data(), p.size());
}

// ---- buffer ----

static void proxy_buffer_free(ggml_backend_buffer_t buffer) {
    proxy_buffer_context * ctx = (proxy_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(ctx->w->mtx);
        proxy_call(ctx->w, CMD_FREE, ctx->handle);
        ctx->w->graphs.clear();   // the worker drops its graphs with any buffer
        ctx->w->by_blob.clear();
    }
    delete ctx;
}

static void * proxy_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) (uintptr_t) ((proxy_buffer_context *) buffer->context)->base;
}

// Planar weights are cut at 128-row blocks.
static void proxy_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                    size_t size) {
    proxy_worker * w = ((proxy_buffer_context *) buffer->context)->w;
    std::lock_guard<std::mutex> lock(w->mtx);
    const enum ggml_backend_buffer_usage usage = ggml_backend_buffer_get_usage(buffer);
    if (size <= PROXY_SMALL_CHUNK && usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        auto & sm = g_proxy.small;
        std::lock_guard<std::mutex> lock_small(sm.mtx);
        int i = sm.last;
        if (i < 0 || sm.last_size != size || memcmp(g_proxy.base + g_proxy.hdr->small_off + (uint64_t) i * PROXY_SMALL_CHUNK, data, size) != 0) {
            i = sm.next;
            sm.next = (i + 1) % PROXY_SMALL_N;
            for (proxy_worker * o : g_proxy.workers) {
                if (sm.seq[i][o->index]) {
                    proxy_wait_seq(o, sm.seq[i][o->index]);   // the chunk's last readers
                    sm.seq[i][o->index] = 0;
                }
            }
            memcpy(g_proxy.base + g_proxy.hdr->small_off + (uint64_t) i * PROXY_SMALL_CHUNK, data, size);
            sm.last = i;
            sm.last_size = size;
        }
        proxy_slot * mb = proxy_begin(w, CMD_SET_RAW);
        mb->a[0] = ((proxy_buffer_context *) buffer->context)->handle;
        mb->a[1] = (uint64_t) (uintptr_t) tensor->data + offset;
        mb->a[2] = size;
        mb->a[3] = g_proxy.hdr->small_off + (uint64_t) i * PROXY_SMALL_CHUNK;
        mb->a[4] = usage;
        proxy_issue(w);
        sm.seq[i][w->index] = w->seq;
        return;
    }
    proxy_finish_reads(w);   // (their data may sit in either staging half)
    const size_t half = g_proxy.hdr->staging_size / 2;
    size_t cap = half;
    bool in_place = false;
    static const bool noalign = getenv("GGML_SYNAPSE_PROXY_NOALIGN") != nullptr;
    if (!noalign && ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_synapse_device_context::weight_kind(tensor, in_place) == SYN_W_PLANAR && in_place) {
        const size_t rbb = ggml_row_size(tensor->type, tensor->ne[0]) * 128;
        if (rbb <= half && offset % rbb == 0) {
            cap = half / rbb * rbb;
        }
    }
    const std::vector<uint8_t> p = proxy_ser_tensor(tensor);
    const uint64_t t_in = now_ns();
    for (size_t done = 0; done < size; done += cap) {
        const size_t n = std::min(cap, size - done);
        uint8_t * buf = w->staging + w->half * half;
        const uint64_t buf_off = w->staging_off + w->half * half;
        const uint64_t t0 = now_ns();
        proxy_wait_seq(w, w->half_seq[w->half]);         // the last reader of this half finished
        const uint64_t t1 = now_ns();
        memcpy(buf, (const uint8_t *) data + done, n);
        g_set_stat.wait_ns += t1 - t0;
        g_set_stat.copy_ns += now_ns() - t1;
        proxy_slot * mb = proxy_begin(w, CMD_SET);
        mb->a[0] = offset + done;
        mb->a[1] = n;
        mb->a[2] = buf_off;
        proxy_payload(w, mb, p.data(), p.size());
        proxy_issue(w);
        w->half_seq[w->half] = w->seq;
        w->half ^= 1;
        static const bool wait = getenv("GGML_SYNAPSE_PROXY_SYNC_SET") != nullptr;
        if (wait) {
            proxy_wait(w);
        }
    }
    const uint64_t t_out = now_ns();
    if (g_set_stat.t_last) {
        g_set_stat.between_ns += t_in - g_set_stat.t_last;   // the caller's time between calls (file reads)
    }
    g_set_stat.t_last = t_out;
    g_set_stat.in_ns += t_out - t_in;
    g_set_stat.bytes += size;
    g_set_stat.n++;
}

// ---- shared host buffers ----
// The proxy devices' host buffer type: memory shared with the workers (a memfd each worker maps on first use and pins).
struct proxy_host_buffer {
    uint32_t  id;
    int       fd;
    uint8_t * base;
    size_t    len;
    std::atomic<uint32_t> mapped{0};   // workers holding a mapping (bit per worker index)
    std::atomic<uint32_t> failed{0};   // workers that could not map it
};

static std::mutex g_host_mtx;
static std::vector<proxy_host_buffer *> g_host;

// the host buffer holding [p, p + n), mapped by worker w (nullptr: none, or w cannot map it); w->mtx held
static proxy_host_buffer * proxy_host_mapped(proxy_worker * w, const void * p, size_t n) {
    proxy_host_buffer * hb = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_host_mtx);
        for (proxy_host_buffer * h : g_host) {
            if ((const uint8_t *) p >= h->base && (const uint8_t *) p + n <= h->base + h->len) {
                hb = h;
                break;
            }
        }
    }
    const uint32_t bit = 1u << w->index;
    if (!hb || (hb->failed & bit)) {
        return nullptr;
    }
    if (!(hb->mapped & bit)) {
        proxy_slot * mb = proxy_call(w, CMD_HOST_MAP, hb->id, (uint64_t) hb->fd, hb->len, (uint64_t) getpid());
        if (mb->r[0] != 0) {
            hb->failed.fetch_or(bit);
            return nullptr;
        }
        hb->mapped.fetch_or(bit);
    }
    return hb;
}

// wait for the queued reads and copy their data out of the staging area; w->mtx held
static void proxy_finish_reads(proxy_worker * w) {
    if (w->reads.empty()) {
        return;
    }
    proxy_wait_seq(w, w->reads_seq, true);
    for (const auto & r : w->reads) {
        if (!r.dst) {
            continue;
        }
        if (r.dst_stride == r.row_size) {
            memcpy(r.dst, w->staging + r.off, r.row_size * r.rows);
        } else {
            for (size_t i = 0; i < r.rows; i++) {
                memcpy(r.dst + i * r.dst_stride, w->staging + r.off + i * r.row_size, r.row_size);
            }
        }
    }
    w->reads.clear();
    w->reads_off = 0;
}

// queue a read of rows rows of row_size bytes, row_stride apart in the tensor from offset, to data (rows dst_stride
// apart): in place into a host buffer, else through the staging area (copied out by proxy_finish_reads; the rows
// must fit it); w->mtx held
static void proxy_read(proxy_worker * w, const ggml_tensor * t, void * data, size_t offset, size_t row_size, size_t rows,
                       size_t row_stride, size_t dst_stride) {
    proxy_host_buffer * hb = proxy_host_mapped(w, data, (rows - 1) * dst_stride + row_size);
    const size_t packed = row_size * rows;
    if (!hb) {
        GGML_ASSERT(packed <= g_proxy.hdr->staging_size);
        if (w->reads_off + packed > g_proxy.hdr->staging_size) {
            proxy_finish_reads(w);
        }
    }
    const std::vector<uint8_t> p = proxy_ser_tensor(t);
    const uint64_t dst = hb ? (uint64_t) ((uint8_t *) data - hb->base) : w->staging_off + w->reads_off;
    proxy_slot * mb = proxy_begin(w, rows == 1 ? CMD_GET : CMD_GET2D);
    mb->a[0] = offset;
    mb->a[1] = row_size;
    mb->a[2] = dst;
    if (rows == 1) {
        mb->a[3] = hb ? hb->id : 0;
    } else {
        mb->a[3] = rows;
        mb->a[4] = row_stride;
        mb->a[5] = hb ? hb->id : 0;
        mb->a[6] = dst_stride;
    }
    proxy_payload(w, mb, p.data(), p.size());
    proxy_issue(w);
    w->reads.push_back({ hb ? nullptr : (uint8_t *) data, w->reads_off, row_size, rows, dst_stride });
    w->reads_seq = w->seq;
    if (!hb) {
        w->reads_off += GGML_PAD(packed, 64);
    }
}

// queue reads of n_copies rows of size bytes (stride_tensor apart in the tensor, stride_data apart in data), in pieces
// that fit the staging area; w->mtx held
static void proxy_read_rows(proxy_worker * w, const ggml_tensor * t, void * data, size_t offset, size_t size, size_t n_copies,
                            size_t stride_tensor, size_t stride_data) {
    const size_t cap = g_proxy.hdr->staging_size;
    if (size == 0 || n_copies == 0) {
        return;
    }
    if (size > cap) {
        for (size_t i = 0; i < n_copies; i++) {
            for (size_t done = 0; done < size; done += cap) {
                const size_t n = std::min(cap, size - done);
                proxy_read(w, t, (uint8_t *) data + i * stride_data + done, offset + i * stride_tensor + done, n, 1, n, n);
            }
        }
        return;
    }
    const size_t per = cap / size;
    for (size_t i0 = 0; i0 < n_copies; i0 += per) {
        const size_t n = std::min(per, n_copies - i0);
        proxy_read(w, t, (uint8_t *) data + i0 * stride_data, offset + i0 * stride_tensor, size, n, stride_tensor, stride_data);
    }
}

static void proxy_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                    size_t size) {
    proxy_worker * w = ((proxy_buffer_context *) buffer->context)->w;
    std::lock_guard<std::mutex> lock(w->mtx);
    proxy_read_rows(w, tensor, data, offset, size, 1, size, size);
    proxy_finish_reads(w);
}

// strided rows: packed through the staging area, as many rows per command as fit
static void proxy_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                       size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    const size_t half = g_proxy.hdr->staging_size / 2;
    if (size > half || (stride_tensor == size && stride_data == size)) {
        if (stride_tensor == size && stride_data == size) {
            proxy_buffer_set_tensor(buffer, tensor, data, offset, size * n_copies);
            return;
        }
        for (size_t i = 0; i < n_copies; i++) {
            proxy_buffer_set_tensor(buffer, tensor, (const uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
        }
        return;
    }
    proxy_worker * w = ((proxy_buffer_context *) buffer->context)->w;
    std::lock_guard<std::mutex> lock(w->mtx);
    proxy_finish_reads(w);
    const std::vector<uint8_t> p = proxy_ser_tensor(tensor);
    size_t per = half / size;
    // planar weights: pieces of whole 128-row blocks
    bool in_place = false;
    if (ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS && stride_tensor == size &&
        ggml_backend_synapse_device_context::weight_kind(tensor, in_place) == SYN_W_PLANAR && in_place) {
        const size_t rbb = ggml_row_size(tensor->type, tensor->ne[0]) * 128;
        if (rbb % size == 0 && per >= rbb / size) {
            per = per / (rbb / size) * (rbb / size);
        }
    }
    for (size_t i0 = 0; i0 < n_copies; i0 += per) {
        const size_t n = std::min(per, n_copies - i0);
        uint8_t * buf = w->staging + w->half * half;
        const uint64_t buf_off = w->staging_off + w->half * half;
        proxy_wait_seq(w, w->half_seq[w->half]);   // the last reader of this half finished
        for (size_t i = 0; i < n; i++) {
            memcpy(buf + i * size, (const uint8_t *) data + (i0 + i) * stride_data, size);
        }
        proxy_call(w, CMD_SET2D, offset + i0 * stride_tensor, size, buf_off, n, p.data(), p.size(), stride_tensor, false);
        w->half_seq[w->half] = w->seq;
        w->half ^= 1;
    }
}

static void proxy_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                       size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    proxy_worker * w = ((proxy_buffer_context *) buffer->context)->w;
    std::lock_guard<std::mutex> lock(w->mtx);
    proxy_read_rows(w, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    proxy_finish_reads(w);
}

static void proxy_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset,
                                       size_t size) {
    proxy_worker * w = ((proxy_buffer_context *) buffer->context)->w;
    std::lock_guard<std::mutex> lock(w->mtx);
    const std::vector<uint8_t> p = proxy_ser_tensor(tensor);
    proxy_call(w, CMD_MEMSET, offset, size, value, 0, p.data(), p.size());
}

// card to card through the source worker's staging area
static bool proxy_copy(const ggml_tensor * src, ggml_tensor * dst) {
    if (!src->buffer || !dst->buffer || !proxy_buffer_is(src->buffer) || !proxy_buffer_is(dst->buffer) ||
        ggml_nbytes(src) != ggml_nbytes(dst)) {
        return false;
    }
    proxy_worker * ws = ((proxy_buffer_context *) src->buffer->context)->w;
    proxy_worker * wd = ((proxy_buffer_context *) dst->buffer->context)->w;
    const size_t size = ggml_nbytes(src);
    const size_t cap = g_proxy.hdr->staging_size;
    const std::vector<uint8_t> ps = proxy_ser_tensor(src);
    // lock order by worker index
    std::unique_lock<std::mutex> l0(ws->index <= wd->index ? ws->mtx : wd->mtx);
    std::unique_lock<std::mutex> l1;
    if (ws != wd) {
        l1 = std::unique_lock<std::mutex>(ws->index <= wd->index ? wd->mtx : ws->mtx);
    }
    proxy_finish_reads(ws);
    for (size_t done = 0; done < size; done += cap) {
        const size_t n = std::min(cap, size - done);
        proxy_call(ws, CMD_GET, done, n, ws->staging_off, 0, ps.data(), ps.size());
        proxy_set_from(wd, dst, done, n, ws->staging_off);
    }
    return true;
}

static bool proxy_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    return proxy_copy(src, dst);
}

static void proxy_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    proxy_buffer_context * ctx = (proxy_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(ctx->w->mtx);
    proxy_call(ctx->w, CMD_CLEAR, ctx->handle, value);
}

static const ggml_backend_buffer_i proxy_buffer_interface = {
    /* .free_buffer     = */ proxy_buffer_free,
    /* .get_base        = */ proxy_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ proxy_buffer_memset_tensor,
    /* .set_tensor      = */ proxy_buffer_set_tensor,
    /* .get_tensor      = */ proxy_buffer_get_tensor,
    /* .set_tensor_2d   = */ proxy_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ proxy_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ proxy_buffer_cpy_tensor,
    /* .clear           = */ proxy_buffer_clear,
    /* .reset           = */ NULL,
};

static bool proxy_buffer_is(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_base == proxy_buffer_get_base;
}

// ---- buffer type ----

static const char * proxy_buft_get_name(ggml_backend_buffer_type_t buft) {
    return ((proxy_worker *) buft->context)->name.c_str();
}

static ggml_backend_buffer_t proxy_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    proxy_worker * w = (proxy_worker *) buft->context;
    if (!proxy_start(w)) {
        return nullptr;
    }
    uint64_t handle, base;
    {
        std::lock_guard<std::mutex> lock(w->mtx);
        proxy_slot * mb = proxy_call(w, CMD_ALLOC, size);
        handle = mb->r[0];
        base = mb->r[1];
    }
    if (handle == 0) {
        GGML_LOG_ERROR("%s: %s: allocation of %zu bytes failed\n", __func__, w->name.c_str(), size);
        return nullptr;
    }
    proxy_buffer_context * ctx = new proxy_buffer_context{ w, handle, base, size };
    return ggml_backend_buffer_init(buft, proxy_buffer_interface, ctx, size);
}

static size_t proxy_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t proxy_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i proxy_buft_interface = {
    /* .get_name         = */ proxy_buft_get_name,
    /* .alloc_buffer     = */ proxy_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ NULL,
    /* .get_alignment    = */ proxy_buft_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ proxy_buft_get_alloc_size,
    /* .get_alloc_size_n = */ NULL,
    /* .is_host          = */ NULL,
};

// ---- host buffer type (see proxy_host_buffer) ----

static void proxy_host_buffer_free(ggml_backend_buffer_t buffer) {
    proxy_host_buffer * hb = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_host_mtx);
        for (auto it = g_host.begin(); it != g_host.end(); ++it) {
            if ((*it)->base == buffer->context) {
                hb = *it;
                g_host.erase(it);
                break;
            }
        }
    }
    GGML_ASSERT(hb);
    for (proxy_worker * w : g_proxy.workers) {
        std::lock_guard<std::mutex> lock(w->mtx);
        if (w->pid > 0) {
            proxy_finish_reads(w);   // (some may write here)
            if (hb->mapped & (1u << w->index)) {
                proxy_call(w, CMD_HOST_UNMAP, hb->id);
            }
        }
    }
    munmap(hb->base, hb->len);
    close(hb->fd);
    delete hb;
}

static ggml_backend_buffer_t proxy_host_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    static uint32_t next_id = 1;
    const size_t len = GGML_PAD(std::max<size_t>(size, 1), 4096);
    const int fd = (int) syscall(SYS_memfd_create, "ggml-synapse-host", MFD_CLOEXEC);
    void * p = MAP_FAILED;
    if (fd >= 0 && ftruncate(fd, (off_t) len) == 0) {
        p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    if (p == MAP_FAILED) {
        if (fd >= 0) {
            close(fd);
        }
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    proxy_host_buffer * hb = new proxy_host_buffer();
    hb->fd = fd;
    hb->base = (uint8_t *) p;
    hb->len = len;
    {
        std::lock_guard<std::mutex> lock(g_host_mtx);
        hb->id = next_id++;
        g_host.push_back(hb);
    }
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(p, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = proxy_host_buffer_free;
    return buffer;
}

static const char * proxy_host_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_SYNAPSE_NAME "_Host";
}

static size_t proxy_host_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static bool proxy_host_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

// one type for every card: the meta backend (tensor split) offers a host buffer type only when its devices share one.
// No device: llama starts a backend for the device of each buffer type holding weights, and this type is no card's
static ggml_backend_buffer_type_t proxy_dev_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    if (getenv("GGML_SYNAPSE_NO_HOSTBUF")) {
        return nullptr;
    }
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ {
            /* .get_name         = */ proxy_host_buft_get_name,
            /* .alloc_buffer     = */ proxy_host_buft_alloc_buffer,
            /* .alloc_buffer_n   = */ NULL,
            /* .get_alignment    = */ proxy_host_buft_get_alignment,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL,
            /* .get_alloc_size_n = */ NULL,
            /* .is_host          = */ proxy_host_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &buft;
}

// ---- backend ----

struct proxy_device_context {
    proxy_worker * w;
    ggml_backend_buffer_type buft;
};

static ggml_guid_t proxy_guid() {
    static ggml_guid guid = { 0x5c, 0x1e, 0x93, 0x0a, 0x7b, 0x24, 0x4e, 0x61,
                              0x8f, 0xd2, 0x3a, 0x05, 0xc9, 0x6e, 0x11, 0xb7 };
    return &guid;
}

static const char * proxy_backend_get_name(ggml_backend_t backend) {
    return ((proxy_worker *) backend->context)->name.c_str();
}

static void proxy_backend_free(ggml_backend_t backend) {
    delete backend;
}

static void proxy_backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset,
                                           size_t size) {
    GGML_UNUSED(backend);
    proxy_buffer_set_tensor(tensor->buffer, tensor, data, offset, size);
}

// queued: the data is there after the next synchronize
static void proxy_backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset,
                                           size_t size) {
    proxy_worker * w = (proxy_worker *) backend->context;
    GGML_ASSERT(((proxy_buffer_context *) tensor->buffer->context)->w == w);
    std::lock_guard<std::mutex> lock(w->mtx);
    proxy_read_rows(w, tensor, data, offset, size, 1, size, size);
}

static void proxy_backend_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_UNUSED(backend);
    proxy_buffer_set_tensor_2d(tensor->buffer, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static void proxy_backend_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    proxy_worker * w = (proxy_worker *) backend->context;
    GGML_ASSERT(((proxy_buffer_context *) tensor->buffer->context)->w == w);
    std::lock_guard<std::mutex> lock(w->mtx);
    proxy_read_rows(w, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static bool proxy_backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src,
                                           ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    return proxy_copy(src, dst);
}

static void proxy_backend_synchronize(ggml_backend_t backend) {
    proxy_worker * w = (proxy_worker *) backend->context;
    bool busy;
    {
        std::lock_guard<std::mutex> lock(w->mtx);
        busy = w->dirty || !w->reads.empty();
    }
    for (proxy_worker * p : w->peers) {
        if (busy && p != w) {
            std::lock_guard<std::mutex> lock(p->mtx);
            if (p->dirty) {
                proxy_call(p, CMD_SYNC, 0, 0, 0, 0, nullptr, 0, 0, false);
            }
        }
    }
    std::lock_guard<std::mutex> lock(w->mtx);
    if (w->dirty) {
        proxy_call(w, CMD_SYNC, 0, 0, 0, 0, nullptr, 0, 0, false);
    }
    proxy_wait_seq(w, w->sync_seq, true);
    proxy_finish_reads(w);
}

// what a reused graph must match: per node its identity, data, op, flags, shape, op params and sources
static void proxy_graph_sig(const ggml_cgraph * cg, std::vector<uint8_t> & sig) {
    struct node_sig {
        const void * t;
        const void * data;
        const void * src[GGML_MAX_SRC];
        int32_t op, flags;
        int64_t ne[GGML_MAX_DIMS];
        size_t nb[GGML_MAX_DIMS];
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    };
    sig.resize((size_t) cg->n_nodes * sizeof(node_sig));
    node_sig * ns = (node_sig *) sig.data();
    for (int i = 0; i < cg->n_nodes; i++) {
        const ggml_tensor * t = cg->nodes[i];
        node_sig & n = ns[i];
        memset(&n, 0, sizeof(n));
        n.t = t;
        n.data = t->data;
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            n.src[k] = t->src[k];
        }
        n.op = t->op;
        n.flags = t->flags;
        memcpy(n.ne, t->ne, sizeof(n.ne));
        memcpy(n.nb, t->nb, sizeof(n.nb));
        memcpy(n.op_params, t->op_params, sizeof(n.op_params));
    }
}

static enum ggml_status proxy_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    proxy_worker * w = (proxy_worker *) backend->context;
    ggml_synapse_trace("p compute");
    std::lock_guard<std::mutex> lock(w->mtx);
    const int nn = cgraph->n_nodes;
    auto tag = [&](proxy_worker::graph_sig & e) {
        for (int i = 0; i < nn; i++) {
            cgraph->nodes[i]->extra = &e;
        }
    };
    if (nn > 0 && cgraph->nodes[nn - 1]->extra) {
        const void * e = cgraph->nodes[nn - 1]->extra;
        for (auto it = w->graphs.begin(); it != w->graphs.end(); ++it) {
            if (&*it != e) {
                continue;
            }
            bool ok = it->cg == cgraph && it->n_nodes == nn;
            for (int i = 0; ok && i < nn; i += 32) {
                ok = cgraph->nodes[i]->extra == e;
            }
            if (ok) {
                w->graphs.splice(w->graphs.begin(), w->graphs, it);
                proxy_call(w, CMD_COMPUTE, it->id, 0, 0, 0, nullptr, 0, 0, false);
                return GGML_STATUS_SUCCESS;
            }
            break;
        }
    }
    // a graph the worker already holds: its id only
    std::vector<uint8_t> sig;
    proxy_graph_sig(cgraph, sig);
    for (auto it = w->graphs.begin(); it != w->graphs.end(); ++it) {
        if (it->cg == cgraph && it->sig == sig) {
            w->graphs.splice(w->graphs.begin(), w->graphs, it);
            tag(*it);
            proxy_call(w, CMD_COMPUTE, it->id, 0, 0, 0, nullptr, 0, 0, false);
            return GGML_STATUS_SUCCESS;   // queued; a failure surfaces at the next wait
        }
    }
    proxy_ser s;
    const std::unordered_set<const ggml_tensor *> in_graph(cgraph->nodes, cgraph->nodes + cgraph->n_nodes);
    s.w = w;
    s.nodes = &in_graph;
    std::vector<int32_t> nodes;
    nodes.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * t = cgraph->nodes[i];
        // a view of a tensor on another card is left in this split's node range by the scheduler
        // (it copies the view itself for its consumers here): nothing in this graph reads it
        if (t->view_src && t->view_src->buffer && proxy_buffer_is(t->view_src->buffer) &&
            ((proxy_buffer_context *) t->view_src->buffer->context)->w != w &&
            (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE)) {
            continue;
        }
        nodes.push_back(s.add(t));
    }
    // the worker's copy of the graph has no use counts: nodes read outside this graph (as the
    // lowering decides from cgraph->use_counts) are flagged as outputs
    if (cgraph->use_counts) {
        std::unordered_map<const ggml_tensor *, int> uses;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            if (t->flags & GGML_TENSOR_FLAG_COMPUTE) {
                for (int k = 0; k < GGML_MAX_SRC; k++) {
                    if (t->src[k]) {
                        uses[t->src[k]]++;
                    }
                }
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            const size_t h = ggml_hash_find(&cgraph->visited_hash_set, t);
            const auto si = s.idx.find(t);
            if (si != s.idx.end() && h != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, h)) {
                const int u = uses.count(t) ? uses[t] : 0;
                if (cgraph->use_counts[h] == 0 || cgraph->use_counts[h] > u) {
                    s.ts[si->second].flags |= GGML_TENSOR_FLAG_OUTPUT;
                }
            }
        }
    }
    std::vector<uint8_t> p(8 + s.ts.size() * sizeof(proxy_tensor) + nodes.size() * 4);
    const int32_t nt = (int32_t) s.ts.size(), nnodes = (int32_t) nodes.size();
    memcpy(p.data(), &nt, 4);
    memcpy(p.data() + 4, &nnodes, 4);
    memcpy(p.data() + 8, s.ts.data(), s.ts.size() * sizeof(proxy_tensor));
    memcpy(p.data() + 8 + s.ts.size() * sizeof(proxy_tensor), nodes.data(), nodes.size() * 4);
    const size_t h = std::hash<std::string_view>{}(std::string_view((const char *) p.data(), p.size()));
    for (auto [b, e] = w->by_blob.equal_range(h); b != e; ++b) {
        auto it = b->second;
        if (it->blob == p) {
            it->cg = cgraph;
            it->n_nodes = nn;
            it->sig = std::move(sig);
            w->graphs.splice(w->graphs.begin(), w->graphs, it);
            tag(*it);
            proxy_call(w, CMD_COMPUTE, it->id, 0, 0, 0, nullptr, 0, 0, false);
            return GGML_STATUS_SUCCESS;   // queued; a failure surfaces at the next wait
        }
    }
    uint32_t drop = 0;
    if (w->graphs.size() >= 2048) {
        auto last = std::prev(w->graphs.end());
        const size_t hl = std::hash<std::string_view>{}(std::string_view((const char *) last->blob.data(), last->blob.size()));
        for (auto [b, e] = w->by_blob.equal_range(hl); b != e; ++b) {
            if (b->second == last) {
                w->by_blob.erase(b);
                break;
            }
        }
        drop = last->id;
        w->graphs.pop_back();
    }
    const uint32_t id = w->next_graph++;
    w->graphs.push_front({ id, cgraph, nn, std::move(sig), p });
    w->by_blob.emplace(h, w->graphs.begin());
    tag(w->graphs.front());
    proxy_call(w, CMD_COMPUTE, id, drop, 0, 0, p.data(), p.size(), 0, false);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i proxy_backend_interface = {
    /* .get_name                = */ proxy_backend_get_name,
    /* .free                    = */ proxy_backend_free,
    /* .set_tensor_async        = */ proxy_backend_set_tensor_async,
    /* .get_tensor_async        = */ proxy_backend_get_tensor_async,
    /* .set_tensor_2d_async     = */ proxy_backend_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ proxy_backend_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ proxy_backend_cpy_tensor_async,
    /* .synchronize             = */ proxy_backend_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ proxy_backend_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

// ---- device ----

static const char * proxy_dev_get_name(ggml_backend_dev_t dev) {
    return ((proxy_device_context *) dev->context)->w->name.c_str();
}

static const char * proxy_dev_get_description(ggml_backend_dev_t dev) {
    return ((proxy_device_context *) dev->context)->w->description.c_str();
}

// a started worker reports its own free memory (an open card shows 0 free to the driver: Synapse reserves it all);
// before that the driver's figures, where a card held by another process shows 0 free
static void proxy_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    proxy_worker * w = ((proxy_device_context *) dev->context)->w;
    {
        std::lock_guard<std::mutex> lock(w->mtx);
        if (w->pid > 0) {
            proxy_slot * mb = proxy_call(w, CMD_HELLO);
            *total = mb->r[0];
            *free = mb->r[1];
            return;
        }
    }
    if (!ggml_synapse_card_memory(w->card, free, total)) {
        *free = *total = 0;
    }
}

static enum ggml_backend_dev_type proxy_dev_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void proxy_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = proxy_dev_get_name(dev);
    props->description = proxy_dev_get_description(dev);
    props->type = proxy_dev_get_type(dev);
    props->device_id = ggml_synapse_card_pci(((proxy_device_context *) dev->context)->w->card);
    proxy_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ true,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t proxy_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    if (!proxy_start(((proxy_device_context *) dev->context)->w)) {
        return nullptr;
    }
    return new ggml_backend{
        /* .guid      = */ proxy_guid(),
        /* .interface = */ proxy_backend_interface,
        /* .device    = */ dev,
        /* .context   = */ ((proxy_device_context *) dev->context)->w,
    };
}

static ggml_backend_buffer_type_t proxy_dev_get_buffer_type(ggml_backend_dev_t dev) {
    return &((proxy_device_context *) dev->context)->buft;
}

static bool proxy_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft == proxy_dev_get_buffer_type(dev);
}

static bool proxy_dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    if (op->op == GGML_OP_GET_ROWS) {
        return false;
    }
    return ggml_synapse_supports_op(dev, op) && op->ne[1] >= 32;
}

static const ggml_backend_device_i proxy_dev_interface = {
    /* .get_name                = */ proxy_dev_get_name,
    /* .get_description         = */ proxy_dev_get_description,
    /* .get_memory              = */ proxy_dev_get_memory,
    /* .get_type                = */ proxy_dev_get_type,
    /* .get_props               = */ proxy_dev_get_props,
    /* .init_backend            = */ proxy_dev_init_backend,
    /* .get_buffer_type         = */ proxy_dev_get_buffer_type,
    /* .get_host_buffer_type    = */ proxy_dev_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_synapse_supports_op,
    /* .supports_buft           = */ proxy_dev_supports_buft,
    /* .offload_op              = */ proxy_dev_offload_op,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

// ---- collectives: allreduce through host memory ----
// Each worker copies its tensor into its reduce slot, waits for every rank (arrived[]), sums the
// slots in rank order (identical results on every card) and writes the sum back.

// GGML_SYNAPSE_COMM=host: the host-memory allreduce; default HCCL (in place on every card's stream,
// ordered with its graphs)
struct proxy_comm {
    std::vector<proxy_worker *> ws;
    uint32_t gen = 0;
    uint64_t key = 0;   // HCCL communicator of the workers, 0: host allreduce
};

static void * proxy_comm_init(ggml_backend_t * backends, size_t n) {
    proxy_comm * c = new proxy_comm;
    for (size_t i = 0; i < n; i++) {
        if (!backends[i] || !ggml_guid_matches(backends[i]->guid, proxy_guid())) {
            delete c;
            return nullptr;
        }
        c->ws.push_back((proxy_worker *) backends[i]->context);
    }
    const char * mode = getenv("GGML_SYNAPSE_COMM");
    if (!mode || strcmp(mode, "host") != 0) {
        static uint64_t next_key = 1;
        const uint64_t key = next_key++;
        std::vector<std::unique_lock<std::mutex>> locks;
        for (proxy_worker * w : c->ws) {
            locks.emplace_back(w->mtx);
        }
        proxy_call(c->ws[0], CMD_CCL_ID);
        for (size_t i = 0; i < n; i++) {
            proxy_slot * mb = proxy_begin(c->ws[i], CMD_CCL_INIT);
            mb->a[0] = key;
            mb->a[1] = i;
            mb->a[2] = n;
            proxy_issue(c->ws[i]);
        }
        bool ok = true;
        for (proxy_worker * w : c->ws) {
            proxy_wait(w);
            ok = ok && w->mb->slot[w->seq % PROXY_RING].r[0] == 0;
        }
        if (ok) {
            c->key = key;
        } else {
            GGML_LOG_WARN("%s: HCCL communicator failed; allreduce through host memory\n", __func__);
        }
    }
    for (proxy_worker * w : c->ws) {
        std::lock_guard<std::mutex> lock(w->mtx);
        w->peers = c->ws;
    }
    return c;
}

static void proxy_comm_free(void * ctx) {
    proxy_comm * c = (proxy_comm *) ctx;
    for (proxy_worker * w : c->ws) {
        std::lock_guard<std::mutex> lock(w->mtx);
        if (w->peers == c->ws) {
            w->peers.clear();
        }
    }
    if (c->key) {
        for (proxy_worker * w : c->ws) {
            std::lock_guard<std::mutex> lock(w->mtx);
            proxy_call(w, CMD_CCL_FREE, c->key);
        }
    }
    delete c;
}

// A tensor whose compute flag the meta backend cleared (this card's slice of one of its sources is empty, e.g. a
// split giving it none of an FFN's rows) holds no value: it contributes zeros (the meta backend's own fallback
// all-reduce zeroes it the same way; with a backend all-reduce that is the backend's job).
static bool proxy_comm_allreduce_tensor(void * ctx, ggml_tensor ** tensors) {
    proxy_comm * c = (proxy_comm *) ctx;
    const size_t n = c->ws.size();
    if (c->key) {
        const ggml_type type = tensors[0]->type;
        for (size_t i = 0; i < n; i++) {
            const ggml_tensor * t = tensors[i];
            if ((type != GGML_TYPE_F32 && type != GGML_TYPE_BF16) || t->type != type || !ggml_is_contiguous(t) ||
                ggml_nelements(t) != ggml_nelements(tensors[0]) || !t->buffer || !proxy_buffer_is(t->buffer) ||
                ((proxy_buffer_context *) t->buffer->context)->w != c->ws[i]) {
                return false;
            }
        }
        std::vector<std::unique_lock<std::mutex>> locks;
        for (size_t i = 0; i < n; i++) {
            locks.emplace_back(c->ws[i]->mtx);
        }
        const uint64_t t0 = proxy_stats_on() ? now_ns() : 0;
        ggml_synapse_trace("p ccl");
        for (size_t i = 0; i < n; i++) {
            proxy_worker * w = c->ws[i];
            proxy_slot * mb = proxy_begin(w, CMD_CCL_REDUCE);
            mb->a[0] = c->key;
            mb->a[1] = (uint64_t) (uintptr_t) tensors[i]->data;
            mb->a[2] = (uint64_t) ggml_nelements(tensors[i]);
            mb->a[3] = type == GGML_TYPE_BF16;
            mb->a[4] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;   // see proxy_comm_allreduce_tensor
            proxy_issue(w);   // queued on the card's stream after its graphs; failures surface at the next wait
        }
        ggml_synapse_trace("p ccl end");
        if (t0) {
            g_stats[CMD_CCL_REDUCE].n++;
            g_stats[CMD_CCL_REDUCE].ns += now_ns() - t0;
        }
        return true;
    }
    const size_t nbytes = ggml_nbytes(tensors[0]);
    for (size_t i = 0; i < n; i++) {
        const ggml_tensor * t = tensors[i];
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nbytes(t) != nbytes || !t->buffer ||
            !proxy_buffer_is(t->buffer) || ((proxy_buffer_context *) t->buffer->context)->w != c->ws[i]) {
            return false;
        }
    }
    if (nbytes > g_proxy.hdr->reduce_slot) {
        return false;
    }
    const uint32_t gen = ++c->gen;
    std::vector<std::unique_lock<std::mutex>> locks;
    for (size_t i = 0; i < n; i++) {
        locks.emplace_back(c->ws[i]->mtx);
    }
    const uint64_t t0 = proxy_stats_on() ? now_ns() : 0;
    for (size_t i = 0; i < n; i++) {
        proxy_worker * w = c->ws[i];
        const std::vector<uint8_t> p = proxy_ser_tensor(tensors[i]);
        proxy_slot * mb = proxy_begin(w, CMD_ALLREDUCE);
        mb->a[0] = gen;
        mb->a[1] = nbytes;
        mb->a[2] = i;
        mb->a[3] = n;
        mb->a[4] = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
        proxy_payload(w, mb, p.data(), p.size());
        proxy_issue(w);
    }
    for (size_t i = 0; i < n; i++) {
        proxy_wait(c->ws[i]);
    }
    if (t0) {
        g_stats[CMD_ALLREDUCE].n++;
        g_stats[CMD_ALLREDUCE].ns += now_ns() - t0;
    }
    return true;
}

void * ggml_synapse_proxy_proc_address(const char * name) {
    if (!g_proxy.hdr) {
        return nullptr;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *) proxy_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *) proxy_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *) proxy_comm_allreduce_tensor;
    }
    return nullptr;
}

// ---- startup / shutdown ----

static std::string proxy_worker_path() {
    if (const char * e = getenv("GGML_SYNAPSE_WORKER")) {
        return e;
    }
    Dl_info info;
    if (dladdr((void *) &proxy_worker_path, &info) && info.dli_fname) {
        std::string p = info.dli_fname;
        const size_t s = p.rfind('/');
        return (s == std::string::npos ? std::string(".") : p.substr(0, s)) + "/ggml-synapse-worker";
    }
    return "ggml-synapse-worker";
}

static void proxy_shutdown() {
    if (proxy_stats_on()) {
        static const char * names[] = { "", "hello", "alloc", "free", "clear", "set", "get", "memset", "set2d", "get2d",
                                         "compute", "sync", "exit", "allreduce", "ccl_id", "ccl_init", "ccl_reduce",
                                         "ccl_free", "host_map", "host_unmap", "set_raw" };
        if (g_set_stat.n) {
            fprintf(stderr, "proxy set_tensor n %lu  %.1f GB  inside %.1f s (wait %.1f s copy %.1f s)  between calls %.1f s\n",
                    (unsigned long) g_set_stat.n, g_set_stat.bytes / 1e9, g_set_stat.in_ns / 1e9, g_set_stat.wait_ns / 1e9,
                    g_set_stat.copy_ns / 1e9, g_set_stat.between_ns / 1e9);
        }
        for (int c = 1; c < (int) (sizeof(names) / sizeof(names[0])); c++) {
            if (g_stats[c].n) {
                fprintf(stderr, "proxy %-8s n %8lu  avg %8.1f us  total %9.1f ms  payload %8.1f MB\n", names[c],
                        (unsigned long) g_stats[c].n.load(), g_stats[c].ns / 1e3 / g_stats[c].n, g_stats[c].ns / 1e6,
                        g_stats[c].bytes / 1e6);
            }
        }
    }
    for (proxy_worker * w : g_proxy.workers) {
        if (w->pid > 0) {
            std::lock_guard<std::mutex> lock(w->mtx);
            proxy_begin(w, CMD_EXIT);
            proxy_issue(w);
        }
    }
    for (proxy_worker * w : g_proxy.workers) {
        if (w->pid > 0) {
            int st = 0;
            waitpid(w->pid, &st, 0);
            w->pid = 0;
        }
    }
}

bool ggml_synapse_proxy_init(ggml_backend_reg_t reg, std::vector<ggml_backend_dev_t> & devices) {
    const int n = std::min(ggml_synapse_card_count(), PROXY_MAX_WORKERS);
    if (n <= 0) {
        return false;
    }
    const uint64_t mb = 1ull << 20;
    const uint64_t payload = 64 * mb;
    const uint64_t staging = (getenv("GGML_SYNAPSE_SHM_MB") ? atoll(getenv("GGML_SYNAPSE_SHM_MB")) : 256) * mb;
    const uint64_t area_off = GGML_PAD(sizeof(proxy_header), 4096);
    const uint64_t reduce_slot = 64 * mb;
    const uint64_t reduce_off = area_off + (uint64_t) n * (payload + staging);
    const uint64_t small_off = reduce_off + (uint64_t) n * 2 * reduce_slot;
    const size_t size = small_off + (uint64_t) PROXY_SMALL_N * PROXY_SMALL_CHUNK;

    // pages of the segment are allocated when touched: the areas of unused cards cost nothing
    const int fd = (int) syscall(SYS_memfd_create, "ggml-synapse", 0);
    if (fd < 0 || ftruncate(fd, (off_t) size) != 0) {
        GGML_LOG_ERROR("%s: shared memory: %s\n", __func__, strerror(errno));
        return false;
    }
    uint8_t * base = (uint8_t *) mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        GGML_LOG_ERROR("%s: mmap: %s\n", __func__, strerror(errno));
        return false;
    }
    proxy_header * hdr = new (base) proxy_header();
    hdr->magic = PROXY_MAGIC;
    hdr->n_workers = (uint32_t) n;
    hdr->payload_size = payload;
    hdr->staging_size = staging;
    hdr->area_off = area_off;
    hdr->reduce_off = reduce_off;
    hdr->reduce_slot = reduce_slot;
    hdr->small_off = small_off;
    g_proxy.fd = fd;
    g_proxy.base = base;
    g_proxy.size = size;
    g_proxy.hdr = hdr;

    g_proxy.exe = proxy_worker_path();
    // each worker sees only its own card (HABANA_VISIBLE_MODULES set per worker)
    for (char ** e = environ; *e; e++) {
        if (strncmp(*e, "HABANA_VISIBLE_MODULES=", 23) != 0) {
            g_proxy.env.push_back(*e);
        }
    }
    for (int i = 0; i < n; i++) {
        proxy_worker * w = new proxy_worker;
        w->index = i;
        w->card = i;
        w->module = ggml_synapse_card_module(i);
        w->description = ggml_synapse_card_description(i);
        w->mb = &hdr->mb[i];
        const uint64_t off = area_off + (uint64_t) i * (payload + staging);
        w->payload = base + off;
        w->staging = base + off + payload;
        w->staging_off = off + payload;
        w->name = std::string(GGML_SYNAPSE_NAME) + std::to_string(w->card);
        w->seq = 0;
        g_proxy.workers.push_back(w);
    }
    atexit(proxy_shutdown);

    for (proxy_worker * w : g_proxy.workers) {
        proxy_device_context * dctx = new proxy_device_context;
        dctx->w = w;
        ggml_backend_dev_t dev = new ggml_backend_device{
            /* .iface   = */ proxy_dev_interface,
            /* .reg     = */ reg,
            /* .context = */ dctx,
        };
        dctx->buft = ggml_backend_buffer_type{
            /* .iface   = */ proxy_buft_interface,
            /* .device  = */ dev,
            /* .context = */ w,
        };
        devices.push_back(dev);
    }
    return true;
}

// start the card's worker (first use of the card); false if it cannot open the card
static bool proxy_start(proxy_worker * w) {
    std::lock_guard<std::mutex> lock(g_proxy.start_mtx);
    if (w->pid > 0) {
        return true;
    }
    if (w->failed) {
        return false;
    }
    std::vector<std::string> args = { g_proxy.exe, std::to_string(g_proxy.fd), std::to_string(w->index), std::to_string(w->card) };
    std::vector<std::string> env = g_proxy.env;
    env.push_back("HABANA_VISIBLE_MODULES=" + std::to_string(w->module));
    std::vector<char *> argv, envp;
    for (auto & a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    for (auto & e : env) envp.push_back(e.data());
    envp.push_back(nullptr);
    pid_t pid = 0;
    if (posix_spawn(&pid, g_proxy.exe.c_str(), nullptr, nullptr, argv.data(), envp.data()) != 0) {
        GGML_LOG_ERROR("%s: cannot start %s: %s\n", __func__, g_proxy.exe.c_str(), strerror(errno));
        w->failed = true;
        return false;
    }
    futex_wait_ne(&w->mb->ready, 0, pid);
    if (w->mb->ready.load() != 1) {
        int st = 0;
        waitpid(pid, &st, 0);
        GGML_LOG_ERROR("%s: %s (module %d): the card worker failed to start (card in use by another process?)\n", __func__,
                       w->name.c_str(), w->module);
        w->failed = true;
        return false;
    }
    std::lock_guard<std::mutex> lock_w(w->mtx);
    w->pid = pid;
    proxy_slot * m = proxy_call(w, CMD_HELLO);
    w->mem_total = m->r[0];
    w->mem_free = m->r[1];
    GGML_LOG_INFO("%s: %s: worker pid %d, %.1f GiB free\n", __func__, w->name.c_str(), (int) w->pid,
                  w->mem_free / 1073741824.0);
    return true;
}

// ---------------------------------------------------------------------------
// worker side
// ---------------------------------------------------------------------------

namespace {

struct worker_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
};

// a reduced tensor whose value is held in a slot: the [count, ranks] slices, or (b16) the bf16 sum
struct ag_pending {
    uint64_t  addr;
    int64_t   count;
    ggml_type type;
    int       ranks;
    uint64_t  slot;   // device address of the [count, ranks] slices
    bool      b16 = false;   // slot: the bf16 all-reduced sum
    size_t nbytes() const { return (size_t) count * ggml_type_size(type); }
    bool operator==(const ag_pending & o) const {
        return addr == o.addr && count == o.count && type == o.type && ranks == o.ranks && slot == o.slot && b16 == o.b16;
    }
};

// a graph with pending sums folded in (id: the parent's graph; 0: the sums alone, written back)
struct ag_graph {
    uint32_t id;
    std::vector<ag_pending> sig;
    worker_graph g;            // g.gf null: the graph as it is
    std::vector<bool> dead;    // by pending entry: overwritten by the graph (no longer pending after it)
};

struct worker_state {
    proxy_header * hdr;
    uint8_t * base;
    proxy_mailbox * mb;
    uint8_t * big;        // the payload area
    uint8_t * payload;    // the current command's payload
    ggml_backend_t backend;
    ggml_backend_buffer_type_t buft;
    std::vector<ggml_backend_buffer_t> buffers = { nullptr };   // handle -> buffer (0: none)
    std::unordered_map<uint32_t, worker_graph> graphs;          // graphs by the parent's id
    std::vector<float> reduce;                                  // allreduce sum
    std::map<uint64_t, hcclComm_t> comms;                       // HCCL communicators by the parent's key
    std::map<uint64_t, int> comm_ranks;
    // gather slots (see ag_pending)
    int64_t ag_max = 0;                                         // elements per rank, 0: off
    int ag_ranks = 0;
    size_t ag_slot_bytes = 0;
    ggml_backend_buffer_t ag_buf = nullptr;
    std::unordered_map<uint64_t, int> ag_slots;                 // reduced address -> slot
    std::vector<ag_pending> pend;
    std::unordered_multimap<uint64_t, ag_graph> ag_graphs;      // by ag_key
    // bf16 all-reduces (see ag_pending.b16)
    int64_t ar16_min = 0;                                       // elements, 0: off (set from GGML_SYNAPSE_AR_BF16)
    size_t ar16_slot_bytes = 0;
    ggml_backend_buffer_t ar16_buf = nullptr;
    std::unordered_map<uint64_t, worker_graph> ar16_casts;      // f32 -> slot casts by (addr, count, slot)
    std::vector<ggml_backend_buffer_t> ar16_old;                // outgrown slot buffers (cached graphs may still name them)
    std::vector<ggml_context *> ar16_old_ctx;
    std::unordered_map<uint64_t, worker_graph> zero_fills;      // zeroing graphs of reduced tensors by (addr, count, type)
    std::unordered_map<uint32_t, std::pair<uint8_t *, size_t>> host_bufs;   // the parent's host buffers mapped here, by id
};

// tensors of a message into ctx; returns them by index
std::vector<ggml_tensor *> worker_build(worker_state & ws, ggml_context * ctx, const proxy_tensor * ts, int n) {
    std::vector<ggml_tensor *> out(n);
    for (int i = 0; i < n; i++) {
        const proxy_tensor & p = ts[i];
        ggml_tensor * t = ggml_new_tensor(ctx, (ggml_type) p.type, GGML_MAX_DIMS, p.ne);
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            t->nb[d] = p.nb[d];
        }
        t->op = (ggml_op) p.op;
        t->flags = p.flags;
        memcpy(t->op_params, p.op_params, sizeof(p.op_params));
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            t->src[k] = p.src[k] == i ? t : p.src[k] >= 0 ? out[p.src[k]] : nullptr;   // (a cast's self-reference)
        }
        t->view_src = p.view_src >= 0 ? out[p.view_src] : nullptr;
        t->view_offs = p.view_offs;
        t->data = (void *) (uintptr_t) p.data;
        if (p.buffer) {
            GGML_ASSERT(p.buffer < ws.buffers.size() && ws.buffers[p.buffer]);
            t->buffer = ws.buffers[p.buffer];
            ggml_backend_buffer_set_usage(t->buffer, (ggml_backend_buffer_usage) p.usage);
        }
        memcpy(t->name, p.name, sizeof(t->name));
        out[i] = t;
    }
    return out;
}

// the message's tensor (payload: n, index, tensors) in a scratch context
ggml_tensor * worker_tensor(worker_state & ws, ggml_context ** ctx) {
    int32_t n, i;
    memcpy(&n, ws.payload, 4);
    memcpy(&i, ws.payload + 4, 4);
    ggml_init_params ip = { (size_t) n * ggml_tensor_overhead() + 1024, nullptr, true };
    *ctx = ggml_init(ip);
    return worker_build(ws, *ctx, (const proxy_tensor *) (ws.payload + 8), n)[i];
}

uint64_t g_wstat_hit = 0, g_wstat_miss = 0, g_wstat_ns = 0, g_wstat_get_ns = 0, g_wstat_ag = 0, g_wstat_ag_fallback = 0;

void ag_drop(worker_state & ws, uint32_t id);

// the graph of id (built from the payload when it carries one)
ggml_cgraph * worker_graph_get(worker_state & ws, uint32_t id, size_t len) {
    if (len == 0) {
        auto it = ws.graphs.find(id);
        GGML_ASSERT(it != ws.graphs.end() && "graph id not held by the worker");
        g_wstat_hit++;
        return it->second.gf;
    }
    g_wstat_miss++;
    int32_t nt, nn;
    memcpy(&nt, ws.payload, 4);
    memcpy(&nn, ws.payload + 4, 4);
    worker_graph g;
    ggml_init_params ip = { (size_t) nt * ggml_tensor_overhead() + ggml_graph_overhead_custom(nn, false) + 4096, nullptr, true };
    g.ctx = ggml_init(ip);
    const std::vector<ggml_tensor *> ts = worker_build(ws, g.ctx, (const proxy_tensor *) (ws.payload + 8), nt);
    g.gf = ggml_new_graph_custom(g.ctx, nn, false);
    const int32_t * idx = (const int32_t *) (ws.payload + 8 + (size_t) nt * sizeof(proxy_tensor));
    for (int i = 0; i < nn; i++) {
        ggml_graph_add_node(g.gf, ts[idx[i]]);
    }
    auto old = ws.graphs.find(id);
    if (old != ws.graphs.end()) {
        ggml_free(old->second.ctx);
        ag_drop(ws, id);
    }
    ws.graphs[id] = g;
    return g.gf;
}

ggml_backend_buffer_t worker_buffer_of(worker_state & ws, uint64_t addr) {
    for (ggml_backend_buffer_t b : ws.buffers) {
        if (b) {
            const uint64_t base = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(b);
            if (addr >= base && addr < base + ggml_backend_buffer_get_size(b)) {
                return b;
            }
        }
    }
    return nullptr;
}

uint64_t ag_key(uint32_t id, const std::vector<ag_pending> & pend) {
    uint64_t h = id;
    for (const ag_pending & p : pend) {
        h = h * 1000003u ^ p.addr ^ ((uint64_t) p.count << 40) ^ p.slot ^ (p.b16 ? 1ull << 39 : 0);
    }
    return h;
}

void ag_drop(worker_state & ws, uint32_t id) {
    for (auto it = ws.ag_graphs.begin(); it != ws.ag_graphs.end();) {
        if (it->second.id == id) {
            ggml_free(it->second.g.ctx);
            it = ws.ag_graphs.erase(it);
        } else {
            ++it;
        }
    }
}

void ag_drop_all(worker_state & ws) {
    for (auto & e : ws.ag_graphs) {
        ggml_free(e.second.g.ctx);
    }
    ws.ag_graphs.clear();
}

// the sum of p's slices in the shape of shape_of (contiguous, p.count elements) as nodes appended to order
ggml_tensor * ag_sum(worker_state & ws, ggml_context * ctx, const ag_pending & p, const ggml_tensor * shape_of,
                     std::vector<ggml_tensor *> & order) {
    if (p.b16) {
        ggml_tensor * g = ggml_new_tensor_1d(ctx, GGML_TYPE_BF16, p.count);
        g->data = (void *) (uintptr_t) p.slot;
        g->buffer = ws.ar16_buf;
        ggml_set_name(g, "ar16_sum");
        ggml_tensor * v = g;
        if (shape_of) {
            const int64_t * ne = shape_of->ne;
            const size_t es = sizeof(ggml_bf16_t);
            v = ggml_view_4d(ctx, g, ne[0], ne[1], ne[2], ne[3], ne[0] * es, ne[0] * ne[1] * es, ne[0] * ne[1] * ne[2] * es, 0);
            order.push_back(v);
        }
        ggml_tensor * r = ggml_cast(ctx, v, GGML_TYPE_F32);
        order.push_back(r);
        return r;
    }
    ggml_tensor * g = ggml_new_tensor_1d(ctx, p.type, p.count * p.ranks);
    g->data = (void *) (uintptr_t) p.slot;
    g->buffer = ws.ag_buf;
    ggml_set_name(g, "ag_slices");
    const int64_t * ne = shape_of ? shape_of->ne : nullptr;
    const int64_t ne1[4] = { p.count, 1, 1, 1 };
    if (!ne) {
        ne = ne1;
    }
    const size_t es = ggml_type_size(p.type);
    ggml_tensor * r = nullptr;
    for (int k = 0; k < p.ranks; k++) {
        ggml_tensor * v = ggml_view_4d(ctx, g, ne[0], ne[1], ne[2], ne[3], ne[0] * es, ne[0] * ne[1] * es,
                                       ne[0] * ne[1] * ne[2] * es, (size_t) k * p.count * es);
        order.push_back(v);
        if (r) {
            r = ggml_add(ctx, r, v);
            order.push_back(r);
        } else {
            r = v;
        }
    }
    return r;
}

void ag_why(const ggml_tensor * t, const ag_pending & p, const char * why) {
    static int n = 0;
    if (getenv("GGML_SYNAPSE_AG_DEBUG") && n++ < 20) {
        fprintf(stderr, "ag fallback: %s: %s %s op %s ne [%lld %lld %lld %lld] data %p view_src %s | pending %p count %lld\n", why, t->name,
                ggml_type_name(t->type), ggml_op_desc(t), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                t->data, t->view_src ? t->view_src->name : "-", (void *) (uintptr_t) p.addr, (long long) p.count);
    }
}

// gf with the pending sums its nodes read folded in (cached by the pending set), or nullptr when gf
// touches a pending tensor other than by reading it whole or overwriting its memory; gf == nullptr:
// the sums alone, written back to their tensors
const ag_graph * ag_graph_for(worker_state & ws, uint32_t id, const ggml_cgraph * gf) {
    const uint64_t key = ag_key(id, ws.pend);
    auto range = ws.ag_graphs.equal_range(key);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.id == id && it->second.sig == ws.pend) {
            return &it->second;
        }
    }
    const size_t np = ws.pend.size();
    std::vector<std::vector<ggml_tensor *>> exact(np);   // tensors read whole in place of the sum
    std::vector<bool> dead(np, gf == nullptr);
    if (gf) {
        std::unordered_set<const ggml_tensor *> nodes(gf->nodes, gf->nodes + gf->n_nodes);
        std::unordered_set<const ggml_tensor *> seen;
        auto overlaps = [&](const ggml_tensor * t, const ag_pending & p) {
            const uint64_t lo = (uint64_t) (uintptr_t) t->data;
            return t->data && ggml_nelements(t) > 0 && lo < p.addr + p.nbytes() && p.addr < lo + ggml_nbytes(t);
        };
        auto whole = [&](const ggml_tensor * t, const ag_pending & p) {
            return (uint64_t) (uintptr_t) t->data == p.addr && t->type == p.type && ggml_nelements(t) == p.count &&
                   ggml_is_contiguous(t);
        };
        // inputs: the nodes' sources from outside the graph and the views they are of
        auto input = [&](ggml_tensor * t) {
            for (; t && !nodes.count(t); t = t->view_src) {
                if (!seen.insert(t).second) {
                    return true;
                }
                for (size_t i = 0; i < np; i++) {
                    if (overlaps(t, ws.pend[i])) {
                        if (!whole(t, ws.pend[i])) {
                            ag_why(t, ws.pend[i], "part of it read");
                            return false;
                        }
                        exact[i].push_back(t);
                    }
                }
            }
            return true;
        };
        for (int j = 0; j < gf->n_nodes; j++) {
            ggml_tensor * n = gf->nodes[j];
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                if (n->src[k] && !input(n->src[k])) {
                    return nullptr;
                }
            }
            if (n->view_src && !input(n->view_src)) {
                return nullptr;
            }
            for (size_t i = 0; i < np; i++) {
                if (!overlaps(n, ws.pend[i])) {
                    continue;
                }
                const ggml_tensor * r = n;
                while (r->view_src) {
                    r = r->view_src;
                }
                if (!nodes.count(r)) {
                    // a view node of an input: the pending tensor whole (replaced by the sum), or of
                    // another tensor in the same memory, which is fine only once a node rewrites it
                    if (whole(r, ws.pend[i]) && whole(n, ws.pend[i])) {
                        exact[i].push_back(n);
                    } else if (whole(r, ws.pend[i]) || !dead[i]) {
                        ag_why(n, ws.pend[i], "view node of it");
                        return nullptr;
                    }
                } else if (!n->view_src) {
                    dead[i] = true;   // a node writes there: the memory is reused
                }
            }
        }
    }
    ag_graph e;
    e.id = id;
    e.sig = ws.pend;
    e.dead = dead;
    size_t n_sums = 0;
    for (size_t i = 0; i < np; i++) {
        n_sums += gf ? exact[i].size() : 1;
    }
    if (n_sums == 0) {
        ggml_cgraph * none = nullptr;
        e.g.gf = none;
        auto it = ws.ag_graphs.emplace(key, std::move(e));
        return &it->second;
    }
    const size_t n_new = (gf ? gf->n_nodes : 0) + n_sums * (2 * (size_t) ws.ag_ranks + 4);
    ggml_init_params ip = { n_new * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_new, false) + 4096, nullptr, true };
    e.g.ctx = ggml_init(ip);
    std::vector<ggml_tensor *> order;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> map;
    std::vector<ggml_tensor *> sums(np, nullptr);
    // readers of a pending tensor get its sum (one per shape), made on first use
    std::unordered_map<const ggml_tensor *, size_t> reader;
    for (size_t i = 0; i < np; i++) {
        for (ggml_tensor * x : exact[i]) {
            reader[x] = i;
        }
        if (!gf) {
            sums[i] = ag_sum(ws, e.g.ctx, ws.pend[i], nullptr, order);
        }
    }
    std::vector<std::pair<size_t, ggml_tensor *>> made;
    auto mapped = [&](ggml_tensor * t) -> ggml_tensor * {
        if (!t) {
            return t;
        }
        auto it = map.find(t);
        if (it != map.end()) {
            return it->second;
        }
        auto ri = reader.find(t);
        if (ri == reader.end()) {
            return t;
        }
        ggml_tensor * r = nullptr;
        for (const auto & m : made) {
            if (m.first == ri->second && ggml_are_same_shape(m.second, t)) {
                r = m.second;
            }
        }
        if (!r) {
            r = ag_sum(ws, e.g.ctx, ws.pend[ri->second], t, order);
            made.emplace_back(ri->second, r);
        }
        map[t] = r;
        return r;
    };
    for (int i = 0; gf && i < gf->n_nodes; i++) {
        ggml_tensor * n = gf->nodes[i];
        if (reader.count(n)) {
            continue;   // a whole view of a pending tensor: replaced by the sum where used
        }
        ggml_tensor * c = ggml_new_tensor(e.g.ctx, n->type, GGML_MAX_DIMS, n->ne);
        memcpy(c->nb, n->nb, sizeof(c->nb));
        c->op = n->op;
        memcpy(c->op_params, n->op_params, sizeof(c->op_params));
        c->flags = n->flags;
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            c->src[k] = n->src[k] == n ? c : mapped(n->src[k]);   // (a cast's self-reference)
        }
        c->view_src = mapped(n->view_src);
        c->view_offs = n->view_offs;
        c->data = n->data;
        c->buffer = n->buffer;
        memcpy(c->name, n->name, sizeof(c->name));
        map[n] = c;
        order.push_back(c);
    }
    if (!gf) {
        // the sums written back
        for (size_t i = 0; i < np; i++) {
            const ag_pending & p = ws.pend[i];
            ggml_tensor * x = ggml_new_tensor_1d(e.g.ctx, p.type, p.count);
            x->data = (void *) (uintptr_t) p.addr;
            x->buffer = worker_buffer_of(ws, p.addr);
            ggml_tensor * cp = ggml_cpy(e.g.ctx, sums[i], x);
            cp->flags |= GGML_TENSOR_FLAG_OUTPUT;
            order.push_back(cp);
        }
    }
    e.g.gf = ggml_new_graph_custom(e.g.ctx, (int) order.size(), false);
    for (ggml_tensor * t : order) {
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_graph_add_node(e.g.gf, t);
    }
    auto it = ws.ag_graphs.emplace(key, std::move(e));
    return &it->second;
}

// pending sums written to their tensors (before any command other than a graph or a collective)
enum ggml_status ag_materialize(worker_state & ws) {
    if (ws.pend.empty()) {
        return GGML_STATUS_SUCCESS;
    }
    const ag_graph * e = ag_graph_for(ws, 0, nullptr);
    ws.pend.clear();
    return ggml_backend_graph_compute_async(ws.backend, e->g.gf);
}

// a slot of the gather buffer for the reduce at addr, 0: none (the plain all-reduce then)
uint64_t ag_slot(worker_state & ws, uint64_t addr, int64_t count, int ranks) {
    if (ws.ag_max <= 0 || ranks < 2 || count > ws.ag_max || !worker_buffer_of(ws, addr)) {
        return 0;
    }
    static const int n_slots = 256;
    if (!ws.ag_buf) {
        ws.ag_ranks = ranks;
        ws.ag_slot_bytes = GGML_PAD((size_t) ws.ag_max * ranks * sizeof(float), 256);
        ws.ag_buf = ggml_backend_buft_alloc_buffer(ws.buft, ws.ag_slot_bytes * n_slots);
        if (!ws.ag_buf) {
            ws.ag_max = 0;
            return 0;
        }
        ggml_backend_buffer_set_usage(ws.ag_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    if (ranks != ws.ag_ranks) {
        return 0;
    }
    auto it = ws.ag_slots.find(addr);
    if (it == ws.ag_slots.end()) {
        if ((int) ws.ag_slots.size() >= n_slots) {
            return 0;
        }
        it = ws.ag_slots.emplace(addr, (int) ws.ag_slots.size()).first;
    }
    return (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(ws.ag_buf) + (uint64_t) it->second * ws.ag_slot_bytes;
}

// a bf16 slot for an all-reduce of count f32 elements (the lowest one no pending sum holds), 0: none
// (the f32 all-reduce then)
uint64_t ar16_slot(worker_state & ws, int64_t count) {
    static const int n_slots = 8;
    if (ws.ar16_min <= 0 || count < ws.ar16_min) {
        return 0;
    }
    const size_t need = GGML_PAD((size_t) count * sizeof(ggml_bf16_t), 256);
    if (need > ws.ar16_slot_bytes) {
        for (const ag_pending & p : ws.pend) {
            if (p.b16) {
                return 0;   // still read from the old buffer
            }
        }
        if (ws.ar16_buf) {
            ws.ar16_old.push_back(ws.ar16_buf);
        }
        ag_drop_all(ws);
        for (auto & c : ws.ar16_casts) {
            ws.ar16_old_ctx.push_back(c.second.ctx);
        }
        ws.ar16_casts.clear();
        ws.ar16_slot_bytes = 0;
        ws.ar16_buf = ggml_backend_buft_alloc_buffer(ws.buft, need * n_slots);
        if (!ws.ar16_buf) {
            ws.ar16_min = 0;
            return 0;
        }
        ggml_backend_buffer_set_usage(ws.ar16_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ws.ar16_slot_bytes = need;
    }
    const uint64_t base = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(ws.ar16_buf);
    bool used[n_slots] = {};
    for (const ag_pending & p : ws.pend) {
        if (p.b16) {
            used[(p.slot - base) / ws.ar16_slot_bytes] = true;
        }
    }
    for (int i = 0; i < n_slots; i++) {
        if (!used[i]) {
            return base + (uint64_t) i * ws.ar16_slot_bytes;
        }
    }
    return 0;
}

// the graph casting the f32 tensor at addr into its bf16 slot
ggml_cgraph * ar16_cast(worker_state & ws, uint64_t addr, int64_t count, uint64_t slot) {
    const uint64_t key = addr ^ ((uint64_t) count << 40) ^ (slot * 1000003u);
    auto it = ws.ar16_casts.find(key);
    if (it != ws.ar16_casts.end()) {
        return it->second.gf;
    }
    worker_graph g;
    ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false) + 1024, nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, count);
    x->data = (void *) (uintptr_t) addr;
    x->buffer = worker_buffer_of(ws, addr);
    ggml_tensor * y = ggml_new_tensor_1d(g.ctx, GGML_TYPE_BF16, count);
    y->data = (void *) (uintptr_t) slot;
    y->buffer = ws.ar16_buf;
    ggml_tensor * cp = ggml_cpy(g.ctx, x, y);
    cp->flags |= GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_COMPUTE;
    g.gf = ggml_new_graph_custom(g.ctx, 1, false);
    ggml_graph_add_node(g.gf, cp);
    return ws.ar16_casts.emplace(key, g).first->second.gf;
}

// the graph zeroing count elements of type at addr (a reduced tensor with no value on this card)
ggml_cgraph * zero_fill(worker_state & ws, uint64_t addr, int64_t count, ggml_type type) {
    const uint64_t key = addr ^ ((uint64_t) count << 40) ^ ((uint64_t) type << 34);
    auto it = ws.zero_fills.find(key);
    if (it != ws.zero_fills.end()) {
        return it->second.gf;
    }
    worker_graph g;
    ggml_init_params ip = { 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false) + 1024, nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(g.ctx, type, count);
    x->data = (void *) (uintptr_t) addr;
    x->buffer = worker_buffer_of(ws, addr);
    ggml_tensor * z = ggml_fill(g.ctx, x, 0.0f);
    z->data = x->data;   // in place
    z->buffer = x->buffer;
    z->flags |= GGML_TENSOR_FLAG_OUTPUT | GGML_TENSOR_FLAG_COMPUTE;
    g.gf = ggml_new_graph_custom(g.ctx, 1, false);
    ggml_graph_add_node(g.gf, z);
    return ws.zero_fills.emplace(key, g).first->second.gf;
}

// pin to the CPUs near the card (the worker's only visible card, device 0)
void worker_pin_numa() {
    cpu_set_t set;
    if (ggml_synapse_card_affinity(0, &set) && sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("ggml-synapse-worker: sched_setaffinity");
    }
}

void worker_drop_graphs(worker_state & ws) {
    ag_drop_all(ws);
    for (auto & z : ws.zero_fills) {
        ggml_free(z.second.ctx);
    }
    ws.zero_fills.clear();
    for (auto & c : ws.ar16_casts) {
        ggml_free(c.second.ctx);   // (their source tensors name a buffer that may be gone)
    }
    ws.ar16_casts.clear();
    for (auto & g : ws.graphs) {
        ggml_free(g.second.ctx);
    }
    ws.graphs.clear();
}

}   // namespace

// GGML_SYNAPSE_EVT=1: device time of the graph launches and collectives between synchronizations
// (steps), from timed events recorded after each; per step position with =2 (printed at exit)
struct worker_evt {
    int level = 0;
    synDeviceId dev = 0;
    synStreamHandle stream = nullptr;
    std::vector<synEventHandle> pool;
    std::vector<std::pair<int, synEventHandle>> rec;   // kind (0 graph, 1 collective), event
    struct step_kind {
        uint64_t n = 0;
        std::vector<double> sum;   // by position
        std::vector<int> kind;
        std::vector<double> total;   // per step (us)
    };
    std::map<size_t, step_kind> steps;   // by the number of events
    uint64_t issued[2] = {}, starved[2] = {};   // issues that found the device done with all earlier work
    void mark(int kind) {
        if (!level) {
            return;
        }
        if (!rec.empty()) {
            issued[kind]++;
            starved[kind] += synEventQuery(rec.back().second) == synSuccess;
        }
        if (rec.size() >= pool.size()) {
            synEventHandle e = nullptr;
            if (synEventCreate(&e, dev, EVENT_COLLECT_TIME) != synSuccess) {
                return;
            }
            pool.push_back(e);
        }
        synEventHandle e = pool[rec.size()];
        if (synEventRecord(e, stream) == synSuccess) {
            rec.emplace_back(kind, e);
        }
    }
    // after a stream synchronize
    void collect() {
        if (rec.size() > 1) {
            step_kind & sk = steps[rec.size()];
            sk.sum.resize(rec.size());
            sk.kind.resize(rec.size());
            double t = 0;
            for (size_t i = 1; i < rec.size(); i++) {
                uint64_t ns = 0;
                if (synEventElapsedTime(&ns, rec[i - 1].second, rec[i].second) == synSuccess && ns < 10000000000ull) {
                    sk.sum[i] += ns / 1e3;
                    sk.kind[i] = rec[i].first;
                    t += ns / 1e3;
                }
            }
            sk.total.push_back(t);
            sk.n++;
        }
        rec.clear();
    }
    void print(int card) {
        const step_kind * best = nullptr;
        for (const auto & e : steps) {
            if (!best || e.second.n > best->n) {
                best = &e.second;
            }
        }
        if (!best) {
            return;
        }
        double tot[2] = {};
        for (size_t i = 1; i < best->sum.size(); i++) {
            tot[best->kind[i]] += best->sum[i] / best->n;
            if (level >= 2) {
                fprintf(stderr, "worker card %d: step pos %zu %s %.1f us\n", card, i, best->kind[i] ? "collective" : "graph", best->sum[i] / best->n);
            }
        }
        std::vector<double> tt = best->total;
        std::sort(tt.begin(), tt.end());
        fprintf(stderr, "worker card %d: %lu steps of %zu events: graphs %.2f ms, collectives %.2f ms per step (mean); step median %.2f ms; device idle at issue: graphs %.1f%%, collectives %.1f%%\n", card,
                (unsigned long) best->n, best->sum.size(), tot[0] / 1e3, tot[1] / 1e3, tt[tt.size() / 2] / 1e3,
                100.0 * starved[0] / std::max<uint64_t>(issued[0], 1), 100.0 * starved[1] / std::max<uint64_t>(issued[1], 1));
    }
};

extern "C" __attribute__((visibility("default"))) int ggml_synapse_worker_main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ggml-synapse-worker <fd> <index> <card>\n");
        return 2;
    }
    const int fd = atoi(argv[1]);
    const int index = atoi(argv[2]);
    const int card = atoi(argv[3]);
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("ggml-synapse-worker: fstat");
        return 2;
    }
    uint8_t * base = (uint8_t *) mmap(nullptr, (size_t) st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        perror("ggml-synapse-worker: mmap");
        return 2;
    }
    proxy_header * hdr = (proxy_header *) base;
    GGML_ASSERT(hdr->magic == PROXY_MAGIC && index < (int) hdr->n_workers);
    // exit with the parent
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    // a profile per card (GGML_SYNAPSE_PROF_WORKER=i: only worker i profiles, the profiler's host buffers are large)
    if (const char * prof = getenv("GGML_SYNAPSE_PROF")) {
        const char * only = getenv("GGML_SYNAPSE_PROF_WORKER");
        if (only && atoi(only) != index) {
            unsetenv("GGML_SYNAPSE_PROF");
            unsetenv("HABANA_PROFILE");
        } else {
            setenv("GGML_SYNAPSE_PROF", (std::string(prof) + ".card" + std::to_string(card)).c_str(), 1);
        }
    }

    // GGML_SYNAPSE_NO_PIN=1: no CPU affinity to the card's NUMA node
    if (!getenv("GGML_SYNAPSE_NO_PIN")) {
        worker_pin_numa();
    }

    worker_state ws;
    ws.hdr = hdr;
    ws.base = base;
    ws.mb = &hdr->mb[index];
    g_worker_log = &ws.mb->log;
    ws.big = base + hdr->area_off + (uint64_t) index * (hdr->payload_size + hdr->staging_size);
    ws.backend = ggml_backend_synapse_init(0);   // the parent made only this worker's card visible
    if (!ws.backend) {
        ws.mb->ready.store(2);
        futex(&ws.mb->ready, FUTEX_WAKE, INT_MAX);
        return 1;
    }
    ws.buft = ggml_backend_get_default_buffer_type(ws.backend);
    {
        const char * e = getenv("GGML_SYNAPSE_AG_MAX");   // elements per rank; 0: plain all-reduces
        ws.ag_max = e ? atoll(e) : 16384;
        const char * b = getenv("GGML_SYNAPSE_AR_BF16");   // min elements of a bf16 all-reduce; 0: off
        ws.ar16_min = b ? atoll(b) : 1 << 20;
    }
    ggml_backend_synapse_device_context * dctx = (ggml_backend_synapse_device_context *) ws.backend->context;
    {
        const uint64_t area = hdr->area_off;
        const size_t len = (size_t) st.st_size - area;
        if (synHostMap(dctx->sdev, len, base + area) == synSuccess) {
            dctx->pinned_add(base + area, len);
        }
    }
    ws.mb->ready.store(1);
    futex(&ws.mb->ready, FUTEX_WAKE, INT_MAX);

    worker_evt evt;
    evt.level = getenv("GGML_SYNAPSE_EVT") ? atoi(getenv("GGML_SYNAPSE_EVT")) : 0;
    evt.dev = dctx->sdev;
    evt.stream = dctx->stream;
    uint32_t seen = 0;   // the last command finished
    uint64_t w_idle = 0, w_cmd_ns[32] = {}, w_cmd_n[32] = {};
    const bool w_stats = getenv("GGML_SYNAPSE_PROXY_STATS") != nullptr;
    for (;;) {
        if (ws.mb->req.load(std::memory_order_acquire) == seen) {
            const uint64_t ti = w_stats ? now_ns() : 0;
            futex_wait_ne(&ws.mb->req, seen, 0, &ws.mb->req_sleepers);
            if (w_stats) {
                w_idle += now_ns() - ti;
            }
        }
        const uint64_t tc = w_stats ? now_ns() : 0;
        const uint32_t cur = seen + 1;
        proxy_slot * mb = &ws.mb->slot[cur % PROXY_RING];
        ws.payload = mb->big ? ws.big : mb->inl;
        const uint64_t * a = mb->a;
        bool exit_now = false;
        if (!ws.pend.empty() && mb->cmd != CMD_COMPUTE && mb->cmd != CMD_CCL_REDUCE) {
            const enum ggml_status st = ag_materialize(ws);
            if (st != GGML_STATUS_SUCCESS) {
                int32_t none = 0;
                ws.mb->err.compare_exchange_strong(none, (int32_t) st);
            }
        }
        switch (mb->cmd) {
            case CMD_HELLO: {
                size_t free = 0, total = 0;
                ggml_backend_dev_memory(ggml_backend_get_device(ws.backend), &free, &total);
                mb->r[0] = total;
                mb->r[1] = free;
                break;
            }
            case CMD_ALLOC: {
                ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ws.buft, (size_t) a[0]);
                if (b) {
                    ws.buffers.push_back(b);
                    mb->r[0] = ws.buffers.size() - 1;
                    mb->r[1] = (uint64_t) (uintptr_t) ggml_backend_buffer_get_base(b);
                } else {
                    mb->r[0] = 0;
                    mb->r[1] = 0;
                }
                break;
            }
            case CMD_FREE: {
                worker_drop_graphs(ws);
                ggml_backend_synchronize(ws.backend);
                ggml_backend_buffer_free(ws.buffers[a[0]]);
                ws.buffers[a[0]] = nullptr;
                break;
            }
            case CMD_CLEAR:
                ggml_backend_buffer_clear(ws.buffers[a[0]], (uint8_t) a[1]);
                break;
            case CMD_SET:
            case CMD_GET:
            case CMD_MEMSET: {
                ggml_context * ctx = nullptr;
                ggml_tensor * t = worker_tensor(ws, &ctx);
                if (mb->cmd == CMD_SET) {
                    ggml_backend_tensor_set(t, base + a[2], (size_t) a[0], (size_t) a[1]);
                } else if (mb->cmd == CMD_GET) {
                    uint8_t * dst = a[3] ? ws.host_bufs.at((uint32_t) a[3]).first + a[2] : base + a[2];
                    ggml_backend_tensor_get(t, dst, (size_t) a[0], (size_t) a[1]);
                    evt.collect();
                } else {
                    ggml_backend_tensor_memset(t, (uint8_t) a[2], (size_t) a[0], (size_t) a[1]);
                }
                ggml_free(ctx);
                break;
            }
            case CMD_SET2D:
            case CMD_GET2D: {
                ggml_context * ctx = nullptr;
                ggml_tensor * t = worker_tensor(ws, &ctx);
                const size_t off = (size_t) a[0], size = (size_t) a[1], n = (size_t) a[3], stride = (size_t) a[4];
                uint8_t * rows = base + a[2];
                size_t dstride = size;   // rows in the segment: packed
                if (mb->cmd == CMD_GET2D && a[5]) {
                    rows = ws.host_bufs.at((uint32_t) a[5]).first + a[2];
                    dstride = (size_t) a[6];
                }
                if (stride == size && dstride == size) {
                    // adjacent rows: one transfer
                    if (mb->cmd == CMD_SET2D) {
                        ggml_backend_tensor_set(t, rows, off, size * n);
                    } else {
                        ggml_backend_tensor_get(t, rows, off, size * n);
                    }
                } else {
                    for (size_t i = 0; i < n; i++) {
                        if (mb->cmd == CMD_SET2D) {
                            ggml_backend_tensor_set(t, rows + i * size, off + i * stride, size);
                        } else {
                            ggml_backend_tensor_get(t, rows + i * dstride, off + i * stride, size);
                        }
                    }
                }
                if (mb->cmd == CMD_GET2D) {
                    evt.collect();   // (the reads waited for the card)
                }
                ggml_free(ctx);
                break;
            }
            case CMD_ALLREDUCE: {
                ggml_context * ctx = nullptr;
                ggml_tensor * t = worker_tensor(ws, &ctx);
                const uint32_t gen = (uint32_t) a[0];
                const size_t nbytes = (size_t) a[1];
                const int rank = (int) a[2], nr = (int) a[3];
                auto slot = [&](int r) {
                    return (float *) (base + hdr->reduce_off + ((uint64_t) r * 2 + (gen & 1)) * hdr->reduce_slot);
                };
                if (a[4]) {
                    memset(slot(rank), 0, nbytes);   // no value on this card: zeros
                } else {
                    ggml_backend_tensor_get(t, slot(rank), 0, nbytes);   // synchronizes the card first
                }
                hdr->arrived[rank].store(gen, std::memory_order_release);
                futex(&hdr->arrived[rank], FUTEX_WAKE, INT_MAX);
                for (int r = 0; r < nr; r++) {
                    for (uint32_t g; (int32_t) ((g = hdr->arrived[r].load(std::memory_order_acquire)) - gen) < 0;) {
                        futex_wait_ne(&hdr->arrived[r], g, 0);
                    }
                }
                const size_t ne = nbytes / sizeof(float);
                ws.reduce.resize(ne);
                float * acc = ws.reduce.data();
                memcpy(acc, slot(0), nbytes);
                for (int r = 1; r < nr; r++) {
                    const float * x = slot(r);
                    for (size_t i = 0; i < ne; i++) {
                        acc[i] += x[i];
                    }
                }
                ggml_backend_tensor_set(t, acc, 0, nbytes);
                ggml_free(ctx);
                break;
            }
            case CMD_CCL_ID: {
                const hcclResult_t r = hcclGetUniqueId(&hdr->ccl_id);
                mb->r[0] = (uint64_t) r;
                break;
            }
            case CMD_CCL_INIT: {
                hcclComm_t comm = nullptr;
                const hcclResult_t r = hcclCommInitRank(&comm, (int) a[2], hdr->ccl_id, (int) a[1]);
                if (r == hcclSuccess) {
                    ws.comms[a[0]] = comm;
                    ws.comm_ranks[a[0]] = (int) a[2];
                } else {
                    fprintf(stderr, "ggml-synapse-worker: card %d: hcclCommInitRank failed: %d\n", card, (int) r);
                }
                mb->r[0] = (uint64_t) r;
                break;
            }
            case CMD_CCL_REDUCE: {
                ggml_synapse_trace("ccl begin");
                dctx->flush_uploads();   // queued input uploads come first on the stream
                static const bool skip = getenv("GGML_SYNAPSE_CCL_SKIP") != nullptr;
                const uint64_t addr = a[1];
                const int64_t count = (int64_t) a[2];
                const ggml_type type = a[3] ? GGML_TYPE_BF16 : GGML_TYPE_F32;
                const hcclDataType_t dt = a[3] ? hcclBfloat16 : hcclFloat32;
                hcclComm_t comm = ws.comms.at(a[0]);
                const int ranks = ws.comm_ranks[a[0]];
                mb->r[0] = 0;
                for (const ag_pending & p : ws.pend) {
                    if (addr < p.addr + p.nbytes() && p.addr < addr + (uint64_t) count * ggml_type_size(type)) {
                        mb->r[0] = (uint64_t) (int64_t) ag_materialize(ws);   // a pending one is reduced again
                        break;
                    }
                }
                if (a[4] && worker_buffer_of(ws, addr)) {
                    // no value on this card (meta backend: compute disabled): zeros first, on the same stream
                    const enum ggml_status st = ggml_backend_graph_compute_async(ws.backend, zero_fill(ws, addr, count, type));
                    if (st != GGML_STATUS_SUCCESS && mb->r[0] == 0) {
                        mb->r[0] = (uint64_t) (int64_t) st;
                    }
                }
                const uint64_t slot = skip ? 0 : ag_slot(ws, addr, count, ranks);
                const uint64_t s16 = !slot && !skip && type == GGML_TYPE_F32 && worker_buffer_of(ws, addr) ? ar16_slot(ws, count) : 0;
                hcclResult_t r = hcclSuccess;
                if (s16) {
                    const enum ggml_status st = ggml_backend_graph_compute_async(ws.backend, ar16_cast(ws, addr, count, s16));
                    if (st != GGML_STATUS_SUCCESS && mb->r[0] == 0) {
                        mb->r[0] = (uint64_t) (int64_t) st;
                    }
                    r = hcclAllReduce((void *) (uintptr_t) s16, (void *) (uintptr_t) s16, (size_t) count, hcclBfloat16, hcclSum, comm, dctx->stream);
                    ag_pending p = { addr, count, type, 1, s16 };
                    p.b16 = true;
                    ws.pend.push_back(p);
                } else if (slot) {
                    r = hcclAllGather((void *) (uintptr_t) addr, (void *) (uintptr_t) slot, (size_t) count, dt, comm, dctx->stream);
                    g_wstat_ag++;
                    ws.pend.push_back({ addr, count, type, ranks, slot });
                } else if (!skip) {
                    r = hcclAllReduce((void *) (uintptr_t) addr, (void *) (uintptr_t) addr, (size_t) count, dt, hcclSum, comm, dctx->stream);
                }
                dctx->set_pending();
                evt.mark(1);
                if (mb->r[0] == 0) {
                    mb->r[0] = (uint64_t) r;
                }
                ggml_synapse_trace("ccl end");
                break;
            }
            case CMD_SET_RAW: {
                ggml_backend_buffer_t b = ws.buffers.at(a[0]);
                ggml_backend_buffer_set_usage(b, (ggml_backend_buffer_usage) a[4]);
                ggml_tensor t;
                memset(&t, 0, sizeof(t));
                t.type = GGML_TYPE_I8;
                t.ne[0] = (int64_t) a[2];
                t.ne[1] = t.ne[2] = t.ne[3] = 1;
                t.nb[0] = 1;
                t.nb[1] = t.nb[2] = t.nb[3] = (size_t) a[2];
                t.data = (void *) (uintptr_t) a[1];
                t.buffer = b;
                ggml_backend_tensor_set(&t, base + a[3], 0, (size_t) a[2]);
                break;
            }
            case CMD_HOST_MAP: {
                char path[64];
                snprintf(path, sizeof(path), "/proc/%d/fd/%d", (int) a[3], (int) a[1]);
                const int hfd = open(path, O_RDWR);
                void * p = hfd >= 0 ? mmap(nullptr, (size_t) a[2], PROT_READ | PROT_WRITE, MAP_SHARED, hfd, 0) : MAP_FAILED;
                if (hfd >= 0) {
                    close(hfd);
                }
                mb->r[0] = 1;
                if (p != MAP_FAILED) {
                    if (synHostMap(dctx->sdev, (size_t) a[2], p) == synSuccess) {
                        dctx->pinned_add(p, (size_t) a[2]);
                        ws.host_bufs[(uint32_t) a[0]] = { (uint8_t *) p, (size_t) a[2] };
                        mb->r[0] = 0;
                    } else {
                        munmap(p, (size_t) a[2]);
                    }
                }
                if (mb->r[0] != 0) {
                    fprintf(stderr, "ggml-synapse-worker: card %d: cannot map host buffer %s (%s)\n", card, path, strerror(errno));
                }
                break;
            }
            case CMD_HOST_UNMAP: {
                auto it = ws.host_bufs.find((uint32_t) a[0]);
                if (it != ws.host_bufs.end()) {
                    ggml_backend_synchronize(ws.backend);
                    dctx->pinned_drop(it->second.first);
                    synHostUnmap(dctx->sdev, it->second.first);
                    munmap(it->second.first, it->second.second);
                    ws.host_bufs.erase(it);
                }
                break;
            }
            case CMD_CCL_FREE: {
                auto it = ws.comms.find(a[0]);
                if (it != ws.comms.end()) {
                    ggml_backend_synchronize(ws.backend);
                    hcclCommDestroy(it->second);
                    ws.comms.erase(it);
                }
                break;
            }
            case CMD_COMPUTE: {
                const uint64_t t0 = now_ns();
                ggml_synapse_trace("cmd compute");
                if (a[1]) {
                    auto it = ws.graphs.find((uint32_t) a[1]);
                    if (it != ws.graphs.end()) {
                        ggml_free(it->second.ctx);
                        ws.graphs.erase(it);
                        ag_drop(ws, (uint32_t) a[1]);
                    }
                }
                ggml_cgraph * gf = worker_graph_get(ws, (uint32_t) a[0], (size_t) mb->payload_len);
                mb->r[0] = 0;
                if (!ws.pend.empty()) {
                    // the pending sums this graph reads folded in, else materialized first
                    if (const ag_graph * e = ag_graph_for(ws, (uint32_t) a[0], gf)) {
                        if (e->g.gf) {
                            gf = e->g.gf;
                        }
                        size_t k = 0;
                        for (size_t i = 0; i < ws.pend.size(); i++) {
                            if (!e->dead[i]) {
                                ws.pend[k++] = ws.pend[i];
                            }
                        }
                        ws.pend.resize(k);
                    } else {
                        g_wstat_ag_fallback++;
                        mb->r[0] = (uint64_t) (int64_t) ag_materialize(ws);
                    }
                }
                g_wstat_get_ns += now_ns() - t0;
                if (evt.level && evt.rec.empty()) {
                    dctx->flush_uploads();
                    evt.mark(0);   // the start
                }
                const enum ggml_status st = ggml_backend_graph_compute_async(ws.backend, gf);
                if (mb->r[0] == 0) {
                    mb->r[0] = (uint64_t) (int64_t) st;
                }
                evt.mark(0);
                g_wstat_ns += now_ns() - t0;
                ggml_synapse_trace("compute end");
                break;
            }
            case CMD_SYNC:
                ggml_backend_synchronize(ws.backend);
                evt.collect();
                break;
            case CMD_EXIT:
                exit_now = true;
                break;
            default:
                GGML_ABORT("ggml-synapse-worker: unknown command %u", mb->cmd);
        }
        // a failure of a command the parent does not wait for is kept for its next wait
        if ((mb->cmd == CMD_COMPUTE || mb->cmd == CMD_CCL_REDUCE) && mb->r[0] != 0) {
            int32_t none = 0;
            ws.mb->err.compare_exchange_strong(none, (int32_t) mb->r[0]);
        }
        if (w_stats && mb->cmd < 32) {
            w_cmd_ns[mb->cmd] += now_ns() - tc;
            w_cmd_n[mb->cmd]++;
        }
        seen = cur;
        futex_store(&ws.mb->done, seen, &ws.mb->done_sleepers);
        if (exit_now) {
            break;
        }
    }
    evt.print(card);
    if (w_stats) {
        const synapse_load_stat & ls = g_load_stat;
        fprintf(stderr, "worker card %d: load: planar %.1f GB convert %.1f s dma wait %.1f s; whole %.1f GB convert %.1f s upload %.1f s\n",
                card, ls.bytes / 1e9, ls.conv_ns / 1e9, ls.wait_ns / 1e9, ls.whole_bytes / 1e9, ls.whole_conv_ns / 1e9,
                ls.whole_h2d_ns / 1e9);
        fprintf(stderr, "worker card %d: idle %.1f ms;", card, w_idle / 1e6);
        for (int c = 1; c < 32; c++) {
            if (w_cmd_n[c]) {
                fprintf(stderr, " cmd%d n %lu avg %.1f us;", c, (unsigned long) w_cmd_n[c], w_cmd_ns[c] / 1e3 / w_cmd_n[c]);
            }
        }
        fprintf(stderr, "\n");
    }
    if (getenv("GGML_SYNAPSE_PROXY_STATS")) {
        fprintf(stderr, "worker card %d: graphs %lu cached / %lu built, compute %.1f ms (graph lookup %.1f ms); all-gathers %lu, materialized for a graph %lu\n", card,
                (unsigned long) g_wstat_hit, (unsigned long) g_wstat_miss, g_wstat_ns / 1e6, g_wstat_get_ns / 1e6,
                (unsigned long) g_wstat_ag, (unsigned long) g_wstat_ag_fallback);
    }
    worker_drop_graphs(ws);
    ggml_backend_synchronize(ws.backend);
    if (ws.ag_buf) {
        ggml_backend_buffer_free(ws.ag_buf);
    }
    for (auto & c : ws.ar16_casts) {
        ws.ar16_old_ctx.push_back(c.second.ctx);
    }
    for (ggml_context * c : ws.ar16_old_ctx) {
        ggml_free(c);
    }
    if (ws.ar16_buf) {
        ws.ar16_old.push_back(ws.ar16_buf);
    }
    for (ggml_backend_buffer_t b : ws.ar16_old) {
        ggml_backend_buffer_free(b);
    }
    for (ggml_backend_buffer_t b : ws.buffers) {
        if (b) {
            ggml_backend_buffer_free(b);
        }
    }
    ggml_backend_free(ws.backend);
    return 0;
}

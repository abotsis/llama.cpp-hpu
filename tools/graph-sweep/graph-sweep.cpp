// llama-graph-sweep: fill the recipe cache of a backend that compiles one graph per shape (SYNAPSE), so a later
// llama-server / llama-cli run with the same settings loads every graph instead of stalling on a compile.
//
// It runs dummy batches through the model: at every KV length llama.cpp pads to (LLAMA_KV_PAD_GEOM) up to --max-ctx,
// one batch of every ubatch size the batch splitter produces (LLAMA_UBATCH_PAD: 1, every padded size and n_ubatch - 1,
// the padded full ubatch; LLAMA_UBATCH_POW2: n_ubatch, 256 .. 8, 1 .. 7; else n_ubatch, 4 and 1: prompt remainders of
// other sizes are not covered), with and without an output. A graph's key holds the context size, the number of sequences, the ubatch size and the KV
// types, so pass the arguments the server will get (-m, -c, -np, -ub, -b, -fa, -ctk, -ctv, -sm, -dev, ...): it
// takes the server's arguments and defaults (-np auto = 4 sequences with a unified KV cache).
//
// Own options (the rest go to the server's argument parser):
//   --max-ctx N      sweep KV lengths up to N (default: the context size)
//   --cache-dir DIR  recipe cache directory (sets GGML_SYNAPSE_CACHE; default: GGML_SYNAPSE_CACHE, else
//                    ~/.cache/llama.cpp/synapse)
//   --sizes LIST     ubatch sizes to sweep, comma separated (default: all the splitter produces)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "speculative.h"
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: llama_n_ubatch_pad

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// compile log lines of the backend (ggml-synapse-graph.cpp, any card): counted, not printed
static std::atomic<int> g_compiled{0};
static std::atomic<int> g_loaded{0};
static std::atomic<double> g_compile_ms{0};
static bool g_verbose = false;   // -lv 4: print each compile log line too

static void sweep_log(ggml_log_level level, const char * text, void * /*user*/) {
    const char * c = strstr(text, "synapse: compiled graph");
    const char * l = strstr(text, "synapse: loaded graph");
    if ((c || l) && g_verbose) {
        fputs(text, stderr);
    }
    if (c) {
        g_compiled++;
        const char * ms = strstr(c, " ms (");
        if (ms) {
            const char * p = ms;
            while (p > c && p[-1] >= '0' && p[-1] <= '9') {
                p--;
            }
            g_compile_ms = g_compile_ms + atof(p);
        }
        return;
    }
    if (l) {
        g_loaded++;
        return;
    }
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

// the KV length llama.cpp pads `used` cells to (llama_kv_cache::get_n_kv: 256-cell steps, LLAMA_KV_PAD_GEOM:
// max(256, 1/4 of the largest power of two <= used))
static uint32_t padded_kv(uint32_t used, uint32_t size, bool geom) {
    uint32_t pad = 256;
    if (geom) {
        uint32_t p2 = 1;
        while (used > 0 && p2 <= used / 2) {
            p2 *= 2;
        }
        pad = std::max(pad, p2 / 4);
    }
    return std::min(size, std::max(pad, (used + pad - 1) / pad * pad));
}

static bool env_on(const char * name) {
    const char * v = getenv(name);
    return v && atoi(v) != 0;
}

static std::string hms(double s) {
    char buf[32];
    const int t = (int) s;
    snprintf(buf, sizeof(buf), "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60);
    return buf;
}

int main(int argc, char ** argv) {
    uint32_t max_ctx = 0;
    std::string cache_dir;
    std::vector<uint32_t> sizes_opt;
    std::vector<char *> args = { argv[0] };
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if ((a == "--max-ctx" || a == "--cache-dir" || a == "--sizes") && i + 1 < argc) {
            const std::string v = argv[++i];
            if (a == "--max-ctx") {
                max_ctx = (uint32_t) std::stoul(v);
            } else if (a == "--cache-dir") {
                cache_dir = v;
            } else {
                for (size_t p = 0; p < v.size();) {
                    const size_t q = v.find(',', p);
                    sizes_opt.push_back((uint32_t) std::stoul(v.substr(p, q == std::string::npos ? std::string::npos : q - p)));
                    p = q == std::string::npos ? v.size() : q + 1;
                }
            }
            continue;
        }
        if (a == "-h" || a == "--help") {
            printf("llama-graph-sweep [--max-ctx N] [--cache-dir DIR] [--sizes LIST] <llama-server arguments>\n\n");
        }
        args.push_back(argv[i]);
    }
    if (!cache_dir.empty()) {
        // before the backend registers: card workers inherit it
        setenv("GGML_SYNAPSE_CACHE", cache_dir.c_str(), 1);
    }

    common_params params;
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    // as llama-server
    if (params.n_parallel < 0) {
        params.n_parallel = 4;
        params.kv_unified = true;
    }
    if (params.kv_unified_per_slot > 0 && params.n_ctx == 0 && (uint32_t) params.fit_params_min_ctx != UINT32_MAX) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
    }
    // the server's output limits (server_output_limits, not the mmproj audio case): they size llama's compute buffer,
    // which places the graph inputs, and their sections are part of each recipe's key
    if (params.embedding || (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        params.n_outputs_max         = params.n_batch;
        params.n_outputs_max_per_seq = 1;
    } else {
        const auto lim = common_speculative_get_output_limits(params.n_batch, params.n_parallel,
                                                              common_speculative_n_max(&params.speculative));
        params.n_outputs_max         = std::max<int32_t>(1, lim.total);
        params.n_outputs_max_per_seq = std::max<int32_t>(1, lim.per_seq);
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    const llama_model * model = init->model();
    if (ctx == nullptr || model == nullptr) {
        fprintf(stderr, "%s: failed to load the model\n", __func__);
        return 1;
    }
    g_verbose = params.verbosity >= 4;
    llama_log_set(sweep_log, nullptr);

    const char * cd = getenv("GGML_SYNAPSE_CACHE");
    if (cd && strcmp(cd, "0") == 0) {
        fprintf(stderr, "%s: the recipe cache is off (GGML_SYNAPSE_CACHE=0): nothing would be kept\n", __func__);
        return 1;
    }
    const std::string cache_where = cd && *cd ? cd : std::string(getenv("HOME") ? getenv("HOME") : "~") + "/.cache/llama.cpp/synapse";

    const uint32_t n_ctx    = llama_n_ctx(ctx);
    const uint32_t n_batch  = llama_n_batch(ctx);
    const uint32_t n_ubatch = llama_n_ubatch(ctx);
    max_ctx = max_ctx == 0 ? n_ctx : std::min(max_ctx, n_ctx);

    // the backend sets these at registration (ggml-synapse.cpp), i.e. by now
    const bool geom = env_on("LLAMA_KV_PAD_GEOM");
    const bool pow2 = env_on("LLAMA_UBATCH_POW2");

    // LLAMA_UBATCH_PAD: the prompt ubatches of one sequence are padded to a few sizes
    bool tok_pad = false;
    for (uint32_t t = 2; t <= n_ubatch; t++) {
        tok_pad = tok_pad || llama_n_ubatch_pad(ctx, t) != t;
    }

    std::vector<uint32_t> sizes = sizes_opt;
    if (sizes.empty()) {
        std::set<uint32_t> s = { n_ubatch };
        if (tok_pad) {
            // decode and the padded sizes. A padded full ubatch has its own graph (the pad masks of recurrent
            // models; a full one without pads keeps the plain graph): n_ubatch - 1 tokens
            s.insert(1);
            for (uint32_t t = 2; t < n_ubatch; t++) {
                s.insert(llama_n_ubatch_pad(ctx, t));
            }
            if (n_ubatch > 2 && llama_n_ubatch_pad(ctx, n_ubatch - 1) == n_ubatch) {
                s.insert(n_ubatch - 1);
            }
        } else if (pow2) {
            for (uint32_t p = 8; p < n_ubatch; p *= 2) {
                s.insert(p);
            }
            for (uint32_t p = 1; p < 8 && p < n_ubatch; p++) {
                s.insert(p);
            }
        } else {
            // decode, and llama-server's last prompt chunk before a context checkpoint (SWA / recurrent models)
            s.insert(1);
            s.insert(std::min(4u, n_ubatch));
        }
        sizes.assign(s.rbegin(), s.rend());
    }
    for (uint32_t s : sizes) {
        if (s == 0 || s > n_ubatch) {
            fprintf(stderr, "%s: ubatch size %u not in 1 .. n_ubatch (%u)\n", __func__, s, n_ubatch);
            return 1;
        }
    }

    std::vector<uint32_t> kvs;   // the KV lengths, ascending
    for (uint32_t u = 1; u <= max_ctx; u++) {
        const uint32_t k = padded_kv(u, n_ctx, geom);
        if (kvs.empty() || k != kvs.back()) {
            kvs.push_back(k);
        }
    }

    struct target { uint32_t size; bool out; };
    int n_total = 0;
    for (uint32_t k : kvs) {
        for (uint32_t s : sizes) {
            n_total += s <= k ? 2 : 0;
        }
    }

    fprintf(stderr, "\n%s: n_ctx %u, %d sequences%s, n_batch %u, n_ubatch %u; KV lengths %zu (%u .. %u%s), ubatch sizes %zu%s: %d graphs\n",
            __func__, n_ctx, params.n_parallel, params.kv_unified ? " (unified KV)" : "", n_batch, n_ubatch, kvs.size(), kvs.front(),
            kvs.back(), geom ? ", geometric" : "", sizes.size(), tok_pad ? " (padded)" : "", n_total);
    fprintf(stderr, "%s: recipe cache %s\n\n", __func__, cache_where.c_str());

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token tok = llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_bos(vocab) : 0;
    common_batch batch(ctx);

    uint32_t pos = 0;   // cells used by sequence 0
    // n tokens at pos, logits for the last one when out
    auto run = [&](uint32_t n, bool out) {
        batch.clear();
        for (uint32_t i = 0; i < n; i++) {
            batch.add(tok, (llama_pos) (pos + i), 0, out && i + 1 == n);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "\n%s: decode failed (%u tokens at %u)\n", __func__, n, pos);
            exit(1);
        }
        pos += n;
    };
    // up to pos == to, without outputs, in whole ubatches and then the largest swept sizes that fit (a remainder of
    // another size would compile a graph nobody uses)
    const uint32_t full = std::max(n_ubatch, n_batch / n_ubatch * n_ubatch);
    auto fill = [&](uint32_t to) {
        if (pos > to) {
            llama_memory_clear(mem, true);
            pos = 0;
        }
        while (pos < to) {
            const uint32_t d = to - pos;
            uint32_t n = d >= n_ubatch ? std::min(full, d / n_ubatch * n_ubatch) : 0;
            for (uint32_t s : sizes) {   // descending
                if (n == 0 && s <= d) {
                    n = s;
                }
            }
            run(n == 0 ? d : n, false);
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    int n_done = 0;
    auto progress = [&](uint32_t k, uint32_t s, bool out) {
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const double eta = n_done > 0 ? el / n_done * (n_total - n_done) : 0.0;
        fprintf(stderr, "\r[%5d/%d] kv %7u, %3u tokens%s | compiled %d (%.0f s), loaded %d | %s elapsed, ~%s left   ",
                n_done, n_total, k, s, out ? ", out" : "     ", g_compiled.load(), g_compile_ms.load() / 1000.0, g_loaded.load(),
                hms(el).c_str(), hms(eta).c_str());
        fflush(stderr);
    };

    uint32_t k_prev = 0;
    for (uint32_t k : kvs) {
        // each ubatch of s tokens has to end in (k_prev, k] to get this KV length. Pieces up to the window width go in
        // one after another from k_prev; wider ones (s > k - k_prev) start at k - s. Clear + refill when behind
        std::vector<target> pend;
        for (uint32_t s : sizes) {
            if (s <= k) {
                pend.push_back({ s, false });
                pend.push_back({ s, true });
            }
        }
        const uint32_t w = k - k_prev;
        for (auto it = pend.begin(); it != pend.end();) {
            if (it->size > w) {
                fill(k - it->size);
                progress(k, it->size, it->out);
                run(it->size, it->out);
                n_done++;
                it = pend.erase(it);
            } else {
                ++it;
            }
        }
        while (!pend.empty()) {
            fill(k_prev);
            for (auto it = pend.begin(); it != pend.end();) {
                if (pos + it->size <= k) {
                    progress(k, it->size, it->out);
                    run(it->size, it->out);
                    n_done++;
                    it = pend.erase(it);
                } else {
                    ++it;
                }
            }
        }
        k_prev = k;
    }
    progress(kvs.back(), 0, false);

    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "\n\n%s: done in %s: %d graphs compiled (%.0f s of compiling), %d loaded from the cache; recipes in %s\n",
            __func__, hms(el).c_str(), g_compiled.load(), g_compile_ms.load() / 1000.0, g_loaded.load(), cache_where.c_str());

    return 0;
}

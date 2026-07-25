#include "llama-model-loader.h"

#include "ggml-alloc.h"
#include "ggml.h"
#include "ggml-cpu.h"   // ggml_temporal_pool_register (expert slot-pool)
#include "gguf.h"
#include "llama-hparams.h"

#if defined(__linux__)
#include <fcntl.h>   // open() the repacked expert side-file for the temporal pool
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <future>
#include <regex>

#if defined(_POSIX_MAPPED_FILES) || defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>   // posix_madvise, for the LLAMA_TEMPORAL_MMAP=2 per-tensor policy
#include <fcntl.h>      // posix_fadvise: the second half of a real eviction
#include <unistd.h>     // dup: the controller outlives the loader's fd
#include <cstring>      // strerror
#include <atomic>
#include <random>
#include <thread>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Forced expert eviction -- the Android analogue of the CUDA TEMPORAL_SWAP_PROB path.
//
// Why this exists. On a random-weight model the router has almost no diversity, so the
// same experts are selected every token and the natural page-fault rate is
// unrepresentative (the CUDA side hit this too, which is why it drives swaps at a
// prescribed rate via TEMPORAL_SWAP_PROB instead of trusting the router). Measuring
// "temporal residency" against that workload measures a fixed slice staying cached.
//
// So we force the turnover instead of hoping for it: a background thread evicts randomly
// chosen expert slices at a prescribed rate with madvise(MADV_DONTNEED), which drops the
// clean file-backed pages and makes the next use of that expert a real fault from UFS.
// That reproduces the mechanism under test -- stream ~1 expert/layer/token -- without
// depending on routing behaviour the model cannot produce.
//
// NOTE: this must be madvise(MADV_DONTNEED), not posix_madvise(POSIX_MADV_DONTNEED),
// which is a documented no-op on Linux and would have silently evicted nothing.
//
//   LLAMA_TEMPORAL_EVICT_HZ=<n>  expert slices to evict per second (0/unset = off)
namespace {
struct expert_region { uint8_t * addr; size_t expert_bytes; int n_experts; size_t file_off; int fd; };
std::vector<expert_region> g_expert_regions;
// Attention / norm / embedding weights. These are touched on EVERY token, so they must be
// resident in ALL three regimes -- only expert residency is supposed to vary. Without
// re-asserting them the kernel reclaims them under memory pressure and the "streamed
// experts" regime silently becomes "stream everything", making decode slow for the wrong
// reason. Observed: R=0 gave 9.52% file residency, BELOW the 18.5% the non-expert weights
// alone occupy, i.e. attention was being evicted and the measurement was invalid.
struct hot_region { uint8_t * addr; size_t bytes; };
std::vector<hot_region>    g_hot_regions;
std::atomic<long>          g_hot_resident_pages{0};
std::atomic<long>          g_hot_total_pages{0};
std::atomic<bool>          g_evictor_started{false};
std::atomic<long>          g_evictions{0};
std::atomic<long>          g_evicted_bytes{0};
}

void llama_temporal_register_hot(uint8_t * addr, size_t bytes) {
    g_hot_regions.push_back({addr, bytes});
}

extern "C" void llama_temporal_hot_residency(long * resident, long * total) {
    if (resident) *resident = g_hot_resident_pages.load();
    if (total)    *total    = g_hot_total_pages.load();
}

extern "C" void llama_temporal_evict_stats(long * n, long * bytes) {
    if (n)     *n     = g_evictions.load();
    if (bytes) *bytes = g_evicted_bytes.load();
}

void llama_temporal_register_experts(uint8_t * addr, size_t total_bytes, int n_experts,
                                     size_t file_off, int fd) {
    if (n_experts <= 1) return;
    // dup() the fd. The one passed in belongs to the loader's llama_file, which is
    // destroyed (fd closed) as soon as load finishes -- but the controller thread runs
    // for the process lifetime. fadvise on the dead fd fails EBADF, and since madvise
    // alone drops nothing from the page cache, eviction silently degrades to the exact
    // no-op of §4. This is why the R=0 "streamed" regime decoded at 12.7 tok/s against
    // a 3.75 tok/s storage roofline: nothing was being evicted.
    int own = dup(fd);
    if (own < 0) return;
    g_expert_regions.push_back({addr, total_bytes / (size_t) n_experts, n_experts, file_off, own});
}

// Real eviction requires BOTH calls, in this order. Measured on-device:
//   madvise(MADV_DONTNEED) alone  -> next touch reads 0 B from disk (page stayed in cache)
//   fadvise(DONTNEED) alone       -> refused while the range is mapped, 0 B
//   madvise THEN fadvise          -> next touch reads exactly the slice size from disk
// Using either one alone silently evicts nothing, which is what the first version of this
// controller did.
static void evict_range(uint8_t * addr, size_t len, int fd, size_t file_off) {
    uint8_t * al  = (uint8_t *) ((uintptr_t) addr & ~(uintptr_t) 4095);
    size_t    fo  = file_off & ~(size_t) 4095;
    // fadvise(DONTNEED) only drops pages FULLY covered by [offset, offset+len): the
    // kernel rounds the start up and the end down. fo is aligned down, so the start is
    // exact; the end must be rounded UP or the tail page silently survives.
    size_t    flen = (((file_off + len + 4095) & ~(size_t) 4095)) - fo;
    unsigned char probe[1];
    if (mincore(al, 4096, probe) != 0) return;      // range no longer mapped
    madvise(al, (size_t)(addr + len - al), MADV_DONTNEED);   // 1. drop our PTEs (kernel rounds len up)
    int rc = posix_fadvise(fd, (off_t) fo, (off_t) flen, POSIX_FADV_DONTNEED);   // 2. drop page cache
    if (rc != 0) {
        // A failing fadvise means eviction has silently degraded to the §4 madvise-only
        // no-op and every regime number is void. Loud, once.
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LLAMA_LOG_WARN("temporal: posix_fadvise(DONTNEED) FAILED: %s -- eviction is NOT happening\n",
                           strerror(rc));
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic expert residency control -- the three regimes.
//
// Rate-based random eviction gives statistical turnover, not control: it cannot express
// "exactly R of E experts are resident". These are the regimes we actually need:
//
//   LLAMA_TEMPORAL_R=0        streamed   -- no experts resident; every use faults from UFS
//   LLAMA_TEMPORAL_R=<E>      resident   -- all experts held; the ceiling
//   LLAMA_TEMPORAL_R=<k>      temporal   -- a rolling window of k experts resident
//   LLAMA_TEMPORAL_ROLL_HZ=<n>           -- window advances n times/s (0 = static window)
//
// Enforcement, and its honest limit. `ulimit -l` on this device is 64 KiB, so mlock()
// cannot pin gigabytes and residency CANNOT be hard-guaranteed. Instead a controller
// thread continuously re-asserts the target: MADV_DONTNEED on everything outside the
// window (a hard drop of clean file pages) and MADV_WILLNEED on everything inside it.
// The resident set is a CONTIGUOUS window, so enforcing a region costs at most three
// madvise calls rather than one per expert.
//
// Because enforcement is best-effort, achieved residency is MEASURED with mincore() and
// reported -- never assumed to equal the requested R.
namespace {
std::atomic<long> g_resident_pages{0};
std::atomic<long> g_expert_pages{0};
}

extern "C" void llama_temporal_residency(long * resident, long * total) {
    if (resident) *resident = g_resident_pages.load();
    if (total)    *total    = g_expert_pages.load();
}

static void llama_temporal_start_residency_controller() {
    const char * rs = getenv("LLAMA_TEMPORAL_R");
    if (!rs || g_expert_regions.empty() || g_evictor_started.exchange(true)) return;
    const int    R    = atoi(rs);
    const char * hs   = getenv("LLAMA_TEMPORAL_ROLL_HZ");
    const double roll = hs ? atof(hs) : 0.0;

    const int E = g_expert_regions[0].n_experts;
    LLAMA_LOG_INFO("%s: residency control R=%d of E=%d, roll=%.1f Hz, %zu regions\n",
                   __func__, R, E, roll, g_expert_regions.size());

    const std::vector<expert_region> regions = g_expert_regions;
    const std::vector<hot_region>    hot     = g_hot_regions;
    LLAMA_LOG_INFO("%s: holding %zu hot (attn/norm/embed) regions resident in all regimes\n",
                   __func__, hot.size());
    std::thread([regions, hot, R, roll, E]() {
        const auto interval = std::chrono::duration<double>(roll > 0 ? 1.0 / roll : 0.05);
        size_t w = 0;
        for (;;) {
            // Hot weights are re-asserted every pass in EVERY regime, so the only thing
            // that varies between streamed / temporal / resident is expert residency.
            long hot_res = 0, hot_tot = 0;
            for (const auto & h : hot) {
                uint8_t * al = (uint8_t *)((uintptr_t) h.addr & ~(uintptr_t) 4095);
                unsigned char probe[1];
                if (mincore(al, 4096, probe) != 0) continue;
                madvise(al, h.bytes, MADV_WILLNEED);
                std::vector<unsigned char> v((h.bytes + 4095) / 4096);
                if (mincore(al, h.bytes, v.data()) == 0) {
                    for (unsigned char c : v) hot_res += (c & 1);
                    hot_tot += (long) v.size();
                }
            }
            g_hot_resident_pages = hot_res;
            g_hot_total_pages    = hot_tot;

            long res = 0, tot = 0;
            for (const auto & r : regions) {
                const size_t lo = w, hi = std::min<size_t>(w + R, r.n_experts);
                auto slice = [&](size_t a, size_t b, int advice) {
                    if (b <= a) return;
                    uint8_t * p   = r.addr + a * r.expert_bytes;
                    size_t    len = (b - a) * r.expert_bytes;
                    if (advice == MADV_DONTNEED) {
                        evict_range(p, len, r.fd, r.file_off + a * r.expert_bytes);
                    } else {
                        uint8_t * al = (uint8_t *)((uintptr_t) p & ~(uintptr_t) 4095);
                        unsigned char probe[1];
                        if (mincore(al, 4096, probe) != 0) return;
                        madvise(al, len, advice);
                    }
                };
                slice(0,  lo,           MADV_DONTNEED);          // outside window: drop
                slice(hi, r.n_experts,  MADV_DONTNEED);
                slice(lo, hi,           MADV_WILLNEED);          // inside window: fetch
                if (R > 0) { g_evictions++; }
                tot += (long)(r.expert_bytes * r.n_experts / 4096);
            }
            // measure what we ACTUALLY achieved, rather than trusting the advice
            for (const auto & r : regions) {
                std::vector<unsigned char> vec((r.expert_bytes * r.n_experts + 4095) / 4096);
                uint8_t * al = (uint8_t *)((uintptr_t) r.addr & ~(uintptr_t) 4095);
                if (mincore(al, r.expert_bytes * r.n_experts, vec.data()) == 0) {
                    for (unsigned char c : vec) res += (c & 1);
                }
            }
            g_resident_pages = res;
            g_expert_pages   = tot;
            if (roll > 0) { w = (w + 1) % (size_t) std::max(1, E); }
            std::this_thread::sleep_for(interval);
        }
    }).detach();
}

void llama_temporal_start_evictor() {
    llama_temporal_start_residency_controller();
    const char * s = getenv("LLAMA_TEMPORAL_EVICT_HZ");
    const double hz = s ? atof(s) : 0.0;
    if (hz <= 0.0 || g_expert_regions.empty() || g_evictor_started.exchange(true)) return;

    LLAMA_LOG_INFO("%s: forced expert eviction at %.0f slices/s over %zu regions\n",
                   __func__, hz, g_expert_regions.size());

    // The thread gets its OWN COPY of the region list. llama-bench loads the model more
    // than once per invocation, so the global vector keeps growing and reallocating; a
    // detached thread holding a reference into it is a use-after-free, which is what
    // segfaulted at high eviction rates (and only rarely at 1 Hz, because the window is
    // narrow). The snapshot is immutable for the thread's lifetime.
    const std::vector<expert_region> regions = g_expert_regions;

    std::thread([hz, regions]() {
        std::mt19937 rng(1234);
        const auto period = std::chrono::duration<double>(1.0 / hz);
        for (;;) {
            const auto & r = regions[rng() % regions.size()];
            const size_t e = rng() % (size_t) r.n_experts;
            uint8_t * p = r.addr + e * r.expert_bytes;
            // align down to a page so madvise accepts it
            uint8_t * aligned = (uint8_t *) ((uintptr_t) p & ~(uintptr_t) 4095);

            // SAFETY: llama_mmap::unmap_fragment() releases the unused head/tail of the
            // mapping after load. If such a range is later reused by an ANONYMOUS mapping,
            // MADV_DONTNEED there does not drop file pages -- it ZERO-FILLS live memory,
            // which segfaulted at high eviction rates. mincore() fails with ENOMEM on an
            // unmapped range, so it is a cheap check that the slice is still mapped before
            // we touch it. Anything that fails the check is dropped permanently.
            unsigned char probe[1];
            if (mincore(aligned, 4096, probe) != 0) {
                continue;
            }
            // Two-step eviction (madvise+fadvise) -- madvise alone is the §4 silent no-op:
            // it zaps PTEs but leaves the page in cache, so the next touch reads 0 B.
            evict_range(p, r.expert_bytes, r.fd, r.file_off + e * r.expert_bytes);
            g_evictions++;
            g_evicted_bytes += (long) r.expert_bytes;
            std::this_thread::sleep_for(period);
        }
    }).detach();
}
#endif

static const size_t kiB = 1024;
static const size_t MiB = 1024*kiB;
static const size_t GiB = 1024*MiB;

const char * llama_file_version_name(llama_fver version) {
    switch (version) {
        case GGUF_FILE_VERSION_V1: return "GGUF V1 (support until nov 2023)";
        case GGUF_FILE_VERSION_V2: return "GGUF V2";
        case GGUF_FILE_VERSION_V3: return "GGUF V3 (latest)";
    }

    return "unknown";
}

#define LLAMA_FTYPE_PREFIX "(guessed) "

const char * llama_ftype_name(llama_ftype ftype) {
    static constexpr size_t guessed_prefix_len = sizeof(LLAMA_FTYPE_PREFIX) - 1;
    const char * name;
    switch ((enum llama_ftype) (ftype & ~LLAMA_FTYPE_GUESSED)) {
        case LLAMA_FTYPE_ALL_F32:          name = LLAMA_FTYPE_PREFIX "all F32"; break;
        case LLAMA_FTYPE_MOSTLY_F16:       name = LLAMA_FTYPE_PREFIX "F16"; break;
        case LLAMA_FTYPE_MOSTLY_BF16:      name = LLAMA_FTYPE_PREFIX "BF16"; break;
        case LLAMA_FTYPE_MOSTLY_Q1_0:      name = LLAMA_FTYPE_PREFIX "Q1_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_0:      name = LLAMA_FTYPE_PREFIX "Q2_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_0:      name = LLAMA_FTYPE_PREFIX "Q4_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1:      name = LLAMA_FTYPE_PREFIX "Q4_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0:      name = LLAMA_FTYPE_PREFIX "Q5_0"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1:      name = LLAMA_FTYPE_PREFIX "Q5_1"; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0:      name = LLAMA_FTYPE_PREFIX "Q8_0"; break;
        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: name = LLAMA_FTYPE_PREFIX "MXFP4 MoE"; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     name = LLAMA_FTYPE_PREFIX "NVFP4"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K:      name = LLAMA_FTYPE_PREFIX "Q2_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:    name = LLAMA_FTYPE_PREFIX "Q2_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:    name = LLAMA_FTYPE_PREFIX "Q3_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:    name = LLAMA_FTYPE_PREFIX "Q3_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:    name = LLAMA_FTYPE_PREFIX "Q3_K - Large"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:    name = LLAMA_FTYPE_PREFIX "Q4_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:    name = LLAMA_FTYPE_PREFIX "Q4_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:    name = LLAMA_FTYPE_PREFIX "Q5_K - Small"; break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:    name = LLAMA_FTYPE_PREFIX "Q5_K - Medium"; break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:      name = LLAMA_FTYPE_PREFIX "Q6_K"; break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:     name = LLAMA_FTYPE_PREFIX "TQ1_0 - 1.69 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS:   name = LLAMA_FTYPE_PREFIX "IQ2_XXS - 2.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:    name = LLAMA_FTYPE_PREFIX "IQ2_XS - 2.3125 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:     name = LLAMA_FTYPE_PREFIX "IQ2_S - 2.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:     name = LLAMA_FTYPE_PREFIX "IQ2_M - 2.7 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:    name = LLAMA_FTYPE_PREFIX "IQ3_XS - 3.3 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS:   name = LLAMA_FTYPE_PREFIX "IQ3_XXS - 3.0625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:     name = LLAMA_FTYPE_PREFIX "IQ1_S - 1.5625 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:     name = LLAMA_FTYPE_PREFIX "IQ1_M - 1.75 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:    name = LLAMA_FTYPE_PREFIX "IQ4_NL - 4.5 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:    name = LLAMA_FTYPE_PREFIX "IQ4_XS - 4.25 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:     name = LLAMA_FTYPE_PREFIX "IQ3_S - 3.4375 bpw"; break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:     name = LLAMA_FTYPE_PREFIX "IQ3_S mix - 3.66 bpw"; break;
        default:                           name = LLAMA_FTYPE_PREFIX "unknown, may not work"; break;
    }
    return (ftype & LLAMA_FTYPE_GUESSED) ? name : name + guessed_prefix_len;
}

#undef LLAMA_FTYPE_PREFIX

// return a list of splits for a given path
// for example, given "<name>-00002-of-00004.gguf", returns list of all 4 splits
static std::vector<std::string> llama_get_list_splits(const std::string & path, const int idx, const int n_split) {
    std::vector<std::string> paths;
    std::string split_prefix;
    std::vector<char> buf(llama_path_max(), 0);

    {
        int ret = llama_split_prefix(buf.data(), buf.size(), path.c_str(), idx, n_split);
        if (!ret) {
            throw std::runtime_error(format("invalid split file name: %s", path.c_str()));
        }
        split_prefix = std::string(buf.data(), ret);
    }

    if (split_prefix.empty()) {
        throw std::runtime_error(format("invalid split file: %s", path.c_str()));
    }

    for (int idx = 0; idx < n_split; ++idx) {
        int ret = llama_split_path(buf.data(), buf.size(), split_prefix.c_str(), idx, n_split);
        paths.push_back(std::string(buf.data(), ret));
    }

    return paths;
}

namespace GGUFMeta {
    template <typename T, gguf_type gt_, T (*gfun)(const gguf_context *, const int64_t)>
    struct GKV_Base_Type {
        static constexpr gguf_type gt = gt_;

        static T getter(const gguf_context * ctx, const int kid) {
            return gfun(ctx, kid);
        }
    };

    template<typename T> struct GKV_Base;

    template<> struct GKV_Base<bool        >: GKV_Base_Type<bool,         GGUF_TYPE_BOOL,    gguf_get_val_bool> {};
    template<> struct GKV_Base<uint8_t     >: GKV_Base_Type<uint8_t,      GGUF_TYPE_UINT8,   gguf_get_val_u8  > {};
    template<> struct GKV_Base<uint16_t    >: GKV_Base_Type<uint16_t,     GGUF_TYPE_UINT16,  gguf_get_val_u16 > {};
    template<> struct GKV_Base<uint32_t    >: GKV_Base_Type<uint32_t,     GGUF_TYPE_UINT32,  gguf_get_val_u32 > {};
    template<> struct GKV_Base<uint64_t    >: GKV_Base_Type<uint64_t,     GGUF_TYPE_UINT64,  gguf_get_val_u64 > {};
    template<> struct GKV_Base<int8_t      >: GKV_Base_Type<int8_t,       GGUF_TYPE_INT8,    gguf_get_val_i8  > {};
    template<> struct GKV_Base<int16_t     >: GKV_Base_Type<int16_t,      GGUF_TYPE_INT16,   gguf_get_val_i16 > {};
    template<> struct GKV_Base<int32_t     >: GKV_Base_Type<int32_t,      GGUF_TYPE_INT32,   gguf_get_val_i32 > {};
    template<> struct GKV_Base<int64_t     >: GKV_Base_Type<int64_t,      GGUF_TYPE_INT64,   gguf_get_val_i64 > {};
    template<> struct GKV_Base<float       >: GKV_Base_Type<float,        GGUF_TYPE_FLOAT32, gguf_get_val_f32 > {};
    template<> struct GKV_Base<double      >: GKV_Base_Type<double,       GGUF_TYPE_FLOAT64, gguf_get_val_f64 > {};
    template<> struct GKV_Base<const char *>: GKV_Base_Type<const char *, GGUF_TYPE_STRING,  gguf_get_val_str > {};

    template<> struct GKV_Base<std::string> {
        static constexpr gguf_type gt = GGUF_TYPE_STRING;

        static std::string getter(const gguf_context * ctx, const int kid) {
            return gguf_get_val_str(ctx, kid);
        }
    };

    struct ArrayInfo {
        const gguf_type gt;
        const size_t length;
        const void * data;
    };

    template<> struct GKV_Base<ArrayInfo> {
        public:
        static constexpr gguf_type gt = GGUF_TYPE_ARRAY;
        static ArrayInfo getter(const gguf_context *ctx, const int k) {
            const enum gguf_type arr_type = gguf_get_arr_type(ctx, k);
            return ArrayInfo {
                arr_type,
                gguf_get_arr_n(ctx, k),
                arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, k),
            };
        }
    };

    template<typename T>
    class GKV : public GKV_Base<T> {
        GKV() = delete;

        public:
        static T get_kv(const gguf_context * ctx, const int k) {
            const enum gguf_type kt = gguf_get_kv_type(ctx, k);

            if (kt != GKV::gt) {
                throw std::runtime_error(format("key %s has wrong type %s but expected type %s",
                    gguf_get_key(ctx, k), gguf_type_name(kt), gguf_type_name(GKV::gt)));
            }
            return GKV::getter(ctx, k);
        }

        static const char * override_type_to_str(const llama_model_kv_override_type ty) {
            switch (ty) {
                case LLAMA_KV_OVERRIDE_TYPE_BOOL:  return "bool";
                case LLAMA_KV_OVERRIDE_TYPE_INT:   return "int";
                case LLAMA_KV_OVERRIDE_TYPE_FLOAT: return "float";
                case LLAMA_KV_OVERRIDE_TYPE_STR:   return "str";
            }
            return "unknown";
        }

        static bool validate_override(const llama_model_kv_override_type expected_type, const struct llama_model_kv_override * ovrd) {
            if (!ovrd) { return false; }
            if (ovrd->tag == expected_type) {
                LLAMA_LOG_INFO("%s: Using metadata override (%5s) '%s' = ",
                    __func__, override_type_to_str(ovrd->tag), ovrd->key);
                switch (ovrd->tag) {
                    case LLAMA_KV_OVERRIDE_TYPE_BOOL:  {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_bool ? "true" : "false");
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_INT:   {
                        LLAMA_LOG_INFO("%" PRId64 "\n", ovrd->val_i64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_FLOAT: {
                        LLAMA_LOG_INFO("%.6f\n", ovrd->val_f64);
                    } break;
                    case LLAMA_KV_OVERRIDE_TYPE_STR: {
                        LLAMA_LOG_INFO("%s\n", ovrd->val_str);
                    } break;
                    default:
                        // Shouldn't be possible to end up here, but just in case...
                        throw std::runtime_error(
                            format("Unsupported attempt to override %s type for metadata key %s\n",
                                override_type_to_str(ovrd->tag), ovrd->key));
                }
                return true;
            }
            LLAMA_LOG_WARN("%s: Warning: Bad metadata override type for key '%s', expected %s but got %s\n",
                __func__, ovrd->key, override_type_to_str(expected_type), override_type_to_str(ovrd->tag));
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, bool>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_BOOL, ovrd)) {
                target = ovrd->val_bool;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<!std::is_same<OT, bool>::value && std::is_integral<OT>::value, bool>::type
        try_override(OT & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_INT, ovrd)) {
                target = ovrd->val_i64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_floating_point<OT>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_FLOAT, ovrd)) {
                target = ovrd->val_f64;
                return true;
            }
            return false;
        }

        template<typename OT>
        static typename std::enable_if<std::is_same<OT, std::string>::value, bool>::type
        try_override(T & target, const struct llama_model_kv_override * ovrd) {
            if (validate_override(LLAMA_KV_OVERRIDE_TYPE_STR, ovrd)) {
                target = ovrd->val_str;
                return true;
            }
            return false;
        }

        static bool set(const gguf_context * ctx, const int k, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            if (try_override<T>(target, ovrd)) {
                return true;
            }
            if (k < 0) { return false; }
            target = get_kv(ctx, k);
            return true;
        }

        static bool set(const gguf_context * ctx, const char * key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, gguf_find_key(ctx, key), target, ovrd);
        }

        static bool set(const gguf_context * ctx, const std::string & key, T & target, const struct llama_model_kv_override * ovrd = nullptr) {
            return set(ctx, key.c_str(), target, ovrd);
        }
    };
}

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(const std::string & key, T & result, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);


        result = arr_info.length;
        return true;
    }

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    llama_model_loader::get_arr_n(enum llm_kv kid, T & result, bool required) {
        return get_arr_n(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr_n(enum llm_kv kid, uint32_t & result, bool required);
    template std::enable_if<std::is_integral<uint32_t>::value, bool>::type
    llama_model_loader::get_arr_n<uint32_t>(const std::string & key, uint32_t & result, bool required);

    template<typename T>
    bool llama_model_loader::get_arr(const std::string & key, std::vector<T> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);
            result.clear();

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result.emplace_back(value);
            }
        } else {
            result.resize(arr_info.length);
            result.assign((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length);
        }

        return true;
    }

    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required) {
        const gguf_context * ctx = metadata;
        const int kid = gguf_find_key(ctx, key.c_str());

        if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("array key not found in model: %s", key.c_str()));
            }
            return false;
        }

        struct GGUFMeta::ArrayInfo arr_info =
            GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(ctx, kid);

        switch (arr_info.gt) {
            case GGUF_TYPE_BOOL:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:   GGML_ASSERT((std::is_same<T,     int32_t>::value) ||
                                                (std::is_same<T,    uint32_t>::value)); break;
            case GGUF_TYPE_FLOAT32: GGML_ASSERT((std::is_same<T,       float>::value)); break;
            case GGUF_TYPE_STRING:  GGML_ASSERT((std::is_same<T, std::string>::value)); break;
            default:
                throw std::runtime_error(format("%s is not a string/float32/uint32/int32 array", key.c_str()));
        }

        if (arr_info.length > N_MAX) {
            throw std::runtime_error(format("array length %u for key %s exceeds max %u", (uint32_t) arr_info.length, key.c_str(), (uint32_t) N_MAX));
        }

        if constexpr (std::is_same<T, std::string>::value) {
            const size_t n_items = gguf_get_arr_n(ctx, kid);

            for (size_t i = 0; i < n_items; i++) {
                const T value = gguf_get_arr_str(ctx, kid, i);
                result[i] = value;
            }
        } else {
            if (arr_info.gt == GGUF_TYPE_BOOL) {
                const int8_t * values = (const int8_t *) arr_info.data;
                std::transform(values, values + arr_info.length, result.begin(), [](int8_t x) {
                    return static_cast<T>(x != 0);
                });
            } else {
                std::copy((const T*)arr_info.data, (const T *)arr_info.data + arr_info.length, result.begin());
            }
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_arr(enum llm_kv kid, T & result, bool required) {
        return get_arr(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_arr<std::vector<std::string>>(enum llm_kv kid, std::vector<std::string> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<int32_t, 512>>(enum llm_kv kid, std::array<int32_t, 512> & result, bool required);
    template bool llama_model_loader::get_arr<std::vector<int32_t>>(enum llm_kv kid, std::vector<int32_t> & result, bool required);
    template bool llama_model_loader::get_arr<std::array<uint32_t, LLAMA_MAX_LAYERS>>(enum llm_kv kid, std::array<uint32_t, LLAMA_MAX_LAYERS> & result, bool required);

    template<typename T>
    bool llama_model_loader::get_key(const std::string & key, T & result, bool required) {
        auto it = kv_overrides.find(key);

        const struct llama_model_kv_override * override =
            it != kv_overrides.end() ? &it->second : nullptr;

        const bool found = GGUFMeta::GKV<T>::set(metadata, key, result, override);

        if (required && !found) {
            throw std::runtime_error(format("key not found in model: %s", key.c_str()));
        }

        return found;
    }

    template<typename T>
    bool llama_model_loader::get_key(enum llm_kv kid, T & result, bool required) {
        return get_key(llm_kv(kid), result, required);
    }

    template bool llama_model_loader::get_key<bool>       (enum llm_kv kid, bool & result,        bool required);
    template bool llama_model_loader::get_key<float>      (enum llm_kv kid, float & result,       bool required);
    template bool llama_model_loader::get_key<uint32_t>   (enum llm_kv kid, uint32_t & result,    bool required);
    template bool llama_model_loader::get_key<std::string>(enum llm_kv kid, std::string & result, bool required);

    template<>
    bool llama_model_loader::get_key(enum llm_kv kid, enum llama_pooling_type & result, bool required) {
        uint32_t tmp;
        const bool found = get_key(kid, tmp, required);
        if (found) {
            result = (enum llama_pooling_type) tmp;
        } else {
            result = LLAMA_POOLING_TYPE_UNSPECIFIED;
        }
        return found;
    }

    // get array of n <= N_MAX elements, or a single element repeated n times
    template<typename T, size_t N_MAX>
    bool llama_model_loader::get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required) {
        const int kid = gguf_find_key(metadata, key.c_str());

        if (kid < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        if (n > N_MAX) {
            throw std::runtime_error(format("n > N_MAX: %u > %u for key %s", n, (uint32_t) N_MAX, key.c_str()));
        }

        if (gguf_get_kv_type(metadata, kid) == GGUF_TYPE_ARRAY) {
            struct GGUFMeta::ArrayInfo arr_info =
                GGUFMeta::GKV<GGUFMeta::ArrayInfo>::get_kv(metadata, kid);

            if (n != arr_info.length) {
                throw std::runtime_error(format("key %s has wrong array length; expected %u, got %u", key.c_str(), n, (uint32_t) arr_info.length));
            }

            return get_arr(key, result, required);
        }

        T value;

        bool ok = get_key(key, value, required);
        if (!ok) {
            return false;
        }

        for (uint32_t i = 0; i < n; i++) {
            result[i] = value;
        }

        return true;
    }

    template<typename T>
    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    bool llama_model_loader::get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required) {
        const std::string key = llm_kv(kid);

        const int id = gguf_find_key(metadata, key.c_str());

        if (id < 0) {
            if (required) {
                throw std::runtime_error(format("key not found in model: %s", key.c_str()));
            }
            return false;
        }

        // throw and error if type is an array
        if (gguf_get_kv_type(metadata, id) == GGUF_TYPE_ARRAY) {
            if (required) {
                throw std::runtime_error(format("expected scalar, found array for key: %s", key.c_str()));
            }
            return false;
        }

        return get_key(key, result, required);
    }

    // TODO: this is not very clever - figure out something better
    template bool llama_model_loader::get_key_or_arr<std::array<int,      4>>  (enum llm_kv kid, std::array<int,      4>   & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<uint32_t, 512>>(enum llm_kv kid, std::array<uint32_t, 512> & result, uint32_t n, bool required);
    template bool llama_model_loader::get_key_or_arr<std::array<float,    512>>(enum llm_kv kid, std::array<float,    512> & result, uint32_t n, bool required);


llama_model_loader::llama_model_loader(
        struct gguf_context * meta,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits,
        FILE * file,
        bool use_mmap,
        bool use_direct_io,
        bool check_tensors,
        bool no_alloc,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p)
        : metadata(meta), set_tensor_data(set_tensor_data), set_tensor_data_ud(set_tensor_data_ud) {
    int trace = 0;
    if (getenv("LLAMA_TRACE")) {
        trace = atoi(getenv("LLAMA_TRACE"));
    }

    if (param_overrides_p != nullptr) {
        for (const struct llama_model_kv_override * p = param_overrides_p; p->key[0] != 0; p++) {
            kv_overrides.insert({std::string(p->key), *p});
        }
    }

    tensor_buft_overrides = param_tensor_buft_overrides_p;

    if (!fname.empty()) {
        // Load the main GGUF
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file(fname.c_str(), params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from %s", __func__, fname.c_str()));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(fname.c_str(), "rb", use_direct_io));
        contexts.emplace_back(ctx);

        if (use_mmap && use_direct_io) {
            if (files.back()->has_direct_io()) {
                LLAMA_LOG_WARN("%s: direct I/O is enabled, disabling mmap\n", __func__);
                use_mmap = false;
            } else {
                LLAMA_LOG_WARN("%s: direct I/O is not available, using mmap\n", __func__);
                use_direct_io = false;

                // reopen file using std::fopen for mmap
                files.pop_back();
                files.emplace_back(new llama_file(fname.c_str(), "rb", false));
            }
        }

        // Save tensors data offset of the main file.
        // For subsidiary files, `meta` tensor data offset must not be used,
        // so we build a unified tensors index for weights.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
        uint16_t n_split = 0;
        get_key(llm_kv(LLM_KV_SPLIT_COUNT), n_split, false);

        // Load additional GGML contexts
        if (n_split > 1) {
            // make sure the main file is loaded first
            uint16_t idx = 0;
            const std::string kv_split_no = llm_kv(LLM_KV_SPLIT_NO);
            get_key(kv_split_no, idx);
            if (idx != 0) {
                throw std::runtime_error(format("illegal split file idx: %d (file: %s), model must be loaded with the first split", idx, fname.c_str()));
            }

            // generate list of splits if needed
            if (splits.empty()) {
                splits = llama_get_list_splits(fname, idx, n_split);
            }

            // in case user give a custom list of splits, check if it matches the expected number
            if (n_split != (uint16_t)splits.size()) {
                throw std::runtime_error(format("invalid split count, given: %zu splits, but expected %d", splits.size(), n_split));
            }

            if (trace > 0) {
                LLAMA_LOG_INFO("%s: loading additional %d GGUFs\n", __func__, n_split);
            }

            // load other splits
            for (idx = 1; idx < n_split; idx++) {
                const char * fname_split = splits[idx].c_str();

                struct gguf_init_params split_params = {
                    /*.no_alloc = */ true,
                    /*.ctx      = */ &ctx,
                };
                gguf_context_ptr ctx_gguf { gguf_init_from_file(fname_split, split_params) };
                if (!ctx_gguf) {
                    throw std::runtime_error(format("%s: failed to load GGUF split from %s", __func__, fname_split));
                }

                // check idx
                {
                    const int kid = gguf_find_key(ctx_gguf.get(), kv_split_no.c_str());
                    if (kid < 0) {
                        throw std::runtime_error(format("missing key %s in GGUF split %s", kv_split_no.c_str(), fname_split));
                    }
                    int idx_gguf = gguf_get_val_u16(ctx_gguf.get(), kid);
                    if (idx_gguf != idx) {
                        throw std::runtime_error(format("invalid split file idx: %d (file: %s), expected %d", idx_gguf, fname_split, idx));
                    }
                }

                files.emplace_back(new llama_file(fname_split, "rb", use_direct_io));
                contexts.emplace_back(ctx);

                // Save tensors data offset info of the shard.
                for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
                    std::string tensor_name = std::string(cur->name);
                    // make sure there is no duplicated tensor names
                    if (weights_map.find(tensor_name) != weights_map.end()) {
                        throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
                    }
                    n_elements += ggml_nelements(cur);
                    n_bytes    += ggml_nbytes(cur);
                    weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), idx, ctx_gguf.get(), cur));
                }
            }

            get_key(llm_kv(LLM_KV_SPLIT_TENSORS_COUNT), n_tensors);

            // sanity check
            {
                const int n_tensors_loaded = (int) weights_map.size();
                if (n_tensors != n_tensors_loaded) {
                    throw std::runtime_error(format("corrupted model: %d tensors expected but %d found", n_tensors, n_tensors_loaded));
                }
            }

            LLAMA_LOG_INFO("%s: additional %d GGUFs metadata loaded.\n",  __func__, n_split - 1);
        }
    } else if (file != nullptr) {
        struct ggml_context * ctx = NULL;
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &ctx,
        };

        metadata_ptr.reset(gguf_init_from_file_ptr(file, params));
        metadata = metadata_ptr.get();
        if (metadata == nullptr) {
            throw std::runtime_error(format("%s: failed to load model from file pointer", __func__));
        }

        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));

        files.emplace_back(new llama_file(file));
        contexts.emplace_back(ctx);

        // Save tensors data offset info of the main file.
        for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur)) {
            std::string tensor_name = std::string(cur->name);
            // make sure there is no duplicated tensor names
            if (weights_map.find(tensor_name) != weights_map.end()) {
                throw std::runtime_error(format("invalid model: tensor '%s' is duplicated", ggml_get_name(cur)));
            }
            n_elements += ggml_nelements(cur);
            n_bytes    += ggml_nbytes(cur);
            weights_map.emplace(tensor_name, llama_tensor_weight(files.back().get(), 0, metadata, cur));
        }
    } else {
        get_key(llm_kv(LLM_KV_GENERAL_ARCHITECTURE), arch_name, false);
        llm_kv = LLM_KV(llm_arch_from_string(arch_name));
    }

    n_kv      = gguf_get_n_kv(metadata);
    n_tensors = weights_map.size();

    fver = (enum llama_fver) gguf_get_version(metadata);

    LLAMA_LOG_INFO("%s: loaded meta data with %d key-value pairs and %d tensors from %s (version %s)\n",
            __func__, n_kv, n_tensors, fname.empty() ? "(file*)" : fname.c_str(), llama_file_version_name(fver));

    // determine file type based on the number of tensors for each quantization and print meta data
    // TODO: make optional
    {
        std::map<enum ggml_type, uint32_t> n_type;

        uint32_t n_type_max = 0;
        enum ggml_type type_max = GGML_TYPE_F32;

        for (const auto & it : weights_map) {
            const llama_tensor_weight & w = it.second;
            const ggml_tensor * tensor = w.tensor;

            enum ggml_type type = tensor->type;

            n_type[type]++;

            if (n_type_max < n_type[type]) {
                n_type_max = n_type[type];
                type_max   = type;
            }

            if (trace > 0) {
                const uint16_t sid = w.idx;
                LLAMA_LOG_INFO("%s: - tensor split %2d: %32s %-8s [ %s ] %8.2f MiB\n", __func__,
                        sid, ggml_get_name(tensor), ggml_type_name(type), llama_format_tensor_shape(tensor).c_str(),
                        ggml_nbytes(tensor)/1024.0f/1024.0f);
            }
        }

        switch (type_max) {
            case GGML_TYPE_F32:     ftype = LLAMA_FTYPE_ALL_F32;        break;
            case GGML_TYPE_F16:     ftype = LLAMA_FTYPE_MOSTLY_F16;     break;
            case GGML_TYPE_BF16:    ftype = LLAMA_FTYPE_MOSTLY_BF16;    break;
            case GGML_TYPE_Q4_0:    ftype = LLAMA_FTYPE_MOSTLY_Q4_0;    break;
            case GGML_TYPE_Q4_1:    ftype = LLAMA_FTYPE_MOSTLY_Q4_1;    break;
            case GGML_TYPE_Q5_0:    ftype = LLAMA_FTYPE_MOSTLY_Q5_0;    break;
            case GGML_TYPE_Q5_1:    ftype = LLAMA_FTYPE_MOSTLY_Q5_1;    break;
            case GGML_TYPE_Q8_0:    ftype = LLAMA_FTYPE_MOSTLY_Q8_0;    break;
            case GGML_TYPE_Q2_K:    ftype = LLAMA_FTYPE_MOSTLY_Q2_K;    break;
            case GGML_TYPE_Q3_K:    ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;  break;
            case GGML_TYPE_Q4_K:    ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;  break;
            case GGML_TYPE_Q5_K:    ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;  break;
            case GGML_TYPE_Q6_K:    ftype = LLAMA_FTYPE_MOSTLY_Q6_K;    break;
            case GGML_TYPE_TQ1_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ1_0;   break;
            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;
            case GGML_TYPE_IQ2_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ2_XXS; break;
            case GGML_TYPE_IQ2_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ2_XS;  break;
            case GGML_TYPE_IQ2_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ2_S;   break;
            case GGML_TYPE_IQ3_XXS: ftype = LLAMA_FTYPE_MOSTLY_IQ3_XXS; break;
            case GGML_TYPE_IQ1_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_S;   break;
            case GGML_TYPE_IQ1_M:   ftype = LLAMA_FTYPE_MOSTLY_IQ1_M;   break;
            case GGML_TYPE_IQ4_NL:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_NL;  break;
            case GGML_TYPE_IQ4_XS:  ftype = LLAMA_FTYPE_MOSTLY_IQ4_XS;  break;
            case GGML_TYPE_IQ3_S:   ftype = LLAMA_FTYPE_MOSTLY_IQ3_S;   break;
            case GGML_TYPE_NVFP4:   ftype = LLAMA_FTYPE_MOSTLY_NVFP4;   break;
            case GGML_TYPE_Q1_0:    ftype = LLAMA_FTYPE_MOSTLY_Q1_0;    break;
            case GGML_TYPE_Q2_0:    ftype = LLAMA_FTYPE_MOSTLY_Q2_0;    break;
            default:
                {
                    LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, ggml_type_name(type_max));
                    ftype = LLAMA_FTYPE_ALL_F32;
                } break;
        }

        // this is a way to mark that we have "guessed" the file type
        ftype = (llama_ftype) (ftype | LLAMA_FTYPE_GUESSED);

        {
            uint32_t ftype_val = 0;
            if (get_key(LLM_KV_GENERAL_FILE_TYPE, ftype_val, false)) {
                ftype = (llama_ftype) ftype_val;
            }
        }

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values. Note: KV overrides do not apply in this output.\n", __func__);

        for (int i = 0; i < n_kv; i++) {
            const char * name           = gguf_get_key(metadata, i);
            const enum gguf_type type   = gguf_get_kv_type(metadata, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(metadata, i)), gguf_get_arr_n(metadata, i))
                : gguf_type_name(type);

            std::string value          = gguf_kv_to_str(metadata, i);
            const size_t MAX_VALUE_LEN = 40;
            if (value.size() > MAX_VALUE_LEN) {
                value = format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str());
            }
            replace_all(value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), value.c_str());
        }

        // print type counts
        for (auto & kv : n_type) {
            if (kv.second == 0) {
                continue;
            }

            LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, ggml_type_name(kv.first), kv.second);
        }
    }

    if (!llama_mmap::SUPPORTED) {
        LLAMA_LOG_WARN("%s: mmap is not supported on this platform\n", __func__);
        use_mmap = false;
    }

    this->use_mmap = use_mmap;
    this->use_direct_io = use_direct_io;
    this->check_tensors = check_tensors;
    this->no_alloc = no_alloc;
}

std::string llama_model_loader::get_arch_name() const {
    return arch_name;
}

enum llm_arch llama_model_loader::get_arch() const {
    return llm_kv.arch;
}

const llama_model_loader::llama_tensor_weight * llama_model_loader::get_weight(const char * name) const {
    auto pos = weights_map.find(name);
    if (pos != weights_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

const llama_model_loader::llama_tensor_weight & llama_model_loader::require_weight(const char * name) const {
    const llama_tensor_weight * weight = get_weight(name);
    if (!weight) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name));
    }
    return *weight;
}

struct ggml_tensor * llama_model_loader::get_tensor_meta(const char * name) const {
    const auto * weight = get_weight(name);
    if (!weight) {
        return nullptr;
    }
    return weight->tensor;
}

struct ggml_tensor * llama_model_loader::require_tensor_meta(const std::string & name) const {
    struct ggml_tensor * tensor = get_tensor_meta(name.c_str());
    if (!tensor) {
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }
    return tensor;
}

const struct ggml_tensor * llama_model_loader::check_tensor_dims(const std::string & name, const std::vector<int64_t> & ne, bool required) const {
    const struct ggml_tensor * cur = get_tensor_meta(name.c_str());

    if (cur == NULL) {
        if (!required) {
            return NULL;
        }
        throw std::runtime_error(format("%s: tensor '%s' not found", __func__, name.c_str()));
    }

    {
        bool is_ok = true;
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            if ((i < ne.size() && ne[i] != cur->ne[i]) || (i >= ne.size() && cur->ne[i] != 1)) {
                is_ok = false;
                break;
            }
        }
        if (!is_ok) {
            throw std::runtime_error(
                    format("%s: tensor '%s' has wrong shape; expected %s, got %s",
                        __func__, name.c_str(),
                        llama_format_tensor_shape(ne).c_str(),
                        llama_format_tensor_shape(cur).c_str()));
        }
    }

    return cur;
}

// checks if the weight tensor can be used with the specified buffer type and device
static bool weight_buft_supported(const llama_hparams & hparams, ggml_tensor * w, ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) {
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * op_tensor = nullptr;

    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                const int n_expert_used = hparams.n_expert_used;
                GGML_ASSERT(n_expert_used > 0);
                ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_ADD_ID:
            {
                const int n_expert_used = hparams.n_expert_used;
                GGML_ASSERT(n_expert_used > 0);
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_expert_used, 512);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, 512);
                op_tensor = ggml_add_id(ctx, a, w, c);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], w->ne[1], w->ne[2], w->ne[3]);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_DIV:
            {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w->ne[0]);
                op_tensor = ggml_div(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                const int n_embd_head = hparams.n_embd_head_v();
                const int n_head = hparams.n_head();
                ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_head, n_head, 512);
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 512);
                op_tensor = ggml_rope_ext(
                    ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_SSM_CONV:
            {
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * conv_x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0] - 1 + n_seq_tokens, w->ne[1], n_seqs);
                op_tensor = ggml_ssm_conv(ctx, conv_x, w);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                // w is ssm_a, which is used to distinguish Mamba-1 and Mamba-2
                const int64_t d_state      = w->ne[0] == 1 ? hparams.ssm_d_state : w->ne[0];
                const int64_t n_head       = w->ne[1];
                const int64_t head_dim     = hparams.ssm_d_inner / n_head;
                const int64_t n_group      = hparams.ssm_n_group ? hparams.ssm_n_group : 1;
                const int64_t n_seq_tokens = 512;
                const int64_t n_seqs       = 3;
                ggml_tensor * s   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, head_dim, n_head, n_seqs);
                ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * dt  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_head, n_seq_tokens, n_seqs);
                ggml_tensor * B   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * C   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_state, n_group, n_seq_tokens, n_seqs);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
                op_tensor = ggml_ssm_scan(ctx, s, x, dt, w, B, C, ids);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                // FIXME
                const int64_t S = 123;
                const int64_t H = 123;
                const int64_t n_tokens = 123;
                const int64_t n_seqs = 123;
                ggml_tensor  * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * tf = w;
                ggml_tensor  * td = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, H, n_tokens);
                ggml_tensor  * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_seqs, S, H);
                op_tensor = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state);
            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd_inp = hparams.n_embd_inp();
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_embd_inp, w->ne[1], 1, 1);
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        case GGML_OP_SCALE:
            {
                op_tensor = ggml_scale(ctx, w, 1.0f);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == nullptr);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = nullptr;

    return op_supported;
}

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type_t select_weight_buft(const llama_hparams & hparams, ggml_tensor * tensor, ggml_op op, const buft_list_t * buft_list) {
    GGML_ASSERT(!buft_list->empty());
    for (const auto & cur : *buft_list) {
        ggml_backend_dev_t cur_dev = cur.first;
        ggml_backend_buffer_type_t cur_buft = cur.second;
        if (weight_buft_supported(hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
    }

    return nullptr;
}

struct ggml_tensor * llama_model_loader::create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags) {
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // one ggml context per buffer type
            int max_n_tensors = n_tensors;
            max_n_tensors += 1;                   // duplicated output tensor
            max_n_tensors += hparams.n_layer()*2; // duplicated rope freq tensors
            if (files.empty()) {
                max_n_tensors += hparams.n_layer()*256; // this should be well above what any model actually uses
            }
            const size_t ctx_size = ggml_tensor_overhead()*max_n_tensors;

            ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error(format("failed to create ggml context"));
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }
        return it->second.get();
    };

    auto buft_for_tensor = [&](ggml_tensor * t_meta) -> ggml_backend_buffer_type_t {
        if (!t_meta) {
            if (flags & TENSOR_NOT_REQUIRED) {
                return nullptr;
            }
            throw std::runtime_error(format("missing tensor '%s'", tn.str().c_str()));
        }

        // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
        // the tensor is duplicated
        // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
        llm_tensor tn_tensor = tn.tensor;
        if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && (flags & TENSOR_DUPLICATED)) {
            tn_tensor = LLM_TENSOR_OUTPUT;
        }

        llm_tensor_info info;
        try {
            info = llm_tensor_info_for(tn_tensor);
        } catch (const std::out_of_range & e) {
            throw std::runtime_error(format("missing tensor info mapping for %s", tn.str().c_str()));
        }

        // skip unused tensors
        if (info.op == GGML_OP_NONE || (flags & TENSOR_SKIP)) {
            const size_t nbytes = ggml_nbytes(t_meta);
            LLAMA_LOG_WARN("model has unused tensor %s (size = %zu bytes) -- ignoring\n", tn.str().c_str(), nbytes);

            size_data -= nbytes;
            n_created++;

            return nullptr;
        }

        // tensors with "bias" suffix are always used with GGML_OP_ADD or GGML_OP_ADD_ID
        ggml_op op;
        bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
        if (bias) {
            if (info.op == GGML_OP_MUL_MAT_ID) {
                op = GGML_OP_ADD_ID;
            } else {
                op = GGML_OP_ADD;
            }
        } else {
            op = info.op;
        }

        // sanity checks
        if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
            if (tn.bid != -1) {
                GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
            }
        } else {
            if (tn.bid == -1) {
                GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
            }
        }

        // select the buffer type for this tensor
        const buft_list_t * buft_list;
        switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                buft_list = buft_list_input;
                break;
            case LLM_TENSOR_LAYER_OUTPUT:
                buft_list = buft_list_output;
                break;
            case LLM_TENSOR_LAYER_REPEATING:
                GGML_ASSERT(buft_list_layer != nullptr);
                buft_list = buft_list_layer;
                break;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
        }

        ggml_backend_buffer_type_t buft = nullptr;

        // check overrides
        if (tensor_buft_overrides) {
            std::string tensor_name = tn.str();
            for (const auto * overrides = tensor_buft_overrides; overrides->pattern != nullptr; ++overrides) {
                std::regex pattern(overrides->pattern);
                if (std::regex_search(tensor_name, pattern)) {
                    if (overrides->buft == ggml_backend_cpu_buffer_type()) {
                        if ((getenv("LLAMA_TEMPORAL_R") || getenv("LLAMA_NO_REPACK")) && !getenv("LLAMA_TEMPORAL_REPACK")) {
                            // temporal slot-pool: honor the CPU override LITERALLY. The
                            // reconsideration below routes q4_K/q8_0 experts back into
                            // CPU_REPACK (anonymous repacked copies), which the pool
                            // cannot pread into or evict -- measured: only the 23 q6_K
                            // down_exps of 135 expert tensors ended up pool-managed.
                            // LLAMA_NO_REPACK gives the same literal-CPU placement with
                            // the pool INACTIVE: the like-for-like numerics baseline.
                            //
                            // LLAMA_TEMPORAL_REPACK opts back INTO CPU_REPACK: experts
                            // land in the repacked buffer so mul_mat_id takes the fast
                            // interleaved GEMM, and the pool streams pre-repacked slices
                            // from a side-file (LLAMA_TEMPORAL_REPACK_FILE, same byte
                            // layout as the gguf -- repack is a per-plane permutation).
                            buft = overrides->buft;
                        } else {
                        // when overriding to a CPU buffer, consider the extra buffer types
                        buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);
                        }
                        if (use_mmap) {
                            static std::once_flag once;
                            std::call_once(once, [] {
                                LLAMA_LOG_WARN("llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance\n");
                            });
                        }
                    } else {
                        buft = overrides->buft;
                    }

                    LLAMA_LOG_DEBUG("tensor %s (%zu MiB %s) buffer type overridden to %s\n",
                            tensor_name.c_str(),
                            ggml_nbytes(t_meta) / 1024 / 1024, ggml_type_name(t_meta->type),
                            ggml_backend_buft_name(buft));
                    break;
                }
            }
        }

        if (!buft) {
            buft = select_weight_buft(hparams, t_meta, op, buft_list);
            if (!buft) {
                throw std::runtime_error(format("failed to find a compatible buffer type for tensor %s", tn.str().c_str()));
            }
        }

        // avoid using a host buffer when using mmap
        auto * buft_dev = ggml_backend_buft_get_device(buft);
        if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
            auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu_dev) {
                throw std::runtime_error("no CPU backend found");
            }
            buft = ggml_backend_dev_buffer_type(cpu_dev);
        }

        if (buft != buft_list->front().second) {
            if (n_tensors_moved == 0) {
                first_tensor_moved_name = t_meta->name;
                first_tensor_moved_type_name = ggml_type_name(t_meta->type);
                first_moved_from_buft = buft_list->front().second;
                first_moved_to_buft   = buft;
            }
            n_tensors_moved++;
        }

        return buft;
    };

    if (files.empty()) {
        if (flags & TENSOR_SKIP_IF_VIRTUAL) {
            return nullptr;
        }
        ggml_type type = GGML_TYPE_F32;
        const int64_t tid = gguf_find_tensor(metadata, tn.str().c_str());
        if (tid != -1) {
            type = gguf_get_tensor_type(metadata, tid);
        }

        // for tensors that are not required some of the dimensions can be invalid:
        if (flags & TENSOR_NOT_REQUIRED) {
            for (size_t dim = 0; dim < ne.size(); dim++) {
                if (ne.begin()[dim] <= 0) {
                    return nullptr;
                }
            }
        }

        ggml_tensor t_meta;
        memset(&t_meta, 0, sizeof(ggml_tensor));
        t_meta.type = type;
        for (size_t dim = 0; dim < GGML_MAX_DIMS; dim++) {
            t_meta.ne[dim] = dim < ne.size() ? ne.begin()[dim] : 1;
            GGML_ASSERT(t_meta.ne[dim] >= 1);
            t_meta.nb[dim] = dim == 0 ? ggml_type_size(type) : t_meta.ne[dim-1]*t_meta.nb[dim-1];
            GGML_ASSERT(t_meta.nb[dim] >= 1);
        }
        ggml_set_name(&t_meta, tn.str().c_str());

        ggml_backend_buffer_type_t buft = buft_for_tensor(&t_meta);
        GGML_ASSERT(buft != nullptr);
        ggml_context * ctx = ctx_for_buft(buft);
        ggml_tensor * ret = ggml_dup_tensor(ctx, &t_meta);
        ggml_set_name(ret, tn.str().c_str());
        return ret;
    }

    ggml_tensor * t_meta = get_tensor_meta(tn.str().c_str());
    ggml_backend_buffer_type_t buft = buft_for_tensor(t_meta);
    if (buft == nullptr) {
        return nullptr; // return type is ggml_tensor *
    }
    ggml_context * ctx = ctx_for_buft(buft);

    // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
    if (flags & TENSOR_DUPLICATED) {
        ggml_tensor * t = ggml_get_tensor(ctx, tn.str().c_str());
        if (t) {
            return t;
        }
    }

    LLAMA_LOG_DEBUG("%s: loading tensor %s\n", __func__, tn.str().c_str());
    const struct ggml_tensor * cur = check_tensor_dims(tn.str(), ne, !(flags & TENSOR_NOT_REQUIRED));

    if (cur == NULL) {
        return NULL;
    }

    const bool duplicated = flags & TENSOR_DUPLICATED;

    struct ggml_tensor * tensor = ggml_dup_tensor(ctx, cur);
    ggml_set_name(tensor, ggml_get_name(cur));

    if (duplicated) {
        size_data += ggml_nbytes(cur);
    } else {
        n_created++;
    }

    return tensor;
}

struct ggml_tensor * llama_model_loader::create_tensor_as_view(struct ggml_context * ctx, struct ggml_tensor * base, const std::string & name, const std::initializer_list<int64_t> & ne, size_t offset, bool required) {
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, required);

    if (cur == NULL) {
        return NULL;
    }

    if (cur->type != base->type) {
        throw std::runtime_error(format("%s: tensor '%s' has wrong type; expected %s, got %s", __func__, name.c_str(), ggml_type_name(base->type), ggml_type_name(cur->type)));
    }

    std::array<int64_t, GGML_MAX_DIMS> dims;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        dims[i] = i < ne.size() ? ne.begin()[i] : 1;
    }

    struct ggml_tensor * tensor = ggml_view_4d(ctx, base,
                                    dims[0], dims[1], dims[2], dims[3],
                                    cur->nb[1], cur->nb[2], cur->nb[3],
                                    offset);

    ggml_set_name(tensor, name.c_str());

    n_created++;

    return tensor;
}

void llama_model_loader::done_getting_tensors(bool partial) const {
    if (n_created > n_tensors) {
        throw std::runtime_error(format("%s: too many tensors created; expected %d, got %d", __func__, n_tensors, n_created));
    }
    if (n_created < n_tensors) {
        if (!partial) {
            throw std::runtime_error(format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
        }
        LLAMA_LOG_INFO("%s: partial load — used %d of %d tensors in the file (rest belong to a sibling model on the same .gguf)\n",
                __func__, n_created, n_tensors);
    }
    if (n_tensors_moved > 0) {
        LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %zu others) cannot be used with preferred buffer type %s, using %s instead\n",
            __func__, first_tensor_moved_name.c_str(), first_tensor_moved_type_name.c_str(), n_tensors_moved - 1,
            ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
    }
}

void llama_model_loader::init_mappings(bool prefetch, llama_mlocks * mlock_mmaps) {
    if (use_mmap) {
        mappings.reserve(files.size());
        mmaps_used.reserve(files.size());
        for (const auto & file : files) {
            bool is_numa = false;

            auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (dev) {
                auto * reg = ggml_backend_dev_backend_reg(dev);
                auto * is_numa_fn = (decltype(ggml_is_numa) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
                if (is_numa_fn) {
                    is_numa = is_numa_fn();
                }
            }

            std::unique_ptr<llama_mmap> mapping = std::make_unique<llama_mmap>(file.get(), prefetch ? -1 : 0, is_numa);
            mmaps_used.emplace_back(mapping->size(), 0);
            if (mlock_mmaps) {
                std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                mlock_mmap->init(mapping->addr());
                mlock_mmaps->emplace_back(std::move(mlock_mmap));
            }
            mappings.emplace_back(std::move(mapping));
        }
    }

    // compute the total size of all tensors for progress reporting
    for (const auto & it : weights_map) {
        size_data += ggml_nbytes(it.second.tensor);
    }
}

void llama_model_loader::get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const {
    GGML_ASSERT(!mappings.empty());
    const auto & mapping = mappings.at(idx);

    *first = mapping->size();
    *last  = 0;
    *addr = mapping->addr();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        const auto * weight = get_weight(ggml_get_name(tensor));
        if (!weight || weight->idx != idx) {
            continue;
        }
        *first = std::min(*first, weight->offs);
        *last  = std::max(*last,  weight->offs + ggml_nbytes(tensor));
    }
}

void llama_model_loader::load_data_for(struct ggml_tensor * cur) const {
    const auto & w = require_weight(ggml_get_name(cur));

    if (use_mmap) {
        const auto & mapping = mappings.at(w.idx);
        if (cur->data == nullptr) {
            cur->data = (uint8_t *)mapping->addr() + w.offs;
        } else {
            memcpy(cur->data, (uint8_t *)mapping->addr() + w.offs, ggml_nbytes(cur));
        }
    } else {
        GGML_ASSERT(cur->data != nullptr);
        GGML_ASSERT(w.idx < files.size());
        const auto & file = files.at(w.idx);
        file->seek(w.offs, SEEK_SET);
        file->read_raw(cur->data, ggml_nbytes(cur));
    }

    if (check_tensors && !ggml_validate_row_data(cur->type, cur->data, ggml_nbytes(cur))) {
        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
    }
}

bool llama_model_loader::load_all_data(
        struct ggml_context * ctx,
        llama_buf_map & bufs,
        llama_mlocks * lmlocks,
        llama_progress_callback progress_callback,
        void * progress_callback_user_data) {
    if (files.empty()) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            set_tensor_data(t, set_tensor_data_ud);
        }
        return true;
    }
    GGML_ASSERT(size_data != 0 && "call init_mappings() first");

    std::vector<no_init<uint8_t>> read_buf;
    std::vector<std::future<std::pair<ggml_tensor *, bool>>> validation_result;

    // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
    // NVMe raid configurations might require more / larger buffers.
    constexpr size_t n_buffers = 4;

    size_t alignment = 1;
    for (const auto & file : files) {
        alignment = std::max(file->read_alignment(), alignment);
    }

    // Buffer size: balance between memory usage and I/O efficiency
    // 64MB works well for NVMe drives
    const size_t buffer_size = alignment != 1 ? 64 * 1024 * 1024 + 2 * alignment : 1 * 1024 * 1024;

    std::vector<ggml_backend_buffer_t> host_buffers;
    std::vector<ggml_backend_event_t> events;
    std::vector<void *> host_ptrs;
    size_t buffer_idx = 0; // buffer to use for async loads
    ggml_backend_t upload_backend = [&](const char * func) -> ggml_backend_t {
        if (use_mmap || check_tensors) {
            return nullptr;
        }
        // When not using mmaped io use async uploads from pinned memory to GPU memory.
        // First determine if the backend supports the necessary features for async uploads.
        auto * buf = bufs.count(0) ? bufs.at(0) : nullptr;
        if (!buf) {
            LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
            return nullptr;
        }

        auto * buft = ggml_backend_buffer_get_type(buf);
        auto * dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                ggml_backend_buft_name(buft));
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
            return nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (!host_buft) {
            LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        // If the backend is supported, create pinned memory buffers and events for synchronisation.
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            auto * buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);

            if (!buf) {
                LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            host_buffers.emplace_back(buf);
            host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

            auto * event = ggml_backend_event_new(dev);
            if (!event) {
                LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            events.emplace_back(event);
        }

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                ggml_backend_dev_name(dev));
            return nullptr;
        }

        return backend;
    }(__func__);

    if (upload_backend) {
        LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
            ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
            ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
            ggml_backend_name(upload_backend));
    }

    for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        const auto * weight = get_weight(ggml_get_name(cur));
        if (weight == nullptr) {
            // this can happen with split experts models
            continue;
        }

        if (progress_callback) {
            if (!progress_callback((float) size_done / size_data, progress_callback_user_data)) {
                return false;
            }
        }

        size_t n_size = ggml_nbytes(cur);

        if (use_mmap) {
            const auto & mapping = mappings.at(weight->idx);
            ggml_backend_buffer_t buf_mmap = nullptr;
            if (bufs.count(weight->idx)) {
                buf_mmap = bufs.at(weight->idx);
            }
            uint8_t * data = (uint8_t *) mapping->addr() + weight->offs;

#if defined(_POSIX_MAPPED_FILES)
            // LLAMA_TEMPORAL_MMAP=2: per-tensor residency policy for MoE on devices whose
            // RAM is smaller than the model. Expert tensors (*_exps) are the only ones
            // sparsely used -- top_k of E per token -- so only they are marked evictable.
            // Attention, norm and embedding weights are touched on every token and are
            // marked WILLNEED so the kernel keeps them resident instead of thrashing them
            // alongside the experts (which is what mode 1 does wrong).
            {
                static const char * tm = getenv("LLAMA_TEMPORAL_MMAP");
                static const int temporal_mode = tm ? atoi(tm) : 0;
                const bool is_expert = strstr(cur->name, "_exps") != nullptr;
                if (temporal_mode == 2) {
                    posix_madvise(data, n_size,
                                  is_expert ? POSIX_MADV_RANDOM : POSIX_MADV_WILLNEED);
                }
                // Register expert regions for the forced evictor (see llama_temporal_evict).
                // An *_exps tensor holds ALL experts for a layer, shape [.., .., n_expert],
                // so one expert is the slice n_size/n_expert at index e.
                if (is_expert && cur->ne[2] > 1) {
                    llama_temporal_register_experts((uint8_t *) data, n_size, (int) cur->ne[2],
                                                   weight->offs, files.at(weight->idx)->file_id());
                } else {
                    // everything that is not an expert is hot: needed every token
                    llama_temporal_register_hot((uint8_t *) data, n_size);
                }
            }
#endif

            if (check_tensors) {
                validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                    return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                }));
            }

            GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
            if (buf_mmap && cur->data == nullptr) {
                ggml_backend_tensor_alloc(buf_mmap, cur, data);
                if (lmlocks) {
                    const auto & lmlock = lmlocks->at(weight->idx);
                    lmlock->grow_to(weight->offs + n_size);
                }

                auto & mmap_used = mmaps_used[weight->idx];
                mmap_used.first  = std::min(mmap_used.first,  weight->offs);
                mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
            } else {
                ggml_backend_tensor_set(cur, data, 0, n_size);
            }
        } else {
            const auto & file = files.at(weight->idx);

            // temporal repack-stream: CPU_REPACK expert tensors are not flagged
            // is_host, so they would fall to the GPU-upload path below. Route them
            // through the pool path too -- their data is ordinary host memory the pool
            // preads/madvises, and mul_mat_id takes the fast repacked GEMM because the
            // buffer type is CPU_REPACK.
            const bool tm_repack_exps =
                (strstr(cur->name, "_exps") != nullptr && cur->ne[2] > 1) &&
                ggml_backend_buffer_get_type(cur->buffer) != nullptr &&
                strcmp(ggml_backend_buft_name(ggml_backend_buffer_get_type(cur->buffer)), "CPU_REPACK") == 0;
            if (ggml_backend_buffer_is_host(cur->buffer) || tm_repack_exps) {
#if defined(__linux__)
                // temporal slot-pool: with --mmap 0 and experts overridden to a plain
                // CPU buffer (-ot "_exps=CPU"), expert tensors land here in anonymous
                // memory. Register them for explicit pread/madvise residency control.
                // fd is dup'd because this llama_file closes when the loader is
                // destroyed but the pool lives for the process (the §4/S2-3 lesson).
                //
                // LAZY LOAD: when the pool will manage this tensor with R < n_expert,
                // do NOT read the expert data here -- experts start ABSENT and are
                // fetched on first use. Reading 5.3 GiB of soon-evicted experts is a
                // pointless load-time transient, and on a 7.7 GB-RAM device it drove
                // the system into an unkillable-OOM KERNEL PANIC (Pixel 10a, pstore:
                // "System is deadlocked on memory").
                const bool is_exps = strstr(cur->name, "_exps") != nullptr && cur->ne[2] > 1;
                bool lazy = false;
                if (is_exps) {
                    const char * r = getenv("LLAMA_TEMPORAL_R");
                    lazy = r && atoi(r) < (int) cur->ne[2];
                }
                if (!lazy) {
                    if (tm_repack_exps) {
                        // CPU_REPACK tensors must go through the buffer's set_tensor, which
                        // applies the interleaving repack. Reading raw gguf bytes straight
                        // into cur->data would leave PLAIN Q4_0 in a buffer whose kernel
                        // expects the repacked layout -- fast but numerically garbage.
                        std::vector<uint8_t> tmp(n_size);
                        file->seek(weight->offs, SEEK_SET);
                        file->read_raw(tmp.data(), n_size);
                        ggml_backend_tensor_set(cur, tmp.data(), 0, n_size);
                    } else {
                        file->seek(weight->offs, SEEK_SET);
                        file->read_raw(cur->data, n_size);
                    }
                }
                if (is_exps) {
                    // Fetch source: the gguf by default, or a pre-repacked side-file
                    // (same per-expert byte layout) when streaming repacked experts.
                    // The pool derives its fetch path via readlink on the first fd, so
                    // registering the side-file's fd points all fetches at it.
                    int reg_fd;
                    const char * side = getenv("LLAMA_TEMPORAL_REPACK_FILE");
                    if (side && side[0]) {
                        // same rule the dump tool used: the fused [gate|up|down] region
                        // starts at round_up_4096(gguf_size + 4096). No index file needed.
                        ggml_temporal_pool_set_fused_base(((size_t) file->size() + 4096 + 4095) & ~(size_t) 4095);
                    }
                    if (side && side[0]) {
                        static int side_fd = open(side, O_RDONLY);
                        if (side_fd < 0) {
                            throw std::runtime_error(format("failed to open LLAMA_TEMPORAL_REPACK_FILE '%s': %s", side, strerror(errno)));
                        }
                        reg_fd = dup(side_fd);
                    } else {
                        reg_fd = dup(file->file_id());
                    }
                    // Side-file slices are 4K-aligned (same rule the dump tool applies) so
                    // the pool can O_DIRECT DMA straight into the slot with no bounce memcpy.
                    const size_t fetch_off = (side && side[0])
                        ? ((weight->offs + 4095) & ~(size_t) 4095)
                        : weight->offs;
                    ggml_temporal_pool_register(cur->data, n_size, (int) cur->ne[2],
                                                reg_fd, fetch_off, cur->name);
                }
#else
                file->seek(weight->offs, SEEK_SET);
                file->read_raw(cur->data, n_size);
#endif
                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                    }));
                }
            } else {
                // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                if (upload_backend) {
                    size_t offset = weight->offs;
                    alignment = file->read_alignment();
                    size_t aligned_offset = offset & ~(alignment - 1);
                    size_t offset_from_alignment = offset - aligned_offset;
                    file->seek(aligned_offset, SEEK_SET);

                    // Calculate aligned read boundaries
                    size_t read_start = aligned_offset;
                    size_t read_end = (offset + n_size + alignment - 1) & ~(alignment - 1);

                    size_t bytes_read = 0;
                    size_t data_read = 0;  // Actual tensor data copied (excluding padding)

                    while (bytes_read < read_end - read_start) {
                        size_t read_size = std::min<size_t>(buffer_size, read_end - read_start - bytes_read);

                        // Align the destination pointer within the pinned buffer
                        uintptr_t ptr_dest_aligned = (reinterpret_cast<uintptr_t>(host_ptrs[buffer_idx]) + alignment - 1) & ~(alignment - 1);

                        // Wait for previous upload to complete before reusing buffer
                        ggml_backend_event_synchronize(events[buffer_idx]);

                        // Read aligned chunk from file
                        file->read_raw_unsafe(reinterpret_cast<void *>(ptr_dest_aligned), read_size);

                        // Calculate actual data portion (excluding alignment padding)
                        uintptr_t ptr_data = ptr_dest_aligned;
                        size_t data_to_copy = read_size;

                        // Skip alignment padding at start of first chunk
                        if (bytes_read == 0) {
                            ptr_data += offset_from_alignment;
                            data_to_copy -= offset_from_alignment;
                        }

                        // Trim alignment padding at end of last chunk
                        if (aligned_offset + bytes_read + read_size > offset + n_size) {
                            data_to_copy -= (read_end - (offset + n_size));
                        }

                        // Async upload actual data to GPU
                        ggml_backend_tensor_set_async(upload_backend, cur,
                                                      reinterpret_cast<void *>(ptr_data), data_read, data_to_copy);
                        ggml_backend_event_record(events[buffer_idx], upload_backend);

                        data_read += data_to_copy;
                        bytes_read += read_size;

                        ++buffer_idx;
                        buffer_idx %= n_buffers;
                    }
                } else {
                    read_buf.resize(n_size);
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(read_buf.data(), n_size);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                    if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                        throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                    }
                }
            }
        }

        size_done += n_size;
    }

    // free temporary resources used for async uploads
    for (auto * event : events) {
        ggml_backend_event_synchronize(event);
        ggml_backend_event_free(event);
    }
    for (auto * buf : host_buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_backend_free(upload_backend);

    // check validation results
    bool validation_failed = false;
    for (auto & future : validation_result) {
        auto result = future.get();
        if (!result.second) {
            LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
            validation_failed = true;
        }
    }
    if (validation_failed) {
        throw std::runtime_error("found tensors with invalid data");
    }

    // check if this is the last call and do final cleanup
    if (size_done >= size_data) {
        // unmap offloaded tensors and metadata
        if (use_mmap) {
            for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                const auto & mmap_used = mmaps_used.at(idx);
                auto & mapping = mappings.at(idx);
                mapping->unmap_fragment(0, mmap_used.first);
                if (mmap_used.second != 0) {
                    mapping->unmap_fragment(mmap_used.second, mapping->size());
                }
            }
        }
#if defined(_POSIX_MAPPED_FILES)
        // Start the residency controller HERE, on the load-complete path only.
        // Two bugs lived in the placement of this call:
        //  - the tail call below is skipped entirely when a progress_callback is set
        //    (early return above), so with a single buffer type the controller never
        //    started at all;
        //  - load_all_data runs once per ggml context, and an unguarded tail call let a
        //    NON-final call start the controller with a partial region list (observed:
        //    23 of 135 expert tensors -- only the q6_K down_exps -- under control).
        // Starting only when size_done >= size_data guarantees the snapshot sees every
        // registered region.
        llama_temporal_start_evictor();
#endif
        if (progress_callback) {
            // Even though the model is done loading, we still honor
            // cancellation since we need to free allocations.
            return progress_callback(1.0f, progress_callback_user_data);
        }
    }

    return true;
}

std::string llama_model_loader::ftype_name() const {
    return llama_ftype_name(ftype);
}

void llama_model_loader::print_info() const {
    LLAMA_LOG_INFO("%s: file format = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: file type   = %s\n", __func__, llama_ftype_name(ftype));
    if (n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: file size   = %.2f MiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0,        n_bytes*8.0/n_elements);
    } else {
        LLAMA_LOG_INFO("%s: file size   = %.2f GiB (%.2f BPW) \n", __func__, n_bytes/1024.0/1024.0/1024.0, n_bytes*8.0/n_elements);
    }
}

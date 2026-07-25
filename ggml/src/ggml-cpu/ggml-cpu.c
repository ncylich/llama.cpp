#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// temporal expert slot-pool -- the Android analogue of the A6000 CUDA path
// (cudaMemcpyAsync into a fixed pool + TEMPORAL_SWAP_PROB-prescribed turnover).
//
// Expert tensors live in ANONYMOUS memory (--mmap 0, forced to plain CPU buft so they
// are not repacked). Residency is controlled explicitly:
//   evict = madvise(MADV_DONTNEED) on the expert's anonymous range (frees the pages;
//           reliable, unlike page-cache eviction which the kernel can ignore or exceed)
//   fetch = pread() the expert's exact bytes from the GGUF back into place BEFORE the
//           op computes on it -- numerics are bit-identical to the fully-resident run
//           by construction, and bytes-per-token is the sum of pread sizes, exactly.
//
// Fetches are ASYNC on a worker pool (measured on-device: 216 KiB random O_DIRECT reads
// scale 0.86 -> 2.37 GB/s from QD1 to QD8; a single synchronous pread wastes ~2.7x of
// the device). Two forms of same-token overlap, no speculation:
//   1. queue depth: all experts missing for THIS op are fetched concurrently;
//   2. sibling prefetch: gate/up/down of one layer share one routing decision (the same
//      ids tensor), so when the first of the trio learns the needed set it enqueues the
//      other two tensors' missing experts as well -- their IO hides behind the trio's
//      compute. The op still waits for ITS OWN experts before computing: same-token
//      swap, fetched-expert compute happens after its bytes land.
//
//   LLAMA_TEMPORAL_R=<r>          experts kept resident per expert tensor.
//                                 r >= n_expert  -> ceiling (no traffic)
//                                 r <  top_k     -> streamed (working set evicted and
//                                                   re-fetched every op; r=0 canonical)
//                                 top_k <= r < E -> temporal window (FIFO)
//   LLAMA_TEMPORAL_SWAP_PROB=<p>  per needed expert per op, probability it was force-
//                                 evicted since last use -> real re-fetch. Same
//                                 semantics as the CUDA TEMPORAL_SWAP_PROB.
//   LLAMA_TEMPORAL_FETCH_THREADS=<n>  fetch workers (default 8, max 16; 1 = the old
//                                 synchronous behaviour, for A/B)
//   LLAMA_TEMPORAL_SIBLING_PREFETCH=0 disable form-2 overlap (default on)
//   LLAMA_TEMPORAL_ODIRECT=1      O_DIRECT fetches (bypass page cache by construction)
//   unset LLAMA_TEMPORAL_R        pool inactive (registration is harmless).
#if defined(__linux__)
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>

#ifndef RWF_HIPRI
#define RWF_HIPRI 0x00000001   // polled completion: skip the interrupt+wakeup path
#endif

enum ggml_tm_state {
    GGML_TM_ABSENT   = 0,
    GGML_TM_RESIDENT = 1,
    GGML_TM_FETCHING = 2,   // a worker is pread()ing it
    GGML_TM_EVICTING = 3,   // queued for the janitor's madvise; not fetchable until ABSENT
};

struct ggml_tm_pool_tensor {
    void    * data;
    size_t    nbytes;
    int       n_experts;
    size_t    expert_bytes;
    int       fd;            // dup'd by the loader; owned by the pool for process life
    size_t    file_off;
    int       layer_id;      // parsed from "blk.<n>." in the tensor name; -1 = no group
    int       slot;          // 0=gate 1=up 2=down (-1 other), parsed from name
    uint8_t * state;         // [n_experts] ggml_tm_state
    uint8_t * needed;        // experts referenced by the in-flight op
    int     * order;         // [n_experts] compute order for the in-flight op:
                             // resident-needed first, fetching-needed last -- so the op
                             // computes on-hand experts while the missing ones stream in
    int     * fifo;          // fetch-order ring, for the R-window trim
    int       fifo_head;
    int       fifo_len;
    int       n_resident;    // count of state==RESIDENT
    uint64_t  op_seq;        // ops seen by this tensor; stamps last_use
    uint64_t* last_use;      // [n_experts] op_seq at last reference -- LRU eviction.
                             // FIFO evicted intermittently-reused experts before their
                             // next use and sustained ~99 fetches/token even at R=48.
    uint8_t * pending;       // [n_experts] sub-reads still in flight for a FETCHING
                             // expert (split fetch: 108 KiB halves complete in ~435 us
                             // at QD8 vs ~737 us for one 216 KiB read -- probed)
    uint8_t * evict_pending; // [n_experts] evict requested while still FETCHING; the
                             // fetch completion evicts on arrival. Without this, a swap
                             // that evicts an in-flight expert leaks it resident (the
                             // window drops it from tracking) -- exposed once compute
                             // outran fetch latency (repacked kernel), residency crept up.
};

static struct ggml_tm_pool_tensor g_tm_pool[512];
static int             g_tm_pool_n = 0;
static pthread_mutex_t g_tm_mtx     = PTHREAD_MUTEX_INITIALIZER;  // pool state + queue
static pthread_cond_t  g_tm_work_cv = PTHREAD_COND_INITIALIZER;   // workers: queue non-empty
static pthread_cond_t  g_tm_done_cv = PTHREAD_COND_INITIALIZER;   // ensure: a fetch landed
static _Atomic uint64_t g_tm_fetches;
static _Atomic uint64_t g_tm_fetched_bytes;
static _Atomic uint64_t g_tm_evictions;
static int      g_tm_R = -1;          // -1 = pool inactive
static double   g_tm_swap_prob = 0.0;
static uint64_t g_tm_rng = 0x9E3779B97F4A7C15ull;

static _Atomic uint64_t g_tm_hook_calls;
static _Atomic uint64_t g_tm_hook_miss;   // op ran on a tensor the pool doesn't know
static _Atomic uint64_t g_tm_fetch_ns;    // summed IO time across workers (per-read)
static _Atomic uint64_t g_tm_wait_ns;     // time the compute thread actually BLOCKED in
                                          // ensure -- the true per-op stall; with full
                                          // overlap this goes to ~0 while fetch_ns stays
static bool      g_tm_odirect  = false;
static int       g_tm_nworkers = 8;
static bool      g_tm_sibling  = true;
static char      g_tm_path[1024] = {0};   // the gguf path; workers open their own fds

// fetch queue: {tensor idx, expert} ring. own-needed entries are enqueued ahead of
// sibling prefetches (FIFO workers -> the blocking set completes first).
#define GGML_TM_QCAP 8192
// single ring, per-item class: HI(0) = slices an op blocks on; LO(1) = sibling
// prefetch. Workers pop the oldest HI item, else the oldest LO. ensure() PROMOTES a
// still-queued LO item to HI the moment an op actually needs it -- without promotion,
// prioritization delays the sibling ops it was meant to protect.
static struct { int ti; int e; int part; int cls; uint64_t enq_ns; } g_tm_q[GGML_TM_QCAP];
static int g_tm_q_head = 0, g_tm_q_n = 0;   // ring window [head, head+n) under mutex
static _Atomic int g_tm_q_len = 0;          // == n; atomic for unlocked spin-peek
static _Atomic uint64_t g_tm_qwait_hi_ns;   // enqueue->pop for HI items (the stall path)
static _Atomic uint64_t g_tm_qhi_n;
static _Atomic int g_tm_spinners = 0;   // at most 2 workers busy-poll the queue: 6 of 8
                                        // cores run GEMVs; more spinners would fight
                                        // compute instead of cutting wakeup latency
// enforced 1-swap policy globals (defined here so ggml_tm_pool_report can read them)
#define GGML_TM_MAXLAYER 128
#define GGML_TM_MAXK     512
static bool     g_tm_enforce = false;
static bool     g_tm_twopass = false;   // split FFN into resident(K-1)+new(1) sub-passes
static int      g_tm_ewin[GGML_TM_MAXLAYER][GGML_TM_MAXK];
static int      g_tm_ewin_k[GGML_TM_MAXLAYER];
static uint8_t  g_tm_ein[GGML_TM_MAXLAYER][8192];
static uint64_t g_tm_ernd[GGML_TM_MAXLAYER];
static _Atomic uint64_t g_tm_swaps;

// --- execution trace: 2D timeline (lane x time) of the expert path -----------
// LLAMA_TEMPORAL_TRACE=1 records timestamped spans. Lanes: compute thread ith (0..7),
// fetch worker w (100+w), janitor (200). Dumped as chrome-trace JSON at process exit
// (loadable in perfetto/chrome://tracing; also rendered to a static swimlane offline).
struct tm_ev { double ts, dur; int lane, type, layer, expert; };
#define TM_TRACE_MAX 800000
static struct tm_ev    g_tm_trace[TM_TRACE_MAX];
static _Atomic int     g_tm_trace_n;
static bool            g_tm_trace_on = false;
static struct timespec g_tm_trace_t0;
static inline double tm_us(const struct timespec * t) {
    return (double)(t->tv_sec - g_tm_trace_t0.tv_sec) * 1e6
         + (double)(t->tv_nsec - g_tm_trace_t0.tv_nsec) / 1e3;   // microseconds
}
static inline double tm_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return tm_us(&t); }
static inline void tm_ev(double ts, double dur, int lane, int type, int layer, int expert) {
    if (!g_tm_trace_on) return;
    int i = atomic_fetch_add(&g_tm_trace_n, 1);
    if (i < TM_TRACE_MAX) g_tm_trace[i] = (struct tm_ev){ ts, dur, lane, type, layer, expert };
}
static void tm_trace_dump(void) {
    if (!g_tm_trace_on) return;
    const char * f = getenv("LLAMA_TEMPORAL_TRACE_FILE");
    if (!f) f = "/data/local/tmp/tmoe/trace.json";
    FILE * fp = fopen(f, "w"); if (!fp) return;
    int n = atomic_load(&g_tm_trace_n); if (n > TM_TRACE_MAX) n = TM_TRACE_MAX;
    static const char * tn[] = { "GEMV", "WAIT", "FETCH", "EVICT", "ENSURE", "ROUTER" };
    fprintf(fp, "[\n");
    for (int i = 0; i < n; i++) {
        struct tm_ev * e = &g_tm_trace[i];
        double d = e->dur < 0.05 ? 0.05 : e->dur;
        fprintf(fp, "%s{\"name\":\"%s L%d e%d\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":%d}",
                i ? ",\n" : "", tn[e->type % 6], e->layer, e->expert, tn[e->type % 6], e->ts, d, e->lane);
    }
    fprintf(fp, "\n]\n"); fclose(fp);
    fprintf(stderr, "temporal-trace: wrote %d events to %s\n", n, f);
}
// Fetch phase accounting (LLAMA_TEMPORAL_FETCHPROF=1). Answers, from inside the engine:
// how many syscalls does one expert fetch take, how long is each, and how much wall time
// sits OUTSIDE the syscalls (setup / iovec rebuild / accounting). Built because three
// standalone look-alike harnesses each disagreed with the engine for a different reason.
static _Atomic uint64_t g_tm_pf_calls;      // preadv/pread syscalls issued
static _Atomic uint64_t g_tm_pf_fetches;    // expert fetches profiled
static _Atomic uint64_t g_tm_pf_sys_ns;     // time inside syscalls
static _Atomic uint64_t g_tm_pf_wall_ns;    // total fetch wall time
static _Atomic uint64_t g_tm_pf_first_ns;   // time inside the FIRST syscall of a fetch
static _Atomic uint64_t g_tm_pf_max_ns;     // slowest single syscall seen
static _Atomic uint64_t g_tm_pf_short;      // syscalls that returned less than asked
static bool   g_tm_fetchprof = false;
static bool   g_tm_fused = false;    // LLAMA_TEMPORAL_FUSED: one preadv per expert swap
static size_t g_tm_fused_base = 0;  // byte offset of the fused region in the side-file
// Fused layout: for layer L, expert e the three slices sit contiguously as
// [gate | up | down], each expert_bytes long, so one 648 KiB request replaces six 108 KiB
// ones. Measured (S3-28b, single-burst wall time for one expert): 6x108 KiB = 828 us mean /
// 3248 us worst, 1x648 KiB preadv = 678 us mean / 881 us worst.
static inline size_t ggml_tm_fused_off(int layer, int e, int n_experts, size_t expert_bytes) {
    return g_tm_fused_base + ((size_t) layer * (size_t) n_experts + (size_t) e) * 3 * expert_bytes;
}
// find the pool entry for (layer, slot); slot 0=gate 1=up 2=down
static int ggml_tm_find(int layer, int slot) {
    for (int i = 0; i < g_tm_pool_n; i++) {
        if (g_tm_pool[i].layer_id == layer && g_tm_pool[i].slot == slot) return i;
    }
    return -1;
}
static int g_tm_max_spinners = 2;   // LLAMA_TEMPORAL_SPINNERS: workers allowed to spin-poll
static int g_tm_inflight = 0;       // fetch parts dequeued but not yet completed
static pthread_cond_t g_tm_quiet_cv = PTHREAD_COND_INITIALIZER;  // signalled when they drain
static int g_tm_evict_defer = -1;   // LLAMA_TEMPORAL_EVICT_DEFER: hold evictions until quiet
static int g_tm_split = 1;   // sub-reads per expert fetch (LLAMA_TEMPORAL_SPLIT, 1..4).
                             // Default 1: split=2 measured WORSE in all interleaved
                             // pairs (S2-16) -- the halved request size costs more
                             // bandwidth than the added parallelism recovers in the
                             // shallow per-layer bursts this workload produces.
static bool g_tm_workers_started = false;

// janitor queue: eviction madvise work, moved OFF the compute thread. Measured before
// this existed: ~430 trim evictions/token x ~20us of madvise = ~9ms/token of critical-
// path stall. The evict DECISION stays in ensure (state -> EVICTING, under lock); the
// page-freeing madvise happens here. If the janitor finds the expert is needed by the
// in-flight op once freed (swap-prob forced turnover, streamed evict-all), it resubmits
// the fetch itself, closing the evict->refetch loop.
static struct { int ti; int e; } g_tm_jq[GGML_TM_QCAP];
static int g_tm_jq_head = 0, g_tm_jq_len = 0;
static pthread_cond_t g_tm_jan_cv = PTHREAD_COND_INITIALIZER;

static void ggml_tm_pool_report(void) {
    uint64_t f = atomic_load(&g_tm_fetches);
    if (g_tm_fetchprof) {
        uint64_t nf = atomic_load(&g_tm_pf_fetches);
        if (nf) {
            fprintf(stderr, "temporal-fetchprof: fetches=%llu syscalls/fetch=%.2f "
                    "wall/fetch=%.0fus sys/fetch=%.0fus first_call=%.0fus outside_sys=%.0fus "
                    "max_call=%.0fus short_reads=%llu\n",
                    (unsigned long long) nf,
                    (double) atomic_load(&g_tm_pf_calls) / (double) nf,
                    atomic_load(&g_tm_pf_wall_ns) / 1e3 / nf,
                    atomic_load(&g_tm_pf_sys_ns) / 1e3 / nf,
                    atomic_load(&g_tm_pf_first_ns) / 1e3 / nf,
                    (atomic_load(&g_tm_pf_wall_ns) - atomic_load(&g_tm_pf_sys_ns)) / 1e3 / nf,
                    atomic_load(&g_tm_pf_max_ns) / 1e3,
                    (unsigned long long) atomic_load(&g_tm_pf_short));
        }
    }
    fprintf(stderr, "temporal-pool: fetches=%llu fetched_mib=%.1f evictions=%llu "
            "tensors=%d hook_calls=%llu hook_miss=%llu avg_fetch_us=%.1f qwait_hi_us=%.1f wait_ms=%.1f odirect=%d workers=%d sibling=%d split=%d\n",
            (unsigned long long) f,
            (double) atomic_load(&g_tm_fetched_bytes) / (1024.0 * 1024.0),
            (unsigned long long) atomic_load(&g_tm_evictions),
            g_tm_pool_n,
            (unsigned long long) atomic_load(&g_tm_hook_calls),
            (unsigned long long) atomic_load(&g_tm_hook_miss),
            f ? (double) atomic_load(&g_tm_fetch_ns) / 1e3 / (double) f : 0.0,
            atomic_load(&g_tm_qhi_n) ? (double) atomic_load(&g_tm_qwait_hi_ns) / 1e3 / (double) atomic_load(&g_tm_qhi_n) : 0.0,
            (double) atomic_load(&g_tm_wait_ns) / 1e6,
            g_tm_odirect ? 1 : 0, g_tm_nworkers, g_tm_sibling ? 1 : 0, g_tm_split);
    if (g_tm_enforce) {
        fprintf(stderr, "temporal-pool: ENFORCE on, swaps=%llu\n",
                (unsigned long long) atomic_load(&g_tm_swaps));
    }
}

// worker: pop a fetch, do the IO with a private fd + bounce buffer (no lock held),
// then mark the expert resident and wake any waiter.
// Pin an IO thread to the cores named in LLAMA_TEMPORAL_WORKER_AFFINITY ("lo-hi").
// On asymmetric topologies (Pixel 10a: 1 prime + 3 mid + 4 little) the fetch workers
// otherwise wake on big cores and preempt the GEMV threads -- submission work is
// trivial and IO wait burns no CPU, so the little cores are the right home.
static void ggml_tm_set_io_affinity(void) {
    const char * s = getenv("LLAMA_TEMPORAL_WORKER_AFFINITY");
    if (!s || !*s) return;
    int lo = -1, hi = -1;
    if (sscanf(s, "%d-%d", &lo, &hi) != 2 || lo < 0 || hi < lo || hi >= 64) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c = lo; c <= hi; c++) { CPU_SET(c, &set); }
    sched_setaffinity(0, sizeof(set), &set);   // 0 = calling thread on Linux
}

static void ggml_tm_evict(struct ggml_tm_pool_tensor * t, int e);  // fwd: evict-on-arrival

static void * ggml_tm_worker(void * arg) {
    const int wid = (int)(intptr_t) arg;
    ggml_tm_set_io_affinity();
    int fd = open(g_tm_path, O_RDONLY | (g_tm_odirect ? O_DIRECT : 0));
    if (fd < 0) {
        fprintf(stderr, "temporal-pool: FATAL worker open(%s) failed: %s\n", g_tm_path, strerror(errno));
        abort();
    }
    // IO priority: never queue our fetches behind background system IO. Try RT class
    // first (needs privilege, usually refused on Android), fall back to best-effort 0.
    // ioprio_set(IOPRIO_WHO_PROCESS=1, 0=calling thread, (class<<13)|level)
    if (syscall(__NR_ioprio_set, 1, 0, (1 << 13) | 0) != 0) {
        syscall(__NR_ioprio_set, 1, 0, (2 << 13) | 0);
    }
    uint8_t * bounce = NULL;
    size_t bounce_sz = 0;
    for (;;) {
        // bounded spin-peek before sleeping: during decode a fetch burst arrives every
        // few hundred us, and the futex sleep/wake round-trip would sit at the FRONT of
        // every fetch's latency. At most 2 workers spin (the cores compute doesn't use).
        if (atomic_load_explicit(&g_tm_q_len, memory_order_relaxed) == 0) {
            // How many workers may spin-poll. Everyone above the cap sleeps on the condvar
            // and pays a futex wake + scheduler dispatch, which lands at the FRONT of that
            // part's latency -- and a layer waits on max(start+duration), so a late start
            // is pure critical path. This cap was hard-coded to 2, which is exactly why
            // 2 parts start immediately and the rest ~150 us later. LLAMA_TEMPORAL_SPINNERS.
            if (atomic_fetch_add(&g_tm_spinners, 1) < g_tm_max_spinners) {
                for (int i = 0; i < 60000; i++) {   // ~200-400 us of polling
                    if (atomic_load_explicit(&g_tm_q_len, memory_order_relaxed) > 0) break;
                }
            }
            atomic_fetch_sub(&g_tm_spinners, 1);
        }
        pthread_mutex_lock(&g_tm_mtx);
        while (g_tm_q_len == 0) {
            pthread_cond_wait(&g_tm_work_cv, &g_tm_mtx);
        }
        // oldest HI item, else oldest LO (O(n) scan; n is small in decode)
        int pick = -1;
        for (int k = 0; k < g_tm_q_n; k++) {
            int idx = (g_tm_q_head + k) % GGML_TM_QCAP;
            if (g_tm_q[idx].cls == 0) { pick = k; break; }
            if (pick < 0) pick = k;   // fallback: oldest item of any class
        }
        int idx  = (g_tm_q_head + pick) % GGML_TM_QCAP;
        int ti   = g_tm_q[idx].ti;
        int e    = g_tm_q[idx].e;
        int part = g_tm_q[idx].part;
        if (g_tm_q[idx].cls == 0) {
            struct timespec tp; clock_gettime(CLOCK_MONOTONIC, &tp);
            uint64_t nowp = (uint64_t) tp.tv_sec * 1000000000ull + tp.tv_nsec;
            atomic_fetch_add(&g_tm_qwait_hi_ns, nowp - g_tm_q[idx].enq_ns);
            atomic_fetch_add(&g_tm_qhi_n, 1);
            // QWAIT span: submit -> dequeue for THIS part, so the timeline shows exactly
            // how long each part sat in the queue before a worker picked it up.
            if (g_tm_trace_on) {
                uint64_t t0ns = (uint64_t) g_tm_trace_t0.tv_sec * 1000000000ull
                              + (uint64_t) g_tm_trace_t0.tv_nsec;
                double enq_us = (double)(g_tm_q[idx].enq_ns - t0ns) / 1e3;
                double deq_us = (double)(nowp - t0ns) / 1e3;
                tm_ev(enq_us, deq_us - enq_us, 100 + wid, 5 /*QWAIT*/,
                      g_tm_pool[ti].layer_id, e);
            }
        }
        // remove idx by shifting the gap toward head (order otherwise preserved)
        for (int k = pick; k > 0; k--) {
            int dst = (g_tm_q_head + k) % GGML_TM_QCAP;
            int src = (g_tm_q_head + k - 1) % GGML_TM_QCAP;
            g_tm_q[dst] = g_tm_q[src];
        }
        g_tm_q_head = (g_tm_q_head + 1) % GGML_TM_QCAP;
        g_tm_q_n--;
        g_tm_q_len--;
        g_tm_inflight++;
        struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
        pthread_mutex_unlock(&g_tm_mtx);

        size_t need = t->expert_bytes + 2 * 4096;
        if (bounce_sz < need) {
            free(bounce);
            if (posix_memalign((void **) &bounce, 4096, need) != 0) { abort(); }
            bounce_sz = need;
        }

        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        // sub-range [plo, phi) of the expert, page-aligned interior cut points
        size_t    step = (t->expert_bytes / (size_t) g_tm_split) & ~(size_t) 4095;
        size_t    plo  = (size_t) part * step;
        size_t    phi  = (part == g_tm_split - 1) ? t->expert_bytes : plo + step;
        size_t    ebeg = t->file_off + (size_t) e * t->expert_bytes + plo;
        uint8_t * dst  = (uint8_t *) t->data + (size_t) e * t->expert_bytes + plo;
        size_t    sub_bytes = phi - plo;
        // FUSED fetch: one request delivers the whole [gate|up|down] triple, scattered by
        // preadv directly into the three destination slots. 648 KiB in one device request
        // instead of six 108 KiB ones: 828 -> 678 us mean, 3248 -> 881 us worst (S3-28b).
        if (g_tm_fused) {
            const int L = t->layer_id;
            int fti[3] = { ggml_tm_find(L, 0), ggml_tm_find(L, 1), ggml_tm_find(L, 2) };
            struct iovec iov[3];
            bool ok = (fti[0] >= 0 && fti[1] >= 0 && fti[2] >= 0);
            for (int k = 0; ok && k < 3; k++) {
                uint8_t * d = (uint8_t *) g_tm_pool[fti[k]].data
                            + (size_t) e * g_tm_pool[fti[k]].expert_bytes;
                iov[k].iov_base = d;
                iov[k].iov_len  = g_tm_pool[fti[k]].expert_bytes;
                if (((uintptr_t) d) % 4096 != 0) { ok = false; }
            }
            size_t foff = ggml_tm_fused_off(L, e, t->n_experts, t->expert_bytes);
            if (!ok || foff % 4096 != 0) {
                fprintf(stderr, "temporal-pool: FATAL fused fetch misaligned (L=%d e=%d off=%zu)\n", L, e, foff);
                abort();
            }
            struct timespec fs0, fs1;
            clock_gettime(CLOCK_MONOTONIC, &fs0);
            size_t total = 3 * t->expert_bytes, done = 0;
            uint64_t pf_sys = 0, pf_first = 0; int pf_n = 0;
            while (done < total) {
                struct iovec cur[3]; int nv = 0; size_t skip = done;
                for (int k = 0; k < 3; k++) {
                    if (skip >= iov[k].iov_len) { skip -= iov[k].iov_len; continue; }
                    cur[nv].iov_base = (uint8_t *) iov[k].iov_base + skip;
                    cur[nv].iov_len  = iov[k].iov_len - skip;
                    nv++; skip = 0;
                }
                struct timespec c0, c1;
                if (g_tm_fetchprof) { clock_gettime(CLOCK_MONOTONIC, &c0); }
                ssize_t r = preadv(fd, cur, nv, (off_t)(foff + done));
                if (g_tm_fetchprof) {
                    clock_gettime(CLOCK_MONOTONIC, &c1);
                    uint64_t dt = (uint64_t)(c1.tv_sec - c0.tv_sec) * 1000000000ull
                                + (uint64_t)(c1.tv_nsec - c0.tv_nsec);
                    pf_sys += dt; if (pf_n == 0) { pf_first = dt; }
                    pf_n++;
                    size_t asked = 0; for (int q = 0; q < nv; q++) { asked += cur[q].iov_len; }
                    if (r > 0 && (size_t) r < asked) { atomic_fetch_add(&g_tm_pf_short, 1); }
                    uint64_t prev = atomic_load(&g_tm_pf_max_ns);
                    while (dt > prev && !atomic_compare_exchange_weak(&g_tm_pf_max_ns, &prev, dt)) { }
                }
                if (r <= 0) {
                    fprintf(stderr, "temporal-pool: FATAL fused preadv failed (off=%zu): %s\n",
                            foff + done, strerror(errno));
                    abort();
                }
                done += (size_t) r;
            }
            clock_gettime(CLOCK_MONOTONIC, &fs1);
            if (g_tm_fetchprof) {
                atomic_fetch_add(&g_tm_pf_calls, (uint64_t) pf_n);
                atomic_fetch_add(&g_tm_pf_fetches, 1);
                atomic_fetch_add(&g_tm_pf_sys_ns, pf_sys);
                atomic_fetch_add(&g_tm_pf_first_ns, pf_first);
                atomic_fetch_add(&g_tm_pf_wall_ns,
                    (uint64_t)(fs1.tv_sec - fs0.tv_sec) * 1000000000ull
                  + (uint64_t)(fs1.tv_nsec - fs0.tv_nsec));
            }
            atomic_fetch_add(&g_tm_fetch_ns, (uint64_t)(fs1.tv_sec - fs0.tv_sec) * 1000000000ull
                                             + (uint64_t)(fs1.tv_nsec - fs0.tv_nsec));
            atomic_fetch_add(&g_tm_fetched_bytes, total);
            tm_ev(tm_us(&fs0), tm_us(&fs1) - tm_us(&fs0), 100 + wid, 2 /*FETCH*/, L, e);
            pthread_mutex_lock(&g_tm_mtx);
            if (--g_tm_inflight == 0 && g_tm_q_n == 0) { pthread_cond_broadcast(&g_tm_quiet_cv); }
            for (int k = 0; k < 3; k++) {
                struct ggml_tm_pool_tensor * ft = &g_tm_pool[fti[k]];
                if (ft->pending[e] && --ft->pending[e] == 0) {
                    __atomic_store_n(&ft->state[e], GGML_TM_RESIDENT, __ATOMIC_RELEASE);
                    ft->n_resident++;
                    ft->fifo[(ft->fifo_head + ft->fifo_len) % ft->n_experts] = e;
                    ft->fifo_len = ft->fifo_len < ft->n_experts ? ft->fifo_len + 1 : ft->fifo_len;
                    atomic_fetch_add(&g_tm_fetches, 1);
                    if (ft->evict_pending[e]) { ft->evict_pending[e] = 0; ggml_tm_evict(ft, e); }
                }
            }
            pthread_cond_broadcast(&g_tm_done_cv);
            pthread_mutex_unlock(&g_tm_mtx);
            continue;
        }

        // ZERO-COPY fast path: when the file offset, the destination and the length are
        // all 4K-aligned, O_DIRECT can DMA straight into the slot -- no bounce, no memcpy.
        // The bounce copy is not free: 3 x 216 KiB per layer is ~66 us of memcpy that runs
        // on a worker thread WHILE the resident experts compute, and it showed up as a
        // 1.19x inflation of every concurrent GEMV (8.11 vs 6.80 us) -- i.e. it was the
        // entire remaining gap to the fully-resident baseline. Measured, S3-23.
        const bool zerocopy = g_tm_odirect
            && (ebeg % 4096 == 0)
            && (((uintptr_t) dst) % 4096 == 0)
            && (sub_bytes % 4096 == 0);
        if (zerocopy) {
            static _Atomic int announced = 0;
            if (atomic_exchange(&announced, 1) == 0) {
                fprintf(stderr, "temporal-pool: O_DIRECT zero-copy fetch ENABLED (no bounce memcpy)\n");
            }
            size_t left = sub_bytes;
            off_t  fo   = (off_t) ebeg;
            uint8_t * dp = dst;
            uint64_t zp_sys = 0, zp_first = 0; int zp_n = 0;
            while (left > 0) {
                struct timespec c0, c1;
                if (g_tm_fetchprof) { clock_gettime(CLOCK_MONOTONIC, &c0); }
                ssize_t r = pread(fd, dp, left, fo);
                if (g_tm_fetchprof) {
                    clock_gettime(CLOCK_MONOTONIC, &c1);
                    uint64_t dt = (uint64_t)(c1.tv_sec - c0.tv_sec) * 1000000000ull
                                + (uint64_t)(c1.tv_nsec - c0.tv_nsec);
                    zp_sys += dt; if (zp_n == 0) { zp_first = dt; }
                    zp_n++;
                    if (r > 0 && (size_t) r < left) { atomic_fetch_add(&g_tm_pf_short, 1); }
                    uint64_t prev = atomic_load(&g_tm_pf_max_ns);
                    while (dt > prev && !atomic_compare_exchange_weak(&g_tm_pf_max_ns, &prev, dt)) { }
                }
                if (r <= 0) {
                    fprintf(stderr, "temporal-pool: FATAL zero-copy pread failed (off=%lld len=%zu): %s\n",
                            (long long) fo, left, strerror(errno));
                    abort();
                }
                dp += r; fo += r; left -= (size_t) r;
            }
            if (g_tm_fetchprof) {
                atomic_fetch_add(&g_tm_pf_calls, (uint64_t) zp_n);
                atomic_fetch_add(&g_tm_pf_fetches, 1);
                atomic_fetch_add(&g_tm_pf_sys_ns, zp_sys);
                atomic_fetch_add(&g_tm_pf_first_ns, zp_first);
            }
        } else if (g_tm_odirect) {
            // O_DIRECT needs 4K-aligned offset/length/buffer: read the aligned superset
            // into the bounce buffer, copy the sub-range's exact bytes into place.
            static _Atomic int warned = 0;
            if (atomic_exchange(&warned, 1) == 0) {
                fprintf(stderr, "temporal-pool: O_DIRECT bounce path (off%%4096=%zu dst%%4096=%zu len%%4096=%zu)"
                                " -- costs a memcpy per fetch\n",
                        ebeg % 4096, ((uintptr_t) dst) % 4096, sub_bytes % 4096);
            }
            size_t abeg = ebeg & ~(size_t) 4095;
            size_t aend = (ebeg + sub_bytes + 4095) & ~(size_t) 4095;
            size_t left = aend - abeg;
            uint8_t * bp = bounce;
            off_t     fo = (off_t) abeg;
            // RWF_HIPRI (polled completion) saves the interrupt+wakeup tail when the
            // block driver supports poll queues; detected once, silent fallback if not.
            static _Atomic int hipri = -1;   // -1 probe, 1 use, 0 unsupported
            while (left > 0) {
                ssize_t r = -1;
                int h = atomic_load(&hipri);
                if (h != 0) {
                    struct iovec iov = { bp, left };
                    // pos_l/pos_h: the kernel assembles pos as (hi << 64)|lo on 64-bit
                    // ABIs, i.e. pos_l must carry the FULL offset and pos_h must be 0.
                    // Splitting lo/hi 32-bit style truncates offsets >= 4 GiB and reads
                    // the wrong file region -- caught by the PPL gate as nan.
                    r = syscall(__NR_preadv2, fd, &iov, 1,
                                (unsigned long) fo, 0ul, RWF_HIPRI);
                    if (r < 0 && h == -1) { atomic_store(&hipri, 0); }
                    else if (h == -1)     { atomic_store(&hipri, 1); }
                }
                if (r < 0) {
                    r = pread(fd, bp, left, fo);
                }
                if (r <= 0) {
                    fprintf(stderr, "temporal-pool: FATAL O_DIRECT pread failed (off=%lld len=%zu): %s\n",
                            (long long) fo, left, strerror(errno));
                    abort();   // computing on stale/zero weights must never be silent
                }
                bp += r; fo += r; left -= (size_t) r;
            }
            memcpy(dst, bounce + (ebeg - abeg), sub_bytes);
        } else {
            size_t left = sub_bytes;
            off_t  fo   = (off_t) ebeg;
            uint8_t * dp = dst;
            while (left > 0) {
                ssize_t r = pread(fd, dp, left, fo);
                if (r <= 0) {
                    fprintf(stderr, "temporal-pool: FATAL pread failed (off=%lld): %s\n",
                            (long long) fo, strerror(errno));
                    abort();
                }
                dp += r; fo += r; left -= (size_t) r;
            }
            // drop what the read left in the gguf's PAGE CACHE, or the next fetch of
            // this expert is a free cache hit and the regime silently stops streaming
            posix_fadvise(fd, (off_t)(ebeg & ~(size_t) 4095),
                          (off_t)(sub_bytes + 4096), POSIX_FADV_DONTNEED);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        atomic_fetch_add(&g_tm_fetch_ns, (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000ull
                                         + (uint64_t)(ts1.tv_nsec - ts0.tv_nsec));
        atomic_fetch_add(&g_tm_fetched_bytes, sub_bytes);
        if (g_tm_fetchprof) {
            atomic_fetch_add(&g_tm_pf_wall_ns, (uint64_t)(ts1.tv_sec - ts0.tv_sec) * 1000000000ull
                                             + (uint64_t)(ts1.tv_nsec - ts0.tv_nsec));
        }
        tm_ev(tm_us(&ts0), tm_us(&ts1) - tm_us(&ts0), 100 + wid, 2 /*FETCH*/, t->layer_id, e);

        pthread_mutex_lock(&g_tm_mtx);
        if (--g_tm_inflight == 0 && g_tm_q_n == 0) {
            pthread_cond_broadcast(&g_tm_quiet_cv);   // fetch burst drained
        }
        if (--t->pending[e] == 0) {
            // release store: pairs with the acquire fast path in ggml_tm_wait_expert --
            // a thread seeing RESIDENT without the mutex also sees ALL sub-reads' bytes
            __atomic_store_n(&t->state[e], GGML_TM_RESIDENT, __ATOMIC_RELEASE);
            t->n_resident++;
            t->fifo[(t->fifo_head + t->fifo_len) % t->n_experts] = e;
            t->fifo_len = t->fifo_len < t->n_experts ? t->fifo_len + 1 : t->fifo_len;
            atomic_fetch_add(&g_tm_fetches, 1);   // counted per EXPERT, not per sub-read
            pthread_cond_broadcast(&g_tm_done_cv);
            if (t->evict_pending[e]) {
                // a swap requested eviction while this was in flight -- honor it now that
                // it is RESIDENT, so residency stays strictly bounded (no leak).
                t->evict_pending[e] = 0;
                ggml_tm_evict(t, e);
            }
        }
        pthread_mutex_unlock(&g_tm_mtx);
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// io_uring fetch path (LLAMA_TEMPORAL_URING=1)
//
// The pread worker pool issues each part of a swap from a DIFFERENT thread, so six
// concurrent 108 KiB reads cost six futex wakeups and six independent entries into a
// block layer whose UFS host exposes nr_hw_queues=1. This path replaces the pool with
// ONE submitter thread that pushes every queued part as an SQE and hands the whole
// burst to the kernel in a single io_uring_enter -- the same six device requests, but
// submitted back-to-back from one context. It is the concurrency test the standalone
// QD1 latency probe (S2, 315 vs 334 us) could not answer.
//
// NOT implemented on purpose: IORING_REGISTER_BUFFERS. Registered buffers are pinned
// with get_user_pages, and the pool's whole eviction mechanism is MADV_FREE on those
// same expert slots. Registering them would silently defeat eviction, residency would
// become unbounded, and the run would get faster by quietly becoming resident -- which
// is exactly the failure mode of pitfall #11. Rejected on design, not on measurement.
static int tm_uring_setup(unsigned entries, struct io_uring_params * p) {
    return (int) syscall(__NR_io_uring_setup, entries, p);
}
static int tm_uring_enter(int fd, unsigned to_submit, unsigned min_complete, unsigned flags) {
    return (int) syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}
static int tm_uring_register(int fd, unsigned op, void * arg, unsigned nr) {
    return (int) syscall(__NR_io_uring_register, fd, op, arg, nr);
}

static bool g_tm_uring        = false;   // LLAMA_TEMPORAL_URING
static bool g_tm_uring_sqpoll = false;   // LLAMA_TEMPORAL_URING_SQPOLL
static bool g_tm_uring_iopoll = false;   // LLAMA_TEMPORAL_URING_IOPOLL

#define TM_UR_DEPTH 64

struct tm_ur_req {
    int       ti, e, part;
    uint8_t * dst;
    off_t     off;
    size_t    left;
    uint64_t  t0;     // submit time, ns
    bool      busy;
};

// byte geometry of one queued part -- identical arithmetic to the pread worker
static void ggml_tm_part_geom(int ti, int e, int part,
                              size_t * ebeg, uint8_t ** dst, size_t * sub_bytes) {
    struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
    size_t step = (t->expert_bytes / (size_t) g_tm_split) & ~(size_t) 4095;
    size_t plo  = (size_t) part * step;
    size_t phi  = (part == g_tm_split - 1) ? t->expert_bytes : plo + step;
    *ebeg      = t->file_off + (size_t) e * t->expert_bytes + plo;
    *dst       = (uint8_t *) t->data + (size_t) e * t->expert_bytes + plo;
    *sub_bytes = phi - plo;
}

// completion bookkeeping for one part -- the tail of ggml_tm_worker, verbatim in effect
static void ggml_tm_finish_part(int ti, int e) {
    struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
    pthread_mutex_lock(&g_tm_mtx);
    if (--g_tm_inflight == 0 && g_tm_q_n == 0) {
        pthread_cond_broadcast(&g_tm_quiet_cv);
    }
    if (--t->pending[e] == 0) {
        __atomic_store_n(&t->state[e], GGML_TM_RESIDENT, __ATOMIC_RELEASE);
        t->n_resident++;
        t->fifo[(t->fifo_head + t->fifo_len) % t->n_experts] = e;
        t->fifo_len = t->fifo_len < t->n_experts ? t->fifo_len + 1 : t->fifo_len;
        atomic_fetch_add(&g_tm_fetches, 1);
        pthread_cond_broadcast(&g_tm_done_cv);
        if (t->evict_pending[e]) {
            t->evict_pending[e] = 0;
            ggml_tm_evict(t, e);
        }
    }
    pthread_mutex_unlock(&g_tm_mtx);
}

static void * ggml_tm_uring_worker(void * arg) {
    (void) arg;
    ggml_tm_set_io_affinity();
    int fd = open(g_tm_path, O_RDONLY | (g_tm_odirect ? O_DIRECT : 0));
    if (fd < 0) {
        fprintf(stderr, "temporal-pool: FATAL uring open(%s) failed: %s\n", g_tm_path, strerror(errno));
        abort();
    }
    if (syscall(__NR_ioprio_set, 1, 0, (1 << 13) | 0) != 0) {
        syscall(__NR_ioprio_set, 1, 0, (2 << 13) | 0);
    }

    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    if (g_tm_uring_sqpoll) { p.flags |= IORING_SETUP_SQPOLL; p.sq_thread_idle = 2000; }
    if (g_tm_uring_iopoll) { p.flags |= IORING_SETUP_IOPOLL; }
    int ring = tm_uring_setup(TM_UR_DEPTH, &p);
    if (ring < 0) {
        fprintf(stderr, "temporal-pool: FATAL io_uring_setup failed: %s "
                        "(sqpoll=%d iopoll=%d)\n", strerror(errno),
                        g_tm_uring_sqpoll ? 1 : 0, g_tm_uring_iopoll ? 1 : 0);
        abort();
    }
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);
    uint8_t * sq = (uint8_t *) mmap(NULL, sq_sz, PROT_READ | PROT_WRITE,
                                    MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQ_RING);
    uint8_t * cq = (uint8_t *) mmap(NULL, cq_sz, PROT_READ | PROT_WRITE,
                                    MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_CQ_RING);
    struct io_uring_sqe * sqes = (struct io_uring_sqe *) mmap(NULL,
            p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) {
        fprintf(stderr, "temporal-pool: FATAL io_uring mmap failed: %s\n", strerror(errno));
        abort();
    }
    unsigned * sq_tail  = (unsigned *)(sq + p.sq_off.tail);
    unsigned * sq_mask  = (unsigned *)(sq + p.sq_off.ring_mask);
    unsigned * sq_array = (unsigned *)(sq + p.sq_off.array);
    unsigned * sq_flags = (unsigned *)(sq + p.sq_off.flags);
    unsigned * cq_head  = (unsigned *)(cq + p.cq_off.head);
    unsigned * cq_tail  = (unsigned *)(cq + p.cq_off.tail);
    unsigned * cq_mask  = (unsigned *)(cq + p.cq_off.ring_mask);
    struct io_uring_cqe * cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);

    bool fixed_file = false;
    if (g_tm_uring_sqpoll) {
        fixed_file = tm_uring_register(ring, IORING_REGISTER_FILES, &fd, 1) == 0;
        if (!fixed_file) {
            fprintf(stderr, "temporal-pool: uring register_files failed: %s (using raw fd)\n",
                    strerror(errno));
        }
    }
    fprintf(stderr, "temporal-pool: io_uring fetch path ENABLED (sq_entries=%u cq_entries=%u "
                    "sqpoll=%d iopoll=%d fixed_file=%d)\n",
            p.sq_entries, p.cq_entries, g_tm_uring_sqpoll ? 1 : 0,
            g_tm_uring_iopoll ? 1 : 0, fixed_file ? 1 : 0);

    struct tm_ur_req req[TM_UR_DEPTH];
    memset(req, 0, sizeof(req));
    int inflight = 0;

    for (;;) {
        // ---- 1. take every queued part we have room for (HI class first) ----
        int fresh[TM_UR_DEPTH];
        int nfresh = 0;
        pthread_mutex_lock(&g_tm_mtx);
        while (g_tm_q_len == 0 && inflight == 0) {
            pthread_cond_wait(&g_tm_work_cv, &g_tm_mtx);
        }
        while (g_tm_q_len > 0 && inflight + nfresh < TM_UR_DEPTH) {
            int pick = -1;
            for (int k = 0; k < g_tm_q_n; k++) {
                int idx = (g_tm_q_head + k) % GGML_TM_QCAP;
                if (g_tm_q[idx].cls == 0) { pick = k; break; }
                if (pick < 0) pick = k;
            }
            if (pick < 0) break;
            int idx = (g_tm_q_head + pick) % GGML_TM_QCAP;
            if (g_tm_q[idx].cls == 0) {
                struct timespec tp; clock_gettime(CLOCK_MONOTONIC, &tp);
                uint64_t nowp = (uint64_t) tp.tv_sec * 1000000000ull + tp.tv_nsec;
                atomic_fetch_add(&g_tm_qwait_hi_ns, nowp - g_tm_q[idx].enq_ns);
                atomic_fetch_add(&g_tm_qhi_n, 1);
            }
            // find a free slot in the in-flight table
            int slot = -1;
            for (int s = 0; s < TM_UR_DEPTH; s++) { if (!req[s].busy) { slot = s; break; } }
            if (slot < 0) break;
            req[slot].ti   = g_tm_q[idx].ti;
            req[slot].e    = g_tm_q[idx].e;
            req[slot].part = g_tm_q[idx].part;
            req[slot].busy = true;
            for (int k = pick; k > 0; k--) {
                int dst = (g_tm_q_head + k) % GGML_TM_QCAP;
                int src = (g_tm_q_head + k - 1) % GGML_TM_QCAP;
                g_tm_q[dst] = g_tm_q[src];
            }
            g_tm_q_head = (g_tm_q_head + 1) % GGML_TM_QCAP;
            g_tm_q_n--;
            g_tm_q_len--;
            g_tm_inflight++;
            fresh[nfresh++] = slot;
        }
        pthread_mutex_unlock(&g_tm_mtx);

        // ---- 2. build one SQE per part, submit the whole burst in one enter ----
        if (nfresh > 0) {
            struct timespec tsub; clock_gettime(CLOCK_MONOTONIC, &tsub);
            uint64_t sub_ns = (uint64_t) tsub.tv_sec * 1000000000ull + tsub.tv_nsec;
            for (int i = 0; i < nfresh; i++) {
                int s = fresh[i];
                size_t ebeg, sub_bytes; uint8_t * dst;
                ggml_tm_part_geom(req[s].ti, req[s].e, req[s].part, &ebeg, &dst, &sub_bytes);
                const bool aligned = (ebeg % 4096 == 0)
                                  && (((uintptr_t) dst) % 4096 == 0)
                                  && (sub_bytes % 4096 == 0);
                if (g_tm_odirect && !aligned) {
                    // O_DIRECT cannot DMA into an unaligned slot; the pread path bounces
                    // here. Rather than build a second bounce mechanism inside the ring,
                    // do this rare part synchronously and complete it immediately.
                    static _Atomic int warned = 0;
                    if (atomic_exchange(&warned, 1) == 0) {
                        fprintf(stderr, "temporal-pool: uring unaligned part -> sync pread "
                                        "fallback (off%%4096=%zu len%%4096=%zu)\n",
                                ebeg % 4096, sub_bytes % 4096);
                    }
                    size_t left = sub_bytes; off_t fo = (off_t) ebeg; uint8_t * dp = dst;
                    while (left > 0) {
                        ssize_t r = pread(fd, dp, left, fo);
                        if (r <= 0) {
                            fprintf(stderr, "temporal-pool: FATAL uring fallback pread: %s\n",
                                    strerror(errno));
                            abort();
                        }
                        dp += r; fo += r; left -= (size_t) r;
                    }
                    atomic_fetch_add(&g_tm_fetched_bytes, sub_bytes);
                    int ti = req[s].ti, e = req[s].e;
                    req[s].busy = false;
                    ggml_tm_finish_part(ti, e);
                    fresh[i] = -1;
                    continue;
                }
                req[s].dst  = dst;
                req[s].off  = (off_t) ebeg;
                req[s].left = sub_bytes;
                req[s].t0   = sub_ns;
                unsigned tail = *sq_tail;
                unsigned idx  = tail & *sq_mask;
                struct io_uring_sqe * sqe = &sqes[idx];
                memset(sqe, 0, sizeof(*sqe));
                sqe->opcode    = IORING_OP_READ;
                sqe->fd        = fixed_file ? 0 : fd;
                sqe->flags     = fixed_file ? IOSQE_FIXED_FILE : 0;
                sqe->addr      = (uint64_t)(uintptr_t) dst;
                sqe->len       = (unsigned) sub_bytes;
                sqe->off       = (uint64_t) ebeg;
                sqe->user_data = (uint64_t) s;
                sq_array[idx]  = idx;
                __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
                inflight++;
            }
            int nsub = 0;
            for (int i = 0; i < nfresh; i++) { if (fresh[i] >= 0) nsub++; }
            if (nsub > 0) {
                if (g_tm_uring_sqpoll) {
                    if (__atomic_load_n(sq_flags, __ATOMIC_ACQUIRE) & IORING_SQ_NEED_WAKEUP) {
                        tm_uring_enter(ring, nsub, 0, IORING_ENTER_SQ_WAKEUP);
                    }
                } else {
                    int r = tm_uring_enter(ring, nsub, 0, 0);
                    if (r < 0) {
                        fprintf(stderr, "temporal-pool: FATAL io_uring_enter(submit=%d): %s\n",
                                nsub, strerror(errno));
                        abort();
                    }
                }
            }
        }

        // ---- 3. reap ----
        if (inflight > 0) {
            bool empty = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE) == *cq_head;
            if (empty) {
                if (g_tm_uring_iopoll) {
                    tm_uring_enter(ring, 0, 1, IORING_ENTER_GETEVENTS);
                } else {
                    // only block when there is nothing else to submit
                    bool more;
                    pthread_mutex_lock(&g_tm_mtx);
                    more = g_tm_q_len > 0;
                    pthread_mutex_unlock(&g_tm_mtx);
                    if (more) { continue; }
                    tm_uring_enter(ring, 0, 1, IORING_ENTER_GETEVENTS);
                }
            }
            struct timespec tc; clock_gettime(CLOCK_MONOTONIC, &tc);
            uint64_t now_ns = (uint64_t) tc.tv_sec * 1000000000ull + tc.tv_nsec;
            while (__atomic_load_n(cq_tail, __ATOMIC_ACQUIRE) != *cq_head) {
                struct io_uring_cqe * c = &cqes[*cq_head & *cq_mask];
                int s   = (int) c->user_data;
                int res = c->res;
                __atomic_store_n(cq_head, *cq_head + 1, __ATOMIC_RELEASE);
                if (s < 0 || s >= TM_UR_DEPTH || !req[s].busy) {
                    fprintf(stderr, "temporal-pool: FATAL uring stray cqe user_data=%d\n", s);
                    abort();
                }
                if (res <= 0) {
                    fprintf(stderr, "temporal-pool: FATAL uring read failed (off=%lld len=%zu): %s\n",
                            (long long) req[s].off, req[s].left, strerror(-res));
                    abort();   // computing on stale/zero weights must never be silent
                }
                if ((size_t) res < req[s].left) {
                    // short read: resubmit the remainder rather than compute on a hole
                    atomic_fetch_add(&g_tm_pf_short, 1);
                    atomic_fetch_add(&g_tm_fetched_bytes, (uint64_t) res);
                    req[s].dst  += res;
                    req[s].off  += res;
                    req[s].left -= (size_t) res;
                    unsigned tail = *sq_tail;
                    unsigned idx  = tail & *sq_mask;
                    struct io_uring_sqe * sqe = &sqes[idx];
                    memset(sqe, 0, sizeof(*sqe));
                    sqe->opcode    = IORING_OP_READ;
                    sqe->fd        = fixed_file ? 0 : fd;
                    sqe->flags     = fixed_file ? IOSQE_FIXED_FILE : 0;
                    sqe->addr      = (uint64_t)(uintptr_t) req[s].dst;
                    sqe->len       = (unsigned) req[s].left;
                    sqe->off       = (uint64_t) req[s].off;
                    sqe->user_data = (uint64_t) s;
                    sq_array[idx]  = idx;
                    __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
                    if (!g_tm_uring_sqpoll) { tm_uring_enter(ring, 1, 0, 0); }
                    continue;                      // still in flight
                }
                uint64_t dt = now_ns - req[s].t0;
                atomic_fetch_add(&g_tm_fetch_ns, dt);
                atomic_fetch_add(&g_tm_fetched_bytes, (uint64_t) res);
                if (g_tm_fetchprof) {
                    // wall == sys by construction here: the "syscall" is the ring, and
                    // what is timed is submit -> completion for one part. outside_sys is
                    // therefore not meaningful in this arm; wall/fetch still is.
                    atomic_fetch_add(&g_tm_pf_calls, 1);
                    atomic_fetch_add(&g_tm_pf_fetches, 1);
                    atomic_fetch_add(&g_tm_pf_sys_ns, dt);
                    atomic_fetch_add(&g_tm_pf_wall_ns, dt);
                    atomic_fetch_add(&g_tm_pf_first_ns, dt);
                    uint64_t prev = atomic_load(&g_tm_pf_max_ns);
                    while (dt > prev && !atomic_compare_exchange_weak(&g_tm_pf_max_ns, &prev, dt)) { }
                }
                int ti = req[s].ti, e = req[s].e;
                if (g_tm_trace_on) {
                    uint64_t t0ns = (uint64_t) g_tm_trace_t0.tv_sec * 1000000000ull
                                  + (uint64_t) g_tm_trace_t0.tv_nsec;
                    tm_ev((double)(req[s].t0 - t0ns) / 1e3, (double) dt / 1e3, 100,
                          2 /*FETCH*/, g_tm_pool[ti].layer_id, e);
                }
                req[s].busy = false;
                inflight--;
                ggml_tm_finish_part(ti, e);
            }
        }
    }
    return NULL;
}

static void * ggml_tm_janitor(void * arg);

void ggml_temporal_pool_set_fused_base(size_t base) { g_tm_fused_base = base; }

void ggml_temporal_pool_register(void * data, size_t nbytes, int n_experts, int fd, size_t file_off,
                                 const char * name) {
    static bool env_read = false;
    if (!env_read) {
        env_read = true;
        const char * r = getenv("LLAMA_TEMPORAL_R");
        const char * p = getenv("LLAMA_TEMPORAL_SWAP_PROB");
        const char * d = getenv("LLAMA_TEMPORAL_ODIRECT");
        const char * w = getenv("LLAMA_TEMPORAL_FETCH_THREADS");
        const char * s = getenv("LLAMA_TEMPORAL_SIBLING_PREFETCH");
        if (r) { g_tm_R = atoi(r); }
        if (p) { g_tm_swap_prob = atof(p); }
        if (d && atoi(d)) { g_tm_odirect = true; }
        if (w) { g_tm_nworkers = atoi(w); }
        if (g_tm_nworkers < 1)  { g_tm_nworkers = 1; }
        if (g_tm_nworkers > 16) { g_tm_nworkers = 16; }
        if (s && !atoi(s)) { g_tm_sibling = false; }
        if (getenv("LLAMA_TEMPORAL_FUSED")) { g_tm_fused = true; }
        if (getenv("LLAMA_TEMPORAL_FETCHPROF")) { g_tm_fetchprof = true; }
        if (getenv("LLAMA_TEMPORAL_URING")) { g_tm_uring = true; }
        if (getenv("LLAMA_TEMPORAL_URING_SQPOLL")) { g_tm_uring_sqpoll = true; }
        if (getenv("LLAMA_TEMPORAL_URING_IOPOLL")) { g_tm_uring_iopoll = true; }
        if (g_tm_uring && g_tm_fused) {
            fprintf(stderr, "temporal-pool: FATAL LLAMA_TEMPORAL_URING and _FUSED are exclusive\n");
            abort();
        }
        const char * ms = getenv("LLAMA_TEMPORAL_SPINNERS");
        if (ms) { g_tm_max_spinners = atoi(ms); }
        if (g_tm_max_spinners < 0)  { g_tm_max_spinners = 0; }
        if (g_tm_max_spinners > 16) { g_tm_max_spinners = 16; }
        const char * sp = getenv("LLAMA_TEMPORAL_SPLIT");
        if (sp) { g_tm_split = atoi(sp); }
        if (g_tm_split < 1) { g_tm_split = 1; }
        if (g_tm_split > 4) { g_tm_split = 4; }
        const char * en = getenv("LLAMA_TEMPORAL_ENFORCE");
        if (en && atoi(en)) { g_tm_enforce = true; }
        if (getenv("LLAMA_TEMPORAL_TWOPASS")) { g_tm_twopass = true; g_tm_enforce = true; }
        if (getenv("LLAMA_TEMPORAL_TRACE")) {
            clock_gettime(CLOCK_MONOTONIC, &g_tm_trace_t0);
            g_tm_trace_on = true;
            atexit(tm_trace_dump);
        }
        if (g_tm_R >= 0) {
            atexit(ggml_tm_pool_report);
            fprintf(stderr, "temporal-pool: active, R=%d swap_prob=%.3f odirect=%d workers=%d sibling=%d split=%d\n",
                    g_tm_R, g_tm_swap_prob, g_tm_odirect ? 1 : 0, g_tm_nworkers, g_tm_sibling ? 1 : 0, g_tm_split);
        }
    }
    if (g_tm_R < 0 || n_experts <= 1 || nbytes % (size_t) n_experts != 0) {
        if (fd >= 0) { close(fd); }
        return;
    }
    pthread_mutex_lock(&g_tm_mtx);
    if (g_tm_pool_n < (int)(sizeof(g_tm_pool)/sizeof(g_tm_pool[0]))) {
        struct ggml_tm_pool_tensor * t = &g_tm_pool[g_tm_pool_n++];
        t->data         = data;
        t->nbytes       = nbytes;
        t->n_experts    = n_experts;
        t->expert_bytes = nbytes / (size_t) n_experts;
        t->fd           = fd;
        t->file_off     = file_off;
        t->layer_id     = -1;
        t->slot         = -1;
        if (name) {
            sscanf(name, "blk.%d.", &t->layer_id);
            if      (strstr(name, "gate_exps")) t->slot = 0;
            else if (strstr(name, "up_exps"))   t->slot = 1;
            else if (strstr(name, "down_exps")) t->slot = 2;
        }
        t->state        = calloc(n_experts, 1);
        t->needed       = calloc(n_experts, 1);
        t->op_seq       = 0;
        t->last_use     = calloc(n_experts, sizeof(uint64_t));
        t->pending      = calloc(n_experts, 1);
        t->evict_pending= calloc(n_experts, 1);
        t->order        = calloc(n_experts, sizeof(int));
        for (int e = 0; e < n_experts; e++) { t->order[e] = e; }
        t->fifo         = calloc(n_experts, sizeof(int));
        t->fifo_head    = 0;
        if (g_tm_R >= 0 && g_tm_R < n_experts) {
            // LAZY: the loader skipped this tensor's data (see llama-model-loader.cpp);
            // every expert starts ABSENT and is fetched on first use. Avoids the
            // full-model anonymous transient that OOM-panicked the 7.7 GB Pixel.
            t->fifo_len   = 0;
            t->n_resident = 0;               // state[] is calloc'd = GGML_TM_ABSENT
        } else {
            t->fifo_len   = n_experts;       // ceiling: loader read everything; all
            t->n_resident = n_experts;       // resident and in the ring for trim
            memset(t->state, GGML_TM_RESIDENT, n_experts);
            for (int e = 0; e < n_experts; e++) { t->fifo[e] = e; }
        }

        // resolve the gguf path once; fetch workers open their own fds from it
        if (!g_tm_path[0]) {
            char linkp[64];
            snprintf(linkp, sizeof(linkp), "/proc/self/fd/%d", fd);
            ssize_t n = readlink(linkp, g_tm_path, sizeof(g_tm_path) - 1);
            if (n <= 0) {
                fprintf(stderr, "temporal-pool: FATAL readlink(%s) failed: %s\n", linkp, strerror(errno));
                abort();
            }
            g_tm_path[n] = '\0';
        }
        if (!g_tm_odirect) {
            // suppress readahead so a fetch of expert e does not warm its neighbours
            posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
        }
        if (!g_tm_workers_started) {
            g_tm_workers_started = true;
            if (g_tm_uring) {
                // ONE submitter: the whole point is that every part of a burst enters
                // the block layer from the same context, in one syscall.
                pthread_t ut;
                pthread_create(&ut, NULL, ggml_tm_uring_worker, NULL);
                pthread_detach(ut);
            } else {
            for (int i = 0; i < g_tm_nworkers; i++) {
                pthread_t th;
                pthread_create(&th, NULL, ggml_tm_worker, (void *)(intptr_t) i);
                pthread_detach(th);
            }
            }
            pthread_t jt;
            pthread_create(&jt, NULL, ggml_tm_janitor, NULL);
            pthread_detach(jt);
        }
    }
    pthread_mutex_unlock(&g_tm_mtx);
}

// the madvise itself: free ONLY pages fully inside the expert's range (align start up,
// end down) -- rounding outward would MADV_DONTNEED into the neighbouring expert's
// anonymous pages, which zero-fills live weights. Caller holds g_tm_mtx.
static void ggml_tm_madvise_range(struct ggml_tm_pool_tensor * t, int e) {
    // LLAMA_TEMPORAL_NOMADV=1: keep every state transition and every fetch identical but
    // never actually release the pages. Diagnostic only -- residency becomes unbounded
    // (the whole expert file ends up in RAM), so it is NOT a valid serving configuration.
    // Isolates the cost of the madvise itself: TLB shootdown across the compute threads
    // plus the soft refault when the refetched pages are touched again.
    static int nomadv = -1;
    if (nomadv < 0) {
        const char * v = getenv("LLAMA_TEMPORAL_NOMADV");
        nomadv = (v && atoi(v)) ? 1 : 0;
    }
    if (nomadv) return;
    uint8_t * beg = (uint8_t *) t->data + (size_t) e * t->expert_bytes;
    uint8_t * end = beg + t->expert_bytes;
    uint8_t * abeg = (uint8_t *) (((uintptr_t) beg + 4095) & ~(uintptr_t) 4095);
    uint8_t * aend = (uint8_t *) ((uintptr_t)  end        & ~(uintptr_t) 4095);
    if (aend <= abeg) return;
    // LLAMA_TEMPORAL_MADV_FREE=1: lazy reclaim instead of eager destroy. DONTNEED forces
    // every refetch to fault + zero 54 fresh pages INSIDE the fetch worker (simpleperf:
    // unmap_page_range/__pi_clear_page/handle_mm_fault) -- ~0.4 ms of the 1.45 ms
    // per-slice cost. FREE leaves pages live until memory pressure reclaims them, so a
    // refetch overwrites in place with no fault. HONESTY: freed pages linger in RSS
    // until pressure, so the memory-cut claim must come from measured RSS under
    // pressure, not from the R x bytes formula. Kernel still reclaims them first.
    static int use_free = -1;
    if (use_free < 0) {
        const char * v = getenv("LLAMA_TEMPORAL_MADV_FREE");
        use_free = (v && atoi(v)) ? 1 : 0;
    }
    madvise(abeg, (size_t)(aend - abeg), use_free ? MADV_FREE : MADV_DONTNEED);
}

static bool ggml_tm_submit(int ti, int e);   // fwd: janitor resubmits evicted-but-needed

// evict decision (caller holds g_tm_mtx): mark EVICTING and hand the madvise to the
// janitor. Only RESIDENT experts are evictable; FETCHING ones are in flight. If the
// janitor ring is full, do the work inline (correct, just slower).
static void ggml_tm_evict(struct ggml_tm_pool_tensor * t, int e) {
    if (t->state[e] == GGML_TM_FETCHING) {
        // Can't madvise bytes still landing. Defer: the fetch completion evicts on
        // arrival. Without this the swap leaks an in-flight expert resident forever
        // (the window has already dropped it), and residency creeps toward all-resident
        // once compute outpaces fetch latency (the repacked kernel).
        t->evict_pending[e] = 1;
        return;
    }
    if (t->state[e] != GGML_TM_RESIDENT) return;
    t->n_resident--;
    atomic_fetch_add(&g_tm_evictions, 1);
    if (g_tm_jq_len >= GGML_TM_QCAP) {
        ggml_tm_madvise_range(t, e);
        t->state[e] = GGML_TM_ABSENT;
        return;
    }
    t->state[e] = GGML_TM_EVICTING;
    g_tm_jq[(g_tm_jq_head + g_tm_jq_len) % GGML_TM_QCAP].ti = (int)(t - g_tm_pool);
    g_tm_jq[(g_tm_jq_head + g_tm_jq_len) % GGML_TM_QCAP].e  = e;
    g_tm_jq_len++;
    pthread_cond_signal(&g_tm_jan_cv);
}

// enqueue a fetch (caller holds g_tm_mtx). Own-needed before siblings, so FIFO order
// completes the blocking set first.
static bool ggml_tm_submit2(int ti, int e, int cls) {
    struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
    // re-admitting: this expert is needed again, so cancel any pending evict-on-arrival
    // (a prior swap may have queued one while it was in flight).
    t->evict_pending[e] = 0;
    // CANCEL a queued-but-not-yet-executed eviction. Its pages are still mapped (the
    // janitor madvises under this same mutex, so it cannot be mid-flight here), and the
    // janitor skips any entry whose state is no longer EVICTING -- so flipping the state
    // back both revives the data and retires the queued job. Without this, deferring
    // evictions widens the window in which a re-admitted expert would be silently freed
    // out from under a compute that believes it is resident.
    if (t->state[e] == GGML_TM_EVICTING) {
        t->state[e] = GGML_TM_RESIDENT;
        t->n_resident++;
        return true;
    }
    if (t->state[e] == GGML_TM_FETCHING && cls == 0) {
        // PROMOTE: the op now blocks on a slice that may still be queued as prefetch
        for (int k = 0; k < g_tm_q_n; k++) {
            int idx = (g_tm_q_head + k) % GGML_TM_QCAP;
            if (g_tm_q[idx].ti == ti && g_tm_q[idx].e == e) { g_tm_q[idx].cls = 0; }
        }
        return true;
    }
    if (t->state[e] != GGML_TM_ABSENT) return true;    // resident or already in flight
    if (g_tm_q_n + g_tm_split > GGML_TM_QCAP) return false;   // full -- caller decides
    struct timespec tp; clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t now = (uint64_t) tp.tv_sec * 1000000000ull + tp.tv_nsec;
    t->state[e]   = GGML_TM_FETCHING;
    t->pending[e] = (uint8_t) g_tm_split;
    for (int part = 0; part < g_tm_split; part++) {
        int slot = (g_tm_q_head + g_tm_q_n) % GGML_TM_QCAP;
        g_tm_q[slot].ti     = ti;
        g_tm_q[slot].e      = e;
        g_tm_q[slot].part   = part;
        g_tm_q[slot].cls    = cls;
        g_tm_q[slot].enq_ns = now;
        g_tm_q_n++;
        g_tm_q_len++;
    }
    pthread_cond_broadcast(&g_tm_work_cv);   // up to `split` workers can start at once
    return true;
}
static bool ggml_tm_submit(int ti, int e) { return ggml_tm_submit2(ti, e, 0); }

// Fused submit (caller holds g_tm_mtx): queue ONE job for the whole [gate|up|down] triple.
// All three slices are marked FETCHING with pending=1 and are published RESIDENT together
// when the single preadv completes, so a waiter on any of the three is satisfied at once.
static bool ggml_tm_submit_fused(int layer, int e) {
    int ti[3] = { ggml_tm_find(layer, 0), ggml_tm_find(layer, 1), ggml_tm_find(layer, 2) };
    if (ti[0] < 0 || ti[1] < 0 || ti[2] < 0) return false;
    int need = 0;
    for (int k = 0; k < 3; k++) {
        struct ggml_tm_pool_tensor * t = &g_tm_pool[ti[k]];
        t->evict_pending[e] = 0;
        if (t->state[e] == GGML_TM_EVICTING) {   // revive a queued-but-unexecuted eviction
            t->state[e] = GGML_TM_RESIDENT;
            t->n_resident++;
        }
        if (t->state[e] == GGML_TM_ABSENT) need++;
    }
    if (need == 0) return true;                  // already resident or already in flight
    if (g_tm_q_n + 1 > GGML_TM_QCAP) return false;
    struct timespec tp; clock_gettime(CLOCK_MONOTONIC, &tp);
    uint64_t now = (uint64_t) tp.tv_sec * 1000000000ull + tp.tv_nsec;
    for (int k = 0; k < 3; k++) {
        struct ggml_tm_pool_tensor * t = &g_tm_pool[ti[k]];
        if (t->state[e] == GGML_TM_ABSENT) { t->state[e] = GGML_TM_FETCHING; t->pending[e] = 1; }
    }
    int slot = (g_tm_q_head + g_tm_q_n) % GGML_TM_QCAP;
    g_tm_q[slot].ti = ti[0]; g_tm_q[slot].e = e; g_tm_q[slot].part = 0;
    g_tm_q[slot].cls = 0;    g_tm_q[slot].enq_ns = now;
    g_tm_q_n++; g_tm_q_len++;
    pthread_cond_broadcast(&g_tm_work_cv);
    return true;
}

static void * ggml_tm_janitor(void * arg) {
    (void) arg;
    ggml_tm_set_io_affinity();
    pthread_mutex_lock(&g_tm_mtx);
    for (;;) {
        while (g_tm_jq_len == 0) {
            pthread_cond_wait(&g_tm_jan_cv, &g_tm_mtx);
        }
        int ti = g_tm_jq[g_tm_jq_head].ti;
        int e  = g_tm_jq[g_tm_jq_head].e;
        g_tm_jq_head = (g_tm_jq_head + 1) % GGML_TM_QCAP;
        g_tm_jq_len--;
        struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
        if (t->state[e] != GGML_TM_EVICTING) {
            continue;   // superseded (overflow fallback already freed it)
        }
        // Drop the lock for the madvise itself. The expert sits in state EVICTING, and
        // ggml_tm_submit2 only submits an ABSENT expert while ggml_tm_evict only evicts a
        // RESIDENT one, so the janitor exclusively owns this expert for the whole madvise
        // -- no worker can read or write the range. That is exactly the invariant the lock
        // was providing, and the state machine already guarantees it.
        //
        // Holding it was NOT free: the janitor drains its queue (the 3 slices of the
        // evicted expert) without releasing, ~130 us, and fetch workers need the same
        // mutex to dequeue. Visible in the trace: of the fetch parts a layer submits, only
        // 2-3 start immediately and the rest stall until the whole evict batch finishes
        // (L0: parts at t=156..180 us behind a batch ending at 130 us). S3-26.
        // DEFERRED EVICTION: a layer waits on max(start+duration) over its fetch parts, so
        // anything that delays a part's START lands directly on the critical path. The
        // madvise batch (~130 us, lock held) was delaying 47% of them. Eviction has no
        // deadline -- the slot is not needed until the NEXT swap -- so hold it until the
        // in-flight fetches drain, then reclaim during the compute window. Bounded by
        // GGML_TM_QCAP/2 queued evictions so residency can never run away.
        if (g_tm_evict_defer < 0) {
            const char * v = getenv("LLAMA_TEMPORAL_EVICT_DEFER");
            g_tm_evict_defer = (v && atoi(v)) ? 1 : 0;
        }
        if (g_tm_evict_defer) {
            while ((g_tm_inflight > 0 || g_tm_q_n > 0) && g_tm_jq_len < GGML_TM_QCAP / 2) {
                pthread_cond_wait(&g_tm_quiet_cv, &g_tm_mtx);
            }
        }
        static int nolock = -1;
        if (nolock < 0) {
            const char * v = getenv("LLAMA_TEMPORAL_JANITOR_NOLOCK");
            nolock = (v && atoi(v)) ? 1 : 0;
        }
        if (nolock) { pthread_mutex_unlock(&g_tm_mtx); }
        { double _t0 = tm_now(); ggml_tm_madvise_range(t, e); tm_ev(_t0, tm_now() - _t0, 200 /*janitor*/, 3 /*EVICT*/, t->layer_id, e); }
        if (nolock) { pthread_mutex_lock(&g_tm_mtx); }
        t->state[e] = GGML_TM_ABSENT;
        // evicted-then-needed (swap-prob turnover, streamed evict-all): refetch now.
        if (t->needed[e]) {
            ggml_tm_submit(ti, e);
        }
        pthread_cond_broadcast(&g_tm_done_cv);
    }
    return NULL;
}

static inline uint64_t ggml_tm_rand(void) {
    uint64_t x = g_tm_rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    g_tm_rng = x;
    return x;
}

// ---- enforced 1-swap policy (the actual Temporal-MoE technique) --------------
// The resident set IS the active set (no cache/pool): exactly K experts resident,
// exactly 1 swapped per layer per token. Selection is RANDOM, not router-driven --
// a randomly-initialised model routes degenerately (same experts every token), so the
// router cannot exercise the mechanism; the discrete swap policy is imposed instead.
// This makes the forward pass compute the WINDOW (a different, approximate model), so
// the correctness gate is determinism + measured swap count, not bit-identity to the
// unconstrained top-K.  LLAMA_TEMPORAL_ENFORCE=1.  (globals declared above.)
static inline uint64_t ggml_tm_lrand(int L) {
    uint64_t x = g_tm_ernd[L]; x ^= x << 13; x ^= x >> 7; x ^= x << 17; g_tm_ernd[L] = x; return x;
}
// advance layer L's resident window by exactly 1 random swap (evict 1, admit 1).
static void ggml_tm_enforce_advance(int L, int K, int E) {
    if (g_tm_ewin_k[L] == 0) {                 // init: first K experts resident
        g_tm_ernd[L] = 0x9E3779B97F4A7C15ull ^ (0x100000001b3ull * (uint64_t)(L + 1));
        for (int j = 0; j < K; j++) { g_tm_ewin[L][j] = j; g_tm_ein[L][j] = 1; }
        g_tm_ewin_k[L] = K;
        return;
    }
    int s = (int)(ggml_tm_lrand(L) % (uint64_t) K);      // slot to evict
    int a;
    do { a = (int)(ggml_tm_lrand(L) % (uint64_t) E); } while (g_tm_ein[L][a]);  // admit non-resident
    g_tm_ein[L][ g_tm_ewin[L][s] ] = 0;                  // evict old
    g_tm_ewin[L][s] = a; g_tm_ein[L][a] = 1;             // admit new
    atomic_fetch_add(&g_tm_swaps, 1);
}

// ---- two-pass enforce: materialize the window into an ids tensor ----------------------
// Runs as a ggml custom op producing selected_experts [K, n_tokens] (I32). Advances the
// layer's resident window by exactly one random swap, pinning the newly-admitted (still-
// fetching) expert to slot K-1 and the K-1 resident experts to slots 0..K-2. build_moe_ffn
// then splits into a resident sub-pass (0..K-2, never waits) and a new sub-pass (K-1,
// waits) so the resident gate+up+down compute overlaps the new expert's fetch. This op
// also submits the new expert's three slice fetches and evicts one resident via the
// janitor -- all residency management for two-pass mode lives here, not in mul_mat_id.
void ggml_temporal_window_fill(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const int L = (int)(intptr_t) userdata;
    const int K = (int) dst->ne[0];
    const int n_tokens = (int) dst->ne[1];
    int32_t * out = (int32_t *) dst->data;
    if (g_tm_R < 0 || L < 0 || L >= GGML_TM_MAXLAYER) {   // pool off: identity window
        for (int t = 0; t < n_tokens; t++) for (int j = 0; j < K; j++) out[t*K+j] = j;
        return;
    }
    int ti_g = -1, ti_u = -1, ti_d = -1, E = 0;
    for (int i = 0; i < g_tm_pool_n; i++) {
        if (g_tm_pool[i].layer_id != L) continue;
        E = g_tm_pool[i].n_experts;
        if      (g_tm_pool[i].slot == 0) ti_g = i;
        else if (g_tm_pool[i].slot == 1) ti_u = i;
        else if (g_tm_pool[i].slot == 2) ti_d = i;
    }
    pthread_mutex_lock(&g_tm_mtx);
    if (g_tm_ewin_k[L] == 0) {                            // init: window {0..K-1}, all fetched
        g_tm_ernd[L] = 0x9E3779B97F4A7C15ull ^ (0x100000001b3ull * (uint64_t)(L + 1));
        for (int j = 0; j < K; j++) { g_tm_ewin[L][j] = j; g_tm_ein[L][j] = 1;
            if (g_tm_fused) { ggml_tm_submit_fused(L, j); } else {
            if (ti_g>=0) ggml_tm_submit2(ti_g, j, 0);
            if (ti_u>=0) ggml_tm_submit2(ti_u, j, 0);
            if (ti_d>=0) ggml_tm_submit2(ti_d, j, 0); }
        }
        g_tm_ewin_k[L] = K;
    } else {
        int s = (int)(ggml_tm_lrand(L) % (uint64_t)(K > 1 ? K-1 : 1));   // evict a resident (0..K-2)
        int a; do { a = (int)(ggml_tm_lrand(L) % (uint64_t)E); } while (g_tm_ein[L][a]);
        int ev = g_tm_ewin[L][s];
        if (ti_g>=0) ggml_tm_evict(&g_tm_pool[ti_g], ev);
        if (ti_u>=0) ggml_tm_evict(&g_tm_pool[ti_u], ev);
        if (ti_d>=0) ggml_tm_evict(&g_tm_pool[ti_d], ev);
        g_tm_ein[L][ev] = 0;
        g_tm_ewin[L][s]   = g_tm_ewin[L][K-1];            // prev-new (now resident) fills the gap
        g_tm_ewin[L][K-1] = a; g_tm_ein[L][a] = 1;        // new expert pinned to slot K-1
        if (g_tm_fused) { ggml_tm_submit_fused(L, a); } else {
        if (ti_g>=0) ggml_tm_submit2(ti_g, a, 0);
        if (ti_u>=0) ggml_tm_submit2(ti_u, a, 0);
        if (ti_d>=0) ggml_tm_submit2(ti_d, a, 0); }
        atomic_fetch_add(&g_tm_swaps, 1);
    }
    pthread_mutex_unlock(&g_tm_mtx);
    for (int t = 0; t < n_tokens; t++) for (int j = 0; j < K; j++) out[t*K+j] = g_tm_ewin[L][j];
}

// called from the ith==0 section of mul_mat_id, before the barrier
static void ggml_tm_ensure(const struct ggml_tensor * src0, const int64_t * row_counts, int n_as) {
    // two-pass mode: residency is driven by ggml_temporal_window_fill; here we only need
    // to build the compute order (resident first, fetching last) so wait_expert stalls
    // only on the new expert. No submit / evict / override.
    if (g_tm_twopass) {
        if (g_tm_R < 0 || g_tm_pool_n == 0) return;
        int ti = -1;
        for (int i = 0; i < g_tm_pool_n; i++) if (g_tm_pool[i].data == src0->data) { ti = i; break; }
        if (ti < 0 || g_tm_pool[ti].n_experts != n_as) return;
        struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];
        int oi = 0;
        for (int e = 0; e < n_as; e++) if (row_counts[e] > 0 && t->state[e] == GGML_TM_RESIDENT) t->order[oi++] = e;
        for (int e = 0; e < n_as; e++) if (row_counts[e] > 0 && t->state[e] != GGML_TM_RESIDENT) t->order[oi++] = e;
        for (int e = 0; e < n_as; e++) if (row_counts[e] == 0) t->order[oi++] = e;
        return;
    }
    if (g_tm_R < 0 || g_tm_pool_n == 0) return;
    if (g_tm_R < 0 || g_tm_pool_n == 0) return;
    atomic_fetch_add(&g_tm_hook_calls, 1);
    int ti = -1;
    for (int i = 0; i < g_tm_pool_n; i++) {
        if (g_tm_pool[i].data == src0->data) { ti = i; break; }
    }
    if (ti < 0 || g_tm_pool[ti].n_experts != n_as) {
        // an expert matmul the pool cannot see would compute on evicted (zero-filled)
        // weights -- that must be loud, not a silently-fast bogus regime
        atomic_fetch_add(&g_tm_hook_miss, 1);
        return;
    }
    struct ggml_tm_pool_tensor * t = &g_tm_pool[ti];

    pthread_mutex_lock(&g_tm_mtx);

    int needed_count = 0;
    t->op_seq++;
    for (int e = 0; e < n_as; e++) {
        t->needed[e] = row_counts[e] > 0;
        needed_count += t->needed[e];
        if (t->needed[e]) { t->last_use[e] = t->op_seq; }
    }

    // prescribed turnover: each needed expert may have been force-evicted since last use
    if (g_tm_swap_prob > 0.0) {
        for (int e = 0; e < n_as; e++) {
            if (t->needed[e] && t->state[e] == GGML_TM_RESIDENT &&
                (double) (ggml_tm_rand() >> 11) / 9007199254740992.0 < g_tm_swap_prob) {
                ggml_tm_evict(t, e);
            }
        }
    }

    // streamed regime: the window cannot hold even one op's working set, so nothing
    // persists between ops -- evict everything left from last time, then fetch fresh
    if (g_tm_R < needed_count) {
        for (int e = 0; e < n_as; e++) {
            ggml_tm_evict(t, e);       // skips FETCHING entries internally
        }
    }

    // fetch this op's missing experts, all in flight at once (queue depth)
    for (int e = 0; e < n_as; e++) {
        if (t->needed[e] && t->state[e] == GGML_TM_ABSENT) {
            if (!ggml_tm_submit(ti, e)) {
                fprintf(stderr, "temporal-pool: FATAL fetch queue overflow\n");
                abort();               // own-needed fetches must never be dropped
            }
        }
    }

    // sibling prefetch: the other expert tensors of this layer share this op's routing
    // (one ids tensor feeds gate/up/down), so their missing experts can start fetching
    // now and hide behind the trio's compute. Purely same-token; no speculation.
    if (g_tm_sibling && t->layer_id >= 0) {
        for (int i = 0; i < g_tm_pool_n; i++) {
            if (i == ti || g_tm_pool[i].layer_id != t->layer_id) continue;
            if (g_tm_pool[i].n_experts != n_as) continue;
            for (int e = 0; e < n_as; e++) {
                if (t->needed[e]) {
                    ggml_tm_submit2(i, e, 1);   // LO class: prefetch never delays a
                                                // blocking fetch; queue-full is fine --
                                                // the sibling's own ensure re-submits
                }
            }
        }
    }

    // NO blocking wait here: the compute loop consumes experts through t->order
    // (resident-needed first, in-flight last) and synchronizes PER EXPERT, so resident-
    // expert GEMVs overlap the remaining fetches -- the missing expert is computed last,
    // after its bytes land. Bit-identity is preserved: no row of expert e is touched
    // before state[e] == RESIDENT.
    {
        int oi = 0;
        for (int e = 0; e < n_as; e++) {
            if (t->needed[e] && t->state[e] == GGML_TM_RESIDENT) { t->order[oi++] = e; }
        }
        for (int e = 0; e < n_as; e++) {
            if (t->needed[e] && t->state[e] != GGML_TM_RESIDENT) { t->order[oi++] = e; }
        }
        for (int e = 0; e < n_as; e++) {
            if (!t->needed[e]) { t->order[oi++] = e; }
        }
    }

    // temporal window: trim non-needed residents down to R. Default is the FIFO
    // rolling window -- the temporal technique's semantics (turnover is prescribed,
    // not policy-optimized). LLAMA_TEMPORAL_LRU=1 selects last-use eviction; kept as
    // an instrumented OBSERVATION only (S2-15): cache-affinity policies are outside
    // the technique's scope, and measured gains were marginal anyway.
    if (g_tm_R >= needed_count) {
        static const char * lru_env = NULL;
        static bool lru = false;
        if (!lru_env) { lru_env = getenv("LLAMA_TEMPORAL_LRU") ?: ""; lru = atoi(lru_env) != 0; }
        if (lru) {
            while (t->n_resident > g_tm_R) {
                int      victim = -1;
                uint64_t oldest = UINT64_MAX;
                for (int e = 0; e < n_as; e++) {
                    if (t->state[e] == GGML_TM_RESIDENT && !t->needed[e] && t->last_use[e] < oldest) {
                        oldest = t->last_use[e];
                        victim = e;
                    }
                }
                if (victim < 0) break;   // everything resident is needed this op
                ggml_tm_evict(t, victim);
            }
        } else {
            int scans = t->fifo_len;
            while (t->n_resident > g_tm_R && scans-- > 0) {
                int e = t->fifo[t->fifo_head];
                t->fifo_head = (t->fifo_head + 1) % t->n_experts;
                t->fifo_len--;
                if (t->state[e] != GGML_TM_RESIDENT) continue;
                if (t->needed[e]) {   // in use this op -- rotate to the back instead
                    t->fifo[(t->fifo_head + t->fifo_len) % t->n_experts] = e;
                    t->fifo_len++;
                    continue;
                }
                ggml_tm_evict(t, e);
            }
        }
    }

    pthread_mutex_unlock(&g_tm_mtx);
}

// ---- compute-side API (called by every thread of mul_mat_id) ----------------

// one lookup per (thread, op); returns NULL when the pool is inactive or does not
// manage this tensor (then the tensor was never evicted and no waits are needed).
static struct ggml_tm_pool_tensor * ggml_tm_lookup(const void * data, int n_as) {
    if (g_tm_R < 0 || g_tm_pool_n == 0) return NULL;
    for (int i = 0; i < g_tm_pool_n; i++) {
        if (g_tm_pool[i].data == data && g_tm_pool[i].n_experts == n_as) {
            return &g_tm_pool[i];
        }
    }
    return NULL;
}

// block until expert e's bytes are in place. Unlocked fast path is safe: during an op,
// a needed expert can only transition TOWARD resident (trim skips needed experts, and
// the next ensure of this tensor cannot run until this op's node completes), so a stale
// read can only cause a harmless slow-path entry.
static void ggml_tm_wait_expert(struct ggml_tm_pool_tensor * t, int e, int ith) {
    if (__atomic_load_n(&t->state[e], __ATOMIC_ACQUIRE) == GGML_TM_RESIDENT) {
        return;
    }
    struct timespec tw0, tw1;
    clock_gettime(CLOCK_MONOTONIC, &tw0);
    // Bounded spin before sleeping: the expected residual wait after overlap is only
    // ~100-300 us, so a futex sleep/wake round-trip (tens of us) is pure overhead on
    // almost every wait. Poll the readiness flag; fall back to the condvar only if the
    // spin budget expires (long stall -- do not burn a core on it).
    bool resident = false;
    for (;;) {
        for (int i = 0; i < 4000; i++) {
            if (__atomic_load_n(&t->state[e], __ATOMIC_ACQUIRE) == GGML_TM_RESIDENT) {
                resident = true;
                break;
            }
        }
        if (resident) break;
        clock_gettime(CLOCK_MONOTONIC, &tw1);
        uint64_t ns = (uint64_t)(tw1.tv_sec - tw0.tv_sec) * 1000000000ull
                      + (uint64_t)(tw1.tv_nsec - tw0.tv_nsec);
        // Spin budget: during a gate wait ALL compute threads are stalled -- the cores
        // have nothing else to run, so spinning is free and the futex sleep/wake +
        // global-broadcast herd (~200 us per gate event x ~283 events/token at R=18)
        // is pure loss. LLAMA_TEMPORAL_SPIN_US overrides (default 300).
        static uint64_t budget_ns = 0;
        if (budget_ns == 0) {
            const char * v = getenv("LLAMA_TEMPORAL_SPIN_US");
            long us = v ? atol(v) : 300;
            if (us < 50) us = 50;
            if (us > 20000) us = 20000;
            budget_ns = (uint64_t) us * 1000ull;
        }
        if (ns > budget_ns) {   // spin budget spent -> sleep
            break;
        }
    }
    if (!resident) {
        pthread_mutex_lock(&g_tm_mtx);
        while (t->state[e] != GGML_TM_RESIDENT) {
            pthread_cond_wait(&g_tm_done_cv, &g_tm_mtx);
        }
        pthread_mutex_unlock(&g_tm_mtx);
    }
    clock_gettime(CLOCK_MONOTONIC, &tw1);
    tm_ev(tm_us(&tw0), tm_us(&tw1) - tm_us(&tw0), ith, 1 /*WAIT*/, t->layer_id, e);
    if (ith == 0) {   // one thread's stall is THE op stall; summing all would overcount
        atomic_fetch_add(&g_tm_wait_ns, (uint64_t)(tw1.tv_sec - tw0.tv_sec) * 1000000000ull
                                        + (uint64_t)(tw1.tv_nsec - tw0.tv_nsec));
    }
}

// Make the REPACKED mul_mat_id pool-aware. That kernel is a separate code path with no
// residency logic of its own, so without this it computes on experts that have not been
// fetched yet -- silently, on whatever bytes the slot happens to hold. The two-pass decode
// barriers only covered decode; PREFILL (n_tokens>1) goes through the single-pass path and
// was corrupting the context before the first token was even sampled. S3-24.
void ggml_tm_wait_src_expert(const struct ggml_tensor * src0, int e, int ith) {
    if (g_tm_R < 0 || g_tm_pool_n == 0 || !src0 || !src0->data) return;
    for (int i = 0; i < g_tm_pool_n; i++) {
        struct ggml_tm_pool_tensor * t = &g_tm_pool[i];
        if (t->data != src0->data) continue;
        if (e < 0 || e >= t->n_experts) return;
        if (__atomic_load_n(&t->state[e], __ATOMIC_ACQUIRE) != GGML_TM_RESIDENT) {
            pthread_mutex_lock(&g_tm_mtx);
            if (t->state[e] == GGML_TM_ABSENT) {
                if (g_tm_fused && t->layer_id >= 0) { ggml_tm_submit_fused(t->layer_id, e); }
                else                               { ggml_tm_submit(i, e); }
            }
            pthread_mutex_unlock(&g_tm_mtx);
            ggml_tm_wait_expert(t, e, ith);
        }
        return;
    }
}

// trace hooks for the REPACKED mul_mat_id (repack.cpp). That kernel is a separate code
// path from the custom mul_mat_id here, so it emits no GEMV events of its own -- the
// temporal timeline would be empty of compute once experts are routed to CPU_REPACK.
int    ggml_tm_trace_on(void)  { return g_tm_trace_on ? 1 : 0; }
double ggml_tm_trace_now(void) { return tm_now(); }
void   ggml_tm_trace_gemv(double ts, double dur, int ith, int layer, int expert) {
    tm_ev(ts, dur, ith, 0 /*GEMV*/, layer, expert);
}

// two-pass wait barrier: block until the newly-swapped expert (window slot K-1) has
// finished streaming, before the new-expert sub-pass computes it. Runs as a ggml custom
// op between the resident sub-pass and the new sub-pass, passing the ids through
// unchanged. Kernel-agnostic: the resident experts (pass A) never wait, but the new
// expert must, and the repacked mul_mat_id has no pool awareness of its own -- without
// this it would compute on not-yet-fetched, madvise'd-to-zero bytes (fast but wrong).
// userdata packs the layer plus which half of the window to wait on:
//   mode 0 = the RESIDENT slots 0..K-2, mode 1 = the NEW expert in slot K-1.
// Both halves need a barrier. In steady state the resident experts are already RESIDENT so
// their wait is a single relaxed load each (free), but on the FIRST token the whole window
// is still streaming -- without this the resident pass computes on bytes that have not
// landed and silently corrupts that token. (Caught by the resident-vs-streamed output gate:
// only the first generated token differed. S3-24.)
void ggml_temporal_wait_new(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const struct ggml_tensor * src = dst->src[0];
    if (src && src->data && dst->data && dst->data != src->data) {
        memcpy(dst->data, src->data, ggml_nbytes(dst));   // pass the ids through
    }
    const intptr_t ud = (intptr_t) userdata;
    const int L    = (int) (ud & 0xffff);
    const int mode = (int) (ud >> 16);
    if (g_tm_R < 0 || L < 0 || L >= GGML_TM_MAXLAYER) return;
    const int K = g_tm_ewin_k[L];
    if (K <= 0) return;
    const int lo = mode ? K - 1 : 0;
    const int hi = mode ? K     : K - 1;
    for (int s = lo; s < hi; s++) {
        const int e = g_tm_ewin[L][s];
        for (int i = 0; i < g_tm_pool_n; i++) {
            if (g_tm_pool[i].layer_id == L) {
                ggml_tm_wait_expert(&g_tm_pool[i], e, ith);
            }
        }
    }
}
#else
void ggml_temporal_pool_register(void * data, size_t nbytes, int n_experts, int fd, size_t file_off,
                                 const char * name) {
    (void) data; (void) nbytes; (void) n_experts; (void) fd; (void) file_off; (void) name;
}
void ggml_temporal_wait_new(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) dst; (void) ith; (void) nth; (void) userdata;
}
int    ggml_tm_trace_on(void)  { return 0; }
double ggml_tm_trace_now(void) { return 0.0; }
void   ggml_tm_trace_gemv(double ts, double dur, int ith, int layer, int expert) {
    (void) ts; (void) dur; (void) ith; (void) layer; (void) expert;
}
#endif

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }

#if defined(__linux__)
        // ENFORCED 1-swap policy: replace the (degenerate random-weight) router
        // selection with the per-layer resident window, advanced by exactly one random
        // swap per layer per token (on the gate op). Decode only (one token row).
        // In two-pass mode the window is already materialized into ids upstream by
        // ggml_temporal_window_fill, so we must NOT override here.
        if (g_tm_enforce && !g_tm_twopass && ids->ne[1] == 1) {
            struct ggml_tm_pool_tensor * te = ggml_tm_lookup(src0->data, n_as);
            if (te && te->layer_id >= 0 && te->layer_id < GGML_TM_MAXLAYER && n_ids <= GGML_TM_MAXK) {
                const int L = te->layer_id, K = n_ids;
                if (te->slot == 0 || g_tm_ewin_k[L] == 0) {
                    ggml_tm_enforce_advance(L, K, n_as);   // gate advances; safety-init if up/down first
                }
                memset(matrix_row_counts, 0, n_as * sizeof(int64_t));
                for (int j = 0; j < K; j++) {
                    const int e = g_tm_ewin[L][j];
                    MMID_MATRIX_ROW(e, 0) = (struct mmid_row_mapping) { j, 0 };
                    matrix_row_counts[e] = 1;
                }
            }
        }
        // temporal slot-pool: start async fetches for every missing expert this op
        // references and build the compute order. Does NOT block: per-expert waits in
        // the compute loop below synchronize before any row of an expert is touched.
        {
            const double _ens_t0 = tm_now();
            ggml_tm_ensure(src0, matrix_row_counts, n_as);
            struct ggml_tm_pool_tensor * _te = ggml_tm_lookup(src0->data, n_as);
            tm_ev(_ens_t0, tm_now() - _ens_t0, 0, 4 /*ENSURE*/, _te ? _te->layer_id : -1, -1);
        }
#endif
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

#if defined(__linux__)
    // temporal slot-pool: iterate experts in the ensure-built order (resident first,
    // in-flight last) and synchronize per expert -- resident-expert GEMVs run while the
    // missing experts' bytes stream in; the fetched expert is computed last.
    struct ggml_tm_pool_tensor * tm_entry = ggml_tm_lookup(src0->data, n_as);
#endif

    for (int ia = 0; ia < n_as; ++ia) {
#if defined(__linux__)
        const int cur_a = tm_entry ? tm_entry->order[ia] : ia;
#else
        const int cur_a = ia;
#endif
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

#if defined(__linux__)
        if (tm_entry) {
            ggml_tm_wait_expert(tm_entry, cur_a, ith);
        }
#endif

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        const double _gemv_t0 = tm_now();
        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
        if (g_tm_trace_on) {
            int _lay = tm_entry ? tm_entry->layer_id : -1;
            if (_lay < 0 && src0->name[0]) sscanf(src0->name, "blk.%d.", &_lay);  // baseline: pool off
            tm_ev(_gemv_t0, tm_now() - _gemv_t0, ith, 0 /*GEMV*/, _lay, cur_a);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards

static int ggml_cpu_try_fuse_ops(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_compute_params * params,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return 0;
    }

    struct ggml_tensor * node = cgraph->nodes[node_n];

    if (node->op == GGML_OP_RMS_NORM) {
        // RMS_NORM + MUL fusion
        const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
        if (ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
            struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
            const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
                ? mul_node->src[1] : mul_node->src[0];
            if (node->src[0]->type  == GGML_TYPE_F32 &&
                mul_node->type      == GGML_TYPE_F32 &&
                mul_w->type         == GGML_TYPE_F32 &&
                mul_w->ne[0]        == node->ne[0]   &&
                mul_w->nb[0]        == sizeof(float)) {

                ggml_compute_forward_rms_norm_mul_fused(params, node, mul_node);
                return 1;
            }
        }
    }

    return 0;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);
        if (n_fused > 0) {
            node_n += n_fused;
        } else {
            ggml_compute_forward(&params, node);
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}

// temporal-port.h -- platform shim for the temporal expert slot-pool in ggml-cpu.c.
//
// Every name below is a macro on Linux that expands to exactly the call the pool made
// before the Windows port, so the Linux preprocessed source (and the built object) is
// unchanged. On Windows the same names map to Win32: unbuffered overlapped ReadFile for
// the O_DIRECT pread fetch path, DiscardVirtualMemory for the madvise eviction, SRW locks
// and condition variables for the pthread mutex/condvar, QueryPerformanceCounter for
// clock_gettime(CLOCK_MONOTONIC), Interlocked* for C11 atomics under MSVC (clang-cl keeps
// <stdatomic.h>). io_uring stays Linux-only in ggml-cpu.c itself.
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
// ---------------------------------------------------------------------------- Windows
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <malloc.h>

// --- threads: SRW lock + condition variable, exclusive mode (the pool never takes the
//     lock shared, unlike ggml's threadpool, so ggml_cond_wait's shared-mode wait is
//     not reused here) ------------------------------------------------------------------
typedef SRWLOCK            tm_mutex_t;
typedef CONDITION_VARIABLE tm_cond_t;
#define TM_MUTEX_INIT SRWLOCK_INIT
#define TM_COND_INIT  CONDITION_VARIABLE_INIT
static inline void tm_mutex_lock(tm_mutex_t * m)   { AcquireSRWLockExclusive(m); }
static inline void tm_mutex_unlock(tm_mutex_t * m) { ReleaseSRWLockExclusive(m); }
static inline void tm_cond_wait(tm_cond_t * c, tm_mutex_t * m) { SleepConditionVariableSRW(c, m, INFINITE, 0); }
static inline void tm_cond_broadcast(tm_cond_t * c) { WakeAllConditionVariable(c); }
static inline void tm_cond_signal(tm_cond_t * c)    { WakeConditionVariable(c); }

#define TM_THREAD_FN(name) static DWORD WINAPI name(LPVOID arg)
#define TM_THREAD_RETURN   return 0
static inline void tm_thread_detached(LPTHREAD_START_ROUTINE fn, void * arg) {
    HANDLE h = CreateThread(NULL, 0, fn, arg, 0, NULL);
    if (!h) { fprintf(stderr, "temporal-pool: FATAL CreateThread failed (%lu)\n", (unsigned long) GetLastError()); abort(); }
    CloseHandle(h);   // detached: the thread runs for the process lifetime
}

// --- monotonic clock in the same struct timespec the pool's ns arithmetic uses ------
static inline void tm_clock(struct timespec * t) {
    static LARGE_INTEGER freq = { { 0, 0 } };
    LARGE_INTEGER c;
    if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); }
    QueryPerformanceCounter(&c);
    t->tv_sec  = (time_t) (c.QuadPart / freq.QuadPart);
    t->tv_nsec = (long) ((c.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
}

// --- atomics ---------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
// MSVC C has no _Atomic: 64-bit counters and 32-bit flags on Interlocked*, byte state
// flags on ReadAcquire8/WriteRelease8.
typedef volatile LONG64 tm_au64_t;
typedef volatile LONG   tm_ai32_t;
static inline uint64_t tm_add64(tm_au64_t * p, uint64_t v) { return (uint64_t) InterlockedExchangeAdd64(p, (LONG64) v); }
static inline uint64_t tm_load64(tm_au64_t * p)             { return (uint64_t) InterlockedCompareExchange64(p, 0, 0); }
static inline void     tm_store64(tm_au64_t * p, uint64_t v){ InterlockedExchange64(p, (LONG64) v); }
static inline bool     tm_cas64(tm_au64_t * p, uint64_t * expected, uint64_t desired) {
    LONG64 old = InterlockedCompareExchange64(p, (LONG64) desired, (LONG64) *expected);
    if ((uint64_t) old == *expected) { return true; }
    *expected = (uint64_t) old;
    return false;
}
static inline int  tm_add32(tm_ai32_t * p, int v)   { return (int) InterlockedExchangeAdd(p, (LONG) v); }
static inline int  tm_sub32(tm_ai32_t * p, int v)   { return (int) InterlockedExchangeAdd(p, -(LONG) v); }
static inline int  tm_load32(tm_ai32_t * p)         { return (int) InterlockedCompareExchange(p, 0, 0); }
static inline int  tm_load32_relaxed(tm_ai32_t * p) { return (int) ReadNoFence((volatile LONG *) p); }
static inline int  tm_xchg32(tm_ai32_t * p, int v)  { return (int) InterlockedExchange(p, (LONG) v); }
static inline void tm_store32(tm_ai32_t * p, int v) { InterlockedExchange(p, (LONG) v); }
static inline uint8_t tm_load8_acq(uint8_t * p)     { return (uint8_t) ReadAcquire8((volatile CHAR *) p); }
static inline void tm_store8_rel(uint8_t * p, uint8_t v) { WriteRelease8((volatile CHAR *) p, (CHAR) v); }
#else
// clang-cl: C11 atomics and the GCC builtins are available, same as on Linux
#include <stdatomic.h>
typedef _Atomic uint64_t tm_au64_t;
typedef _Atomic int      tm_ai32_t;
#define tm_add64(p, v)          atomic_fetch_add(p, v)
#define tm_load64(p)            atomic_load(p)
#define tm_store64(p, v)        atomic_store(p, v)
#define tm_cas64(p, e, d)       atomic_compare_exchange_weak(p, e, d)
#define tm_add32(p, v)          atomic_fetch_add(p, v)
#define tm_sub32(p, v)          atomic_fetch_sub(p, v)
#define tm_load32(p)            atomic_load(p)
#define tm_load32_relaxed(p)    atomic_load_explicit(p, memory_order_relaxed)
#define tm_xchg32(p, v)         atomic_exchange(p, v)
#define tm_store32(p, v)        atomic_store(p, v)
#define tm_load8_acq(p)         __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define tm_store8_rel(p, v)     __atomic_store_n(p, v, __ATOMIC_RELEASE)
#endif

// --- files: the fetch path -------------------------------------------------------------
typedef HANDLE   tm_file_t;
typedef intptr_t tm_ssize_t;
typedef int64_t  tm_off_t;
#define TM_FILE_INVALID INVALID_HANDLE_VALUE
static inline bool tm_file_ok(tm_file_t f) { return f != INVALID_HANDLE_VALUE; }

static __declspec(thread) char  tm_err_buf[96];
static __declspec(thread) DWORD tm_err_last = 0;
static inline const char * tm_errstr(void) {
    snprintf(tm_err_buf, sizeof(tm_err_buf), "win32 error %lu", (unsigned long) tm_err_last);
    return tm_err_buf;
}

// O_DIRECT -> FILE_FLAG_NO_BUFFERING: every read bypasses the cache manager and goes to the
// volume; buffer, offset and length must be sector-aligned (the pool only issues 4 KiB
// aligned reads on this path, as it did for O_DIRECT). Overlapped so the offset travels
// in the OVERLAPPED struct instead of a shared file pointer.
static inline tm_file_t tm_open_read(const char * path, bool direct) {
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, flags, NULL);
    if (h == INVALID_HANDLE_VALUE) { tm_err_last = GetLastError(); }
    return h;
}
static inline void tm_close(tm_file_t f) { CloseHandle(f); }

// pread(): one synchronous-by-completion overlapped read at an explicit offset. Each
// fetch worker owns its handle and has one read in flight, so the per-thread event is
// enough to wait on. Returns bytes read, 0 at EOF, -1 on error (tm_errstr says why).
static __declspec(thread) HANDLE tm_pread_event = NULL;
static inline tm_ssize_t tm_pread(tm_file_t f, void * buf, size_t len, tm_off_t off) {
    if (!tm_pread_event) {
        tm_pread_event = CreateEventA(NULL, TRUE, FALSE, NULL);
        if (!tm_pread_event) { tm_err_last = GetLastError(); return -1; }
    }
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset     = (DWORD) ((uint64_t) off & 0xffffffffu);
    ov.OffsetHigh = (DWORD) ((uint64_t) off >> 32);
    ov.hEvent     = tm_pread_event;
    DWORD want = (len > 0x7fffffffu) ? 0x7fffffffu : (DWORD) len;
    DWORD got  = 0;
    if (!ReadFile(f, buf, want, NULL, &ov)) {
        DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) { return 0; }
        if (e != ERROR_IO_PENDING) { tm_err_last = e; return -1; }
    }
    if (!GetOverlappedResult(f, &ov, &got, TRUE)) {
        DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) { return 0; }
        tm_err_last = e;
        return -1;
    }
    return (tm_ssize_t) got;
}

// the loader hands the pool a CRT descriptor (from _open/_dup); resolve its path so the
// workers can open their own unbuffered handles, as readlink(/proc/self/fd) did on Linux
static inline int tm_fd_path(int fd, char * buf, size_t n) {
    HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { tm_err_last = GetLastError(); return -1; }
    DWORD r = GetFinalPathNameByHandleA(h, buf, (DWORD) n, FILE_NAME_NORMALIZED);
    if (r == 0 || r >= n) { tm_err_last = GetLastError(); return -1; }
    return (int) r;   // "\\?\C:\..." is accepted verbatim by CreateFileA
}
static inline void tm_close_fd(int fd) { _close(fd); }
#define tm_fadvise_dontneed(fd, off, len) ((void) 0)   // no page-cache advice on Windows
#define tm_fadvise_random(fd)             ((void) 0)

// --- memory ------------------------------------------------------------------------------
static inline int  tm_aligned_alloc(void ** pp, size_t sz) { *pp = _aligned_malloc(sz, 4096); return *pp ? 0 : -1; }
static inline void tm_aligned_free(void * p)                { _aligned_free(p); }

// madvise(MADV_DONTNEED | MADV_FREE) -> DiscardVirtualMemory: the pages leave the working
// set and read back as zero on the next touch; the refetch overwrites them in place.
// LLAMA_TEMPORAL_MADV_FREE keeps its name on Windows but both flavours are this one call.
// Resolved at run time because the SDK header hides it below _WIN32_WINNT 0x0603.
typedef DWORD (WINAPI * tm_discard_fn)(PVOID, SIZE_T);
static inline void tm_madvise(void * addr, size_t len, int use_free) {
    static tm_discard_fn fn = NULL;
    static int resolved = 0;
    (void) use_free;
    if (!resolved) {
        resolved = 1;
        HMODULE k32 = GetModuleHandleA("kernel32.dll");
        if (k32) { fn = (tm_discard_fn) (void *) GetProcAddress(k32, "DiscardVirtualMemory"); }
        if (!fn) { fprintf(stderr, "temporal-pool: DiscardVirtualMemory unavailable, eviction falls back to VirtualAlloc(MEM_RESET)\n"); }
    }
    if (fn) { fn(addr, len); }
    else    { VirtualAlloc(addr, len, MEM_RESET, PAGE_READWRITE); }
}

// LLAMA_TEMPORAL_WORKER_AFFINITY "lo-hi": sched_setaffinity -> SetThreadAffinityMask
static inline void tm_set_affinity_range(int lo, int hi) {
    DWORD_PTR mask = 0;
    for (int c = lo; c <= hi && c < 64; c++) { mask |= ((DWORD_PTR) 1) << c; }
    if (mask) { SetThreadAffinityMask(GetCurrentThread(), mask); }
}

#else
// ---------------------------------------------------------------------------- Linux
// exact expansions of the pre-port code; nothing here changes the Linux build
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>

typedef pthread_mutex_t tm_mutex_t;
typedef pthread_cond_t  tm_cond_t;
#define TM_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define TM_COND_INIT  PTHREAD_COND_INITIALIZER
#define tm_mutex_lock(m)    pthread_mutex_lock(m)
#define tm_mutex_unlock(m)  pthread_mutex_unlock(m)
#define tm_cond_wait(c, m)  pthread_cond_wait(c, m)
#define tm_cond_broadcast(c) pthread_cond_broadcast(c)
#define tm_cond_signal(c)   pthread_cond_signal(c)
#define TM_THREAD_FN(name)  static void * name(void * arg)
#define TM_THREAD_RETURN    return NULL
#define tm_thread_detached(fn, arg) do { pthread_t th; pthread_create(&th, NULL, fn, arg); pthread_detach(th); } while (0)

#define tm_clock(t) clock_gettime(CLOCK_MONOTONIC, t)

typedef _Atomic uint64_t tm_au64_t;
typedef _Atomic int      tm_ai32_t;
#define tm_add64(p, v)          atomic_fetch_add(p, v)
#define tm_load64(p)            atomic_load(p)
#define tm_store64(p, v)        atomic_store(p, v)
#define tm_cas64(p, e, d)       atomic_compare_exchange_weak(p, e, d)
#define tm_add32(p, v)          atomic_fetch_add(p, v)
#define tm_sub32(p, v)          atomic_fetch_sub(p, v)
#define tm_load32(p)            atomic_load(p)
#define tm_load32_relaxed(p)    atomic_load_explicit(p, memory_order_relaxed)
#define tm_xchg32(p, v)         atomic_exchange(p, v)
#define tm_store32(p, v)        atomic_store(p, v)
#define tm_load8_acq(p)         __atomic_load_n(p, __ATOMIC_ACQUIRE)
#define tm_store8_rel(p, v)     __atomic_store_n(p, v, __ATOMIC_RELEASE)

typedef int     tm_file_t;
typedef ssize_t tm_ssize_t;
typedef off_t   tm_off_t;
#define TM_FILE_INVALID (-1)
#define tm_file_ok(f)           ((f) >= 0)
#define tm_errstr()             strerror(errno)
#define tm_open_read(path, direct) open(path, O_RDONLY | ((direct) ? O_DIRECT : 0))
#define tm_close(f)             close(f)
#define tm_pread(f, b, l, o)    pread(f, b, l, o)
#define tm_close_fd(fd)         close(fd)
#define tm_fadvise_dontneed(fd, off, len) posix_fadvise(fd, off, len, POSIX_FADV_DONTNEED)
#define tm_fadvise_random(fd)   posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)
#define tm_aligned_alloc(pp, sz) posix_memalign(pp, 4096, sz)
#define tm_aligned_free(p)      free(p)
#define tm_madvise(addr, len, use_free) madvise(addr, len, (use_free) ? MADV_FREE : MADV_DONTNEED)
#endif

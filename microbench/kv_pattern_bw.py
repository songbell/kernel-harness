"""Achievable read bandwidth for the *paged 2D-block* access pattern, not a sequential stream.

Why this exists: the pa_small_q roofline says the KV read runs at 101 GB/s against a 132 GB/s
roof, i.e. 77%, leaving a 2.4x whole-kernel gap that four structural attacks have failed to
close. But 132 GB/s is the *streaming* roof, measured by microbench/dram_bw.py reading uint4
along a contiguous buffer. The kernel does nothing of the sort: it issues LSC 2D block loads of
16 token-rows x 16 bytes at pitch 128, from 256-token pages reached through an index
indirection. If that pattern's own roof is near 101, the KV read is already finished and the
"gap" is an artifact of dividing by the wrong number -- which would also explain why every
attack on the remaining time has come back empty.

Faithful to the kernel's K load (pa_small_q_ov_exp.cm, the b2dK descriptor):
  surface per page   256 rows x 128 B, pitch 128       (block_2d_desc(ptr, H, W, Pitch, x, y))
  block loaded       BlockHeight 16, BlockWidth 16, NBlocks 1
  block_x            thread's chunk * 16   -- 8 threads span the 128 B head dim
  block_y            token offset inside the page
  page hop           through a block-index array, every 256 tokens
  prefetch           distance 1, same shape, matching KV_PREFETCH_DIST=1

Three arms, so the cost can be attributed rather than just observed:
  stream   uint4 grid-stride over the same buffer      -- reproduces the 132 GB/s roof here
  seq      the real 2D-block pattern, identity page map -- isolates the tile shape
  paged    the real 2D-block pattern, shuffled page map -- adds the indirection

    python microbench/kv_pattern_bw.py [size_mb] [rounds]
"""
import sys
import time

import numpy as np

from clops import cl

PAGE_TOKENS = 256          # KV_BLOCK_SIZE
HEAD_SIZE = 128            # bytes per token, uint8 cache
PAGE_BYTES = PAGE_TOKENS * HEAD_SIZE
WG_THREADS = 8             # matches q_len=16: 8 threads span the 128 B head dim in 16 B chunks
REG_N = REG_K = 16
STEPS = PAGE_TOKENS // REG_N

CM_SRC = r'''
#include <cm/cm.h>
#include <cm/cmtl.h>

#define PAGE_TOKENS 256
#define HEAD_SIZE   128
#define REG_N       16
#define REG_K       16
#define STEPS       (PAGE_TOKENS / REG_N)

// STREAMS=1 reproduces the kernel as written: the K chunk loop and the V chunk loop run one
// after the other, so a thread has one 2D-block load outstanding at a time. STREAMS=2 issues
// two independent loads back to back from unrelated pages, which is what merging the K and V
// marshal loops into a single body would produce. Total bytes read is identical -- each
// iteration covers two pages instead of one, over half as many iterations -- so the two arms
// are directly comparable as bandwidth.
#ifndef STREAMS
#define STREAMS 1
#endif

extern "C" _GENX_MAIN_ void kv_pattern(
    uint8_t* kv [[type("svmptr_t")]],
    int* page_map [[type("svmptr_t")]],
    uint* sink [[type("svmptr_t")]],
    int n_iter)
{
    const uint tid      = cm_local_id(0);
    const uint grp      = cm_group_id(0);
    const uint ngroups  = cm_group_count(0);

    matrix<uint8_t, REG_N, REG_K> Kq[STREAMS];
    vector<uint, 4> acc = 0;

    for (int lp = grp; lp < n_iter; lp += ngroups) {
        uint8_t* base[STREAMS];
        #pragma unroll
        for (int s = 0; s < STREAMS; s++) {
            // Plain indexed load, as the kernel does for block_indices.
            base[s] = kv + (uint64_t)page_map[lp + s * n_iter] * (PAGE_TOKENS * HEAD_SIZE);
        }
#if PREFETCH
        const int lp_n = (lp + (int)ngroups < n_iter) ? lp + (int)ngroups : lp;
        uint8_t* base_n[STREAMS];
        #pragma unroll
        for (int s = 0; s < STREAMS; s++) {
            base_n[s] = kv + (uint64_t)page_map[lp_n + s * n_iter] * (PAGE_TOKENS * HEAD_SIZE);
        }
#endif
        #pragma nounroll
        for (int step = 0; step < STEPS; step++) {
            // Issued back to back with no dependency between them, so both are in flight.
            #pragma unroll
            for (int s = 0; s < STREAMS; s++) {
                lsc::block_2d_desc<uint8_t, 1, REG_N, REG_K> d(
                    base[s], PAGE_TOKENS - 1, HEAD_SIZE - 1, HEAD_SIZE - 1, 0, 0);
                d.set_block_x(tid * REG_K);
                d.set_block_y(step * REG_N);
                cm_load<lsc::Normal>(Kq[s].format<uint8_t>(), d);
            }
#if PREFETCH
            // Distance 1, wrapping to the next page on the last step -- the kernel's
            // KV_PREFETCH_DIST=1 crosses pages the same way via its lookahead cache.
            {
                const bool last = (step == STEPS - 1);
                #pragma unroll
                for (int s = 0; s < STREAMS; s++) {
                    lsc::block_2d_desc<uint8_t, 1, REG_N, REG_K> p(
                        last ? base_n[s] : base[s],
                        PAGE_TOKENS - 1, HEAD_SIZE - 1, HEAD_SIZE - 1, 0, 0);
                    p.set_block_x(tid * REG_K);
                    p.set_block_y(last ? 0 : (step + 1) * REG_N);
                    cm_prefetch(p);
                }
            }
#endif
            #pragma unroll
            for (int s = 0; s < STREAMS; s++) {
                acc ^= Kq[s].format<uint>().select<4, 1>(0);
            }
        }
    }
    // Keep every load live without ever storing in the steady state.
    if ((acc[0] ^ acc[1] ^ acc[2] ^ acc[3]) == 0xdeadbeefu) {
        sink[0] = 1;
    }
}
'''

STREAM_SRC = r'''
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void stream_read(const __global uint4* src, int n_vec, __global uint* sink) {
    uint4 acc = (uint4)(0);
    size_t stride = get_global_size(0);
    for (size_t i = get_global_id(0); i < (size_t)n_vec; i += stride)
        acc ^= src[i];
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefu) sink[0] = 1;
}
'''


def timed(fn, nbytes, rounds):
    fn()                      # warm up
    cl.finish()
    best = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        cl.finish()
        dt = time.perf_counter() - t0
        gbs = nbytes / dt / 1e9
        best = gbs if best is None else max(best, gbs)
    return best


def main() -> int:
    size_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    n_pages = (size_mb * 1024 * 1024) // PAGE_BYTES
    nbytes = n_pages * PAGE_BYTES

    kv = cl.tensor(np.random.randint(0, 255, size=nbytes // 4, dtype=np.uint32))
    sink = cl.tensor(np.zeros([16], np.uint32))

    ident = np.arange(n_pages, dtype=np.int32)
    # A shuffled map is what makes this "paged": consecutive logical pages land anywhere in the
    # allocation, which is how a real KV cache is laid out after a few sequences have churned.
    shuf = ident.copy()
    np.random.default_rng(0).shuffle(shuf)
    maps = {"seq": cl.tensor(ident), "paged": cl.tensor(shuf)}

    # 8 threads per group matches the kernel's workgroup; enough groups to saturate while
    # leaving each one many pages to walk.
    groups = 512
    gws, lws = [groups * WG_THREADS], [WG_THREADS]

    print(f"\npages {n_pages} x {PAGE_BYTES // 1024} KiB = {nbytes / 2**20:.0f} MiB, "
          f"{groups} groups x {WG_THREADS} threads, best of {rounds}")
    print(f"{'arm':<26} {'GB/s':>8}")

    stream = cl.kernels(STREAM_SRC, "")
    s = timed(lambda: stream.enqueue("stream_read", [1024 * 16], [16 * 16], kv,
                                     nbytes // 16, sink), nbytes, rounds)
    print(f"{'stream (uint4)':<26} {s:>8.1f}")

    results = {"stream": s}
    for streams in (1, 2):
        for pf in (0, 1):
            kern = cl.kernels(CM_SRC, f'-cmc -Qxcm_jit_option="" '
                                      f'-Qxcm_register_file_size=192 '
                                      f'-DPREFETCH={pf} -DSTREAMS={streams}')
            n_iter = n_pages // streams
            for name, idx in maps.items():
                g = timed(lambda k=kern, i=idx, n=n_iter: k.enqueue(
                    "kv_pattern", gws, lws, kv, i, sink, n), nbytes, rounds)
                label = f"2d-block {name} pf={pf} s={streams}"
                results[label] = g
                print(f"{label:<26} {g:>8.1f}   {g / s * 100:5.1f}% of stream")

    print(f"\nCKH_ROOF {{\"kind\": \"kv_pattern_read\", \"size_mb\": {size_mb}, "
          f"\"gb_s\": {{{', '.join(f'{k!r}: {v:.1f}' for k, v in results.items())}}}}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

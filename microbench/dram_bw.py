"""DRAM streaming-read bandwidth roof.

Measured, not taken from a spec sheet or inferred from the device string -- the reference
device reports "32 EUs", which is ambiguous between EUs and Xe-cores by a factor of 8 and
made an inferred peak figure useless.

Buffer is sized well past LLC and re-allocated per round so nothing is served from cache.
Reads uint4 (16 B/lane) with a grid-stride loop, accumulating into a live value so the loads
cannot be folded away.

    python microbench/dram_bw.py [size_mb] [rounds]
"""
import sys
import time

import numpy as np

from clops import cl

SRC = r'''
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void stream_read(const __global uint4* src, int n_vec, __global uint* sink) {
    uint4 acc = (uint4)(0);
    size_t stride = get_global_size(0);
    for (size_t i = get_global_id(0); i < (size_t)n_vec; i += stride)
        acc ^= src[i];
    // Keep the loads live without ever actually storing.
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xdeadbeefu) sink[0] = 1;
}
'''


def main() -> int:
    size_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    nbytes = size_mb * 1024 * 1024
    n_vec = nbytes // 16

    kern = cl.kernels(SRC, "")
    # Enough work-items to saturate, far fewer than n_vec so each does many iterations.
    gws, lws = [1024 * 16], [16 * 16]
    sink = cl.tensor(np.zeros([16], np.uint32))

    # Distinct buffers, cycled: reusing one lets LLC serve part of it and inflates the number.
    bufs = [cl.tensor(np.random.randint(0, 255, size=nbytes // 4, dtype=np.uint32))
            for _ in range(2)]
    cl.finish()          # drain before collecting, as the working benches in this tree do

    # Wall clock, not cl.finish() profiling: that returns no events for OpenCL-C kernels on
    # this stack (mem-bw.py in the same tree reads cycle counters inside the kernel for the
    # same reason). At hundreds of MB per pass the launch overhead is well under 1%.
    kern.enqueue("stream_read", gws, lws, bufs[0], n_vec, sink)   # warm up
    cl.finish()

    best = None
    for i in range(rounds):
        t0 = time.perf_counter()
        kern.enqueue("stream_read", gws, lws, bufs[i % len(bufs)], n_vec, sink)
        cl.finish()
        dt = time.perf_counter() - t0
        gbs = nbytes / dt / 1e9
        best = gbs if best is None else max(best, gbs)

    print(f"CKH_ROOF {{\"kind\": \"dram_read\", \"size_mb\": {size_mb}, "
          f"\"gb_s\": {best:.1f}, \"rounds\": {rounds}}}")
    print(f"\nDRAM streaming read: {best:.1f} GB/s   ({size_mb} MB, best of {rounds})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

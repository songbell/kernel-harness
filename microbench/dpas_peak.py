"""XMX fp16 DPAS throughput roof, measured.

Needed because the compute roof for this device was only ever *inferred*: the device string
reports "32 EUs", which is ambiguous between EUs and Xe-cores by a factor of 8, and an earlier
attempt to get the DPAS share by ablation was invalid (replacing the systolic op with vector
adds cost 4x more than the op it replaced). So the compute side of the roofline has been an
estimate the whole time.

Four independent accumulator chains: a single chain measures systolic *latency*, not
throughput, which would understate the roof several-fold.

    python microbench/dpas_peak.py [threads] [iters] [rounds]
"""
import sys
import time

import numpy as np

from clops import cl

# RepeatCount 8, SystolicDepth 8, fp16 VNNI -> 8x16 output, K = 8*2 = 16.
# MACs per dpas = 8 * 16 * 16 = 2048  ->  4096 FLOP.
FLOP_PER_DPAS = 8 * 16 * 16 * 2
CHAINS = 4
UNROLL = 8

SRC = r'''
extern "C" _GENX_MAIN_ void dpas_peak(float* out [[type("svmptr_t")]], int iters) {
    matrix<half, 16, 16> a = 1.0f;      // src1, K x N
    matrix<half, 8, 16>  b = 1.0f;      // src2, M x K
    matrix<float, 8, 16> c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int u = 0; u < UNROLL_N; u++) {
            // Four independent chains so the systolic array stays fed; one chain would
            // serialise on the accumulator and measure latency instead.
            c0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(
                     c0.format<float>(), a.format<int32_t>(), b.format<int32_t>());
            c1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(
                     c1.format<float>(), a.format<int32_t>(), b.format<int32_t>());
            c2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(
                     c2.format<float>(), a.format<int32_t>(), b.format<int32_t>());
            c3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(
                     c3.format<float>(), a.format<int32_t>(), b.format<int32_t>());
        }
    }
    // Keep every chain live without ever storing in the steady state.
    float s = c0[0][0] + c1[0][0] + c2[0][0] + c3[0][0];
    if (s == 1234.5f) out[0] = s;
}
'''


def main() -> int:
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    kern = cl.kernels(SRC, f'-cmc -Qxcm_jit_option="" -Qxcm_register_file_size=192 '
                           f'-DUNROLL_N={UNROLL}')
    out = cl.tensor(np.zeros([16], np.float32))
    total_flop = threads * iters * UNROLL * CHAINS * FLOP_PER_DPAS

    kern.enqueue("dpas_peak", [threads], [16], out, iters)   # warm up
    cl.finish()

    best = None
    for _ in range(rounds):
        t0 = time.perf_counter()
        kern.enqueue("dpas_peak", [threads], [16], out, iters)
        cl.finish()
        dt = time.perf_counter() - t0
        tf = total_flop / dt / 1e12
        best = tf if best is None else max(best, tf)

    print(f"CKH_ROOF {{\"kind\": \"dpas_fp16\", \"threads\": {threads}, \"iters\": {iters}, "
          f"\"tflops\": {best:.2f}}}")
    print(f"\nXMX fp16 DPAS: {best:.2f} TFLOPS   "
          f"({threads} threads x {iters} x {UNROLL} x {CHAINS} dpas, best of {rounds})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

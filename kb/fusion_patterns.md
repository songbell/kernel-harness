# CM fusion / split patterns

When to fuse two operations into one kernel vs keep them as separate kernels. One entry so
far, from the first CM kernel that wasn't pa_small_q -- see `kb/README.md` for why this file
(and its siblings) started this small.

## A reduction immediately after a GEMM usually wants its own kernel, not a fused epilogue

Light elementwise epilogues (divide, scale, bias-add) are free to fuse into a GEMM's output
write -- they touch each output element exactly once, no matter how the GEMM tiles the
problem. A *reduction* across the GEMM's N dimension (e.g. row-sum the GEMM's output) is a
different kind of epilogue: it needs to see an entire row before it can produce one output
element, but the GEMM's own parallelism strategy spreads that row's N-tiles across separate
workgroups on purpose. Fusing the reduction in means either serializing one workgroup over
every N-tile of its row (destroying the GEMM's N-direction parallelism) or reducing across
workgroups via atomics/a second pass -- a real synchronization cost, not just a theoretical
one (see `kb/xpu_optimizations.md`'s workgroup-lockstep entry for how expensive cross-thread
agreement tends to be in practice). Splitting into two kernels -- GEMM (+ any light epilogue)
writing a full intermediate, then a separate reduction kernel -- keeps both phases fully
parallel and is usually the better trade despite the extra kernel launch and intermediate
round-trip. Quantify which side of that trade actually wins for a given shape with measured
roofs (`microbench/dpas_peak.py` for the GEMM's compute floor, `microbench/dram_bw.py` for the
intermediate's round-trip cost) rather than assuming -- for a compute-bound GEMM the
intermediate traffic is usually negligible next to the matmul itself.

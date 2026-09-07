# CM kernel knowledge base

Split by concern, not one big file. Consulted during Phase 1 analysis (see
`.claude/agents/algorithm-critic.md`) alongside the measured roofs in `microbench/`.

| File | Concern |
|---|---|
| `correctness.md` | What makes a check wrong or meaningless, not slow |
| `fusion_patterns.md` | When to fuse two ops into one kernel vs split them |
| `memory_patterns.md` | Access-pattern and bandwidth measurement |
| `xpu_optimizations.md` | CM/Xe hardware-execution behavior (sync cost, register file, compile-time-vs-runtime) |
| `harness_design.md` | Meta: building the harness/agents themselves, not kernel content |

## Why this only has a handful of entries, split this small

This project has so far done deep, real-hardware work on exactly one CM kernel family
(`pa_small_q`, a paged-attention decode kernel) plus the start of a second (a
GEMM+divide+sum+scale example) -- not enough breadth yet to fill out every category. Every
entry in every file here was verified by measurement on this hardware, not assumed from
another kernel language's constructs; CM's constructs (`lsc::block_2d_desc`, `cm_dpas`,
workgroup/SLM tiling) and its lack of an in-kernel autotune search (variants are compiled
ahead of time and picked at dispatch, tuned externally via `ckh bench`/`range-tuner`) are
specific to this hardware and language.

Grow these files the same way `xpu_optimizations.md`'s tile/partition-scaling entry and
`fusion_patterns.md`'s GEMM+reduction entry were added: only after a real measurement on this
hardware forces the conclusion, not by assuming a lesson from another domain generalizes. A
category with no entries yet means exactly that -- no CM-verified lesson yet, not "nothing to
say here."

Deliberately plain Markdown -- nothing here is parsed by code yet (no `KernelSpec` reads
these programmatically). If that need arises, structure then; building unused machinery
ahead of a real consumer is exactly the kind of thing `harness_design.md` argues against
doing.

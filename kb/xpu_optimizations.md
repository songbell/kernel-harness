# CM / Xe hardware-execution patterns

Hardware-execution behavior specific to this GPU family -- synchronization cost, register
file sizing, compile-time-vs-runtime tradeoffs, occupancy/parallelism scaling. Not fusion
strategy or access-pattern bandwidth (see `kb/fusion_patterns.md` and `kb/memory_patterns.md`
for those). Populated only from what this project has measured on real CM hardware -- see
`kb/README.md` for why this file (and its siblings) started this small. There is no
in-kernel autotune-search mechanism in CM (kernel variants are compiled ahead of time as
discrete "rungs" and picked at dispatch time, tuned externally via `ckh bench`/
`range-tuner`), so patterns here are about what to measure and pick between, not an
in-kernel search mechanism.

## Workgroup lockstep cost is structural, not instruction-cost

If ablating a barrier shows it costs N% of the kernel, that N% is very often the cost of the
workgroup having to *agree* at all -- not the barrier instruction itself. Three independent
mechanisms tried on the same kernel (SLM double-buffering to halve the barrier count, a split
barrier to overlap independent work across it, moving a prefetch earlier) all failed to
recover more than a fraction of an ablated barrier's measured cost. Before spending more
effort on making a barrier "cheaper," check whether the real cost is the synchronization
requirement itself, which no barrier-side trick removes.

## Runtime-vs-compile-time-constant loop bounds cost real percent, even when the runtime value never changes

A loop trip count or guard condition driven by a runtime scalar (needed so one compiled
kernel can serve several configurations) measurably costs more than the same kernel compiled
with that value baked in as a constant -- even when, for any single dispatch, the scalar's
value is fixed and known ahead of time. Measured at 6-10% across several configurations, not
a one-off. If a kernel needs to serve N discrete configurations, compiling N variants (one
constant each) and selecting among them at dispatch time is a real, measurable win over one
kernel reading a runtime parameter -- this is the same principle as compiling separate rungs
for different tile sizes, just applied to a different axis.

## Register file size has a real, non-monotonic optimum

More registers is not more headroom, and fewer is not more occupancy in any simple way.
Sweep register file size explicitly (e.g. 128/160/192/256) rather than picking a value by
intuition -- the optimum has been the *middle* of a swept range, not either end, because a
smaller file can force spills while a larger one can starve occupancy the kernel actually
needed. A `-Qxcm_register_file_size` sweep is cheap; skipping it and guessing is not.

## A tile/partition-size rule tuned at low parallelism silently under- or over-serves at high parallelism

A rule choosing a tuning parameter (partition size, tile size, ...) based on a single scalar
axis (context length, problem size) that was calibrated with only one unit of independent
work in flight (one sequence, one batch item) will not automatically generalize once several
independent units are batched together -- the *other* axes (batch size, sequence count) also
supply parallelism, "for free," changing where the real optimum sits. A rule that only reads
one axis when the true optimum depends on the product of several will be correct at the
calibration point and measurably suboptimal everywhere else.

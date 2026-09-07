---
name: cm-kernel-opt
description: End-to-end workflow for developing and optimizing an Intel CM (C-for-Metal) GPU kernel in the aboutSHW sandbox and then integrating it into the OpenVINO GPU plugin. Use when asked to optimize, profile, or extend a pa_* / cm_* kernel, when a kernel-level performance regression is reported, or when a new kernel must be built for the intel_gpu CM path. Covers roofline analysis, ablation budgeting, bit-exactness verification, parameter calibration and plugin integration.
---

# CM kernel: develop in the sandbox, then integrate

Flow diagram, artifact/versioning map and the container measurement loop: **`WORKFLOW.md`**
in this directory.

Two-repo workflow. The kernel is developed and measured in `aboutSHW`, then ported to
`openvino`. **They are different files** and results do not automatically transfer.

| | sandbox | production |
|---|---|---|
| kernel | `aboutSHW/opencl/tests/pageatten/pa_small_q_ov_exp.cm` | `openvino/.../impls/cm/pa_small_q.cm` |
| harness | `pytest` + `clops` | the model itself |
| iteration | seconds | a plugin rebuild |

## Environment (get this right first — it is not optional)

- **Measure inside the `llm` container.** Bare metal drifts ~2× within a session; the
  container holds 0.6%.
- **The container mounts only `/home/intel/ceciliapeng` → `/ceciliapeng`.** Anything the
  harness must run has to live there. Canonical copies are version-controlled under
  `aboutSHW/opencl/tests/pageatten/harness/`; run `harness/sync.sh` to push them to
  `/home/intel/ceciliapeng/kernel_harness/` before measuring.
- The container's kernel tree is `/ceciliapeng/bell/aboutSHW/...` — a **separate copy** from
  `/home/intel/bell/aboutSHW/...`. Edit the latter, sync, then measure.
- Bare-metal CM builds need `LD_LIBRARY_PATH=/home/intel/river` (`libclangFEWrapper.so`).

## Phases

Run in order. Each phase's output decides whether the next is worth doing.

**P. `pre-profiler`** — *only when the target kernel is not already fixed by the user.* Run the
real e2e pipeline under cl_intercept (`ckh profile setup|run|report`), split the timeline into
prefill and generate, and report each kernel's share of its phase. That share is the **Amdahl
ceiling** on any e2e win. → share too small : stop here, and say so.
The pipeline command comes from `[profile].pipeline` in `platform.toml`, or you ask the user
for it — in the main conversation, before dispatching any subagent, since a subagent cannot
ask. Always pass `--no-prompt` so the CLI never blocks on stdin.

**0. `rig-warden`** — check for competing GPU work, pick the environment, establish the noise
floor. *Nothing below this line is trustworthy without it.*

**1. `roofline-analyst`** — measure the device's roofs empirically, separate compulsory
traffic from algorithm artifacts, report `current / floor`.
→ gap < 1.2× : go to phase 2 only. gap 1.5–3× : continue to phase 3. gap > 3× : phase 2 first.

**1b. `ckh kernelgen` (OPTIONAL)** — only when the kernel `profile` named has no sandbox copy
yet. Ports the plugin `.cm` plus its include closure into the sandbox and scaffolds a spec
and a compile test. It derives the signature and the full `-D` list, marks anything lifted
from host C++ as a GUESS, and refuses to invent input data. **Skip it** if you already have a
sandbox kernel path — then just point `kernels/<name>.py` at it.

**2. `algorithm-critic`** — is this decomposition right *for this shape*? Especially when a
traffic term turned out to be an artifact, or when performance swings across shapes.

**3. `budget-prober`** — ablation-decompose the implementation into a measured budget. Attack
terms by measured size, never by plausibility.

`ckh kernel-profile` is the cheap first pass at the same question: a pipe-cycle budget from
the IGC dump, with no GPU time spent. It ranks *hypotheses*; ablation is what turns one into
a measured term. Use it to choose what to ablate, never as the budget itself.

**4. Optimization loop**, per candidate:
   `bitexact-classifier` → implement → `equivalence-prover` → measure → ledger.
   Reject on measurement, not on taste. Record negatives with their numbers.

`ckh trial` is this loop with the bookkeeping automated: a branching tree with a budget, each
node pinning its own source snapshot, gates run in cost order (validate → equiv → bench). Its
`finalize` re-measures the shortlist against the baseline in ONE interleaved batch — stored
per-trial deltas came from different moments on a drifting box and are not comparable, so
they shortlist and never rank.

**5. `range-tuner`** — any constant that changed must be calibrated across its whole domain
and shipped with a machine-independent, bidirectionally-verified regression guard.

**6. `integrator`** — port to the plugin, build, run the real model, check the output text
**and** the accepted-token count against baseline.

## Hard rules

1. Measure the noise floor before quoting any effect.
2. An unmeasured hypothesis is not a conclusion. In this codebase four confident ones in a row
   were wrong (SLM bandwidth, marshal ILP, pipelining, scalar lse loads).
3. A test must be shown capable of failing. Prove non-vacuity; verify guards in both
   directions.
4. Constants get calibrated over a domain, never at a point.
5. The sandbox kernel is not the shipped kernel. Diff them code-only before transferring a
   claim.
6. Bisect one variable at a time, and establish the baseline first.
7. Bit-exact changes need no accuracy conversation; anything else needs the user's explicit
   sign-off, with the max_diff distribution measured, not the max alone.

## The ledger

`pa_small_q*.cm` carries its measured history in comments — the convention predates this
workflow and is load-bearing. **Write results into the kernel, including negatives**, with
numbers and the rig they came from. This is what stops a rejected idea being re-tried blind,
and it is how a stale "+1 to +3%" verdict was caught as a rig artifact once a stable rig
existed.

Record: what was measured, on what shape, the numbers, and whether it was adopted — and if
not, whether it was rejected on *measurement* or on *policy* (e.g. fp16 partials were
measured at one ulp and declined for accuracy risk; that distinction matters to whoever reads
it next).

## Known plugin invariants

See `integrator`. The ones that have already caused outages: `_order` vs `_stages` indexing in
`set_kernels`; includes must be inside `namespace KERNEL_NAME`; `add_stage` swallows
exceptions so route on `has_stage()`; JIT constants cannot follow runtime state.

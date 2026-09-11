# Optimizing a kernel with this harness

Written for an engineer picking up a CM kernel they did not write. Follow the order; the
gates matter more than the steps, and most of the value is in *not* proceeding.

Tool support is uneven — `doctor`, `bench`, `ledger`, `profile` and now `equiv` are
implemented; `roofs`, `ablate` and `sweep` are still manual with reference scripts. Each step
says which.

---

## Day 0 — setup (~10 min)

```bash
git clone <this repo> && cd cm-kernel-harness
cp platform.example.toml platform.toml && $EDITOR platform.toml   # your kernel repo paths
pip install -e .
pip install -e /path/to/aboutSHW/opencl                         # the clops binding used by measurements
ckh doctor
```

The second install is required when `exec.clops_path` is empty. It installs the `clops`
package from the sandbox checkout named by `repos.sandbox`; `pybind11` is installed by the
harness itself because clops imports it during measurement setup. If `exec.clops_path` points
at a checkout that already provides clops, keep the harness install and skip the editable
clops install.

`ckh doctor` must print `competing work none`. If it lists anything, stop — a benchmark
running in parallel once produced `ablation_off > ablation_on`, a physically impossible
ordering, and invalidated a whole batch of results.

Optional, if you work with Claude Code: `.claude/skills/cm-kernel-opt/install.sh <workspace>`
links the role definitions into your workspace. They also read fine as human checklists —
each one names the specific failure it exists to prevent.

---

## Step 0 — is this kernel worth optimizing at all? (implemented)

Skip only if someone has already fixed the target kernel *and* justified the choice.

```bash
ckh profile setup --install                                    # find or bootstrap cliloader
ckh profile run --out-dir results/profile --repeat 2 -- <your e2e pipeline command>
ckh profile report --dump-dir results/profile --kernel <kernel>
```

This runs the **real pipeline** under cl_intercept, splits the device timeline into prefill
and generate, and prints each kernel's share of its phase:

```
generate   3479.71 ms over   8532 calls  =  37.2% of the phase   -> Amdahl ceiling: 37.2%
```

That ceiling bounds any end-to-end win. A kernel 3x off its roofline that is 2% of the phase
is not worth a week, and phase matters as much as name — the short-query rungs dominate
*generate* and are nearly absent in *prefill*, so "attention is slow" means different things
depending on whether you are looking at TTFT or TPS.

Two results that are **not** performance results, and the report calls both out: zero matches
for your kernel means it was never enqueued (check the feature flag and the build), and a
concurrency factor above 1 means queues overlap, so the share bounds *device work* and the
wall-clock win is smaller still.

Collection is CLI-only on purpose — `profile run` executes an arbitrary command, so it is not
exposed as an MCP tool.

---

## Step 1 — read what is already settled (5 min, saves days)

```bash
ckh ledger <kernel>                    # everything
ckh ledger <kernel> --only rejected    # do not re-litigate these
ckh ledger <kernel> --only open        # started, not finished — free leads
```

Verdicts mean different things:

| verdict | what to do |
|---|---|
| `adopted` | already in; don't re-derive |
| `rejected-measurement` | revisit **only** if the rig or the kernel changed. One stale verdict was a rig artifact whose real sign was the opposite — so this is not "never", it is "know why you're retrying" |
| `rejected-policy` | measurement is fine, an owner said no. Ask the owner, don't re-measure |
| `invalid-probe` | the *measurement* was broken; the underlying question is still open |
| `open` | prototyped and promising, not finished |

Also read the kernel's own comment ledger. Findings are deliberately recorded in two places:
conclusions in the source where you'll read the code, full records here where they're
queryable.

---

## Step 2 — establish the rig *before* any claim (implemented)

```bash
ckh bench <kernel> --axis <your axes> --rounds 5
```

Any line flagged `<- spread exceeds noise floor` means differences smaller than that spread
are not resolvable on your box. Do not proceed by averaging harder. Fix the environment:
quiesce the machine, pin clocks, isolate the run. On the reference box bare metal drifts ~2x
within a session; that had to be solved before any of the real findings were possible.

The bench interleaves configs round-by-round and reports the minimum, because sequential
A-then-B charges B for A's heat.

---

## Step 3 — describe your kernel (the only real coding, ~1 hour)

Copy `kernels/pa_small_q.py`. You supply five things:

```python
jit(shape)      -> {DEFINE: value}      # anything a change might flip belongs here
dispatch(shape) -> (gws, lws)
args(shape,data)-> [kernel args in signature order]
inputs(shape)   -> {name: host tensor}  # MUST be deterministic per shape
build_options(shape) -> str
```

Then `SPEC = KernelSpec(...)` and `DEFAULT_AXES`.

**The one trap that will bite you:** `inputs` must be cached by shape. The generator in the
sandbox tests re-randomises on every call, and handing the two sides of an A/B separately
generated tensors produced **0/96 spurious mismatches** that looked exactly like a kernel
bug. `pa_small_q.py` uses `functools.lru_cache`; do the same.

Set `prod_source` if the kernel ships in another repo. The sandbox copy and the shipped copy
**are not the same file** — a plugin-only code path was once 17% of the runtime and entirely
absent from the sandbox copy, which invalidated a performance claim.

Sanity-check: `ckh bench <kernel>` and confirm the numbers match whatever you already know.

---

## Step 4 — roofline: how far from the floor? (manual)

Do this before optimizing anything. It decides whether optimizing is even the right activity.

1. **Measure** the roofs; never infer them. The reference device string reports `32 EUs`,
   ambiguous between EUs and Xe-cores by a factor of 8, which made an inferred peak-FLOPS
   figure useless. Write a streaming-bandwidth and a DPAS-only microbench under
   `microbench/`.
2. Count **compulsory** traffic and FLOPs from the *task*. Label everything else an
   **algorithm artifact** — e.g. fp32 split-K partials were ~34% of total DRAM traffic and
   exist only because the work is split over KV.
3. Report `gap = current / floor`, per shape (the binding roof moves: bandwidth at long
   context, *parallelism* at short).

Then branch:

- `gap < 1.2x` → micro-optimization is exhausted. Go to Step 5 (algorithm) only.
- `gap 1.5–3x` → real headroom. Go to Step 6 (budget).
- `gap > 3x` → suspect the decomposition. Step 5 first.

---

## Step 5 — is the algorithm right *for this shape*? (manual)

Read `.claude/agents/algorithm-critic.md`. Trigger signs, all observed in practice:

- A traffic term you labelled "artifact" is large.
- The same constant is optimal at one shape and 3x wrong at another — that is a decomposition
  whose balance point moves with shape, not a badly chosen number.
- Padding costs more than the work it pads (a padded thread once measured *slower* than a
  fully-valid one).
- A design premise in the kernel header that has never been re-measured.

Bound any alternative on paper before prototyping, and prototype in the sandbox only.

---

## Step 6 — budget, then attack the largest term (manual)

Add macro-guarded probes, default off, ablation-off path byte-identical:

```c
#ifndef ABLATE_X
#define ABLATE_X 0
#endif
```

**Two self-checks, both non-negotiable:**

1. **The probe must be cheaper than what it removes.** One probe replaced the DPAS with
   vector adds and measured 0.826 → **3.474 ms** — it measured the cost of *not* using DPAS.
2. **Probe-off must reproduce the baseline.** Hoisting a matrix out of a loop "just for the
   ablation" moved the baseline 0.52 → 1.02 ms and invalidated the comparison.

Attack by measured size, never by plausibility. Four confident hypotheses in a row were
wrong: SLM bandwidth, marshal ILP, barrier pipelining, and scalar loads in the reduce.

---

## Step 7 — per change: classify, prove, measure (implemented via `ckh equiv`)

```
bitexact-classifier → implement → ckh equiv → ckh bench → ckh ledger --add
```

**Classify first.** Bit-exact means every output bit identical, not "within tolerance". If
bit-exact, no accuracy conversation is needed. If not, quantify the cost — the max_diff
**distribution** over the correctness suite, not just the max — and get the owner's sign-off.

**Then prove it**, don't argue it. `ckh equiv <kernel>` runs the kernel against its declared
`KernelSpec.reference` (`TorchReference` — an independent Python/torch ground truth — or
`KernelReference` — another compiled kernel as baseline) and requires a `non_vacuous` note on
that reference: a documented reason the check is capable of failing, not just passing. Three
separate times, before this existed, a hand-rolled equivalence test reported success while
proving nothing:

- the input never exercised the change (an all-ones mask masks nothing),
- the two sides got different inputs,
- the "does it fire?" probe was itself vacuous (the chosen mask was what the causal mask
  already applies).

`ckh equiv` doesn't auto-detect any of these — it can't — but it forces the `non_vacuous`
note to exist and warns loudly when it doesn't, and it shares input data between both sides
by construction (the second failure mode above is no longer possible to get wrong). Worked
examples: `kernels/pa_small_q.py` (TorchReference) and
`kernels/pa_small_q_vs_baseline.py` (KernelReference) — both have a real, mutation-verified
non_vacuous note; copy whichever shape fits, or ask the `kernel-onboarder` agent to do it.
Older, pre-`ckh equiv` scripts are still useful for the axis-sweep patterns they used:
`aboutSHW/opencl/tests/pageatten/harness/equiv_template.py`.

**Record every outcome**, including rejections, with numbers:

```bash
ckh ledger <kernel> --add change="..." verdict=rejected-measurement \
  shape="q=16 15k" numbers="+7.8%" bit_exact=true note="why"
```

---

## Step 8 — calibrate constants over their domain (manual)

Any constant you touched must be swept against **every axis it interacts with**, and scored
`worst` and `mean` against the per-case best. A value tuned at one point was the worst of four
candidates across the domain (+168% worst / +40% mean) and caused a 3.3x regression at short
context.

Ship a **machine-independent** guard: assert a relative property measured in the same run,
not absolute ms. Provide a force-override env var and **verify the guard fails on the known-bad
value** — a guard only ever seen to pass is not known to work. Reference:
`test_15k_perf_comparison_ov_exp.py::test_small_q_partition_choice`.

---

## Step 9 — integrate (manual)

1. `diff` the sandbox and shipped kernels **code-only** (strip comments) and account for every
   difference before transferring any claim.
2. Port the exact hunk; verify your anchor matched **once** before writing.
3. Build, then run the real workload and check **two** things: output sanity *and* a
   quantitative signal (for speculative decoding, the accepted-token count — a broken path
   shows up as 0 accepted long before the text looks obviously wrong).
4. If it breaks, **single-variable bisect**, and establish the baseline first. Four changes
   were in flight once; reverting them one at a time isolated the culprit in four builds.

Known host-side traps are listed in `.claude/agents/integrator.md` — index aliasing between
stage arrays, includes needing to sit inside a namespace, exception-swallowing registration,
and JIT constants that cannot follow runtime state.

---

## If you only remember four things

1. Measure the noise floor before quoting any effect.
2. An unmeasured hypothesis is not a conclusion.
3. A test must be shown capable of failing.
4. Calibrate over a domain, never at a point.

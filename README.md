# cm-kernel-harness

Measurement, verification and calibration harness for Intel CM (C-for-Metal) GPU kernels.
Kernels stay in their own repos; this repo holds the **method**, the **tooling** and the
**findings**.

## Quickstart

```bash
cp platform.example.toml platform.toml   # the only machine-specific file
$EDITOR platform.toml                    # point it at your kernel repos
pip install -e .

ckh doctor                               # validate env, check for competing GPU work
ckh ledger pa_small_q --only rejected    # READ THIS FIRST -- what is already settled
ckh bench pa_small_q --axis q_len=6,16   # interleaved min-of-N over a shape grid
```

## Why it is shaped like this

Distilled from an optimization pass on `pa_small_q` where roughly half the effort went into
wrong hypotheses, self-inflicted regressions and tests that silently proved nothing. Three
design goals follow directly:

**Reusable across kernels.** Everything kernel-specific lives in one descriptor
(`kernels/<name>.py`): signature, jit defines, shape axes, dispatch, input generation.
`bench` / `equiv` / `ablate` / `sweep` are generic over it. Adopting a new kernel means
writing a descriptor, not another one-off script — the previous approach re-derived the same
facts by grepping a 1100-line source about fifteen times.

**Token-efficient.** Verbose output stays in the measurement environment. `runner.py` emits
exactly one machine-readable line; the CLI prints a table. Machine facts live in
`platform.toml` instead of being rediscovered. Results append to `results/*.jsonl` so a
comparison is a command, not a model re-reading logs. And `ckh ledger` exists because two
already-rejected ideas were re-prototyped from scratch — about four measurement round-trips
each — purely because the earlier verdicts were not queryable.

**Team-usable.** One config file per machine, `ckh doctor` to validate it, `tests/` so the
harness does not lie (it did, three times: an ablation that cost more than what it removed, a
generator that re-randomised between the two sides of an A/B, and an "is the mask firing?"
probe that could never fire).

## The discipline is in the tool, not in your memory

Bare metal on the reference box drifts ~2x within a session — the same config measured 0.52 ms
early and 1.59 ms late, which silently inverted the sign of several A/B tests. So `bench`
interleaves configs round-by-round (sequential A-then-B charges B for A's heat), reports the
**minimum**, and **flags any config whose spread exceeds `rig.noise_floor_pct`**:

```
q_len=16 ... | rescale=branch       0.834 ms  spread  48.6%  n=4  <- spread exceeds noise floor
q_len=16 ... | rescale=branchless   0.805 ms  spread  53.4%  n=4  <- spread exceeds noise floor

2 config(s) exceeded the 2.0% noise floor. Differences smaller than the spread are NOT
resolvable -- fix the rig before drawing conclusions.
```

That 3.5% difference is real (it reproduces on a stable rig) but is **not resolvable here**,
and the tool says so rather than letting it become a result. If your box cannot hold a clock,
that is an environment problem to solve separately — isolating the run, pinning clocks, or a
container — not something to paper over with more rounds.

## Layout

```
platform.example.toml     machine facts (paths, noise floor, rounds)
src/ckh/
  platform.py             config + env + competing-work check
  kernel.py               KernelSpec: the entire per-kernel surface
  runner.py               in-environment worker; emits ONE json line
  bench.py                interleaved min-of-N + table rendering
  ledger.py               append-only findings store
  cli.py                  ckh doctor | bench | ledger
kernels/pa_small_q.py     worked example descriptor
ledger/pa_small_q.jsonl   15 seeded findings from the original pass
.claude/                  agents + skill (the method, versioned with the tooling)
```

## Status

Working: `doctor`, `bench` (shape grids, A/B variants in one interleaved batch), `ledger`.

TODO: `equiv` (bit-exact A/B with mandatory non-vacuity proof), `ablate` (probe registry with
the two self-checks), `sweep` (domain calibration + worst/mean penalty scoring), `roofs`
(measured bandwidth and DPAS roofs — note the reference device string reports `32 EUs`, which
is ambiguous between EUs and Xe-cores by a factor of 8, so spec-sheet peaks are unusable).
Reference implementations for all four exist in
`aboutSHW/opencl/tests/pageatten/harness/` and need generalizing over `KernelSpec`.

## The eight roles

`.claude/agents/` — `rig-warden`, `roofline-analyst`, `algorithm-critic`, `budget-prober`,
`bitexact-classifier`, `equivalence-prover`, `range-tuner`, `integrator`. Flow and gates in
`.claude/skills/cm-kernel-opt/WORKFLOW.md`. Each one exists because of a specific failure; the
agent files name it.

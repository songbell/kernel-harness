# cm-kernel-harness

Measurement, verification and calibration harness for Intel CM (C-for-Metal) GPU kernels.
Kernels stay in their own repos; this repo holds the **method**, the **tooling** and the
**findings**.

New to this? **`docs/ONBOARDING.md`** is the step-by-step path for optimizing a kernel you
did not write, including which steps have tool support today and which are still manual.

## Quickstart

```bash
cp platform.example.toml platform.toml   # the only machine-specific file
$EDITOR platform.toml                    # point it at your kernel repos
pip install -e .

ckh doctor                               # validate env, check clops resolves, check for competing GPU work
ckh ledger pa_small_q --only rejected    # READ THIS FIRST -- what is already settled
ckh profile setup                        # e2e first: is this kernel even worth optimizing?
ckh bench pa_small_q --axis q_len=6,16   # interleaved min-of-N over a shape grid
ckh equiv pa_small_q --axis q_len=6      # check the kernel against its declared reference
```

## Before optimizing anything: `ckh profile`

A kernel 3x off its roofline that is 2% of the phase is not worth a week. `ckh profile` runs
the **real pipeline** under cl_intercept, splits the device timeline into prefill and
generate, and prints each kernel's share of its phase — the Amdahl ceiling on any e2e win.

```bash
ckh profile setup --install                                  # find or bootstrap cliloader
ckh profile run --out-dir results/profile --repeat 2 -- <your pipeline command>
ckh profile report --dump-dir results/profile --kernel pa_small_q
```

```
generate   3479.71 ms over   8532 calls  =  37.2% of the phase   -> Amdahl ceiling: 37.2%
```

The split is a decision, not a fact, so the report states the anchor and gap that produced it
and warns when the anchor leaks across the boundary. `clEnqueue*` memory ops are excluded from
kernel totals, and the concurrency factor is printed so a share of *device work* is never
mistaken for a share of wall time. Zero matches for the named kernel is reported as a
**configuration** result — it was never enqueued — not as a performance one.

## Driving it from Copilot (MCP)

The same commands are exposed as MCP tools, so a model can run the loop instead of a human
relaying CLI output into a chat. The server must live **on the box with the GPU** — every
tool compiles and runs a CM kernel — so there are two lanes:

```bash
cp .env.example .env
bash deploy/deploy_ckh.sh --local    # GPU is in this box  -> stdio, nothing on a socket
bash deploy/deploy_ckh.sh            # CKH_REMOTE_HOST set -> rsync + venv + systemd + HTTP
```

Both render `.vscode/mcp.json` from a committed template, so the server location has one
source of truth (`.env`). Tools: `doctor`, `list_kernels`, `bench`, `equiv`, `snapshot`,
`round`, `ledger_query`, `ledger_add`, `profile_status`, `profile_report`. Details and the
security posture (the port is a remote-execution primitive, so the remote lane refuses to
deploy without `MCP_AUTH_TOKEN`) are in [deploy/README.md](deploy/README.md).

`ckh profile run` is deliberately **not** an MCP tool: it executes an arbitrary user-supplied
pipeline command, which over a socket would be a second remote-execution primitive. Collection
stays on the CLI; the model gets detection and analysis.

The refusals are in the server, not in the prompt: `bench` / `equiv` / `round` raise rather
than measure while competing GPU work is running, and a kernel name that is not a plain
module name is rejected before it reaches `importlib`. `tests/test_mcp.py` covers the
handshake, the schemas and both refusals — a broken MCP server otherwise fails silently,
looking exactly like a model that chose not to call anything.


## Why it is shaped like this

Distilled from an optimization pass on `pa_small_q` where roughly half the effort went into
wrong hypotheses, self-inflicted regressions and tests that silently proved nothing. Three
design goals follow directly:

**Not tied to one sandbox repo.** Every measurement ultimately calls `clops` to compile/run a
CM kernel; that used to only resolve because it happened to live inside the reference
sandbox (aboutSHW). `exec.clops_path` in `platform.toml` now points at wherever your own
`clops`-providing checkout is, independently of `repos.sandbox` (the kernel *source* you're
optimizing) -- or leave it unset and `pip install -e <path>/opencl` once, outside this
config, if you'd rather have a normal installed `clops`. `ckh doctor` reports which one
actually resolved and fails loudly if neither does.

**Reusable across kernels.** Everything kernel-specific lives in one descriptor
(`kernels/<name>.py`): signature, jit defines, shape axes, dispatch, input generation, and
now a `reference` — either a `TorchReference` (independent Python/torch ground truth) or a
`KernelReference` (another compiled kernel as baseline). `bench` and `equiv` are generic over
it; `ablate` / `sweep` still need to be. Adopting a new kernel means writing a descriptor, not
another one-off script — the previous approach re-derived the same facts by grepping a
1100-line source about fifteen times. See `.claude/agents/kernel-onboarder.md`.

**Token-efficient.** Verbose output stays in the measurement environment. `runner.py` emits
exactly one machine-readable line; the CLI prints a table. Machine facts live in
`platform.toml` instead of being rediscovered. Results append to `results/*.jsonl` so a
comparison is a command, not a model re-reading logs. And `ckh ledger` exists because two
already-rejected ideas were re-prototyped from scratch — about four measurement round-trips
each — purely because the earlier verdicts were not queryable.

**Team-usable.** One config file per machine, `ckh doctor` to validate it, `tests/` so the
harness does not lie about itself. Currently covers the reference/equiv logic
(`tests/test_reference.py`, no GPU needed) -- the three original failure modes named below
(an ablation that cost more than what it removed, a generator that re-randomised between the
two sides of an A/B, an "is the mask firing?" probe that could never fire) happened in the
sandbox's ad hoc scripts before this harness existed and are guarded against structurally
now (shared cached inputs, `non_vacuous` as a required field) rather than by a retroactive
regression test for each historical incident.

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
.env.example              deployment facts (transport, remote host, token) -- copy to .env
deploy/                   deploy_ckh.sh (local + remote lanes), run_ckh.sh, ckh.service,
                          configure_mcp_client.sh -- see deploy/README.md
src/ckh/
  platform.py             config + env + competing-work check
  kernel.py               KernelSpec: the entire per-kernel surface
  runner.py               in-environment worker (bench); emits ONE json line
  equiv_runner.py         in-environment worker (equiv); emits ONE json line
  bench.py                interleaved min-of-N + table rendering
  equiv.py                reference check orchestration + table rendering
  reference.py            TorchReference / KernelReference -- what "correct" means
  ledger.py               append-only findings store
  clintercept.py          e2e profiling: locate/install cliloader, run, phase-split, Amdahl
  cli.py                  ckh doctor | profile | bench | equiv | snapshot | round | log | ledger
  mcp/                    the same surface as MCP tools (stdio + Streamable-HTTP)
kernels/pa_small_q.py             worked example: TorchReference (independent SDPA)
kernels/pa_small_q_vs_baseline.py worked example: KernelReference (pa_small_q_ov.cm baseline)
kb/                       correctness/fusion/memory/xpu/harness-design patterns, split by
                          concern -- see kb/README.md
ledger/pa_small_q.jsonl   findings from the original pass, plus the reference-mode additions
tests/test_reference.py   pure-Python coverage of reference.py / equiv.py, no GPU needed
tests/test_mcp.py         protocol coverage of the MCP layer, no GPU needed
.claude/                  agents + skill (the method, versioned with the tooling)
```

## Status

Working: `doctor`, `bench` (shape grids, A/B variants in one interleaved batch), `ledger`,
`equiv` (TorchReference and KernelReference, both with a required `non_vacuous` note),
`profile` (cl_intercept e2e budget, prefill/generate split, Amdahl ceiling).

TODO: `ablate` (probe registry with the two self-checks), `sweep` (domain calibration +
worst/mean penalty scoring), `roofs` (measured bandwidth and DPAS roofs — note the reference
device string reports `32 EUs`, which is ambiguous between EUs and Xe-cores by a factor of 8,
so spec-sheet peaks are unusable). Reference implementations for all three exist in
`aboutSHW/opencl/tests/pageatten/harness/` and need generalizing over `KernelSpec`, the same
way `equiv` now is.

## The eleven roles

`.claude/agents/` — `pre-profiler`, `rig-warden`, `roofline-analyst`, `algorithm-critic`,
`budget-prober`, `bitexact-classifier`, `equivalence-prover`, `range-tuner`, `integrator`,
`kernel-onboarder`, `reference-generator`. Flow and gates in
`.claude/skills/cm-kernel-opt/WORKFLOW.md`. Each one
exists because of a specific failure (or, for the last two, a specific missing capability);
the agent files name it. `ckh gen-reference <name>` (a plain script, not an agent) turns a
plain PyTorch `Model` (`kernels/pending/<name>_pytorch.py`) into the reference half of a
`kernels/<name>.py` for zero LLM cost; `reference-generator` is only invoked as a fallback
when it refuses (shape left completely unspecified -- a genuine judgment call). Either way,
`kernel-onboarder` picks up from there once a real CM kernel exists.

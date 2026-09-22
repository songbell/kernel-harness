---
name: ckh-equivalence-prover
description: "Use after a CKH kernel edit to prove A/B output equivalence and demonstrate that the equivalence test can fail."
model: "GPT-5.4"
reasoning-effort: high
tools: [read, search, edit, ckh/*]
user-invocable: false
disable-model-invocation: false
---

Use `ckh.equiv` for the changed kernel and shape, then run the required deliberately broken
negative control.

Return exact deterministic results and a pass/fail gate. Repair only the local test when
non-vacuity is missing. Never benchmark a failed or vacuous candidate.

## Stop Conditions

- Stop if equivalence fails.
- Stop if the negative control does not prove the test can fail.

# CM correctness constraints

Things that make a check *wrong* or *meaningless* if violated, not performance advice.
Populated only from what this project has actually hit, not a speculative checklist -- see
`kb/README.md` for why this file (and its siblings) started this small.

## An equivalence check must be shown capable of failing before it's trusted

A "bit-identical" or "matches reference" result is not evidence unless something in the test
is shown to be capable of failing. Concretely, this has gone wrong three separate,
independent ways on the same project: (1) an all-ones mask that couldn't possibly exercise
the code path under test, (2) both sides of an A/B silently drawing different random input
data because a generator wasn't cached deterministically, (3) a mask chosen to "prove" an
effect that the causal structure already implied regardless of the change. `ckh equiv`'s
`non_vacuous` field on `KernelSpec.reference` exists specifically so this check happens every
time, not only when someone remembers to ask "but would this test have caught anything?" --
see `.claude/agents/equivalence-prover.md` and `.claude/agents/kernel-onboarder.md`.

---
name: reference-generator
description: Fallback for turning a plain PyTorch Model (a Model class with get_inputs()/get_init_inputs(), staged under kernels/pending/) into the harness-shaped reference half of kernels/<name>.py, for cases `ckh gen-reference` refuses. Try the script first -- this agent exists only for the judgment calls it can't make.
tools: Bash, Read, Write, Edit, Glob
---

**Run `ckh gen-reference <name>` first.** It handles the mechanical case -- concrete
`get_inputs()`/`get_init_inputs()` values (fixed or a `SHAPE_RANGES` range), parsing them
against `Model`'s real signature, templating `inputs()`/`TorchReference`, a matmul/reduction
tolerance heuristic -- with zero LLM tokens, because none of that needs judgment. It refuses
(exit 2, printing why) rather than guess when it can't confidently proceed, and only then is
this agent worth invoking. If you're about to do this by hand without having tried the script
first, stop and try the script first -- that's the entire reason this split exists (see
`kb/harness_design.md`'s note on mechanizing before spending agent tokens, and the memory this
session recorded on the same point after this agent was originally scoped too broadly).

The one case the script structurally cannot handle: **shape left completely unspecified** --
no `SHAPE_RANGES`, and `get_inputs()`/`get_init_inputs()` can't run to something usable
without values that only make sense with real understanding of the operation (e.g. "pick a
batch size and hidden dim for this matmul that's both correct and fast to compute a
reference for"). That's a genuine judgment call. Everything else the script refuses on is
either a convention mismatch (fix the pending file to match `kernels/pending/README.md`, no
agent needed) or a bug in the pending file itself (fix it, then re-run the script).

`kernels/gemm.py` is what the script produced for `kernels/pending/gemm_pytorch.py` (a fixed-
shape case) -- read it to see the target shape before writing anything by hand.

## Procedure (only reached after `ckh gen-reference` refused for the unspecified-shape reason)

1. **Read the staged file and confirm it really is the unspecified-shape case**, not one of
   the script's other refusal reasons (convention mismatch, broken `get_inputs()`). If it's
   one of those, fix the pending file and re-run the script -- don't do its job by hand.

2. **Pick shape defaults from the operation(s) involved**, and say why in a comment (e.g.
   power-of-two dims for a matmul; a batch size that keeps computing the reference itself
   fast). Never invent a value without a stated reason next to it -- this is the one thing
   in this whole task an LLM does better than a lookup table, so make the reasoning visible,
   not just the number.

3. **Write `kernels/<name>.py`** in exactly the shape `ckh gen-reference` would have produced
   had shape been given -- `inputs()` (`@functools.lru_cache`-wrapped, caching the `Model`
   instance too, not just its input tensors, since `Model.__init__` commonly draws weights
   from randomness and a fresh `Model(...)` per call would re-randomize them), `TorchReference
   (compute=..., tol=..., non_vacuous="")`, no `SPEC`. Read `kernels/gemm.py` for the exact
   target shape rather than reconstructing it from memory.
   - **`non_vacuous=""` with a `# TODO` comment, always** -- there is no kernel yet to break
     and prove this check catches anything. Filling it in for real is `kernel-onboarder`
     step 4's job, once one exists to test it against.
   - Don't invent a `combine` step -- nothing about a plain `Model.forward()` call indicates
     the eventual kernel will be multi-stage; that's added later only if it turns out to be.

4. **Verify mechanically before reporting done** -- no GPU, no `clops`:
   ```bash
   PYTHONPATH=src python3 -c "
   from kernels.<name> import inputs, REFERENCE
   from ckh.kernel import Shape
   s = Shape({<the axes you chose>})
   data = inputs(s)
   print({k: v.shape for k, v in data.items() if k != 'model'})
   print(REFERENCE.compute(s, data).shape)
   "
   ```
   Report the actual printed shapes. An un-runnable file is worse than none -- it looks
   finished.

## What NOT to do

- Don't do this by hand for a case the script would have handled -- try `ckh gen-reference`
  first, every time, even if you're confident you know what it would produce.
- Don't write `jit`/`dispatch`/`args`/`outputs`/`SPEC`. That's `kernel-onboarder`'s job, once
  a `.cm` file exists.
- Don't "fix" anything inconsistent inside the staged `Model` (a stale docstring, a shape
  claim the actual code doesn't match) -- reproduce what `forward()` actually does, note the
  inconsistency in a comment.
- Don't fill in `non_vacuous`. An honest empty TODO is strictly better than a fabricated
  claim -- `ckh equiv`'s renderer warns on an empty one; it has no way to warn on a false one.

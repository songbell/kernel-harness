# Harness/tooling design patterns

Not a kernel-optimization pattern -- this is meta-knowledge about building the harness and
its agents/scripts, kept separate from `kb/correctness.md` / `fusion_patterns.md` /
`memory_patterns.md` / `xpu_optimizations.md`, which are all about kernel content. This file
exists because building this harness's own tooling has already produced one lesson worth not
re-learning.

## Mechanize the task before making it an agent

Before wrapping a task in a `.claude/agents/*.md` (a full subagent invocation), split it into
the mechanical part (parsing, templating, a heuristic from a lookup table or a source-text
substring match) and the part that needs genuine semantic judgment. Script the former; keep
the agent narrowly scoped to the latter, invoked only as a fallback. `ckh gen-reference` (a
plain script) vs the `reference-generator` agent is the concrete example: the common case --
a PyTorch `Model` with concrete `get_inputs()`/`get_init_inputs()` values -- is entirely
mechanical, and was originally routed through a full agent invocation for zero judgment
actually being exercised, before this was caught and split. A subagent invocation is a real,
non-trivial token cost (a full context load plus its own reasoning/tool-calling loop); a
script costs close to nothing beyond its output being read back.

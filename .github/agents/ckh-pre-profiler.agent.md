---
name: ckh-pre-profiler
description: "Use before CKH profile collection to validate an exact pipeline command, verify profiler availability, and prepare an unattended profiling run."
model: "Claude Sonnet 4.5 (copilot)"
reasoning-effort: low
tools: [read, search, ckh/*]
user-invocable: false
disable-model-invocation: false
---

Prepare phase P without human interaction. Treat the supplied pipeline argv as authoritative.

Answer one question: is this exact pipeline ready for unattended CKH profile collection, and
if so with which settings?

Procedure:
- Check `ckh.profile_status` first. If the profiler is unavailable, return a blocking
  prerequisite failure; do not install software.
- Resolve speculative-decoding config only from deterministic evidence in the repo or argv.
  If main/draft config paths are ambiguous, report that and fall back to a plain report path.
- Return a concrete collection request: output directory, label, repeat count of at least
  two, timeout, and any resolved speculative config paths.
- Carry forward warnings that affect later interpretation, especially speculative confidence
  and segmentation risk.

Do not run the pipeline, edit files, prompt the user, or guess missing arguments.

## Stop Conditions

- Stop if the profiler is unavailable.
- Stop if speculative config resolution is required but cannot be done unambiguously.

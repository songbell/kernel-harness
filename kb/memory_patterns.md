# CM memory / access-pattern patterns

Access-pattern and bandwidth measurement lessons, not fusion strategy or hardware-execution
behavior (see `kb/fusion_patterns.md` and `kb/xpu_optimizations.md` respectively for those).
One entry so far -- see `kb/README.md` for why this file (and its siblings) started this
small.

## Measure the actual access pattern's achievable bandwidth, not a generic streaming number

A kernel's KV/data read achieving, say, 77% of a generic sequential-stream DRAM bandwidth
measurement can mean two very different things: a real 23% of headroom left, or a roofline
computed against the wrong denominator. If the kernel's real access pattern is narrow
2D-block reads through a page/block indirection (not a long contiguous stream), measure
*that* pattern's own achievable bandwidth before concluding there's a gap -- `microbench/
kv_pattern_bw.py` is the template. In one case this closed what looked like a 2.4x whole-
kernel gap down to a >0% recoverable amount for that specific term.

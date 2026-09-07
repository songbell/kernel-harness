# `kernels/pending/`

Staging area for a plain PyTorch reference before it's wired into the harness. Files here are
never imported by `ckh` on their own -- nothing calls `ckh bench pending` -- they only get
read by the `reference-generator` agent, which turns one into `kernels/<name>.py`'s
`inputs()` + `TorchReference` half.

## Contract: `<name>_pytorch.py`

A `Model` class with `get_inputs()`/`get_init_inputs()` functions describing its forward/init
arguments -- nothing else required.

```python
class Model(nn.Module):
    def __init__(self, ...): ...
    def forward(self, x): ...       # the operation, exactly as you'd write it anywhere else

def get_inputs():        # -> list of positional args to Model.forward
    ...

def get_init_inputs():   # -> list of positional args to Model.__init__
    ...
```

That's the whole contract. You do not need to know anything about `Shape`, `KernelSpec`, or
`TorchReference` to write this file. Once it's here, run:

```bash
ckh gen-reference <name>
```

This is a plain script, not an agent -- it does the translation mechanically (parse
`get_inputs()`/`get_init_inputs()`'s concrete values, template `inputs()`/`TorchReference`,
pick a starting tolerance from which ops are present) for zero LLM cost. It refuses, rather
than guessing, only when shape is left completely unspecified -- that one case needs real
judgment and is handled by the `reference-generator` *agent* instead, invoked only then.

## How much shape detail you have to give

Three levels, least to most work:

1. **Nothing.** No shape constants anywhere in the file. The agent picks reasonable defaults
   for the operation(s) involved and says why in the generated file's comments -- never
   silently.
2. **A range.** Add a module-level `SHAPE_RANGES = {"batch_size": (32, 2048), ...}` dict. The
   agent samples a small number of representative points from it, not an exhaustive sweep,
   and says which points and why.
3. **Fixed values**, exactly like `gemm_pytorch.py` below (`batch_size = 1024` etc. at module
   level, consumed by `get_inputs()`/`get_init_inputs()`). Taken as-is, one point.

## Worked example

`gemm_pytorch.py` -- a fixed-shape case (level 3 above): matmul + divide + sum + scale.
`kernels/gemm.py` is what `reference-generator` produced from it; read that file's comments
to see the same three questions (shape source, tolerance, non_vacuous) answered for a real
case, not just described here.

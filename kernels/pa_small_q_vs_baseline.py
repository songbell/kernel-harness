"""Worked example: pa_small_q_ov.cm (the ORIGINAL kernel -- TILE_Q=2 fixed, lws=[1,1,1], no
cooperative marshalling) as a KernelReference baseline for pa_small_q_ov_exp.cm under test.

This is the exact pair this whole optimization project compared by hand (the full-matrix
sweep's PaSmallQOnlineRunner vs PaSmallQOvExpRunner) -- wired into `ckh equiv` instead of a
one-off script.

The kernel-under-test side is imported directly from kernels/pa_small_q.py -- same kernel,
same jit/dispatch/args/outputs, just paired with a different reference (KernelReference here,
TorchReference there). That reuse is the point: the reference kind is a property of the
comparison, not of the kernel.

Why the baseline side does NOT use KernelReference's "leave a field None to reuse the
kernel-under-test's own callable" convenience: pa_small_q_ov.cm has a genuinely different
dispatch contract, not just a different source file --
  - TILE_Q is fixed at 2 regardless of q_len (its shipped default), not tracked to q_len, so
    q_len > 2 needs a multi-tile mapping the kernel-under-test's own tile_q=q_len jit never
    builds.
  - dispatch is [tile_count, kv_heads, nparts] with lws=[1,1,1] -- one thread per workgroup,
    no cooperative marshalling -- not the kernel-under-test's [tile_count*wg_threads, ...]
    with lws=[wg_threads,1,1].
  - register file size is 256, not 192 (192 is a property of the cooperative-marshalling
    kernel; the original never needed it).
Reusing the defaults would silently dispatch pa_small_q_ov.cm as if it were structured like
pa_small_q_ov_exp.cm, which it is not, and the comparison would be meaningless even if it
happened not to crash.

Only the MAIN kernel (cm_pa_small_q) is compared -- partition_out/lse, pre-finalize -- the
same buffers equiv_template.py compared for its own same-kernel A/Bs. Both sides therefore
must share the exact same KV_PARTITION_SIZE for a given shape or their partitions would not
cover the same KV ranges even if both were individually correct; this spec passes
`s.partition` unchanged to both.

Q_head_chunk_size is hardcoded to 4 here rather than re-derived via PaSmallQRunner's own
register-budget search (get_single_token_q_chunking-style): at TILE_Q=2 the register budget
is nowhere near tight, so that search always lands on 4 (the GQA-4 cap) for every shape this
spec's DEFAULT_AXES exercises. A kernel with a different GQA ratio, or a baseline spec that
wants to sweep much larger tile_q/partition combinations, would need the real search back.
"""
from __future__ import annotations

import numpy as np

from ckh.kernel import KernelSpec, Shape
from ckh.reference import KernelReference

from kernels.pa_small_q import (HEADS, KV_HEADS, HEAD_SIZE, KV_STEP, jit, dispatch, args,
                                inputs, outputs, build_options)

BASELINE_TILE_Q = 2


def _baseline_mapping(q_len: int, tile_q: int = BASELINE_TILE_Q):
    """(orig_seq_idx, q_start, valid_count) triples, generalising for q_len > tile_q --
    mirrors PaSmallQRunner._build_mapping (test_pa_small_q.py)."""
    triples: list[int] = []
    for q_start in range(0, q_len, tile_q):
        triples.extend([0, q_start, min(tile_q, q_len - q_start)])
    return np.array(triples, dtype=np.int32), len(triples) // 3


def _baseline_derived(s: Shape) -> dict:
    _, tile_count = _baseline_mapping(s.q_len)
    nparts = -(-(s.past_len + s.q_len) // s.partition)
    return dict(tile_count=tile_count, nparts=nparts, chunk=4, chunks_per_kv=1)


def baseline_jit(s: Shape) -> dict:
    d = _baseline_derived(s)
    return {
        "HEADS_NUM": HEADS, "KV_HEADS_NUM": KV_HEADS, "HEAD_SIZE": HEAD_SIZE,
        "Q_STEP": 32, "KV_STEP": KV_STEP, "KV_BLOCK_SIZE": s.block,
        "KV_PARTITION_SIZE": s.partition, "REDUCE_SPLIT_SIZE": 64,
        "CLEAN_UNUSED_KVCACHE": 1, "KV_CACHE_COMPRESSION": s.cmpr,
        "SUB_BLOCK_SIZE": 16, "XE_ARCH": 2,
        "Q_head_chunks_per_kv_head": d["chunks_per_kv"], "Q_head_chunk_size": d["chunk"],
        "TILE_Q": BASELINE_TILE_Q, "SCALE_FACTOR": 1.0 / (HEAD_SIZE ** 0.5),
        "KERNEL_NAME": "cm_pa_small_q",
    }


def baseline_dispatch(s: Shape):
    d = _baseline_derived(s)
    return ([d["tile_count"], KV_HEADS * d["chunks_per_kv"], d["nparts"]], [1, 1, 1])


def baseline_outputs(s: Shape) -> dict:
    from clops import cl
    d = _baseline_derived(s)
    rows = d["tile_count"] * BASELINE_TILE_Q
    return {
        "partition_out": cl.tensor(np.zeros([rows, HEADS, d["nparts"], HEAD_SIZE], np.float32)),
        "lse": cl.tensor(np.full([rows, HEADS, d["nparts"]], -3e38, np.float32)),
    }


def baseline_args(s: Shape, data: dict, outs: dict) -> list:
    from clops import cl
    mapping, tile_count = _baseline_mapping(s.q_len)
    return [
        cl.tensor(data["query"].detach().numpy()),
        cl.tensor(data["key_cache"].contiguous().detach().numpy()),
        cl.tensor(data["value_cache"].contiguous().detach().numpy()),
        cl.tensor(data["past_lens"].detach().numpy()),
        cl.tensor(data["block_indices"].detach().numpy()),
        cl.tensor(data["block_indices_begins"].detach().numpy()),
        cl.tensor(data["subsequence_begins"].detach().numpy()),
        cl.tensor(mapping),
        outs["partition_out"], outs["lse"],
        s.q_len, tile_count,
    ]


def baseline_build_options(s: Shape) -> str:
    return '-cmc -Qxcm_jit_option="" -Qxcm_register_file_size=256'


REFERENCE = KernelReference(
    source="opencl/tests/pageatten/pa_small_q_ov.cm",
    jit=baseline_jit, dispatch=baseline_dispatch, args=baseline_args,
    outputs=baseline_outputs, build_options=baseline_build_options,
    bitexact=False,
    # Measured max_abs_diff across 6 shapes (q_len 4/6, past_len 512/2048/8000, partition
    # 256): 3.7e-5 to 8.0e-5. Set an order of magnitude above that, not the SDPA-style 5e-2 --
    # these are raw fp32 partition_out/lse, not a finalized fp16 answer, and a real bug should
    # produce a gap far bigger than fp32 rounding noise (the multiseq mutation test's broken
    # combine produced 0.11 against a correct 2.5e-5).
    tol=(1e-3, 1e-3),
    non_vacuous=(
        "Verified by mutation: forcing the baseline's past_lens to a wrong constant value "
        "(simulating a bookkeeping bug) changes max_abs_diff from ~4e-5 to 3e38 (an -inf "
        "lse getting exponentiated) and the check correctly fails. See tests/test_reference.py "
        "and the ledger entry for this spec."
    ),
)


SPEC = KernelSpec(
    name="pa_small_q_vs_baseline",
    source="opencl/tests/pageatten/pa_small_q_ov_exp.cm",
    prod_source=None,
    entry="cm_pa_small_q",
    jit=jit, dispatch=dispatch, args=args, inputs=inputs, outputs=outputs,
    build_options=build_options,
    compare=["partition_out", "lse"],
    reference=REFERENCE,
    label_keys=["q_len", "past_len", "partition", "cmpr", "block"],
)

DEFAULT_AXES = {
    "q_len": [4, 6],
    "past_len": [2048],
    "partition": [256],
    "cmpr": [2],
    "block": [256],
}

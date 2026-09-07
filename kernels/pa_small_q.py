"""Descriptor for the paged-attention small-q decode kernel (q_len > 1 spec-decode window).

This is the whole per-kernel cost of adopting the harness: no bespoke measurement script.
"""
from __future__ import annotations

import functools

import numpy as np

from ckh.kernel import KernelSpec, Shape
from ckh.reference import TorchReference

HEADS, KV_HEADS, HEAD_SIZE, KV_STEP = 32, 8, 128, 16
ROWS_PER_THREAD = 8


@functools.lru_cache(maxsize=64)
def _inputs_cached(q_len: int, past_len: int, block: int, cmpr: int):
    """Deterministic per shape.

    The underlying generator re-randomises on every call. Handing two sides of an A/B
    separately-generated tensors once produced 0/96 spurious mismatches that looked exactly
    like a kernel bug, so the caching here is load-bearing, not an optimisation.

    Uses test_pa_small_q's _build_inputs (not test_15k_perf_comparison's
    _build_small_q_inputs, a perf-only variant with no ground truth) specifically because it
    also computes an SDPA `expected` -- needed for the TorchReference below. This costs
    nothing at measurement time: it runs once per distinct shape (this cache), not once per
    timed loop iteration, so `ckh bench`'s numbers are unaffected.
    """
    from test_pa_small_q import SmallQCase, _build_inputs
    case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                      block_size=block, past_len=past_len, q_len=q_len,
                      kv_cache_compression=cmpr, tile_q=q_len, partition_block_num=1)
    return _build_inputs(case)


def _derived(s: Shape) -> dict:
    """Values the kernel's #defines and the dispatch both need."""
    chunk = s.values.get("chunk", 4)
    tile_q = s.values.get("tile_q", s.q_len)
    q_rows = chunk * tile_q
    wg_threads = max(1, -(-q_rows // ROWS_PER_THREAD))
    nparts = -(-(s.past_len + s.q_len) // s.partition)
    return dict(chunk=chunk, tile_q=tile_q, wg_threads=wg_threads, nparts=nparts,
                chunks_per_kv=(HEADS // KV_HEADS) // chunk)


def jit(s: Shape) -> dict:
    d = _derived(s)
    return {
        "HEADS_NUM": HEADS, "KV_HEADS_NUM": KV_HEADS, "HEAD_SIZE": HEAD_SIZE,
        "KV_STEP": KV_STEP, "KV_BLOCK_SIZE": s.block, "KV_PARTITION_SIZE": s.partition,
        "KV_CACHE_COMPRESSION": s.cmpr, "SUB_BLOCK_SIZE": 16, "XE_ARCH": 2,
        "Q_head_chunks_per_kv_head": d["chunks_per_kv"], "Q_head_chunk_size": d["chunk"],
        "TILE_Q": d["tile_q"], "SCALE_FACTOR": 1.0 / (HEAD_SIZE ** 0.5),
        "HAS_QQ_BIAS": s.values.get("qq_bias", 0),
        "KERNEL_NAME": "cm_pa_small_q",
    }


def dispatch(s: Shape):
    d = _derived(s)
    return ([1 * d["wg_threads"], KV_HEADS * d["chunks_per_kv"], d["nparts"]],
            [d["wg_threads"], 1, 1])


def outputs(s: Shape) -> dict:
    """Built once per measured call (bench: once per loop iteration, same cost as the old
    inline construction it replaces; equiv: once, then read back)."""
    from clops import cl
    d = _derived(s)
    rows = d["tile_q"]
    return {
        "partition_out": cl.tensor(np.zeros([rows, HEADS, d["nparts"], HEAD_SIZE], np.float32)),
        "lse": cl.tensor(np.full([rows, HEADS, d["nparts"]], -3e38, np.float32)),
    }


def args(s: Shape, data: dict, outs: dict) -> list:
    from clops import cl
    d = _derived(s)
    out = [cl.tensor(data["query"].detach().numpy()),
           cl.tensor(data["key_cache"].contiguous().detach().numpy()),
           cl.tensor(data["value_cache"].contiguous().detach().numpy()),
           cl.tensor(data["past_lens"].detach().numpy()),
           cl.tensor(data["block_indices"].detach().numpy()),
           cl.tensor(data["block_indices_begins"].detach().numpy()),
           cl.tensor(data["subsequence_begins"].detach().numpy())]
    if s.values.get("qq_bias", 0):
        spec_n = s.q_len
        out += [cl.tensor(np.ones(spec_n * spec_n, np.uint8)),
                cl.tensor(np.array([0, spec_n * spec_n], np.int32))]
    out += [cl.tensor(np.array([0, 0, s.q_len], np.int32)),
            outs["partition_out"], outs["lse"],
            d["tile_q"], 1]
    return out


def inputs(s: Shape) -> dict:
    return _inputs_cached(s.q_len, s.past_len, s.block, s.cmpr)


def _torch_compute(s: Shape, data: dict):
    return data["expected"]


def _torch_combine(s: Shape, kernel_outputs: dict):
    """Independent torch re-implementation of pa_small_q_finalization.cm's logsumexp merge
    (REDUCE_OPT==2's fused form -- sum unnormalised, divide once): out[row,head] =
    sum_p(partition_out[row,head,p] * exp(lse[row,head,p] - max_p lse)) / sum_p(that weight).

    Deliberately not calling the reduce *kernel* -- that would make this "compare the kernel
    against its own reduce step reimplemented," which proves nothing about either. This is
    the actual algorithm, worked out independently.
    """
    partition_out = kernel_outputs["partition_out"]   # [rows, HEADS, nparts, HEAD_SIZE]
    lse = kernel_outputs["lse"]                       # [rows, HEADS, nparts]
    lse_max = lse.max(axis=-1, keepdims=True)
    w = np.exp(lse - lse_max)
    denom = w.sum(axis=-1, keepdims=True)
    merged = (partition_out * w[..., None]).sum(axis=-2) / denom      # [rows, HEADS, HEAD_SIZE]
    return merged[: s.q_len]


def build_options(s: Shape) -> str:
    # 192 and not 256: measured 0.832 vs 0.963 ms at q_len=16 / partition 512. 128 spills.
    return '-cmc -Qxcm_jit_option="" -Qxcm_register_file_size=192'


REFERENCE = TorchReference(
    compute=_torch_compute,
    combine=_torch_combine,
    bitexact=False,
    tol=(5e-2, 5e-2),           # quantised cmpr modes need this looser than fp16-only SDPA
    non_vacuous=(
        "expected is a real SDPA computation over dequantised, causally-masked K/V (see "
        "test_pa_small_q._build_inputs), not a value derived from the kernel under test -- "
        "a kernel that ignored past_lens, dropped the causal mask, or read the wrong K/V "
        "block would fail this at every shape, not just an edge case. Verified by mutation: "
        "feeding a deliberately wrong past_lens makes `ckh equiv pa_small_q` fail (see the "
        "harness's own tests/test_reference.py and the ledger entry recording the run)."
    ),
)

SPEC = KernelSpec(
    name="pa_small_q",
    source="opencl/tests/pageatten/pa_small_q_ov_exp.cm",
    prod_source="src/plugins/intel_gpu/src/graph/impls/cm/pa_small_q.cm",
    entry="cm_pa_small_q",
    jit=jit, dispatch=dispatch, args=args, inputs=inputs, outputs=outputs,
    build_options=build_options,
    compare=["partition_out", "lse"],
    reference=REFERENCE,
    label_keys=["q_len", "past_len", "partition", "cmpr", "block"],
)

# Defaults for `ckh bench pa_small_q` with no axes given.
DEFAULT_AXES = {
    "q_len": [6, 16],
    "past_len": [15360],
    "partition": [512],
    "cmpr": [2],
    "block": [256],
}

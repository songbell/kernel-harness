"""Descriptor for the paged-attention small-q decode kernel (q_len > 1 spec-decode window).

This is the whole per-kernel cost of adopting the harness: no bespoke measurement script.
"""
from __future__ import annotations

import functools

import numpy as np

from ckh.kernel import KernelSpec, Shape

HEADS, KV_HEADS, HEAD_SIZE, KV_STEP = 32, 8, 128, 16
ROWS_PER_THREAD = 8


@functools.lru_cache(maxsize=64)
def _inputs_cached(q_len: int, past_len: int, block: int, cmpr: int):
    """Deterministic per shape.

    The underlying generator re-randomises on every call. Handing two sides of an A/B
    separately-generated tensors once produced 0/96 spurious mismatches that looked exactly
    like a kernel bug, so the caching here is load-bearing, not an optimisation.
    """
    from test_15k_perf_comparison import _build_small_q_inputs
    from test_pa_small_q import SmallQCase
    case = SmallQCase(num_heads=HEADS, num_kv_heads=KV_HEADS, head_size=HEAD_SIZE,
                      block_size=block, past_len=past_len, q_len=q_len,
                      kv_cache_compression=cmpr, tile_q=q_len, partition_block_num=1)
    return _build_small_q_inputs(case)


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


def args(s: Shape, data: dict) -> list:
    from clops import cl
    d = _derived(s)
    rows = d["tile_q"]
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
            cl.tensor(np.zeros([rows, HEADS, d["nparts"], HEAD_SIZE], np.float32)),
            cl.tensor(np.full([rows, HEADS, d["nparts"]], -3e38, np.float32)),
            d["tile_q"], 1]
    return out


def inputs(s: Shape) -> dict:
    return _inputs_cached(s.q_len, s.past_len, s.block, s.cmpr)


def build_options(s: Shape) -> str:
    # 192 and not 256: measured 0.832 vs 0.963 ms at q_len=16 / partition 512. 128 spills.
    return '-cmc -Qxcm_jit_option="" -Qxcm_register_file_size=192'


SPEC = KernelSpec(
    name="pa_small_q",
    source="opencl/tests/pageatten/pa_small_q_ov_exp.cm",
    prod_source="src/plugins/intel_gpu/src/graph/impls/cm/pa_small_q.cm",
    entry="cm_pa_small_q",
    jit=jit, dispatch=dispatch, args=args, inputs=inputs, build_options=build_options,
    compare=["partition_out", "lse"],
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

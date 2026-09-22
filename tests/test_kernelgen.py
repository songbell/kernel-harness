"""Coverage for `ckh kernelgen`'s parsing. No GPU, no openvino checkout -- synthetic sources.

The point is the two directions the required-`-D` list is built from. Each half misses cases
the other catches, and a silently-incomplete list is the failure mode that matters: it does
not look wrong, it just makes the ported kernel fail to compile with a message that points
at the kernel rather than at the tool.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ckh import kernelgen as kg  # noqa: E402

CM = '''
// Copyright
#include "helper.hpp"

namespace KERNEL_NAME {

#if XE_ARCH==1
#define REG_N 8
#else
#define REG_N 16
#endif

#define STEPS (PARTITION_SIZE / KV_STEP)

#if COMPRESSION
#define ELEM uint8_t
#else
#define ELEM half
#endif

extern "C" _GENX_MAIN_ void KERNEL_NAME(
    half* query [[type("svmptr_t")]],
    ELEM* key [[type("svmptr_t")]],
#if HAS_BIAS
    uchar* bias [[type("svmptr_t")]],
#endif
    float* out [[type("svmptr_t")]],
    int token_count
    ) {
    // HEAD_SIZE only ever appears in ordinary code -- no preprocessor scan can see it.
    const uint n = cm_global_id(0) * HEAD_SIZE;
}
}  // namespace KERNEL_NAME
'''

CL = '''
#include "helper.hpp"

__kernel void foo_cl(
    __global const float* query,
#if HAS_BIAS
    __global const uchar* bias,
#endif
    __global float* out,
    int token_count
) {
    uint n = get_global_id(0) + HEAD_SIZE;
}
'''

DUMPED_CL = '''
void block_store(global half* p, half v) { }

//====================================================
// Kernel template: sdpa_micro
// Kernel name: sdpa_micro__generate_6397930029999432054__sa
#define KERNEL(name) __kernel void sdpa_micro__generate_6397930029999432054__sa

KERNEL(micro_sdpa)(
    __global const int* shape_info,
    const global half *K,
    const global half *Q,
    global half *A,
    const __global int* blocked_indexes_start_and_gws_mapping
) {
    const int x = shape_info[0];
}
'''

HPP = "#define HELPER_CONST 4\n"

CPP = '''
JitConstants FooGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = Base::get_jit_constants(params);
    jit.make("HEAD_SIZE", desc->k_head_size);
    jit.make("PARTITION_SIZE", partition);
    jit.add(make_jit_constant("SCALE", scale_factor));
    jit.make("UNRELATED_SIBLING", 7);
    return jit;
}

DispatchDataFunc FooGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params) {
        auto& wgs = kd.params.workGroups;
        wgs.global = {tile_count, heads, parts};
        wgs.local = {1, 1, 1};
    }};
}

Arguments FooGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;
    args.push_back({Types::INPUT, Idx::QUERY});
    args.push_back({Types::SCALAR, 0});
    return args;
}

JitConstants BarGenerator::get_jit_constants(const kernel_impl_params& params) const {
    jit.make("NOT_FOOS", 1);
    return jit;
}
'''


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    d = tmp_path / "cm"
    d.mkdir()
    (d / "foo.cm").write_text(CM)
    (d / "helper.hpp").write_text(HPP)
    (d / "foo_gen.cpp").write_text(CPP)
    return d


def test_signature_is_parsed_in_order_with_guards(tree):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    assert [p.name for p in src.params] == ["query", "key", "bias", "out", "token_count"]
    assert next(p for p in src.params if p.name == "bias").guard == "HAS_BIAS"
    # A parameter outside the #if must NOT inherit the guard.
    assert next(p for p in src.params if p.name == "out").guard == ""


def test_include_closure_is_walked(tree):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    assert [p.name for p in src.includes] == ["helper.hpp"]
    assert not src.missing_includes
    assert "HELPER_CONST" in src.defined


def test_missing_include_is_reported_not_swallowed(tmp_path):
    p = tmp_path / "x.cm"
    p.write_text('#include "nowhere.hpp"\n' + CM.split("#include", 1)[1].split("\n", 1)[1])
    src = kg.parse_cm(p, [tmp_path])
    assert "nowhere.hpp" in src.missing_includes


def test_undefined_alone_misses_code_only_macros(tree):
    """Guards the reason `required()` is a union and not just the preprocessor scan."""
    src = kg.parse_cm(tree / "foo.cm", [tree])
    assert {"XE_ARCH", "COMPRESSION", "HAS_BIAS", "KV_STEP", "PARTITION_SIZE"} <= src.undefined
    assert "HEAD_SIZE" not in src.undefined          # used only in code -> invisible here
    assert "REG_N" not in src.undefined              # defined locally -> not a -D
    assert not ({"half", "uint8_t", "defined"} & src.undefined)


def test_required_is_the_union_and_excludes_sibling_kernels(tree):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    hints = kg.scrape_host(tree, "FooGenerator")
    req = src.required(hints)
    assert "HEAD_SIZE" in req                        # recovered via the host jit list
    assert "XE_ARCH" in req                          # recovered via the preprocessor scan
    # The host sets these, but this source never mentions them -- admitting them would put
    # a sibling kernel's constants into this kernel's compile line.
    assert "UNRELATED_SIBLING" not in req
    assert "NOT_FOOS" not in req


def test_host_scrape_is_scoped_to_one_generator(tree):
    hints = kg.scrape_host(tree, "FooGenerator")
    assert hints.jit["SCALE"] == "scale_factor"
    assert "NOT_FOOS" not in hints.jit
    assert hints.gws == "tile_count, heads, parts"
    assert hints.lws == "1, 1, 1"
    assert len(hints.args) == 2


def test_entry_macro_resolves_to_the_profiled_name(tree):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    assert src.entry == "KERNEL_NAME"
    src.resolve_entry("cm_foo")
    assert src.entry == "cm_foo"


def test_a_non_entry_file_is_refused_not_half_ported(tmp_path):
    p = tmp_path / "header_only.cm"
    p.write_text("#define A 1\nint helper() { return 0; }\n")
    with pytest.raises(kg.Refused):
        kg.parse_cm(p, [tmp_path])


def test_port_refuses_to_clobber(tree, tmp_path):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    sandbox = tmp_path / "sandbox"
    kg.port(sandbox, "k", src, "foo")
    with pytest.raises(kg.Refused):
        kg.port(sandbox, "k", src, "foo")
    kg.port(sandbox, "k", src, "foo", overwrite=True)


def test_port_records_provenance(tree, tmp_path):
    src = kg.parse_cm(tree / "foo.cm", [tree])
    m = kg.port(tmp_path / "sandbox", "k", src, "foo")
    assert m["origin"] == str(tree / "foo.cm")
    assert m["includes"] == ["helper.hpp"]
    on_disk = json.loads((tmp_path / "sandbox" / "k" / "foo.kernelgen.json").read_text())
    assert on_disk == m
    assert "PORTED by `ckh kernelgen`" in (tmp_path / "sandbox" / "k" / "foo.cm").read_text()


def test_generated_files_are_valid_python_and_fail_loudly(tree, tmp_path):
    """A scaffold that imports but silently returns nothing would be worse than no file:
    `ckh bench` would measure a kernel launched with empty args."""
    import py_compile

    src = kg.parse_cm(tree / "foo.cm", [tree])
    src.resolve_entry("cm_foo")
    hints = kg.scrape_host(tree, "FooGenerator")
    sandbox, repo = tmp_path / "sandbox", tmp_path / "repo"
    (repo / "kernels").mkdir(parents=True)
    generated = kg.prepare(
        repo,
        sandbox,
        "k",
        src,
        "foo",
        hints,
        focus_axes={"q_len": [6, 16], "past_len": [15360]},
    )
    test = Path(generated["compile_test"])
    spec = Path(generated["spec"])
    wrapper = Path(generated["wrapper"])
    correctness = Path(generated["correctness_test"])
    perf = Path(generated["perf_test"])
    for path in (test, spec, wrapper, correctness, perf):
        py_compile.compile(str(path), doraise=True)

    ns: dict = {}
    exec(compile(spec.read_text(), str(spec), "exec"), ns)
    assert ns["SPEC"].entry == "cm_foo"
    assert ns["DEFAULT_AXES"] == {"q_len": [6, 16], "past_len": [15360]}
    for fn in ("dispatch", "inputs", "outputs"):
        with pytest.raises(NotImplementedError):
            ns[fn](None)
    assert 'runner.enqueue(*ARGS)' in test.read_text()
    assert 'TARGET_Q_LENS = tuple(FOCUS_AXES.get("q_len", [6, 16]))' in correctness.read_text()
    assert 'TARGET_PAST_LENS = tuple(FOCUS_AXES.get("past_len", [15360]))' in perf.read_text()


def test_candidate_matching_strips_the_cm_prefix(tmp_path, monkeypatch):
    prod = tmp_path / "ov"
    (prod / kg.CM_SUBDIR).mkdir(parents=True)
    (prod / kg.CL_SUBDIR).mkdir(parents=True)
    for n in ("pa_small_q.cm", "pa_small_q_finalization.cm", "other.cm"):
        (prod / kg.CM_SUBDIR / n).write_text("")
    (prod / kg.CL_SUBDIR / "softmax.cl").write_text("")
    hits = kg.candidates(tmp_path / "ov", "cm_pa_small_q")
    assert [p.name for p in hits] == ["pa_small_q.cm", "pa_small_q_finalization.cm"]
    assert kg.candidates(tmp_path / "ov", "cm_other") == [prod / kg.CM_SUBDIR / "other.cm"]
    assert kg.candidates(tmp_path / "ov", "softmax") == [prod / kg.CL_SUBDIR / "softmax.cl"]


def test_parse_source_accepts_opencl_c_entry(tmp_path):
    src_path = tmp_path / "foo.cl"
    helper = tmp_path / "helper.hpp"
    src_path.write_text(CL)
    helper.write_text(HPP)
    src = kg.parse_source(src_path, [tmp_path])
    assert src.entry == "foo_cl"
    assert [p.name for p in src.params] == ["query", "bias", "out", "token_count"]
    assert next(p for p in src.params if p.name == "bias").guard == "HAS_BIAS"


def test_parse_source_prefers_dumped_kernel_macro_over_helper_void(tmp_path):
    src_path = tmp_path / "dumped.cl"
    src_path.write_text(DUMPED_CL)
    src = kg.parse_source(src_path, [tmp_path])
    assert src.entry == "sdpa_micro__generate_6397930029999432054__sa"
    assert [p.name for p in src.params] == [
        "shape_info", "K", "Q", "A", "blocked_indexes_start_and_gws_mapping"
    ]


def test_dump_candidates_match_runtime_name_suffixes(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    dumped = dump / "sdpa_micro__generate_6397930029999432054__sa.cl"
    dumped.write_text("__kernel void sdpa_micro__generate_6397930029999432054__sa(int x) {}")
    hits = kg.dump_candidates(dump, "sdpa_micro__generate_6397930029999432054__sa")
    assert hits == [dumped]
    hits = kg.dump_candidates(dump, "sdpa_micro__generate")
    assert hits == [dumped]


def test_configured_dump_sources_reads_profile_pipeline_env():
    path = kg.configured_dump_sources({"profile": {"pipeline_env": {"OV_GPU_DUMP_SOURCES_PATH": "C:/tmp/ovdump"}}})
    assert path == Path("C:/tmp/ovdump")


def test_dump_sources_from_profile_dir_reads_profile_run_metadata(tmp_path):
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "profile_run.json").write_text(json.dumps({"dump_sources": "C:/tmp/from-meta"}))
    assert kg.dump_sources_from_profile_dir(profile) == Path("C:/tmp/from-meta")


def test_port_preserves_source_extension(tmp_path):
    src_path = tmp_path / "foo.cl"
    src_path.write_text(CL)
    (tmp_path / "helper.hpp").write_text(HPP)
    src = kg.parse_source(src_path, [tmp_path])
    out = kg.port(tmp_path / "sandbox", "k", src, "foo")
    assert out["source_ext"] == ".cl"
    assert Path(out["sandbox_kernel"]).suffix == ".cl"

"""In-environment worker for `ckh kernel-profile`: compile with the IGC dump flags on and
report where the assembly landed.

Freshness is the whole risk here. IGC writes `<entry>.asm` into the working directory, and a
dump left by an earlier build is indistinguishable from one this compile produced -- analysing
it means reporting on the previous version of the kernel, which is the same trap as measuring
before syncing. So the existing file's mtime is recorded first and a dump that did not move is
refused rather than parsed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

RESULT_PREFIX = "CKH_ASM "
DUMP_FLAGS = "-mdump_asm -mCM_printregusage"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--shape", required=True, help="JSON dict of one shape point")
    ap.add_argument("--source", required=True)
    a = ap.parse_args()

    from clops import cl

    from ckh.kernel import Shape
    spec = importlib.import_module(f"kernels.{a.spec}").SPEC
    shape = Shape(json.loads(a.shape))

    dump = Path.cwd() / f"{spec.entry}.asm"
    before = dump.stat().st_mtime if dump.exists() else 0.0

    opts = (spec.build_options(shape) if spec.build_options else "")
    try:
        cl.kernels(f'#include "{a.source}"', f"{opts} {DUMP_FLAGS} {spec.defines(shape)}")
    except Exception as e:                              # noqa: BLE001
        tail = "\n".join(str(e).strip().splitlines()[-8:])
        print(RESULT_PREFIX + json.dumps({"error": f"compile failed:\n{tail}"}))
        return

    if not dump.exists():
        print(RESULT_PREFIX + json.dumps({
            "error": f"no {dump.name} after compiling with {DUMP_FLAGS}. The build options "
                     f"may already pin a different dump directory."}))
        return
    if dump.stat().st_mtime <= before:
        print(RESULT_PREFIX + json.dumps({
            "error": f"{dump.name} was not rewritten by this compile -- it is left over from "
                     f"an earlier build. Refusing to profile a stale dump."}))
        return
    print(RESULT_PREFIX + json.dumps({"asm": str(dump)}))


if __name__ == "__main__":
    sys.exit(main())

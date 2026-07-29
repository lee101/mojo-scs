"""ctypes bridge to the single mojo-scs shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_SCS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-scs.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mscs_form_system": ([I, I, I, I, I, I, I, F, F], I),
    "mscs_csr_matvec": ([I, I, I, I, I, I], None),
    "mscs_csr_matvec32": ([I, I, I, I, I, I], None),
    "mscs_csr_tmatvec": ([I, I, I, I, I, I, I], None),
    "mscs_csr_tmatvec32": ([I, I, I, I, I, I, I], None),
    "mscs_project_cone": ([I, I, I, I, I], None),
    "mscs_project_cone_copy": ([I, I, I, I, I, I], None),
    "mscs_solve_admm": (
        [I] * 22 + [F, F, F, F, F] + [I],
        I,
    ),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_SCS_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [
            pixi,
            "run",
            "--manifest-path",
            os.path.join(ROOT, "pixi.toml"),
            "mojo",
        ]
    raise BuildError("mojo not found; set MOJO_SCS_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_SCS_LIB") and os.path.exists(LIB) and not force:
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not sources:
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"no Mojo sources found under {SRC}")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= max(map(os.path.getmtime, sources)):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    command = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        "-I",
        SRC,
        os.path.join(SRC, "capi.mojo"),
        "-o",
        LIB,
    ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:5000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null NumPy buffer across the Mojo FFI")
    return address


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

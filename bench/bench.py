"""Reproducible mojo-scs benchmarks against upstream SCS and array libraries."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import scs
from scipy import sparse

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_scs  # noqa: E402


def best_time(function, reps=5):
    function()
    best = float("inf")
    result = None
    for _ in range(reps):
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
    return best, result


def duration(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def run_case(name, mojo_fn, reference_fn, check):
    mojo_time, mojo_result = best_time(mojo_fn)
    reference_time, reference_result = best_time(reference_fn)
    check(mojo_result, reference_result)
    speed = reference_time / mojo_time
    return name, mojo_time, reference_time, speed


def main():
    rng = np.random.default_rng(2026)
    rows = []

    values = rng.normal(size=2_000_000)
    cone = {"l": values.size}
    rows.append(
        run_case(
            "nonnegative projection (2M)",
            lambda: mojo_scs.project_cone(values, cone),
            lambda: np.maximum(values, 0),
            lambda a, b: np.testing.assert_array_equal(a, b),
        )
    )

    A = sparse.random(
        300_000,
        200,
        density=0.015,
        random_state=rng,
        format="csr",
        dtype=np.float64,
    )
    vector = rng.normal(size=A.shape[1])
    rows.append(
        run_case(
            "CSR matvec (900k nnz)",
            lambda: mojo_scs.matvec(A, vector),
            lambda: A @ vector,
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12),
        )
    )

    n = 64
    upper = rng.uniform(0.5, 2.0, size=n)
    c = -rng.uniform(0.2, 2.0, size=n)
    A_lp = sparse.vstack([sparse.eye(n), -sparse.eye(n)], format="csc")
    data_lp = {"A": A_lp, "b": np.r_[upper, np.zeros(n)], "c": c}
    cone_lp = {"l": 2 * n}
    settings = {
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "max_iters": 20_000,
        "verbose": False,
    }

    def check_solution(a, b):
        assert a["info"]["status"] == "solved"
        assert b["info"]["status"] == "solved"
        np.testing.assert_allclose(a["x"], b["x"], rtol=2e-4, atol=2e-4)

    rows.append(
        run_case(
            "bounded LP setup + solve (n=64)",
            lambda: mojo_scs.solve(data_lp, cone_lp, **settings),
            lambda: scs.solve(data_lp, cone_lp, **settings),
            check_solution,
        )
    )

    n = 128
    target = rng.normal(size=n)
    data_qp = {
        "P": sparse.eye(n, format="csc"),
        "A": -sparse.eye(n, format="csc"),
        "b": np.zeros(n),
        "c": -target,
    }
    cone_qp = {"l": n}
    mojo_solver = mojo_scs.SCS(data_qp, cone_qp, **settings)
    reference_solver = scs.SCS(data_qp, cone_qp, **settings)
    rows.append(
        run_case(
            "warm reusable QP solve (n=128)",
            lambda: mojo_solver.solve(warm_start=False),
            lambda: reference_solver.solve(warm_start=False),
            check_solution,
        )
    )

    n = 96
    tail = rng.normal(size=n - 1)
    A_soc = sparse.vstack(
        [
            sparse.hstack(
                [sparse.csc_matrix((n - 1, 1)), sparse.eye(n - 1)], format="csc"
            ),
            -sparse.eye(n),
        ],
        format="csc",
    )
    data_soc = {
        "A": A_soc,
        "b": np.r_[tail, np.zeros(n)],
        "c": np.r_[1.0, np.zeros(n - 1)],
    }
    cone_soc = {"z": n - 1, "q": [n]}
    rows.append(
        run_case(
            "SOC setup + solve (q=96)",
            lambda: mojo_scs.solve(data_soc, cone_soc, **settings),
            lambda: scs.solve(data_soc, cone_soc, **settings),
            check_solution,
        )
    )

    print(
        f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; "
        f"upstream SCS {scs.__version__}"
    )
    print()
    print("| case | mojo-scs | upstream/reference | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, reference_time, speed in rows:
        comparison = f"{speed:.2f}x"
        if speed < 1:
            comparison = f"{1 / speed:.2f}x slower"
        print(
            f"| {name} | {duration(mojo_time)} | "
            f"{duration(reference_time)} | {comparison} |"
        )


if __name__ == "__main__":
    main()

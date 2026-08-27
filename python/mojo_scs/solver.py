"""SCS-compatible Python surface for the covered cone subset."""

from __future__ import annotations

import time
import operator
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ._lib import addr, lib

INFEASIBLE_INACCURATE = -7
UNBOUNDED_INACCURATE = -6
SIGINT = -5
FAILED = -4
INDETERMINATE = -3
INFEASIBLE = -2
UNBOUNDED = -1
UNFINISHED = 0
SOLVED = 1
SOLVED_INACCURATE = 2


@dataclass(frozen=True)
class _Cone:
    zero: int
    nonnegative: int
    q: np.ndarray

    @property
    def size(self) -> int:
        return self.zero + self.nonnegative + int(self.q.sum())


def _cone_spec(cone: dict[str, Any], rows: int) -> _Cone:
    if not isinstance(cone, dict):
        raise TypeError("cone must be a dictionary")
    supported = {"z", "f", "l", "q"}
    active_unsupported = {
        key
        for key in ("b", "bl", "bu", "s", "ep", "ed", "p")
        if key in cone and np.size(cone[key]) and np.any(np.asarray(cone[key]))
    }
    if active_unsupported:
        names = ", ".join(sorted(active_unsupported))
        raise NotImplementedError(f"unsupported cone field(s): {names}")
    unknown = set(cone) - supported - {"b", "bl", "bu", "s", "ep", "ed", "p"}
    if unknown:
        raise ValueError(f"unrecognized cone field(s): {', '.join(sorted(unknown))}")
    if "z" in cone and "f" in cone:
        raise ValueError("use either 'z' or the legacy alias 'f', not both")
    def dimension(value: Any, name: str) -> int:
        try:
            result = operator.index(value)
        except TypeError as error:
            raise TypeError(f"cone dimension {name} must be an integer") from error
        if result > np.iinfo(np.int64).max:
            raise OverflowError(f"cone dimension {name} does not fit in int64")
        return result

    zero = dimension(cone.get("z", cone.get("f", 0)), "z")
    nonnegative = dimension(cone.get("l", 0), "l")
    q_values = cone.get("q", [])
    if np.ndim(q_values) != 1:
        raise ValueError("cone field q must be one-dimensional")
    q = np.ascontiguousarray(
        [dimension(value, "q") for value in q_values], dtype=np.int64
    )
    if zero < 0 or nonnegative < 0 or np.any(q < 2):
        raise ValueError("cone dimensions must be nonnegative and SOC sizes >= 2")
    parsed = _Cone(zero, nonnegative, q)
    if parsed.size != rows:
        raise ValueError(
            f"cone dimensions sum to {parsed.size}, but A has {rows} rows"
        )
    return parsed


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real")
    array = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if array.size != length:
        raise ValueError(f"{name} has length {array.size}; expected {length}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _csr_validation_key(matrix: sparse.csr_matrix) -> tuple[Any, ...]:
    return (
        matrix.shape,
        matrix.data.ctypes.data,
        matrix.indices.ctypes.data,
        matrix.indptr.ctypes.data,
        matrix.data.size,
        matrix.indices.size,
        matrix.indptr.size,
    )


def _matrix(value: Any, index_dtype=np.int64, validate: bool = True) -> sparse.csr_matrix:
    if np.iscomplexobj(value):
        raise TypeError("A must be real")
    if sparse.isspmatrix_csr(value) and value.dtype == np.float64:
        matrix = value
    else:
        matrix = sparse.csr_matrix(value, dtype=np.float64, copy=False)
    validation_key = _csr_validation_key(matrix)
    if getattr(matrix, "_mojo_scs_validation_key", None) != validation_key:
        try:
            matrix.check_format(full_check=True)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid CSR matrix: {error}") from error
        matrix._mojo_scs_validation_key = validation_key
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    if matrix.ndim != 2 or (
        validate and not np.all(np.isfinite(matrix.data))
    ):
        raise ValueError("A must be a finite two-dimensional matrix")
    if matrix.indices.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise TypeError("CSR indices must use int32 or int64")
    if matrix.indptr.dtype != matrix.indices.dtype:
        matrix = matrix.copy()
        common_index_dtype = index_dtype or np.int64
        matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=common_index_dtype)
        matrix.indices = np.ascontiguousarray(matrix.indices, dtype=common_index_dtype)
    if (
        not matrix.data.flags.c_contiguous
        or not matrix.indptr.flags.c_contiguous
        or not matrix.indices.flags.c_contiguous
        or (index_dtype is not None and matrix.indices.dtype != index_dtype)
    ):
        matrix = matrix.copy()
    if index_dtype is not None:
        matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=index_dtype)
        matrix.indices = np.ascontiguousarray(matrix.indices, dtype=index_dtype)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    matrix._mojo_scs_validation_key = _csr_validation_key(matrix)
    return matrix


def _quadratic(value: Any, n: int) -> np.ndarray:
    if value is None:
        return np.zeros((n, n), dtype=np.float64)
    if np.iscomplexobj(value):
        raise TypeError("P must be real")
    matrix = sparse.csc_matrix(value, dtype=np.float64)
    if matrix.shape != (n, n):
        raise ValueError(f"P has shape {matrix.shape}; expected {(n, n)}")
    upper = sparse.triu(matrix, format="csr")
    diagonal = sparse.diags(upper.diagonal())
    symmetric = upper + upper.T - diagonal
    dense = np.ascontiguousarray(symmetric.toarray(), dtype=np.float64)
    if not np.all(np.isfinite(dense)):
        raise ValueError("P contains non-finite values")
    eigenvalues = np.linalg.eigvalsh(dense)
    tolerance = 1e-12 * max(1.0, float(np.linalg.norm(dense, ord=2)))
    if eigenvalues[0] < -tolerance:
        raise ValueError("P must be positive semidefinite")
    return dense


class SCS:
    """Reusable solver with the same constructor and solve/update shape as SCS."""

    def __init__(self, data, cone, **settings):
        if not isinstance(data, dict) or not {"A", "b", "c"} <= set(data):
            raise ValueError("data must contain A, b, and c")
        self.A = _matrix(data["A"])
        self.m, self.n = self.A.shape
        if self.m == 0 or self.n == 0:
            raise ValueError("A must have at least one row and one column")
        self.b = _vector(data["b"], self.m, "b").copy()
        self.c = _vector(data["c"], self.n, "c").copy()
        self.P = _quadratic(data.get("P"), self.n)
        self.cone = _cone_spec(cone, self.m)

        recognized = {
            "max_iters",
            "eps_abs",
            "eps_rel",
            "alpha",
            "rho",
            "rho_x",
            "verbose",
            "normalize",
            "scale",
            "adaptive_scale",
            "acceleration_lookback",
            "acceleration_interval",
            "time_limit_secs",
            "log_csv_filename",
            "write_data_filename",
        }
        unknown = set(settings) - recognized
        if unknown:
            raise ValueError(f"unrecognized setting(s): {', '.join(sorted(unknown))}")
        try:
            self.max_iters = operator.index(settings.get("max_iters", 100_000))
        except TypeError as error:
            raise TypeError("max_iters must be an integer") from error
        self.eps_abs = float(settings.get("eps_abs", 1e-4))
        self.eps_rel = float(settings.get("eps_rel", 1e-4))
        self.alpha = float(settings.get("alpha", 1.5))
        self.rho = float(settings.get("rho", 1.0))
        self.rho_x = float(settings.get("rho_x", 1e-6))
        self.verbose = bool(settings.get("verbose", False))
        numeric_settings = np.array(
            [self.eps_abs, self.eps_rel, self.alpha, self.rho, self.rho_x]
        )
        if not np.all(np.isfinite(numeric_settings)):
            raise ValueError("solver settings must be finite")
        if self.max_iters <= 0 or self.eps_abs < 0 or self.eps_rel < 0:
            raise ValueError("invalid iteration limit or tolerance")
        if not 0 < self.alpha < 2 or self.rho <= 0 or self.rho_x <= 0:
            raise ValueError("require 0 < alpha < 2, rho > 0, and rho_x > 0")

        self._factor = np.empty((self.n, self.n), dtype=np.float64)
        ok = lib().mscs_form_system(
            addr(self.A.indptr),
            addr(self.A.indices),
            addr(self.A.data),
            addr(self.P),
            addr(self._factor),
            self.m,
            self.n,
            self.rho,
            self.rho_x,
        )
        if not ok:
            raise ValueError("P + rho*A.T*A + rho_x*I is not positive definite")
        factor_is_diagonal = np.count_nonzero(self._factor) == self.n
        self._diagonal_factor = (
            np.diagonal(self._factor).copy() if factor_is_diagonal else None
        )
        self._x = np.zeros(self.n, dtype=np.float64)
        self._s = np.zeros(self.m, dtype=np.float64)
        self._u = np.zeros(self.m, dtype=np.float64)
        self._ax = np.empty(self.m, dtype=np.float64)
        self._rhs = np.empty(self.n, dtype=np.float64)
        self._tmp_n = np.empty(self.n, dtype=np.float64)
        self._old_s = np.empty(self.m, dtype=np.float64)
        self._stats = np.empty(6, dtype=np.float64)

    def solve(self, warm_start=True, x=None, y=None, s=None):
        if x is not None:
            self._x[:] = _vector(x, self.n, "x")
        elif not warm_start:
            self._x.fill(0)
        if s is not None:
            self._s[:] = _vector(s, self.m, "s")
        elif not warm_start:
            self._s.fill(0)
        if y is not None:
            self._u[:] = _vector(y, self.m, "y") / self.rho
        elif not warm_start:
            self._u.fill(0)

        started = time.perf_counter()
        solved = lib().mscs_solve_admm(
            addr(self.A.indptr),
            addr(self.A.indices),
            addr(self.A.data),
            addr(self.P),
            addr(
                self._diagonal_factor
                if self._diagonal_factor is not None
                else self._factor
            ),
            addr(self.b),
            addr(self.c),
            addr(self.cone.q),
            addr(self._x),
            addr(self._s),
            addr(self._u),
            addr(self._ax),
            addr(self._rhs),
            addr(self._tmp_n),
            addr(self._old_s),
            addr(self._stats),
            self.m,
            self.n,
            self.cone.zero,
            self.cone.nonnegative,
            self.cone.q.size,
            self.max_iters,
            self.eps_abs,
            self.eps_rel,
            self.rho,
            self.rho_x,
            self.alpha,
            self._diagonal_factor is not None,
        )
        elapsed = time.perf_counter() - started
        stats = self._stats
        if not (
            np.all(np.isfinite(stats))
            and np.all(np.isfinite(self._x))
            and np.all(np.isfinite(self._s))
            and np.all(np.isfinite(self._u))
        ):
            raise RuntimeError("Mojo solver failed with non-finite numerical output")
        status = "solved" if solved else "unfinished (reached max_iters)"
        status_val = SOLVED if solved else UNFINISHED
        info = {
            "status_val": status_val,
            "iter": int(stats[0]),
            "scale_updates": 0,
            "scale": self.rho,
            "pobj": float(stats[4]),
            "dobj": float(stats[5]),
            "res_pri": float(stats[1]),
            "res_dual": float(stats[2]),
            "gap": float(stats[3]),
            "res_infeas": float("nan"),
            "res_unbdd_a": float("nan"),
            "res_unbdd_p": float("nan"),
            "comp_slack": float(abs(self._u @ self._s) * self.rho),
            "solve_time": elapsed * 1000.0,
            "setup_time": 0.0,
            "lin_sys_time": float("nan"),
            "cone_time": float("nan"),
            "accel_time": 0.0,
            "rejected_accel_steps": 0,
            "accepted_accel_steps": 0,
            "status": status,
        }
        if self.verbose:
            print(
                f"mojo-scs: {status}; iter={info['iter']}, "
                f"pri={info['res_pri']:.3e}, dual={info['res_dual']:.3e}"
            )
        return {
            "x": self._x.copy(),
            "y": (self.rho * self._u).copy(),
            "s": self._s.copy(),
            "info": info,
        }

    def update(self, b=None, c=None):
        if b is not None:
            self.b[:] = _vector(b, self.m, "b")
        if c is not None:
            self.c[:] = _vector(c, self.n, "c")


def solve(data, cone, **settings):
    """Solve one conic problem, matching ``scs.solve(data, cone, **settings)``."""
    return SCS(data, cone, **settings).solve(warm_start=False)

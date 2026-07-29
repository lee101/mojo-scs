"""Direct access to the Mojo sparse and cone kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ._lib import addr, lib
from .solver import _cone_spec, _matrix, _vector


def matvec(A: Any, x: Any, transpose: bool = False) -> np.ndarray:
    matrix = _matrix(A, index_dtype=None, validate=False)
    expected = matrix.shape[0] if transpose else matrix.shape[1]
    vector = _vector(x, expected, "x")
    size = matrix.shape[1] if transpose else matrix.shape[0]
    result = np.empty(size, dtype=np.float64)
    if transpose:
        function = (
            lib().mscs_csr_tmatvec32
            if matrix.indices.dtype == np.int32
            else lib().mscs_csr_tmatvec
        )
        function(
            addr(matrix.indptr),
            addr(matrix.indices),
            addr(matrix.data),
            addr(vector),
            addr(result),
            matrix.shape[0],
            matrix.shape[1],
        )
    else:
        function = (
            lib().mscs_csr_matvec32
            if matrix.indices.dtype == np.int32
            else lib().mscs_csr_matvec
        )
        function(
            addr(matrix.indptr),
            addr(matrix.indices),
            addr(matrix.data),
            addr(vector),
            addr(result),
            matrix.shape[0],
        )
    return result


def project_cone(values: Any, cone: dict[str, Any]) -> np.ndarray:
    if np.iscomplexobj(values):
        raise TypeError("values must be real")
    source = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    result = np.empty_like(source)
    parsed = _cone_spec(cone, source.size)
    lib().mscs_project_cone_copy(
        addr(source),
        addr(result),
        parsed.zero,
        parsed.nonnegative,
        addr(parsed.q),
        parsed.q.size,
    )
    return result

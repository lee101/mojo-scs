import numpy as np
from scipy import sparse

import mojo_scs
from mojo_scs.solver import _matrix


def _reference_project(values, cone):
    result = np.array(values, dtype=float, copy=True)
    offset = int(cone.get("z", cone.get("f", 0)))
    result[:offset] = 0
    linear = int(cone.get("l", 0))
    result[offset : offset + linear] = np.maximum(
        result[offset : offset + linear], 0
    )
    offset += linear
    for q in cone.get("q", []):
        block = result[offset : offset + q]
        tail = np.linalg.norm(block[1:])
        if tail <= -block[0]:
            block[:] = 0
        elif tail > block[0]:
            new_head = 0.5 * (tail + block[0])
            block[1:] *= new_head / tail
            block[0] = new_head
        offset += q
    return result


def test_sparse_matvec_matches_scipy():
    rng = np.random.default_rng(4)
    A = sparse.random(
        513, 127, density=0.07, random_state=rng, format="csr", dtype=np.float64
    )
    x = rng.normal(size=A.shape[1])
    np.testing.assert_allclose(
        mojo_scs.matvec(A, x), A @ x, rtol=2e-14, atol=5e-16
    )


def test_sparse_transpose_matvec_matches_scipy():
    rng = np.random.default_rng(7)
    A = sparse.random(
        277, 91, density=0.13, random_state=rng, format="csc", dtype=np.float64
    )
    x = rng.normal(size=A.shape[0])
    np.testing.assert_allclose(
        mojo_scs.matvec(A, x, transpose=True), A.T @ x, rtol=2e-14
    )


def test_product_cone_projection_matches_reference():
    cone = {"z": 3, "l": 5, "q": [2, 3, 7]}
    values = np.array(
        [
            4,
            -2,
            8,
            -1,
            0,
            2,
            -3,
            5,
            4,
            1,
            -4,
            3,
            0,
            2,
            5,
            -1,
            2,
            -3,
            4,
            -2,
        ],
        dtype=float,
    )
    expected = _reference_project(values, cone)
    actual = mojo_scs.project_cone(values, cone)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(values[:3], [4, -2, 8])


def test_soc_projection_covers_interior_polar_and_boundary_cases():
    cone = {"q": [3, 3, 3]}
    values = [4, 1, 2, -4, 1, 2, 0.5, 3, 4]
    actual = mojo_scs.project_cone(values, cone)
    expected = _reference_project(values, cone)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual[:3], values[:3])
    np.testing.assert_array_equal(actual[3:6], 0)
    np.testing.assert_allclose(actual[6], np.linalg.norm(actual[7:]))


def test_legacy_f_alias_is_zero_cone():
    np.testing.assert_array_equal(
        mojo_scs.project_cone([2.0, -1.0, 3.0], {"f": 2, "l": 1}),
        [0.0, 0.0, 3.0],
    )


def test_projection_simd_scalar_tail_lengths():
    rng = np.random.default_rng(17)
    for length in (1, 3, 5, 17, 33):
        values = rng.normal(size=length)
        np.testing.assert_array_equal(
            mojo_scs.project_cone(values, {"l": length}),
            np.maximum(values, 0),
        )


def test_large_projection_parallel_path_matches_numpy():
    length = 8_000_003
    values = np.linspace(-2.0, 3.0, length, dtype=np.float64)
    actual = mojo_scs.project_cone(values, {"l": length})
    np.testing.assert_array_equal(actual, np.maximum(values, 0))


def test_large_csr_matvec_parallel_path_matches_scipy():
    rows = 2_000_003
    indptr = np.arange(rows + 1, dtype=np.int32)
    indices = np.zeros(rows, dtype=np.int32)
    data = np.linspace(0.5, 1.5, rows, dtype=np.float64)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(rows, 1))
    np.testing.assert_array_equal(
        mojo_scs.matvec(matrix, [2.0]),
        matrix @ np.array([2.0]),
    )


def test_canonical_csr_buffers_remain_zero_copy_for_direct_kernels():
    matrix = sparse.eye(11, format="csr", dtype=np.float64)
    normalized = _matrix(matrix, index_dtype=None)
    assert normalized is matrix
    assert normalized.data.ctypes.data == matrix.data.ctypes.data
    assert normalized.indices.ctypes.data == matrix.indices.ctypes.data
    assert normalized.indptr.ctypes.data == matrix.indptr.ctypes.data


def test_mixed_csr_index_dtypes_are_normalized_before_ffi():
    matrix = sparse.eye(5, format="csr", dtype=np.float64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    np.testing.assert_array_equal(mojo_scs.matvec(matrix, np.ones(5)), np.ones(5))


def test_malformed_csr_is_rejected_before_ffi():
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    matrix.indices[0] = 9
    with np.testing.assert_raises(ValueError):
        mojo_scs.matvec(matrix, np.ones(3))


def test_replaced_csr_buffers_invalidate_format_cache():
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    mojo_scs.matvec(matrix, np.ones(3))
    matrix.indices = matrix.indices.copy()
    matrix.indices[0] = 9
    with np.testing.assert_raises(ValueError):
        mojo_scs.matvec(matrix, np.ones(3))


def test_complex_inputs_are_not_silently_narrowed():
    with np.testing.assert_raises(TypeError):
        mojo_scs.matvec(np.eye(2, dtype=complex), np.ones(2))
    with np.testing.assert_raises(TypeError):
        mojo_scs.project_cone([1 + 2j], {"l": 1})

import inspect

import numpy as np
import pytest
import scs
from scipy import sparse

import mojo_scs


def _both(data, cone, **settings):
    common = {
        "eps_abs": 1e-7,
        "eps_rel": 1e-7,
        "max_iters": 50_000,
        "verbose": False,
    }
    common.update(settings)
    upstream = scs.solve(data, cone, **common)
    mojo = mojo_scs.solve(data, cone, **common)
    assert upstream["info"]["status"] == "solved"
    assert mojo["info"]["status"] == "solved"
    return upstream, mojo


def _assert_kkt(data, result, atol=3e-6):
    A = sparse.csr_matrix(data["A"])
    P = sparse.csr_matrix(data.get("P", sparse.csr_matrix((A.shape[1],) * 2)))
    x, y, s = result["x"], result["y"], result["s"]
    np.testing.assert_allclose(A @ x + s, data["b"], atol=atol, rtol=atol)
    np.testing.assert_allclose(P @ x + data["c"] + A.T @ y, 0, atol=atol, rtol=atol)
    assert abs(y @ s) < 2e-5


def test_public_signatures_match_upstream():
    assert inspect.signature(mojo_scs.solve) == inspect.signature(scs.solve)
    assert inspect.signature(mojo_scs.SCS) == inspect.signature(scs.SCS)
    assert inspect.signature(mojo_scs.SCS.solve) == inspect.signature(scs.SCS.solve)
    assert inspect.signature(mojo_scs.SCS.update) == inspect.signature(scs.SCS.update)


def test_linear_program_matches_upstream():
    A = sparse.csc_matrix([[1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    data = {"A": A, "b": np.array([2.0, 0.0, 0.0]), "c": np.array([-3.0, -2.0])}
    upstream, mojo = _both(data, {"l": 3})
    np.testing.assert_allclose(mojo["x"], upstream["x"], atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(
        mojo["info"]["pobj"], upstream["info"]["pobj"], atol=5e-6
    )
    _assert_kkt(data, mojo)
    assert set(mojo) == {"x", "y", "s", "info"}
    assert {
        "status_val", "iter", "pobj", "dobj", "res_pri", "res_dual", "gap",
        "solve_time", "setup_time", "status",
    } <= set(mojo["info"])


def test_dense_matrix_input_matches_sparse_input():
    dense_a = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])
    data = {"A": dense_a, "b": [2.0, 0.0, 0.0], "c": [-3.0, -2.0]}
    dense_result = mojo_scs.solve(data, {"l": 3}, verbose=False)
    sparse_result = mojo_scs.solve(
        {**data, "A": sparse.csc_matrix(dense_a)}, {"l": 3}, verbose=False
    )
    np.testing.assert_allclose(dense_result["x"], sparse_result["x"])


def test_equality_and_nonnegative_cones_match_upstream():
    A = sparse.csc_matrix(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    data = {
        "A": A,
        "b": np.array([1.0, 0.2, 0.0, 0.0]),
        "c": np.array([0.3, -0.1]),
    }
    upstream, mojo = _both(data, {"z": 2, "l": 2})
    np.testing.assert_allclose(mojo["x"], upstream["x"], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(mojo["s"], upstream["s"], atol=2e-6, rtol=2e-6)
    _assert_kkt(data, mojo)


def test_second_order_cone_matches_upstream():
    A = sparse.csc_matrix(
        np.vstack([[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], -np.eye(3)])
    )
    data = {
        "A": A,
        "b": np.array([1.0, 2.0, 0.0, 0.0, 0.0]),
        "c": np.array([1.0, 0.0, 0.0]),
    }
    upstream, mojo = _both(data, {"z": 2, "q": [3]})
    np.testing.assert_allclose(mojo["x"], upstream["x"], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(mojo["x"], [np.sqrt(5), 1, 2], atol=2e-6)
    _assert_kkt(data, mojo)


def test_quadratic_program_matches_upstream():
    target = np.array([2.0, -1.0, 0.5, -3.0])
    n = target.size
    data = {
        "P": sparse.eye(n, format="csc"),
        "A": -sparse.eye(n, format="csc"),
        "b": np.zeros(n),
        "c": -target,
    }
    upstream, mojo = _both(data, {"l": n})
    expected = np.maximum(target, 0)
    np.testing.assert_allclose(mojo["x"], upstream["x"], atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(mojo["x"], expected, atol=3e-6)
    _assert_kkt(data, mojo)


def test_off_diagonal_upper_triangular_quadratic_matches_upstream():
    P = sparse.csc_matrix([[2.0, 0.4], [0.0, 1.5]])
    A = -sparse.eye(2, format="csc")
    data = {
        "P": P,
        "A": A,
        "b": np.zeros(2),
        "c": np.array([-2.4, -1.9]),
    }
    upstream, mojo = _both(data, {"l": 2})
    np.testing.assert_allclose(mojo["x"], upstream["x"], atol=3e-6, rtol=3e-6)
    symmetric_p = sparse.csr_matrix([[2.0, 0.4], [0.4, 1.5]])
    kkt_data = {**data, "P": symmetric_p}
    _assert_kkt(kkt_data, mojo)


def test_reusable_solver_update_matches_fresh_upstream_solve():
    A = sparse.csc_matrix([[1.0], [-1.0]])
    solver = mojo_scs.SCS(
        {"A": A, "b": [3.0, 0.0], "c": [-1.0]},
        {"l": 2},
        eps_abs=1e-8,
        eps_rel=1e-8,
        verbose=False,
    )
    first = solver.solve(warm_start=False)
    np.testing.assert_allclose(first["x"], [3.0], atol=2e-6)
    solver.update(b=[1.5, 0.0], c=[-2.0])
    updated = solver.solve()
    reference = scs.solve(
        {"A": A, "b": np.array([1.5, 0.0]), "c": np.array([-2.0])},
        {"l": 2},
        eps_abs=1e-8,
        eps_rel=1e-8,
        verbose=False,
    )
    np.testing.assert_allclose(updated["x"], reference["x"], atol=2e-6)


def test_explicit_warm_start_is_accepted():
    A = sparse.csc_matrix([[1.0], [-1.0]])
    solver = mojo_scs.SCS(
        {"A": A, "b": [2.0, 0.0], "c": [-1.0]},
        {"l": 2},
        eps_abs=1e-7,
        eps_rel=1e-7,
    )
    result = solver.solve(x=[1.9], y=[1.0, 0.0], s=[0.1, 1.9])
    np.testing.assert_allclose(result["x"], [2.0], atol=2e-6)


def test_iteration_limit_is_reported_honestly():
    A = sparse.csc_matrix([[1.0], [-1.0]])
    result = mojo_scs.solve(
        {"A": A, "b": [2.0, 0.0], "c": [-1.0]},
        {"l": 2},
        max_iters=1,
        eps_abs=0,
        eps_rel=0,
    )
    assert result["info"]["status_val"] == mojo_scs.UNFINISHED
    assert "max_iters" in result["info"]["status"]


@pytest.mark.parametrize(
    "cone",
    [
        {"s": [2]},
        {"ep": 1},
        {"p": [0.3]},
        {"bl": [0.0], "bu": [1.0], "b": 1},
    ],
)
def test_unsupported_cones_are_rejected(cone):
    with pytest.raises(NotImplementedError):
        mojo_scs.SCS(
            {"A": sparse.eye(2), "b": np.ones(2), "c": np.ones(2)}, cone
        )


def test_cone_size_and_shapes_are_validated():
    with pytest.raises(ValueError, match="sum"):
        mojo_scs.solve(
            {"A": sparse.eye(2), "b": np.ones(2), "c": np.ones(2)}, {"l": 1}
        )
    with pytest.raises(ValueError, match="length"):
        mojo_scs.solve(
            {"A": sparse.eye(2), "b": np.ones(3), "c": np.ones(2)}, {"l": 2}
        )


def test_dimensions_and_iteration_limit_do_not_silently_narrow():
    data = {"A": sparse.eye(2), "b": np.ones(2), "c": np.ones(2)}
    with pytest.raises(TypeError, match="integer"):
        mojo_scs.solve(data, {"l": 2.5})
    with pytest.raises(TypeError, match="integer"):
        mojo_scs.solve(data, {"l": 2}, max_iters=2.5)
    with pytest.raises(TypeError, match="real"):
        mojo_scs.solve({**data, "c": np.ones(2, dtype=complex)}, {"l": 2})


def test_nonconvex_quadratic_and_nonfinite_settings_are_rejected():
    data = {
        "P": sparse.diags([1.0, -1.0]),
        "A": sparse.eye(2),
        "b": np.ones(2),
        "c": np.ones(2),
    }
    with pytest.raises(ValueError, match="positive semidefinite"):
        mojo_scs.solve(data, {"l": 2})
    del data["P"]
    with pytest.raises(ValueError, match="finite"):
        mojo_scs.solve(data, {"l": 2}, rho=np.nan)


def test_compatibility_settings_are_accepted():
    data = {"A": sparse.eye(1), "b": np.ones(1), "c": -np.ones(1)}
    solver = mojo_scs.SCS(
        data,
        {"l": 1},
        normalize=False,
        scale=0.2,
        adaptive_scale=False,
        acceleration_lookback=0,
        acceleration_interval=5,
        time_limit_secs=1.0,
        log_csv_filename="unused.csv",
        write_data_filename="unused.dat",
    )
    assert solver.solve(warm_start=False)["info"]["status"] == "solved"

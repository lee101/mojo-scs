# mojo-scs

`mojo-scs` is a standalone conic splitting solver whose numerical core is
written in [Mojo](https://www.modular.com/mojo). It solves the same standard
form used by the Python package [SCS](https://github.com/cvxgrp/scs):

```text
minimize    1/2 x' P x + c' x
subject to  A x + s = b
            s in K
```

The Python entry points intentionally match upstream: `solve(data, cone,
**settings)`, `SCS(data, cone, **settings)`, `SCS.solve(...)`, and
`SCS.update(b=None, c=None)`. Upstream `scs` is used only by the tests and
benchmarks; it is not a runtime dependency.

This is a focused port, not a wrapper around upstream SCS. It is useful for
small and medium feasible, bounded LPs, SOCPs, and convex QPs, and as a compact
Mojo implementation of the sparse/cone kernels behind a splitting solver.

## Coverage

| area | covered |
| --- | --- |
| objectives | linear and convex quadratic; `P` uses upstream's upper-triangular convention |
| cones | zero/equality (`z`, legacy `f`), nonnegative (`l`), any product of second-order cones (`q`) |
| matrices | dense or SciPy sparse input, normalized once to float64 CSR |
| solver API | one-shot solve, reusable workspace, explicit/default warm starts, `b`/`c` updates |
| results | `x`, `y`, `s`, KKT residuals, objectives, gap, iteration count, SCS-style status/info fields |
| direct kernels | CSR matvec, transpose matvec, and product-cone projection |

Not covered are infeasibility or unboundedness certificates, semidefinite,
exponential, power, and box cones, non-convex `P`, float32, the indirect and
GPU linear solvers, adaptive scaling, Anderson acceleration, or CSV/data
logging. Unsupported cone types raise `NotImplementedError`. Reaching
`max_iters` returns `status_val == UNFINISHED`; it is not mislabeled as a
certificate or successful solve.

For constructor compatibility, upstream tuning/logging names such as
`normalize`, `scale`, `adaptive_scale`, `acceleration_lookback`,
`acceleration_interval`, `time_limit_secs`, `log_csv_filename`, and
`write_data_filename` are accepted but currently have no effect. The active
settings are `max_iters`, `eps_abs`, `eps_rel`, `alpha`, `rho_x`, and the
mojo-scs extension `rho`.

## Install

The repository pins its own Mojo nightly and all Python dependencies:

```bash
pixi install
pixi run build
pixi run test
```

The build writes `dist/libmojo-scs.so`. The Python package also rebuilds it on
first import if the sources are newer. An external deployment can set
`MOJO_SCS_LIB=/absolute/path/to/libmojo-scs.so`.

## Usage

This LP maximizes `3*x + 2*y` subject to `x + 2*y <= 2` and nonnegative
variables:

```python
import numpy as np
from scipy import sparse
import mojo_scs as scs

A = sparse.csc_matrix([
    [ 1.0,  2.0],
    [-1.0,  0.0],
    [ 0.0, -1.0],
])
data = {
    "A": A,
    "b": np.array([2.0, 0.0, 0.0]),
    "c": np.array([-3.0, -2.0]),
}

solution = scs.solve(data, {"l": 3}, verbose=False)
assert solution["info"]["status"] == "solved"
print(solution["x"])  # approximately [2.0, 0.0]
```

The checked-in version is [examples/quickstart.py](examples/quickstart.py) and
runs with:

```bash
pixi run python examples/quickstart.py
```

A reusable factorization follows upstream's API:

```python
solver = scs.SCS(data, {"l": 3}, eps_abs=1e-6, eps_rel=1e-6)
first = solver.solve(warm_start=False)
solver.update(b=np.array([1.0, 0.0, 0.0]))
second = solver.solve()
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, using upstream SCS 3.2.9. Solver rows compare the
same problem and tolerances against upstream SCS. Direct kernel rows compare
against the corresponding NumPy/SciPy operation.

| case | mojo-scs | upstream/reference | speedup |
| --- | ---: | ---: | ---: |
| nonnegative projection (2M) | 3.78 ms | 1.85 ms | 2.04x slower |
| CSR matvec (900k nnz) | 3.41 ms | 2.77 ms | 1.23x slower |
| bounded LP setup + solve (n=64) | 426.2 us | 507.8 us | 1.19x |
| warm reusable QP solve (n=128) | 205.1 us | 225.2 us | 1.10x |
| SOC setup + solve (q=96) | 1.12 ms | 1.06 ms | 1.06x slower |

These are best-of-five timings from one run, not general performance
guarantees. The core uses native-width SIMD with scalar tails, SIMD gathers
for long sparse rows, and parallel execution only above internal size
thresholds. Canonical SciPy CSR buffers keep their native int32 indices for
direct kernels; solver workspaces normalize indices to int64 and allocate
their buffers once.

No GPU path is included. The measured optimization targets do not have enough
arithmetic intensity to justify one: projection performs one comparison while
reading and writing each float64 value, and CSR matvec performs two floating-
point operations per nonzero while moving a value, an index, and irregular
vector data. The benchmarked solver systems use the diagonal factorization
fast path, where device transfers and launch synchronization would dominate.
`pixi run bench` prints a fresh Markdown table and verifies that every timed
solver result agrees numerically.

## How it works

The Python layer validates an SCS data dictionary, converts `A` once to sorted
CSR, expands the symmetric quadratic matrix, and allocates all work buffers.
Mojo assembles

```text
P + rho * A' A + rho_x * I
```

directly from sparse rows and computes an in-place Cholesky factor. One FFI
call then runs the complete proximal, over-relaxed ADMM loop: sparse matvecs,
triangular solves, product-cone projection, dual updates, and KKT checks all
stay in Mojo. No allocation occurs inside Mojo.

Python and Mojo communicate through a C ABI in
[`src/capi.mojo`](src/capi.mojo). NumPy buffers cross as integer addresses and
are reconstructed as `UnsafePointer[..., AnyOrigin[mut=True]]`, avoiding
parametric exported functions. Floating-point arrays are contiguous float64.
Direct kernels accept matched int32 or int64 CSR offsets and indices; the
solver uses contiguous int64. Python retains ownership of every NumPy buffer
for the full synchronous call, so there is no cross-language allocator
pairing.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The parity suite compares every covered cone/objective combination against
the real upstream package, checks KKT conditions, and separately validates
the Mojo kernels against SciPy or a direct NumPy cone projection.

## License

MIT

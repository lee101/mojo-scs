"""Numerical core for the Python conic solver.

All storage belongs to the caller.  Sparse matrices use SciPy-compatible CSR:
`indptr[rows + 1]`, `indices[nnz]`, and `data[nnz]`.
"""

from std.math import sqrt
from max.algorithm import parallelize
from std.sys.info import simd_width_of as simdwidthof

comptime W = simdwidthof[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime PARALLEL_ROWS = 2_000_000
comptime PARALLEL_VALUES = 8_000_000
comptime PARALLEL_TASKS = 4


def dot(a: FPtr, b: FPtr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def norm(a: FPtr, n: Int) -> Float64:
    return sqrt(dot(a, a, n))


def copy_values(dst: FPtr, src: FPtr, n: Int):
    var i = 0
    while i + W <= n:
        dst.store(i, src.load[width=W](i))
        i += W
    while i < n:
        dst[i] = src[i]
        i += 1


def zero_values(dst: FPtr, n: Int):
    var zeros = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        dst.store(i, zeros)
        i += W
    while i < n:
        dst[i] = 0.0
        i += 1


def csr_matvec_range(
    indptr: IPtr, indices: IPtr, data: FPtr, x: FPtr, dst: FPtr, rows: Int
):
    for r in range(rows):
        var total = 0.0
        var k = Int(indptr[r])
        var end = Int(indptr[r + 1])
        while k + W <= end:
            total += (
                data.load[width=W](k)
                * x.gather[width=W](indices.load[width=W](k))
            ).reduce_add()
            k += W
        while k < end:
            total += data[k] * x[Int(indices[k])]
            k += 1
        dst[r] = total


def csr_matvec(
    indptr: IPtr, indices: IPtr, data: FPtr, x: FPtr, dst: FPtr, rows: Int
):
    if rows < PARALLEL_ROWS:
        csr_matvec_range(indptr, indices, data, x, dst, rows)
        return
    var chunk_size = (rows + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @parameter
    def work(task: Int):
        var first = task * chunk_size
        var end = min(first + chunk_size, rows)
        if first < end:
            csr_matvec_range(
                indptr + first,
                indices,
                data,
                x,
                dst + first,
                end - first,
            )

    parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)


def csr_matvec32_range(
    indptr: I32Ptr, indices: I32Ptr, data: FPtr, x: FPtr, dst: FPtr, rows: Int
):
    for r in range(rows):
        var total = 0.0
        var k = Int(indptr[r])
        var end = Int(indptr[r + 1])
        while k + W <= end:
            total += (
                data.load[width=W](k)
                * x.gather[width=W](indices.load[width=W](k))
            ).reduce_add()
            k += W
        while k < end:
            total += data[k] * x[Int(indices[k])]
            k += 1
        dst[r] = total


def csr_matvec32(
    indptr: I32Ptr, indices: I32Ptr, data: FPtr, x: FPtr, dst: FPtr, rows: Int
):
    if rows < PARALLEL_ROWS:
        csr_matvec32_range(indptr, indices, data, x, dst, rows)
        return
    var chunk_size = (rows + PARALLEL_TASKS - 1) // PARALLEL_TASKS

    @parameter
    def work(task: Int):
        var first = task * chunk_size
        var end = min(first + chunk_size, rows)
        if first < end:
            csr_matvec32_range(
                indptr + first,
                indices,
                data,
                x,
                dst + first,
                end - first,
            )

    parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)


def csr_tmatvec(
    indptr: IPtr,
    indices: IPtr,
    data: FPtr,
    x: FPtr,
    dst: FPtr,
    rows: Int,
    cols: Int,
):
    zero_values(dst, cols)
    for r in range(rows):
        var value = x[r]
        var k = Int(indptr[r])
        var end = Int(indptr[r + 1])
        while k < end:
            dst[Int(indices[k])] += data[k] * value
            k += 1


def csr_tmatvec32(
    indptr: I32Ptr,
    indices: I32Ptr,
    data: FPtr,
    x: FPtr,
    dst: FPtr,
    rows: Int,
    cols: Int,
):
    zero_values(dst, cols)
    for r in range(rows):
        var value = x[r]
        var k = Int(indptr[r])
        var end = Int(indptr[r + 1])
        while k < end:
            dst[Int(indices[k])] += data[k] * value
            k += 1


def form_system(
    indptr: IPtr,
    indices: IPtr,
    data: FPtr,
    p_dense: FPtr,
    matrix: FPtr,
    rows: Int,
    cols: Int,
    rho: Float64,
    sigma: Float64,
):
    """matrix = P + rho A.T A, accumulated from sparse rows."""
    copy_values(matrix, p_dense, cols * cols)
    for i in range(cols):
        matrix[i * cols + i] += sigma
    for r in range(rows):
        var first = Int(indptr[r])
        var end = Int(indptr[r + 1])
        var ka = first
        while ka < end:
            var ia = Int(indices[ka])
            var va = data[ka] * rho
            var kb = first
            while kb < end:
                matrix[ia * cols + Int(indices[kb])] += va * data[kb]
                kb += 1
            ka += 1


def cholesky(matrix: FPtr, n: Int) -> Bool:
    var is_diagonal = True
    for i in range(n):
        for j in range(i):
            if matrix[i * n + j] != 0.0:
                is_diagonal = False
                break
        if not is_diagonal:
            break
    if is_diagonal:
        for i in range(n):
            if matrix[i * n + i] <= 0.0:
                return False
            matrix[i * n + i] = sqrt(matrix[i * n + i])
        return True

    for i in range(n):
        for j in range(i + 1):
            var value = matrix[i * n + j] - dot(
                matrix + i * n, matrix + j * n, j
            )
            if i == j:
                if value <= 0.0:
                    return False
                matrix[i * n + i] = sqrt(value)
            else:
                matrix[i * n + j] = value / matrix[j * n + j]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i * n + j] = matrix[j * n + i]
    return True


def cholesky_solve(factor: FPtr, rhs: FPtr, n: Int):
    for i in range(n):
        var value = rhs[i] - dot(factor + i * n, rhs, i)
        rhs[i] = value / factor[i * n + i]
    for reverse_i in range(n):
        var i = n - 1 - reverse_i
        var value = rhs[i] - dot(
            factor + i * n + i + 1, rhs + i + 1, n - i - 1
        )
        rhs[i] = value / factor[i * n + i]


def diagonal_solve(factor: FPtr, rhs: FPtr, n: Int):
    var i = 0
    while i + W <= n:
        var diagonal = factor.load[width=W](i)
        rhs.store(
            i, rhs.load[width=W](i) / (diagonal * diagonal)
        )
        i += W
    while i < n:
        rhs[i] /= factor[i] * factor[i]
        i += 1


def project_soc(values: FPtr, offset: Int, q_sizes: IPtr, q_count: Int):
    var current = offset
    for cone_i in range(q_count):
        var q = Int(q_sizes[cone_i])
        var t = values[current]
        var tail_norm = norm(values + current + 1, q - 1)
        if tail_norm <= -t:
            zero_values(values + current, q)
        elif tail_norm > t:
            var new_t = 0.5 * (tail_norm + t)
            values[current] = new_t
            if tail_norm > 0.0:
                var scale = new_t / tail_norm
                var j = 1
                var scales = SIMD[DType.float64, W](scale)
                while j + W <= q:
                    values.store(
                        current + j,
                        values.load[width=W](current + j) * scales,
                    )
                    j += W
                while j < q:
                    values[current + j] *= scale
                    j += 1
        current += q


def project_cone(
    values: FPtr, zero: Int, nonnegative: Int, q_sizes: IPtr, q_count: Int
):
    var zeros = SIMD[DType.float64, W](0.0)
    var zero_i = 0
    while zero_i + W <= zero:
        values.store(zero_i, zeros)
        zero_i += W
    while zero_i < zero:
        values[zero_i] = 0.0
        zero_i += 1
    var offset = zero
    var i = 0
    while i + W <= nonnegative:
        values.store(
            offset + i, max(values.load[width=W](offset + i), zeros)
        )
        i += W
    while i < nonnegative:
        if values[offset + i] < 0.0:
            values[offset + i] = 0.0
        i += 1
    offset += nonnegative
    project_soc(values, offset, q_sizes, q_count)


def project_cone_copy(
    src: FPtr,
    values: FPtr,
    zero: Int,
    nonnegative: Int,
    q_sizes: IPtr,
    q_count: Int,
):
    var zeros = SIMD[DType.float64, W](0.0)
    var zero_i = 0
    while zero_i + W <= zero:
        values.store(zero_i, zeros)
        zero_i += W
    while zero_i < zero:
        values[zero_i] = 0.0
        zero_i += 1

    var i = 0
    if nonnegative < PARALLEL_VALUES:
        while i + 4 * W <= nonnegative:
            values.store(
                zero + i, max(src.load[width=W](zero + i), zeros)
            )
            values.store(
                zero + i + W,
                max(src.load[width=W](zero + i + W), zeros),
            )
            values.store(
                zero + i + 2 * W,
                max(src.load[width=W](zero + i + 2 * W), zeros),
            )
            values.store(
                zero + i + 3 * W,
                max(src.load[width=W](zero + i + 3 * W), zeros),
            )
            i += 4 * W
        while i + W <= nonnegative:
            values.store(
                zero + i, max(src.load[width=W](zero + i), zeros)
            )
            i += W
    else:
        var vectors = nonnegative // W
        var chunk_size = (vectors + PARALLEL_TASKS - 1) // PARALLEL_TASKS

        @parameter
        def work(task: Int):
            var first = task * chunk_size
            var end = min(first + chunk_size, vectors)
            for vector_i in range(first, end):
                var i = vector_i * W
                values.store(
                    zero + i, max(src.load[width=W](zero + i), zeros)
                )

        parallelize[work](PARALLEL_TASKS, PARALLEL_TASKS)
        i = vectors * W

    while i < nonnegative:
        values[zero + i] = max(src[zero + i], 0.0)
        i += 1

    var offset = zero + nonnegative
    for cone_i in range(q_count):
        var q = Int(q_sizes[cone_i])
        var j = 0
        while j + W <= q:
            values.store(offset + j, src.load[width=W](offset + j))
            j += W
        while j < q:
            values[offset + j] = src[offset + j]
            j += 1
        offset += q
    project_soc(values, zero + nonnegative, q_sizes, q_count)


def dense_matvec(matrix: FPtr, x: FPtr, dst: FPtr, n: Int):
    for i in range(n):
        dst[i] = dot(matrix + i * n, x, n)


def solve_admm(
    indptr: IPtr,
    indices: IPtr,
    data: FPtr,
    p_dense: FPtr,
    factor: FPtr,
    b: FPtr,
    c: FPtr,
    q_sizes: IPtr,
    x: FPtr,
    s: FPtr,
    u: FPtr,
    ax: FPtr,
    rhs: FPtr,
    tmp_n: FPtr,
    old_s: FPtr,
    stats: FPtr,
    rows: Int,
    cols: Int,
    zero: Int,
    nonnegative: Int,
    q_count: Int,
    max_iters: Int,
    eps_abs: Float64,
    eps_rel: Float64,
    rho: Float64,
    sigma: Float64,
    alpha: Float64,
    factor_is_diagonal: Bool,
) -> Int:
    var converged = False
    var iteration = 0
    var primal_residual = 0.0
    var dual_residual = 0.0
    var gap = 0.0
    var primal_objective = 0.0
    var dual_objective = 0.0
    var b_norm = norm(b, rows)
    var c_norm = norm(c, cols)

    while iteration < max_iters:
        var i = 0
        while i + W <= rows:
            var s_v = s.load[width=W](i)
            old_s.store(i, s_v)
            ax.store(
                i,
                b.load[width=W](i) - s_v - u.load[width=W](i),
            )
            i += W
        while i < rows:
            old_s[i] = s[i]
            ax[i] = b[i] - s[i] - u[i]
            i += 1
        csr_tmatvec(indptr, indices, data, ax, rhs, rows, cols)
        i = 0
        while i + W <= cols:
            rhs.store(
                i,
                rho * rhs.load[width=W](i)
                - c.load[width=W](i)
                + sigma * x.load[width=W](i),
            )
            i += W
        while i < cols:
            rhs[i] = rho * rhs[i] - c[i] + sigma * x[i]
            i += 1
        if factor_is_diagonal:
            diagonal_solve(factor, rhs, cols)
        else:
            cholesky_solve(factor, rhs, cols)
        copy_values(x, rhs, cols)

        csr_matvec(indptr, indices, data, x, ax, rows)
        i = 0
        while i + W <= rows:
            var relaxed_ax = (
                alpha * ax.load[width=W](i)
                + (1.0 - alpha)
                * (b.load[width=W](i) - old_s.load[width=W](i))
            )
            s.store(
                i,
                b.load[width=W](i)
                - relaxed_ax
                - u.load[width=W](i),
            )
            old_s.store(i, relaxed_ax)
            i += W
        while i < rows:
            var relaxed_ax = (
                alpha * ax[i] + (1.0 - alpha) * (b[i] - old_s[i])
            )
            s[i] = b[i] - relaxed_ax - u[i]
            old_s[i] = relaxed_ax
            i += 1
        project_cone(s, zero, nonnegative, q_sizes, q_count)
        i = 0
        while i + W <= rows:
            u.store(
                i,
                u.load[width=W](i)
                + old_s.load[width=W](i)
                + s.load[width=W](i)
                - b.load[width=W](i),
            )
            i += W
        while i < rows:
            u[i] += old_s[i] + s[i] - b[i]
            i += 1

        iteration += 1
        if iteration % 25 == 0 or iteration == max_iters:
            csr_matvec(indptr, indices, data, x, ax, rows)
            i = 0
            while i + W <= rows:
                old_s.store(
                    i,
                    ax.load[width=W](i)
                    + s.load[width=W](i)
                    - b.load[width=W](i),
                )
                i += W
            while i < rows:
                old_s[i] = ax[i] + s[i] - b[i]
                i += 1
            primal_residual = norm(old_s, rows)

            i = 0
            while i + W <= rows:
                old_s.store(i, rho * u.load[width=W](i))
                i += W
            while i < rows:
                old_s[i] = rho * u[i]
                i += 1
            csr_tmatvec(indptr, indices, data, old_s, rhs, rows, cols)
            dense_matvec(p_dense, x, tmp_n, cols)
            i = 0
            while i + W <= cols:
                rhs.store(
                    i,
                    rhs.load[width=W](i)
                    + c.load[width=W](i)
                    + tmp_n.load[width=W](i),
                )
                i += W
            while i < cols:
                rhs[i] += c[i] + tmp_n[i]
                i += 1
            dual_residual = norm(rhs, cols)

            var ax_norm = norm(ax, rows)
            var s_norm = norm(s, rows)
            var p_scale = max(max(ax_norm, s_norm), b_norm)
            var d_scale = max(max(norm(tmp_n, cols), c_norm), norm(rhs, cols))
            var eps_primal = eps_abs * sqrt(Float64(rows)) + eps_rel * p_scale
            var eps_dual = eps_abs * sqrt(Float64(cols)) + eps_rel * d_scale

            var px = dot(x, tmp_n, cols)
            primal_objective = dot(c, x, cols) + 0.5 * px
            dual_objective = -dot(b, old_s, rows) - 0.5 * px
            gap = abs(primal_objective - dual_objective)
            var eps_gap = eps_abs + eps_rel * max(
                abs(primal_objective), abs(dual_objective)
            )
            if (
                primal_residual <= eps_primal
                and dual_residual <= eps_dual
                and gap <= eps_gap
            ):
                converged = True
                break

    var i = 0
    while i + W <= rows:
        old_s.store(i, rho * u.load[width=W](i))
        i += W
    while i < rows:
        old_s[i] = rho * u[i]
        i += 1
    stats[0] = Float64(iteration)
    stats[1] = primal_residual
    stats[2] = dual_residual
    stats[3] = gap
    stats[4] = primal_objective
    stats[5] = dual_objective
    return 1 if converged else 0

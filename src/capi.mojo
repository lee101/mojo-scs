"""C ABI for mojo-scs."""

from scs import (
    FPtr,
    I32Ptr,
    IPtr,
    cholesky,
    csr_matvec,
    csr_matvec32,
    csr_tmatvec,
    csr_tmatvec32,
    form_system,
    project_cone,
    project_cone_copy,
    solve_admm,
)


def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def i32p(address: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=address)


@export("mscs_form_system")
def mscs_form_system(
    indptr: Int,
    indices: Int,
    data: Int,
    p_dense: Int,
    matrix: Int,
    rows: Int,
    cols: Int,
    rho: Float64,
    sigma: Float64,
) abi("C") -> Int:
    form_system(
        ip(indptr),
        ip(indices),
        fp(data),
        fp(p_dense),
        fp(matrix),
        rows,
        cols,
        rho,
        sigma,
    )
    return 1 if cholesky(fp(matrix), cols) else 0


@export("mscs_csr_matvec")
def mscs_csr_matvec(
    indptr: Int,
    indices: Int,
    data: Int,
    x: Int,
    dst: Int,
    rows: Int,
) abi("C"):
    csr_matvec(ip(indptr), ip(indices), fp(data), fp(x), fp(dst), rows)


@export("mscs_csr_matvec32")
def mscs_csr_matvec32(
    indptr: Int,
    indices: Int,
    data: Int,
    x: Int,
    dst: Int,
    rows: Int,
) abi("C"):
    csr_matvec32(i32p(indptr), i32p(indices), fp(data), fp(x), fp(dst), rows)


@export("mscs_csr_tmatvec")
def mscs_csr_tmatvec(
    indptr: Int,
    indices: Int,
    data: Int,
    x: Int,
    dst: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    csr_tmatvec(ip(indptr), ip(indices), fp(data), fp(x), fp(dst), rows, cols)


@export("mscs_csr_tmatvec32")
def mscs_csr_tmatvec32(
    indptr: Int,
    indices: Int,
    data: Int,
    x: Int,
    dst: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    csr_tmatvec32(
        i32p(indptr),
        i32p(indices),
        fp(data),
        fp(x),
        fp(dst),
        rows,
        cols,
    )


@export("mscs_project_cone")
def mscs_project_cone(
    values: Int,
    zero: Int,
    nonnegative: Int,
    q_sizes: Int,
    q_count: Int,
) abi("C"):
    project_cone(fp(values), zero, nonnegative, ip(q_sizes), q_count)


@export("mscs_project_cone_copy")
def mscs_project_cone_copy(
    src: Int,
    values: Int,
    zero: Int,
    nonnegative: Int,
    q_sizes: Int,
    q_count: Int,
) abi("C"):
    project_cone_copy(
        fp(src), fp(values), zero, nonnegative, ip(q_sizes), q_count
    )


@export("mscs_solve_admm")
def mscs_solve_admm(
    indptr: Int,
    indices: Int,
    data: Int,
    p_dense: Int,
    factor: Int,
    b: Int,
    c: Int,
    q_sizes: Int,
    x: Int,
    s: Int,
    u: Int,
    ax: Int,
    rhs: Int,
    tmp_n: Int,
    old_s: Int,
    stats: Int,
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
    factor_is_diagonal: Int,
) abi("C") -> Int:
    return solve_admm(
        ip(indptr),
        ip(indices),
        fp(data),
        fp(p_dense),
        fp(factor),
        fp(b),
        fp(c),
        ip(q_sizes),
        fp(x),
        fp(s),
        fp(u),
        fp(ax),
        fp(rhs),
        fp(tmp_n),
        fp(old_s),
        fp(stats),
        rows,
        cols,
        zero,
        nonnegative,
        q_count,
        max_iters,
        eps_abs,
        eps_rel,
        rho,
        sigma,
        alpha,
        Bool(factor_is_diagonal),
    )

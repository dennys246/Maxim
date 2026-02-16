"""Pure-Python linear algebra for small matrices.

No numpy dependency.  Designed for systems up to ~20×20 where the
overhead of a C extension would dominate.  All functions are pure
(no in-place mutation) and operate on plain Python lists.

Brain mapping:
  Linear algebra operations underpin the coupled oscillator dynamics
  in the SCN (Suprachiasmatic Nucleus) module.  The coupling matrix ×
  phase vector evolution is the flagship use case.

Numerical stability:
  - Partial pivoting in Gaussian elimination
  - Householder reflections for QR decomposition
  - Wilkinson shifts for eigenvalue convergence
  - Tolerance constants: 1e-12 (singularity), 1e-15 (zero-vector)
"""

from __future__ import annotations

import logging
import math
from typing import Final

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec = list[float]
Mat = list[list[float]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SINGULAR_TOL: Final[float] = 1e-12
_ZERO_VEC_TOL: Final[float] = 1e-15
_SYMMETRY_TOL: Final[float] = 1e-10
_MAX_QR_ITERS_FACTOR: Final[int] = 100  # max iterations = factor * n


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------


def vec_add(a: Vec, b: Vec) -> Vec:
    """Element-wise vector addition."""
    return [ai + bi for ai, bi in zip(a, b)]


def vec_sub(a: Vec, b: Vec) -> Vec:
    """Element-wise vector subtraction."""
    return [ai - bi for ai, bi in zip(a, b)]


def vec_scale(v: Vec, s: float) -> Vec:
    """Scalar multiplication."""
    return [vi * s for vi in v]


def vec_dot(a: Vec, b: Vec) -> float:
    """Dot product."""
    return math.fsum(ai * bi for ai, bi in zip(a, b))


def vec_norm(v: Vec) -> float:
    """Euclidean (L2) norm."""
    return math.sqrt(math.fsum(vi * vi for vi in v))


def vec_normalize(v: Vec) -> Vec:
    """Unit vector.  Returns zero vector if norm < tolerance."""
    n = vec_norm(v)
    if n < _ZERO_VEC_TOL:
        return [0.0] * len(v)
    return [vi / n for vi in v]


def vec_elementwise_mul(a: Vec, b: Vec) -> Vec:
    """Hadamard (element-wise) product."""
    return [ai * bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------


def mat_zeros(rows: int, cols: int) -> Mat:
    """Zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def mat_identity(n: int) -> Mat:
    """Identity matrix."""
    m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def mat_from_diag(diag: Vec) -> Mat:
    """Diagonal matrix from vector."""
    n = len(diag)
    m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = diag[i]
    return m


# ---------------------------------------------------------------------------
# Matrix arithmetic
# ---------------------------------------------------------------------------


def mat_add(a: Mat, b: Mat) -> Mat:
    """Element-wise matrix addition."""
    return [[aij + bij for aij, bij in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def mat_sub(a: Mat, b: Mat) -> Mat:
    """Element-wise matrix subtraction."""
    return [[aij - bij for aij, bij in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def mat_scale(m: Mat, s: float) -> Mat:
    """Scalar multiplication."""
    return [[mij * s for mij in row] for row in m]


def mat_transpose(m: Mat) -> Mat:
    """Transpose."""
    if not m:
        return []
    cols = len(m[0])
    return [[m[r][c] for r in range(len(m))] for c in range(cols)]


def mat_shape(m: Mat) -> tuple[int, int]:
    """Returns (rows, cols)."""
    if not m:
        return (0, 0)
    return (len(m), len(m[0]))


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------


def mat_mul(a: Mat, b: Mat) -> Mat:
    """Matrix multiplication A @ B."""
    rows_a, cols_a = mat_shape(a)
    rows_b, cols_b = mat_shape(b)
    if cols_a != rows_b:
        raise ValueError(f"Incompatible shapes: ({rows_a}×{cols_a}) @ ({rows_b}×{cols_b})")

    # Transpose b for cache-friendly column access
    bt = mat_transpose(b)
    return [[math.fsum(a[i][k] * bt[j][k] for k in range(cols_a)) for j in range(cols_b)] for i in range(rows_a)]


def mat_vec_mul(m: Mat, v: Vec) -> Vec:
    """Matrix-vector multiplication M @ v."""
    rows, cols = mat_shape(m)
    if cols != len(v):
        raise ValueError(f"Incompatible shapes: ({rows}×{cols}) @ ({len(v)},)")
    return [math.fsum(m[i][k] * v[k] for k in range(cols)) for i in range(rows)]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def mat_frobenius_norm(m: Mat) -> float:
    """Frobenius norm (sqrt of sum of squared elements)."""
    return math.sqrt(math.fsum(mij * mij for row in m for mij in row))


def mat_is_symmetric(m: Mat, tol: float = _SYMMETRY_TOL) -> bool:
    """Check if matrix is symmetric within tolerance."""
    n = len(m)
    if any(len(row) != n for row in m):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(m[i][j] - m[j][i]) > tol:
                return False
    return True


def mat_clamp(m: Mat, lo: float, hi: float) -> Mat:
    """Clamp all elements to [lo, hi]."""
    return [[max(lo, min(hi, mij)) for mij in row] for row in m]


# ---------------------------------------------------------------------------
# Determinant
# ---------------------------------------------------------------------------


def determinant(m: Mat) -> float:
    """Determinant via LU-style elimination with partial pivoting."""
    n = len(m)
    if n == 0:
        return 1.0
    if any(len(row) != n for row in m):
        raise ValueError("Matrix must be square")

    # Special cases for small matrices
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    # Work on a copy
    a = [row[:] for row in m]
    det = 1.0

    for col in range(n):
        # Partial pivoting: find row with largest absolute value in column
        max_val = abs(a[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(a[row][col]) > max_val:
                max_val = abs(a[row][col])
                max_row = row

        if max_val < _SINGULAR_TOL:
            return 0.0

        # Swap rows if needed
        if max_row != col:
            a[col], a[max_row] = a[max_row], a[col]
            det *= -1.0

        det *= a[col][col]
        pivot = a[col][col]

        # Eliminate below
        for row in range(col + 1, n):
            factor = a[row][col] / pivot
            for k in range(col + 1, n):
                a[row][k] -= factor * a[col][k]

    return det


# ---------------------------------------------------------------------------
# Linear solve: Ax = b
# ---------------------------------------------------------------------------


def solve(a: Mat, b: Vec) -> Vec:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Raises ValueError if the system is singular.
    """
    n = len(a)
    if n == 0:
        return []
    if len(b) != n or any(len(row) != n for row in a):
        raise ValueError("Incompatible dimensions for Ax = b")

    # Augmented matrix [A|b]
    aug = [a[i][:] + [b[i]] for i in range(n)]

    # Forward elimination with partial pivoting
    for col in range(n):
        # Find pivot
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < _SINGULAR_TOL:
            raise ValueError("Singular matrix: no unique solution")

        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug[i][n] - s) / aug[i][i]

    return x


# ---------------------------------------------------------------------------
# Matrix inverse
# ---------------------------------------------------------------------------


def inverse(m: Mat) -> Mat:
    """Matrix inverse via Gauss-Jordan elimination.

    Raises ValueError if the matrix is singular.
    """
    n = len(m)
    if n == 0:
        return []
    if any(len(row) != n for row in m):
        raise ValueError("Matrix must be square")

    # Augmented matrix [A|I]
    aug = [m[i][:] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # Partial pivoting
        max_val = abs(aug[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row

        if max_val < _SINGULAR_TOL:
            raise ValueError("Singular matrix: cannot invert")

        if max_row != col:
            aug[col], aug[max_row] = aug[max_row], aug[col]

        # Scale pivot row
        pivot = aug[col][col]
        for k in range(2 * n):
            aug[col][k] /= pivot

        # Eliminate column in all other rows
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for k in range(2 * n):
                aug[row][k] -= factor * aug[col][k]

    # Extract right half
    return [aug[i][n:] for i in range(n)]


# ---------------------------------------------------------------------------
# QR decomposition (Householder reflections)
# ---------------------------------------------------------------------------


def _qr_decompose(m: Mat) -> tuple[Mat, Mat]:
    """QR decomposition via Householder reflections.

    Returns (Q, R) where Q is orthogonal and R is upper triangular.
    Only for square matrices (used internally for eigenvalue computation).
    """
    n = len(m)
    r = [row[:] for row in m]
    q = mat_identity(n)

    for j in range(n - 1):
        # Extract column below diagonal
        x = [r[i][j] if i >= j else 0.0 for i in range(n)]

        # Compute Householder vector
        alpha = vec_norm(x[j:])
        if x[j] > 0:
            alpha = -alpha

        # v = x - alpha * e_j
        v = [0.0] * n
        for i in range(j, n):
            v[i] = x[i]
        v[j] -= alpha

        v_norm = vec_norm(v)
        if v_norm < _ZERO_VEC_TOL:
            continue
        v = vec_scale(v, 1.0 / v_norm)

        # Apply Householder: R = (I - 2vv^T) @ R
        for k in range(j, n):
            dot = math.fsum(v[i] * r[i][k] for i in range(j, n))
            for i in range(j, n):
                r[i][k] -= 2.0 * v[i] * dot

        # Accumulate Q: Q = Q @ (I - 2vv^T)
        for k in range(n):
            dot = math.fsum(v[i] * q[k][i] for i in range(j, n))
            for i in range(j, n):
                q[k][i] -= 2.0 * v[i] * dot

    return q, r


# ---------------------------------------------------------------------------
# Eigenvalues (symmetric matrices)
# ---------------------------------------------------------------------------


def eigenvalues_symmetric(m: Mat) -> Vec:
    """Eigenvalues of a real symmetric matrix via QR algorithm with Wilkinson shifts.

    Returns eigenvalues sorted in ascending order.  Falls back to
    returning the diagonal after max iterations with a warning.
    """
    n = len(m)
    if n == 0:
        return []
    if not mat_is_symmetric(m):
        log.warning("eigenvalues_symmetric called on non-symmetric matrix; results may be inaccurate")

    # Special cases
    if n == 1:
        return [m[0][0]]
    if n == 2:
        a, b = m[0][0], m[0][1]
        d = m[1][1]
        trace = a + d
        det_val = a * d - b * b
        disc = trace * trace - 4.0 * det_val
        if disc < 0:
            disc = 0.0
        sqrt_disc = math.sqrt(disc)
        e1 = (trace - sqrt_disc) / 2.0
        e2 = (trace + sqrt_disc) / 2.0
        return sorted([e1, e2])

    # Work on a copy — iterative QR
    a = [row[:] for row in m]
    max_iters = _MAX_QR_ITERS_FACTOR * n

    for iteration in range(max_iters):
        # Check convergence: off-diagonal elements small enough?
        off_diag = math.fsum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off_diag < _SINGULAR_TOL:
            break

        # Wilkinson shift: eigenvalue of bottom-right 2×2 closer to a[n-1][n-1]
        am = a[n - 2][n - 2]
        bm = a[n - 2][n - 1]
        cm = a[n - 1][n - 1]
        delta = (am - cm) / 2.0
        sign_delta = 1.0 if delta >= 0 else -1.0
        denom = abs(delta) + math.sqrt(delta * delta + bm * bm)
        if denom < _ZERO_VEC_TOL:
            shift = cm
        else:
            shift = cm - sign_delta * bm * bm / denom

        # Shifted QR step: A - shift*I = QR, then A = RQ + shift*I
        shifted = [row[:] for row in a]
        for i in range(n):
            shifted[i][i] -= shift

        q, r = _qr_decompose(shifted)
        a = mat_add(mat_mul(r, q), mat_scale(mat_identity(n), shift))
    else:
        log.warning(
            "eigenvalues_symmetric did not converge after %d iterations; "
            "returning diagonal as approximation",
            max_iters,
        )

    eigenvals = [a[i][i] for i in range(n)]
    return sorted(eigenvals)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Types
    "Vec",
    "Mat",
    # Vector ops
    "vec_add",
    "vec_sub",
    "vec_scale",
    "vec_dot",
    "vec_norm",
    "vec_normalize",
    "vec_elementwise_mul",
    # Matrix construction
    "mat_zeros",
    "mat_identity",
    "mat_from_diag",
    # Matrix arithmetic
    "mat_add",
    "mat_sub",
    "mat_scale",
    "mat_transpose",
    "mat_shape",
    # Matrix multiplication
    "mat_mul",
    "mat_vec_mul",
    # Utilities
    "mat_frobenius_norm",
    "mat_is_symmetric",
    "mat_clamp",
    # Solvers
    "determinant",
    "solve",
    "inverse",
    # Decomposition
    "eigenvalues_symmetric",
]

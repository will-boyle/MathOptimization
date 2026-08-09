"""
solver4.py — D.C. (Difference of Convex) Programming: Simplicial Branch and Bound

Solves:
    min  f0(x) - g0(x)
    s.t. fi(x) - gi(x) <= 0    i = 1, ..., m

where f0, g0, fi, gi are all convex (so the objective and each constraint are D.C.).

At each node (simplex S with vertices v_0 .. v_n, centroid c):
  1. Build affine lower bounds for the convex parts via subgradient at c:
         li(x) = fi(c) + grad_fi(c)^T (x - c)     [li(x) <= fi(x) everywhere]
  2. Build affine lower bounds for the concave parts -gi via vertex interpolation:
         mi(v_j) = -gi(v_j)  →  fit unique affine  mi  through these n+1 points
         [mi(x) <= -gi(x) for all x in S, since -gi is concave on S]
  3. Solve LP:  min l0(x)+m0(x)  s.t. li(x)+mi(x) <= 0,  x in S  →  lower bound L
  4. Check x*, centroid, and vertices for d.c. feasibility  →  update upper bound U
  5. If U - L <= tol: done. Else: bisect S on its longest edge, push two children.

References: Tuy (1998) "D.C. Optimization: Theory, Methods and Algorithms."
"""

import numpy as np
import sympy as sp
from scipy.optimize import linprog
import heapq


# ─────────────────────────────────────────────────── geometry helpers


def _centroid(verts):
    """Mean of the n+1 simplex vertices. verts: (n+1, n) -> (n,)"""
    return np.mean(verts, axis=0)


def _affine_fit(verts, vals):
    """
    Fit the unique affine  m(x) = a @ x + b  through {(v_j, vals[j])}.

    Solves the (n+1)x(n+1) system  [V | 1] @ [a; b] = vals,
    where each row of [V | 1] is [v_j^T, 1].

    Returns (a, b).   verts: (n+1, n),  vals: (n+1,)
    """
    mat = np.column_stack([verts, np.ones(len(vals))])   # (n+1) x (n+1)
    try:
        coeffs = np.linalg.solve(mat, vals)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(mat, vals, rcond=None)
    return np.asarray(coeffs[:-1], float), float(coeffs[-1])


def _simplex_halfspaces(verts):
    """
    Returns (A_ub, b_ub) s.t.  A_ub @ x <= b_ub  <=>  x in conv(verts).

    Derivation: barycentric coords lambda = M_inv @ [x; 1] must be >= 0,
    where M = [verts.T; 1^T] (each column is [v_j; 1]).
    Rewriting lambda_j >= 0 gives one affine inequality per vertex.
    """
    k, n = verts.shape                                    # k = n+1
    M     = np.vstack([verts.T, np.ones((1, k))])         # (n+1) x (n+1)
    M_inv = np.linalg.inv(M)
    # lambda_j = M_inv[j, :n] @ x + M_inv[j, n]  >= 0
    return -M_inv[:, :n], M_inv[:, n]                     # A_ub, b_ub


def _longest_edge(verts):
    k = len(verts)
    best, ii, jj = -1.0, 0, 1
    for i in range(k):
        for j in range(i + 1, k):
            d = np.linalg.norm(verts[i] - verts[j])
            if d > best:
                best, ii, jj = d, i, j
    return ii, jj


def _bisect(verts):
    """Split the simplex along its longest edge. Returns two child vertex arrays."""
    i, j = _longest_edge(verts)
    mid   = (verts[i] + verts[j]) / 2.0
    s1, s2 = verts.copy(), verts.copy()
    s1[i] = mid
    s2[j] = mid
    return s1, s2


def initial_simplex(lb, ub):
    """
    Construct an (n+1)-vertex simplex S that contains [lb, ub]^n.

    Construction:
        v_0    = lb
        v_i[j] = lb[j]  for j != i-1
        v_i[i-1] = lb[i-1] + n * (ub[i-1] - lb[i-1])   for i = 1..n

    Every corner of [lb, ub]^n has barycentric coords 1/n for each non-zero
    direction vertex, so all coords are non-negative — the hypercube lies in S.
    """
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    n  = len(lb)
    verts = np.tile(lb, (n + 1, 1)).copy()                # (n+1) x n, all rows lb
    for i in range(1, n + 1):
        verts[i, i - 1] += n * (ub[i - 1] - lb[i - 1])
    return verts


# ─────────────────────────────────────────────────── LP relaxation


def _solve_lp_node(verts, f0, g0, grad_f0, grad_g0,
                   fi_fns, gi_fns, grad_fi_fns, grad_gi_fns):
    """
    Build and solve the LP relaxation for one simplex node.

    Returns (x_opt, L) where L is the lower bound on this subtree,
    or (None, +inf) if the LP is infeasible (no original solutions here).
    """
    cpt = _centroid(verts)
    n   = len(cpt)

    def linearise(fn, gfn):
        # l(x) = fn(c) + grad_fn(c) @ (x - c)  =  g @ x + (fn(c) - g @ c)
        g  = np.asarray(gfn(cpt), float)
        fc = float(fn(cpt))
        return g, fc - float(g @ cpt)      # (a_vec, scalar_const)

    def underest_concave(fn):
        # Affine m(x) with m(v_j) = -fn(v_j): unique fit at simplex vertices.
        vals = np.array([-float(fn(v)) for v in verts])
        return _affine_fit(verts, vals)    # (a_vec, scalar_const)

    l0_a, l0_c = linearise(f0, grad_f0)
    m0_a, m0_c = underest_concave(g0)

    obj_a   = l0_a + m0_a           # gradient of LP objective
    obj_const = l0_c + m0_c         # constant (not passed to linprog)

    A_rows, b_rows = [], []

    for fi, gi, gfi in zip(fi_fns, gi_fns, grad_fi_fns):
        la, lc = linearise(fi, gfi)
        ma, mc = underest_concave(gi)
        # (la+ma) @ x + (lc+mc) <= 0  ->  row @ x <= rhs
        A_rows.append(la + ma)
        b_rows.append(-(lc + mc))

    A_s, b_s = _simplex_halfspaces(verts)
    A_rows.extend(A_s)
    b_rows.extend(b_s)

    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)

    res = linprog(obj_a, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * n, method='highs')

    if res.status == 0:
        return res.x, float(res.fun) + obj_const
    return None, np.inf


# ─────────────────────────────────────────────────── feasibility check


def _dc_feasible(x, fi_fns, gi_fns, tol=1e-6):
    """True if fi(x) - gi(x) <= tol for every constraint."""
    return all(float(fi(x)) - float(gi(x)) <= tol
               for fi, gi in zip(fi_fns, gi_fns))


# ─────────────────────────────────────────────────── main B&B loop


def solve_dc(f0, g0, grad_f0, grad_g0,
             fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
             lb, ub, tol=1e-4, max_iter=500, verbose=True):
    """
    Solve a D.C. program by simplicial branch and bound.

    Parameters
    ----------
    f0, g0             callable (n,) -> float   objective = f0(x) - g0(x)
    grad_f0, grad_g0   callable (n,) -> (n,)    gradients of f0 and g0
    fi_fns, gi_fns     lists of callables        constraint: fi(x) - gi(x) <= 0
    grad_fi_fns,
    grad_gi_fns        matching gradient lists
    lb, ub             (n,) variable bounds; the initial simplex covers [lb, ub]^n
    tol                convergence: stop when U - L <= tol
    max_iter           maximum number of B&B nodes explored
    verbose            print progress

    Returns
    -------
    x_best : (n,) best feasible point found, or None
    U      : best objective upper bound (inf if no feasible point found)
    """
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    verts0  = initial_simplex(lb, ub)
    U       = np.inf
    x_best  = None
    counter = 0

    heap = []
    heapq.heappush(heap, (-np.inf, counter, verts0))

    for it in range(max_iter):
        if not heap:
            if verbose:
                print("    Branch-and-bound complete: heap empty.")
            break

        L_heap, _, verts = heapq.heappop(heap)

        if L_heap >= U - tol:
            continue     # pruned: this subtree cannot improve U

        x_lp, L = _solve_lp_node(
            verts, f0, g0, grad_f0, grad_g0,
            fi_fns, gi_fns, grad_fi_fns, grad_gi_fns
        )

        if L == np.inf:
            continue     # LP infeasible -> no original feasible points here

        # Prune again with the tighter LP bound
        if L >= U - tol:
            continue

        # ── Upper bound: check candidate points for feasibility ──
        candidates = list(verts) + [_centroid(verts)]
        if x_lp is not None:
            candidates.append(x_lp)

        for x_cand in candidates:
            if _dc_feasible(x_cand, fi_fns, gi_fns):
                val = float(f0(x_cand)) - float(g0(x_cand))
                if val < U:
                    U, x_best = val, x_cand.copy()
                    if verbose:
                        print(f"    iter {it:4d}: new upper bound  U = {U:+.6f}")

        gap = U - L
        if verbose and it % 25 == 0:
            print(f"    iter {it:4d}: L = {L:+.6f}  U = {U:+.6f}  "
                  f"gap = {gap:.2e}  |heap| = {len(heap) + 1}")

        if gap <= tol:
            if verbose:
                print(f"    iter {it:4d}: converged  gap = {gap:.2e}")
            break

        # ── Branch: bisect on longest edge ──
        s1, s2 = _bisect(verts)
        counter += 1;  heapq.heappush(heap, (L, counter, s1))
        counter += 1;  heapq.heappush(heap, (L, counter, s2))

    else:
        if verbose:
            print(f"    max_iter = {max_iter} reached.  U = {U:.6f}")

    return x_best, U


# ─────────────────────────────────────────────────── string parsing


def parse_dc(f0_str, g0_str, dc_ineq_strs):
    """
    Parse a D.C. program from SymPy-compatible strings.

    Parameters
    ----------
    f0_str       : str   convex part of objective,  e.g. "x1**2 + x2**2"
    g0_str       : str   concave part (subtracted), e.g. "x1 + x2"
    dc_ineq_strs : list of (fi_str, gi_str)  for constraints fi(x) - gi(x) <= 0

    Returns
    -------
    f0, g0, grad_f0, grad_g0,
    fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
    syms
    """
    f0_expr = sp.sympify(f0_str)
    g0_expr = sp.sympify(g0_str)
    fi_exprs = [sp.sympify(f) for f, _ in dc_ineq_strs]
    gi_exprs = [sp.sympify(g) for _, g in dc_ineq_strs]

    all_exprs = [f0_expr, g0_expr] + fi_exprs + gi_exprs
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name,
    )

    def _scalar(expr):
        fn = sp.lambdify(syms, expr, 'numpy')
        return lambda x: float(fn(*x))

    def _grad(expr):
        diffs = [sp.lambdify(syms, sp.diff(expr, s), 'numpy') for s in syms]
        return lambda x: np.array([float(d(*x)) for d in diffs])

    return (
        _scalar(f0_expr), _scalar(g0_expr),
        _grad(f0_expr),   _grad(g0_expr),
        [_scalar(e) for e in fi_exprs],
        [_scalar(e) for e in gi_exprs],
        [_grad(e)   for e in fi_exprs],
        [_grad(e)   for e in gi_exprs],
        syms,
    )

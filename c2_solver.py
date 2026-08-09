"""
solver5.py — αBB Global Optimizer

1. Set upper bound U = ∞ and best point x* = none. Place the initial box
   [lb, ub] in a list ordered by lower bound.

2. Take the box with the smallest lower bound from the list.

   a. Evaluate the Hessian at all 2ⁿ corners of the box. Find the most
      negative eigenvalue seen. Set α = max(0, −λ_min / 2) · 1.1. A larger
      α makes the underestimator more convex but looser; this formula gives
      the smallest α that guarantees convexity over the box.

   b. Build the underestimator f̃(x) = f(x) + α · Σᵢ(xᵢ−lᵢ)(xᵢ−uᵢ). The
      correction term is zero at every corner and negative in the interior,
      so f̃(x) ≤ f(x) everywhere on the box. Because f̃ is convex, minimize
      it subject to the (similarly convexified) constraints and the box bounds
      using a convex solver. Let L = f̃(x_sub) be the lower bound.

   c. If the subproblem has no feasible point: discard this box and go back
      to step 2. Its entire region is infeasible for the original problem.

   d. Evaluate f at x_sub, the box center, and every corner. For any that
      satisfy the original (unrelaxed) constraints, update U = min(U, f(x))
      and x*.

   e. If U − L ≤ tolerance: stop. Return x* and U.

   f. If L ≥ U: discard this box and go back to step 2. It cannot improve
      on the current best.

   g. Split the box along its widest dimension. Add both halves to the list
      with lower bound L, in their sorted positions.

3. If the list is empty, return x* and U.

Reference: Androulakis, Maranas, Floudas (1995)
           "αBB: A global optimization method for general constrained
            nonconvex problems." J. Global Optimization.
"""

import numpy as np
import sympy as sp
from scipy.optimize import minimize
from itertools import product  # used in _compute_alpha corner sweep
import heapq


# ─────────────────────────────────────────────────── α computation


def _compute_alpha(hess_fn, lb, ub, safety=0.1):
    """
    Estimate α = max(0, −λ_min(∇²f0) / 2) over [lb, ub].

    Evaluates the Hessian at all 2ⁿ corners and the centroid.
    For polynomial Hessians (the typical case with SymPy input), corners
    capture all extremes of each entry, so this is exact.
    A `safety` fraction is added as a buffer against numerical error.
    Falls back to 512 random samples when n > 15.
    """
    n = len(lb)
    pts = list(product(*zip(lb, ub))) if n <= 15 else []
    if n > 15:
        rng = np.random.default_rng(42)
        pts = [lb + rng.random(n) * (ub - lb) for _ in range(512)]
    pts.append((np.asarray(lb) + np.asarray(ub)) / 2.0)

    lam_min = np.inf
    for pt in pts:
        H = np.asarray(hess_fn(np.asarray(pt, float)), float)
        lam_min = min(lam_min, np.linalg.eigvalsh(H).min())

    alpha = max(0.0, -lam_min / 2.0)
    return alpha * (1.0 + safety)


# ─────────────────────────────────────────────────── subproblem


def _solve_subproblem(f0_fn, grad_f0_fn, lb, ub, alpha,
                       fi_fns=None, grad_fi_fns=None, hess_fi_fns=None):
    """
    Minimize the convex underestimator f̃(x) = f0(x) + α·Σ(xᵢ−lᵢ)(xᵢ−uᵢ)
    over the box [lb, ub] subject to convexified constraints g̃_i(x) ≤ 0, where:

        g̃_i(x) = fi(x) + αᵢ·Σ(xᵢ−lᵢ)(xᵢ−uᵢ)

    αᵢ is chosen so that ∇²g̃_i ⪰ 0, making each constraint convex.  Because
    g̃_i ≤ fi on the box, the relaxed feasible set contains the original one,
    so the subproblem minimum is a valid lower bound.

    If no feasible point exists (relaxed problem infeasible), returns
    (None, +inf) so the caller can prune the box.
    """
    fi_fns      = fi_fns      or []
    grad_fi_fns = grad_fi_fns or []
    hess_fi_fns = hess_fi_fns or [None] * len(fi_fns)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    def ftilde(x):
        return float(f0_fn(x)) + alpha * float(np.dot(x - lb, x - ub))

    def grad_ftilde(x):
        return np.asarray(grad_f0_fn(x), float) + alpha * (2.0 * x - lb - ub)

    # Build convexified underestimators for each constraint.
    # scipy 'ineq' convention: fun(x) >= 0  →  we need -g̃_i(x) >= 0
    constraints = []
    for fi, gfi, hfi in zip(fi_fns, grad_fi_fns, hess_fi_fns):
        ai = _compute_alpha(hfi, lb, ub) if hfi is not None else 0.0
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, fi=fi, ai=ai:
                       -(float(fi(x)) + ai * float(np.dot(x - lb, x - ub))),
            'jac': lambda x, gfi=gfi, ai=ai:
                       -(np.asarray(gfi(x), float) + ai * (2.0 * x - lb - ub)),
        })

    bounds = list(zip(lb, ub))

    # f̃ is convex, so a single starting point (center) is sufficient —
    # any locally optimal solution is globally optimal.
    x0  = (lb + ub) / 2.0
    res = minimize(ftilde, x0, jac=grad_ftilde,
                   bounds=bounds, constraints=constraints,
                   method='SLSQP',
                   options={'ftol': 1e-10, 'maxiter': 500})

    if res.status in (0, 1, 9):
        return res.x.copy(), float(ftilde(res.x))
    return None, np.inf   # subproblem infeasible


# ─────────────────────────────────────────────────── feasibility


def _feasible(x, fi_fns, tol=1e-6):
    return all(float(fi(x)) <= tol for fi in fi_fns)


# ─────────────────────────────────────────────────── box splitting


def _split_box(lb, ub):
    """Bisect the box along its widest dimension."""
    idx = int(np.argmax(np.asarray(ub) - np.asarray(lb)))
    mid = (lb[idx] + ub[idx]) / 2.0
    lb1, ub1 = lb.copy(), ub.copy(); ub1[idx] = mid
    lb2, ub2 = lb.copy(), ub.copy(); lb2[idx] = mid
    return (lb1, ub1), (lb2, ub2)


# ─────────────────────────────────────────────────── main B&B loop


def solve_abb(f0_fn, grad_f0_fn, hess_f0_fn,
              fi_fns, grad_fi_fns, hess_fi_fns=None,
              lb=None, ub=None,
              tol=1e-4, max_iter=1000, verbose=True):
    """
    αBB global optimizer — box branch and bound.

    Parameters
    ----------
    f0_fn, grad_f0_fn, hess_f0_fn   objective callables
    fi_fns, grad_fi_fns             inequality constraint lists (fi(x) <= 0)
    hess_fi_fns                     Hessian callables for each constraint;
                                    used to convexify constraints per box
    lb, ub                          (n,) bounding box
    tol                             convergence: stop when U − L <= tol
    max_iter                        maximum B&B nodes explored

    Returns
    -------
    x_best : (n,) global minimiser found, or None
    U      : best objective value (inf if no feasible point found)
    """
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    U       = np.inf
    x_best  = None
    counter = 0
    heap    = []

    def push(L_bound, lb_box, ub_box):
        nonlocal counter
        counter += 1
        heapq.heappush(heap, (L_bound, counter, lb_box, ub_box))

    push(-np.inf, lb.copy(), ub.copy())

    for it in range(max_iter):
        if not heap:
            if verbose:
                print("    B&B complete: heap empty.")
            break

        L_heap, _, lb_box, ub_box = heapq.heappop(heap)

        if L_heap >= U - tol:
            continue     # pruned by incumbent

        # ── α for this box ──
        alpha = _compute_alpha(hess_f0_fn, lb_box, ub_box)

        # ── Solve convex subproblem (constraints convexified per box) ──
        x_sub, L = _solve_subproblem(
            f0_fn, grad_f0_fn, lb_box, ub_box, alpha,
            fi_fns, grad_fi_fns, hess_fi_fns,
        )

        if L >= U:
            continue     # prune: box cannot improve on current best

        # ── Check center and subproblem solution for upper bound ──
        candidates = [(lb_box + ub_box) / 2.0]
        if x_sub is not None:
            candidates.append(np.clip(x_sub, lb_box, ub_box))

        for x_cand in candidates:
            if _feasible(x_cand, fi_fns):
                val = float(f0_fn(x_cand))
                if val < U:
                    U, x_best = val, x_cand.copy()
                    if verbose:
                        print(f"    iter {it:4d}: new upper bound  U = {U:+.8f}")

        gap = U - L
        if verbose and it % 50 == 0:
            print(f"    iter {it:4d}: L = {L:+.6f}  U = {U:+.6f}  "
                  f"gap = {gap:.2e}  a = {alpha:.4f}  |heap| = {len(heap) + 1}")

        if gap <= tol:
            if verbose:
                print(f"    iter {it:4d}: converged  gap = {gap:.2e}")
            break

        # ── Branch on widest dimension ──
        (lb1, ub1), (lb2, ub2) = _split_box(lb_box, ub_box)
        push(L, lb1, ub1)
        push(L, lb2, ub2)

    else:
        if verbose:
            print(f"    max_iter = {max_iter} reached.  U = {U:.8f}")

    return x_best, U


# ─────────────────────────────────────────────────── parsing


def parse_abb(f0_str, ineq_strs=None):
    """
    Parse a global optimisation problem from SymPy-compatible strings.

    Parameters
    ----------
    f0_str    : str  objective, e.g. "x1**4 - 4*x1**2 + x2**2"
    ineq_strs : list of str  constraints fi(x) <= 0, e.g. ["x1 - 1"]

    Returns
    -------
    f0_fn, grad_f0_fn, hess_f0_fn, fi_fns, grad_fi_fns, syms
    """
    ineq_strs = ineq_strs or []

    f0_expr  = sp.sympify(f0_str)
    fi_exprs = [sp.sympify(s) for s in ineq_strs]

    all_exprs = [f0_expr] + fi_exprs
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name,
    )
    n = len(syms)

    def _hess_mat(expr):
        return sp.Matrix([
            [sp.diff(sp.diff(expr, si), sj) for sj in syms]
            for si in syms
        ])

    def _scalar(expr):
        fn = sp.lambdify(syms, expr, 'numpy')
        return lambda x: float(fn(*x))

    def _grad(expr):
        diffs = [sp.lambdify(syms, sp.diff(expr, s), 'numpy') for s in syms]
        return lambda x: np.array([float(d(*x)) for d in diffs])

    def _hess(mat):
        rows = [[sp.lambdify(syms, mat[i, j], 'numpy') for j in range(n)]
                for i in range(n)]
        return lambda x: np.array([[float(rows[i][j](*x)) for j in range(n)]
                                   for i in range(n)])

    return (
        _scalar(f0_expr),
        _grad(f0_expr),
        _hess(_hess_mat(f0_expr)),
        [_scalar(e) for e in fi_exprs],
        [_grad(e)   for e in fi_exprs],
        [_hess(_hess_mat(e)) for e in fi_exprs],
        syms,
    )

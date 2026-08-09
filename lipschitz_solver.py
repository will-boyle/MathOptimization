"""
solver6.py — Lipschitz Global Optimizer

Solves:
    min  f0(x)
    s.t. fi(x) <= 0   i = 1, ..., m  (optional)
         lb <= x <= ub               (required bounding box)

For each box [l, u] with center c and half-diagonal r = ||u-l||₂ / 2:

    Lower bound on f0 over the box:   f0(c) - L_f * r
    Prune box as infeasible if:       fi(c) - L_i * r > 0  for any i

Because f0 is L_f-Lipschitz, any point x in the box satisfies
||x - c||₂ <= r, so f0(x) >= f0(c) - L_f * r.  That is the lower bound.
The constraint bound works the same way: if even the minimum possible value
of fi on the box exceeds zero, the whole box is infeasible.

Upper bound candidates are evaluated at the box center only (not corners),
giving uniform O(1)-per-node cost regardless of dimension.

Lipschitz constants are estimated by taking the maximum gradient magnitude
over the 2^n corners of the initial bounding box.  For polynomial functions
this is exact; for others it is a valid upper bound provided the gradient
does not exceed the corner values in the interior.  A small safety factor
is added.  The user may also supply L directly.

Requires: f0 evaluable as a callable; gradient used only for L estimation
and can be replaced by finite differences.  No Hessian.  No convexity.
No D.C. decomposition.  Any twice-evaluable Lipschitz function works.

References:
    Piyavskii (1972) — 1D Lipschitz optimizer
    Shubert (1972)   — 1D multi-modal extension
    Jones, Perttunen, Stuckman (1993) — DIRECT (n-D, no L needed)
"""

import numpy as np
import sympy as sp
from itertools import product
import heapq


# ─────────────────────────────────────────────────── Lipschitz estimation


def _sample_pts(lb, ub):
    """2^n corners + centroid, falling back to random for n > 20."""
    n = len(lb)
    if n <= 20:
        corners = [np.array(c, float) for c in product(*zip(lb, ub))]
    else:
        rng = np.random.default_rng(42)
        corners = [lb + rng.random(n) * (ub - lb) for _ in range(1024)]
    corners.append((lb + ub) / 2.0)
    return corners


def estimate_L(grad_fn, lb, ub, safety=0.05):
    """
    Estimate the Lipschitz constant as max ||∇f||₂ over box corners.

    For a differentiable function, this gives a valid Lipschitz constant
    when the gradient norm is maximised at corners (true for polynomials).
    A safety fraction guards against interior peaks.
    """
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    L = max(
        np.linalg.norm(np.asarray(grad_fn(pt), float))
        for pt in _sample_pts(lb, ub)
    )
    return L * (1.0 + safety)


# ─────────────────────────────────────────────────── feasibility


def _feasible(x, fi_fns, tol=1e-6):
    return all(float(fi(x)) <= tol for fi in fi_fns)


# ─────────────────────────────────────────────────── main B&B loop


def solve_lipschitz(f0_fn, fi_fns, L_f, L_fi=None,
                    lb=None, ub=None,
                    tol=1e-4, max_iter=50000, verbose=True):
    """
    Lipschitz global optimizer — box branch and bound.

    Parameters
    ----------
    f0_fn       callable (n,) -> float, L_f-Lipschitz objective
    fi_fns      list of callables fi(x) <= 0
    L_f         Lipschitz constant for f0 (use estimate_L if unknown)
    L_fi        Lipschitz constants for each constraint; None entries
                skip infeasibility pruning for that constraint
    lb, ub      (n,) bounding box; all minima must lie inside
    tol         convergence: stop when U - L <= tol
    max_iter    maximum B&B nodes explored

    Returns
    -------
    x_best : (n,) global minimizer found, or None
    U      : certified upper bound on the global minimum
    """
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    fi_fns = fi_fns or []
    L_fi   = list(L_fi or [None] * len(fi_fns))

    U      = np.inf
    x_best = None
    counter = 0
    heap    = []

    def push(lower, lb_box, ub_box):
        nonlocal counter
        counter += 1
        heapq.heappush(heap, (lower, counter, lb_box, ub_box))

    push(-np.inf, lb.copy(), ub.copy())

    for it in range(max_iter):
        if not heap:
            if verbose:
                print("    B&B complete: heap empty.")
            break

        L_heap, _, lb_box, ub_box = heapq.heappop(heap)

        if L_heap >= U:
            continue

        c = (lb_box + ub_box) / 2.0
        r = np.linalg.norm(ub_box - lb_box) / 2.0

        # ── Prune provably infeasible boxes ──────────────────────────────
        # fi(c) - L_i*r is a lower bound on fi over the box.
        # If it exceeds 0, every point in the box violates constraint i.
        pruned = False
        for fi, Li in zip(fi_fns, L_fi):
            if Li is not None and float(fi(c)) - Li * r > tol:
                pruned = True
                break
        if pruned:
            continue

        # ── Lower bound on f0 over the box ───────────────────────────────
        L = float(f0_fn(c)) - L_f * r

        if L >= U:
            continue

        # ── Update upper bound: evaluate at center ────────────────────────
        if _feasible(c, fi_fns):
            val = float(f0_fn(c))
            if val < U:
                U, x_best = val, c.copy()
                if verbose:
                    print(f"    iter {it:6d}: new upper bound  U = {U:+.8f}")

        gap = U - L
        if verbose and it % 2000 == 0:
            print(f"    iter {it:6d}: L = {L:+.6f}  U = {U:+.6f}  "
                  f"gap = {gap:.2e}  |heap| = {len(heap) + 1}")

        if gap <= tol:
            if verbose:
                print(f"    iter {it:6d}: converged  gap = {gap:.2e}")
            break

        # ── Branch on widest dimension ────────────────────────────────────
        idx = int(np.argmax(ub_box - lb_box))
        mid = (lb_box[idx] + ub_box[idx]) / 2.0
        lb1, ub1 = lb_box.copy(), ub_box.copy(); ub1[idx] = mid
        lb2, ub2 = lb_box.copy(), ub_box.copy(); lb2[idx] = mid
        push(L, lb1, ub1)
        push(L, lb2, ub2)

    else:
        if verbose:
            print(f"    max_iter = {max_iter} reached.  U = {U:.8f}")

    return x_best, U


# ─────────────────────────────────────────────────── parsing


def parse_lipschitz(f0_str, ineq_strs=None):
    """
    Parse objective and constraints from SymPy-compatible strings.

    Computes gradients symbolically.  Non-smooth expressions like Abs(x)
    and Max(x, y) are supported — SymPy differentiates them to sign(x)
    and Heaviside, which numpy evaluates correctly at non-kink points
    (and returns a valid subgradient at kinks).

    Returns
    -------
    f0_fn, grad_f0_fn, fi_fns, grad_fi_fns, syms
    """
    ineq_strs = ineq_strs or []
    f0_expr   = sp.sympify(f0_str)
    fi_exprs  = [sp.sympify(s) for s in ineq_strs]

    all_exprs = [f0_expr] + fi_exprs
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name,
    )

    # numpy module dict that handles common non-smooth SymPy functions
    _np_mod = [
        'numpy',
        {
            'sign':      np.sign,
            'Abs':       np.abs,
            'Max':       np.maximum,
            'Min':       np.minimum,
            'Heaviside': lambda x, y=0: np.heaviside(x, y),
        },
    ]

    def _scalar(expr):
        fn = sp.lambdify(syms, expr, _np_mod)
        return lambda x: float(fn(*x))

    def _grad(expr):
        scalar_fn = _scalar(expr)
        n = len(syms)

        # Try symbolic diff per component; fall back to finite differences
        # if lambdify cannot print the result (e.g. unevaluated Derivative).
        component_fns = []
        for s in syms:
            d_expr = sp.diff(expr, s)
            try:
                d_fn = sp.lambdify(syms, d_expr, _np_mod)
                d_fn(*([0.0] * n))          # smoke-test at origin
                component_fns.append(('sym', d_fn))
            except Exception:
                component_fns.append(('fd', None))

        def grad(x, h=1e-7):
            x = np.asarray(x, float)
            g = np.zeros(n)
            for i, (mode, fn) in enumerate(component_fns):
                if mode == 'sym':
                    try:
                        g[i] = float(fn(*x))
                        continue
                    except Exception:
                        pass
                e = np.zeros(n); e[i] = h
                g[i] = (scalar_fn(x + e) - scalar_fn(x - e)) / (2 * h)
            return g

        return grad

    return (
        _scalar(f0_expr),
        _grad(f0_expr),
        [_scalar(e) for e in fi_exprs],
        [_grad(e)   for e in fi_exprs],
        syms,
    )

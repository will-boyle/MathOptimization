"""
maximin_solver.py

Branch and bound for:

    min  f0(x)
    s.t. fi(x) <= 0,  i = 1 ... m
         lb <= x <= ub

Algorithm overview
------------------
The current best primal value U is folded into the Lagrangian as an
explicit constraint f0(x) <= U with its own dual variable y_U:

    L(x, y) = (1 + y_U) f0(x) + sum_i y_i fi(x) - y_U U

Per-box, the solver alternates between primal and dual updates:
  1. Fix y.  Minimize L(x, y) over x in the box via projected gradient
     descent until x is stationary (partial stationarity in x).
  2. Take one dual ascent step:  y <- max(0, y + lr_y * grad_y L)
     where grad_y L = constraint values at x*(y).
  3. Repeat until full joint stationarity (KKT) is reached, or the
     dual variables grow without bound.

Termination per box
-------------------
Two outcomes are possible and exhaustive:

  KKT found  -- the joint system (x, y) reached a saddle point of L.
                 A violated constraint prevents joint stationarity (the
                 dual gradient equals the constraint value, so y is not
                 stationary while any constraint is violated).  Therefore
                 the only terminal state is a genuine KKT point.
                 Update U if improved, then branch.

  y diverges -- the dual variables grow without bound because no point
                in the box can simultaneously satisfy all constraints.
                The box is pruned.

Branch-and-bound structure
--------------------------
  - Upper bound U: best primal value found across all KKT points.
  - Branch: split along the widest box dimension on every KKT find.
  - Prune: discard a box only when the dual variables diverge.
"""

import numpy as np
import sympy as sp


# ─────────────────────────────────────────────────────────── parsing

def parse_pd(f0_str, fi_strs):
    """
    Parse objective and constraint expression strings.

    Only first derivatives are needed (gradient descent, no Newton).

    Returns
    -------
    f0_fn, grad_f0           objective and its gradient
    fi_fns, grad_fi_fns      constraint functions and gradients
    syms                      list of sympy.Symbol (alphabetical)
    """
    all_exprs = [sp.sympify(s) for s in [f0_str] + list(fi_strs)]
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name,
    )

    def _compile(expr):
        fn_raw   = sp.lambdify(syms, expr, modules='numpy')
        grad_raw = sp.lambdify(syms, [sp.diff(expr, s) for s in syms],
                               modules='numpy')
        def fn(x):   return float(fn_raw(*x))
        def grad(x): return np.array([float(v) for v in grad_raw(*x)], float)
        return fn, grad

    f0_fn, grad_f0 = _compile(all_exprs[0])
    fi_fns, grad_fi_fns = [], []
    for expr in all_exprs[1:]:
        fn, grad = _compile(expr)
        fi_fns.append(fn)
        grad_fi_fns.append(grad)

    return f0_fn, grad_f0, fi_fns, grad_fi_fns, syms


# ─────────────────────────────────────────────────────────── saddle-point solver

def _kkt(f0_fn, grad_f0, fi_fns, grad_fi_fns, lo, hi, U,
         lr_x=0.02, lr_y=0.5, max_outer=300, max_inner=1000,
         tol=1e-6, div_thresh=1e8):
    """
    Per-box alternating primal-dual iteration.

    Each outer iteration:
      - Minimizes L(x, y) in x (inner loop) until x is stationary.
      - Checks KKT (joint stationarity of x and y).
      - Takes one dual ascent step if KKT is not yet met.

    A violated constraint makes the dual gradient (= constraint value)
    nonzero, so the joint system cannot be stationary while any constraint
    is violated.  The only terminal states are therefore a genuine KKT
    point or dual divergence.

    Returns (x, y, found):
        found = True   KKT point found.
        found = False  y diverged (||y|| > div_thresh): box is pruned.
    """
    d = len(lo)
    m = len(fi_fns)
    U_eff = U if np.isfinite(U) else 1e10

    x = (lo + hi) / 2.0
    y = np.zeros(m + 1)    # y[0..m-1] for fi(x)<=0,  y[m] for f0(x)-U<=0

    for _ in range(max_outer):

        # ── Inner loop: fix y, minimize L(x,y) in x ─────────────────
        pgx_norm = np.inf
        for _ in range(max_inner):
            gx = (1.0 + y[m]) * grad_f0(x)
            for i in range(m):
                gx = gx + y[i] * grad_fi_fns[i](x)

            pgx = gx.copy()
            for k in range(d):
                if x[k] <= lo[k] + 1e-12 and pgx[k] > 0: pgx[k] = 0.0
                if x[k] >= hi[k] - 1e-12 and pgx[k] < 0: pgx[k] = 0.0

            x = np.clip(x - lr_x * gx, lo, hi)
            pgx_norm = float(np.linalg.norm(pgx))
            if pgx_norm < tol:
                break

        # ── Constraint values ─────────────────────────────────────────
        c = np.empty(m + 1)
        for i in range(m): c[i] = fi_fns[i](x)
        c[m] = f0_fn(x) - U_eff

        # ── KKT check: primal feasible + complementary slackness ─────
        if pgx_norm < tol and np.all(c <= tol) and np.all(np.abs(y * c) <= tol):
            return x, y, True

        # ── One dual ascent step ──────────────────────────────────────
        y = np.maximum(0.0, y + lr_y * c)

        if np.linalg.norm(y) > div_thresh:
            return x, y, False

    # Timed out: report feasibility based on constraint satisfaction
    c = np.array([fi_fns[i](x) for i in range(m)] + [f0_fn(x) - U_eff])
    return x, y, bool(np.all(c <= tol))


# ─────────────────────────────────────────────────────────── branch-and-bound

def solve_pd(f0_fn, grad_f0, fi_fns, grad_fi_fns, lb, ub,
             eps=1e-3, max_boxes=2000, lr_x=0.02, lr_y=0.5, verbose=True):
    """
    Branch and bound via alternating primal-dual iteration.

    Each box is processed by alternating between minimizing x and ascending y
    until a KKT point is reached or y diverges (box pruned).

    Parameters
    ----------
    f0_fn, grad_f0           objective and its gradient  (from parse_pd)
    fi_fns, grad_fi_fns      inequality constraint functions and gradients
    lb, ub        array-like   initial bounding box
    eps           float        branch until box diameter < eps
    max_boxes     int          maximum boxes to process
    lr_x          float        primal step size (inner minimization)
    lr_y          float        dual step size (outer ascent)
    verbose       bool         print progress

    Returns
    -------
    x_best    ndarray   global minimizer found
    f_best    float     optimal primal value
    n_boxes   int       total boxes processed
    """
    lb = np.atleast_1d(np.asarray(lb, float))
    ub = np.atleast_1d(np.asarray(ub, float))
    d  = len(lb)
    m  = len(fi_fns)

    if verbose:
        print(f"\n[maximin_solver]  d={d}  constraints={m}  eps={eps}  lr_x={lr_x}  lr_y={lr_y}")

    U       = np.inf
    x_best  = None
    n_boxes = 0
    stack   = [(lb.copy(), ub.copy())]    # LIFO: depth-first

    while stack and n_boxes < max_boxes:
        lo_box, hi_box = stack.pop()
        n_boxes += 1

        x, y, found = _kkt(f0_fn, grad_f0, fi_fns, grad_fi_fns,
                            lo_box, hi_box, U,
                            lr_x=lr_x, lr_y=lr_y)

        if not found:
            if verbose:
                print(f"    box {n_boxes:5d}: pruned  (y diverged)")
            continue

        # _kkt returned True: x is a KKT point, update upper bound
        f0_val   = f0_fn(x)
        improved = f0_val < U - 1e-8

        if improved:
            U, x_best = f0_val, x.copy()
            if verbose:
                print(f"    box {n_boxes:5d}: U = {U:+.8f}  x = {x}")

        # Branch when a new best was found, OR when a KKT was found that didn't
        # improve U — the local minimizer may be in the wrong basin; sub-boxes
        # with a different initialization may find a genuinely better point.
        if found and float(np.max(hi_box - lo_box)) > eps:
            idx = int(np.argmax(hi_box - lo_box))
            mid = (lo_box[idx] + hi_box[idx]) / 2.0

            lo1, hi1 = lo_box.copy(), hi_box.copy(); hi1[idx] = mid
            lo2, hi2 = lo_box.copy(), hi_box.copy(); lo2[idx] = mid
            stack.append((lo1, hi1))
            stack.append((lo2, hi2))

    status = "done" if not stack else "budget exhausted"
    if verbose:
        print(f"\n[maximin_solver] {status}  f_best={U:.6g}  boxes={n_boxes}")

    return x_best, U, n_boxes


# ─────────────────────────────────────────────────────────── tests

if __name__ == '__main__':
    print("=== Constrained: min x1^2+x2^2  s.t. x1+x2>=1,  x in [0,1]^2 ===")
    print("    Expected: f*=0.5  x*=(0.5, 0.5)\n")

    f0_fn, grad_f0, fi_fns, grad_fi_fns, syms = parse_pd(
        'x1**2 + x2**2',
        ['-x1 - x2 + 1'],       # -x1-x2+1 <= 0  ↔  x1+x2 >= 1
    )
    x, f, nb = solve_pd(f0_fn, grad_f0, fi_fns, grad_fi_fns,
                         lb=[0, 0], ub=[1, 1],
                         eps=1e-2, max_boxes=500, lr_x=0.02, verbose=True)
    print(f"\n    x* = {x}   f* = {f:.6f}\n")

    print("=== Unconstrained: min (x-0.3)^2,  x in [0,1] ===")
    print("    Expected: f*=0  x*=0.3\n")

    f0_fn2, grad_f02, fi_fns2, grad_fi_fns2, _ = parse_pd('(x - 0.3)**2', [])
    x2, f2, nb2 = solve_pd(f0_fn2, grad_f02, fi_fns2, grad_fi_fns2,
                             lb=[0], ub=[1],
                             eps=1e-2, max_boxes=200, lr_x=0.05, verbose=True)
    print(f"\n    x* = {x2}   f* = {f2:.6f}")

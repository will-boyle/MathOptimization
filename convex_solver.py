import numpy as np

'''
Active-Set Method for General Convex Programs.

=============================================================================
PROBLEM
=============================================================================

We want to solve:

    min  f_0(x)
    s.t. f_i(x) <= 0   for i = 1, ..., m      (inequality constraints)
         Ax = b                                 (equality constraints)

=============================================================================
KEY IDEA: KKT CONDITIONS
=============================================================================

For a convex program, a point x* is optimal if and only if the KKT conditions
hold.  These are a system of equations that must all be zero at the solution:

    1) STATIONARITY:
       grad_f0(x) + sum_i lam_i * grad_f_i(x) + A^T nu = 0

       This says the gradient of the Lagrangian is zero.  In other words,
       the gradient of the objective is cancelled out by a weighted combination
       of the constraint gradients (the weights are the dual variables lam, nu).

    2) PRIMAL FEASIBILITY:
       f_i(x) <= 0  for all i,    Ax = b

    3) DUAL FEASIBILITY:
       lam_i >= 0  for all i

    4) COMPLEMENTARY SLACKNESS:
       lam_i * f_i(x) = 0  for all i

       This says: if a constraint is not tight (f_i(x) < 0), its dual must be
       zero (lam_i = 0).  Equivalently, only ACTIVE constraints (where
       f_i(x) = 0) can have a nonzero dual.

=============================================================================
THE ACTIVE SET IDEA
=============================================================================

Complementary slackness tells us that inactive constraints don't appear in
the stationarity condition.  The equality constraints Ax = b are always
enforced — they are permanent.  The question is only which INEQUALITY
constraints happen to be tight (active) at the solution.

We start with no inequalities active and add them as we discover violations.
At each outer iteration we solve the equality-constrained subproblem:

    min  f_0(x)
    s.t. f_i(x) = 0   for i in W    (active inequalities treated as equalities)
         Ax = b                       (always enforced)

If the solution satisfies:
  (a) f_i(x) <= 0 for ALL i (feasible for the full problem), AND
  (b) lam_i >= 0 for ALL i in W (dual feasible),
then ALL four KKT conditions hold and we are done.

If (a) fails: some constraint is violated — add ALL such constraints to W.
If (b) fails: some active constraint has a negative dual, which means
removing it from W would decrease the objective — remove ALL such constraints.

We repeat until W is correct.

=============================================================================
THE INNER PROBLEM: NEWTON'S METHOD
=============================================================================

At each outer iteration we solve an equality-constrained subproblem.
Its KKT conditions reduce to two things that must both be zero:

    STATIONARITY:     grad_f0(x) + Df_W(x)^T lam + A^T nu = 0
    PRIMAL FEASIBILITY:  f_i(x) = 0 for i in W,   and   Ax = b

Stack these into one residual vector r(x, lam, nu):

    r = [ grad_f0(x) + Df_W(x)^T lam + A^T nu ]   stationarity   (n eqs)
        [ f_W(x)                               ]   primal         (W eqs)
        [ Ax - b                               ]   primal         (p eqs)

The bottom two blocks are both primal feasibility — one for the active
inequalities and one for the fixed equalities.  They are split only because
they enter the matrix differently (Df_W vs A), but they are the same idea.

We want r = 0.  Since f_0 and f_i are nonlinear, r is nonlinear, so we use
Newton's method.  Linearise r around the current point and solve:

    r(y + h) ≈ r(y) + Dr(y) h = 0
    =>  Dr(y) h = -r(y)

The Jacobian Dr has two conceptual block rows — one for stationarity,
one for primal feasibility — split across three matrix rows in practice:

    Dr = d/d(x, lam, nu) of r

                   dx          dlam        dnu
               ┌──────────┬───────────┬────────┐
    stationary │ hess_lag │ Df_W(x)^T │  A^T  │  n rows
               ├──────────┼───────────┼────────┤
    primal     │ Df_W(x)  │     0     │   0   │  W rows
               │ A        │     0     │   0   │  p rows
               └──────────┴───────────┴────────┘

where hess_lag = hess_f0(x) + sum_{i in W} lam_i * hess_f_i(x)

The stationarity row differentiates grad_f0 + Df_W^T lam + A^T nu w.r.t.
each variable:
    w.r.t. x   -> hess_lag     (Hessian of the Lagrangian)
    w.r.t. lam -> Df_W(x)^T   (transpose of constraint Jacobian)
    w.r.t. nu  -> A^T

The primal rows differentiate f_W(x) and Ax-b w.r.t. each variable:
    w.r.t. x   -> Df_W(x) and A
    w.r.t. lam -> 0   (constraints don't depend on duals)
    w.r.t. nu  -> 0

This gives a (n+W+p) x (n+W+p) linear system.  After solving for
h = (dx, dlam, dnu), we take the full Newton step:

    x   <- x   + dx
    lam <- lam + dlam
    nu  <- nu  + dnu

Repeat until ||r|| < tolerance.

Note: this is identical to solver.py EXCEPT we drop the centrality block.
solver.py adds "-diag(lam)*f - 1/t" to keep iterates strictly inside the
feasible region (barrier method).  Here we have no such block — active
constraints are enforced as hard equalities instead.

=============================================================================
OUTER ALGORITHM (ACTIVE SET LOOP)
=============================================================================

    W = {}            (start with equalities Ax=b only; no inequalities active)
    x = x0 s.t. Ax=b (starting point on the equality constraint surface)

    repeat:
        solve the KKT system above with Newton's method
        evaluate f_i(x) for all i

        if any f_i(x) > 0:
            add ALL violated constraints to W
            (rank-check each Jacobian row first to avoid a singular Dr)
        else:
            if all lam_i >= 0 for i in W:
                DONE — x is optimal
            else:
                remove ALL i in W where lam_i < 0

=============================================================================
COMPARISON WITH solver.py
=============================================================================

solver.py uses the primal-dual INTERIOR POINT (barrier) method.  It always
keeps x strictly inside the feasible region (f_i(x) < 0 for all i), which
requires a feasible starting point from Phase I.  It handles ALL constraints
at once via the barrier parameter t.

convex_solver.py uses the ACTIVE SET method.  It starts from any x0, allows
infeasibility during the iteration, and explicitly tracks WHICH constraints
are active.  The two methods will find the same answer but via very different
paths.
'''


def kkt_eq(x, lam, nu, grad_f0, f_active, Df_active, A, b):
    """KKT residual for the equality-constrained subproblem (no barrier term)."""
    dual   = grad_f0(x) + Df_active(x).T @ lam + A.T @ nu
    act    = f_active(x)
    primal = A @ x - b
    return np.concatenate([dual, act, primal])


def D_kkt_eq(x, lam, hess_f0, Hf_active, Df_active, A):
    """
    Jacobian of kkt_eq with respect to (x, lam, nu).

    Args:
        x         : (n,)   primal variable
        lam       : (m_a,) dual variables for active constraints
        hess_f0   : callable x -> (n,n)    Hessian of objective
        Hf_active : callable x -> (m_a,n,n) Hessians of active constraints
        Df_active : callable x -> (m_a,n)   Jacobian of active constraints
        A         : (p,n)  equality constraint matrix

    Returns:
        J : (n+m_a+p, n+m_a+p) Newton matrix
    """
    m_a = len(lam)
    p   = A.shape[0]
    Dfx = Df_active(x)
    Hfx = Hf_active(x)
    hess_lag = hess_f0(x) + (np.tensordot(lam, Hfx, axes=([0], [0])) if m_a > 0 else 0)

    row_dual = np.block([hess_lag, Dfx.T,                  A.T               ])
    row_act  = np.block([Dfx,      np.zeros((m_a, m_a)),   np.zeros((m_a, p))])
    row_eq   = np.block([A,        np.zeros((p,  m_a)),    np.zeros((p,  p)) ])
    return np.vstack([row_dual, row_act, row_eq])


def _solve_inner(x0, lam0, nu0, grad_f0, hess_f0, f_active, Df_active, Hf_active, A, b,
                 eps=1e-8):
    """Newton's method to solve the equality-constrained subproblem."""
    x, lam, nu = x0.copy(), lam0.copy(), nu0.copy()
    n, m_a = len(x), len(lam0)

    for _ in range(500):
        r = kkt_eq(x, lam, nu, grad_f0, f_active, Df_active, A, b)
        if np.linalg.norm(r) < eps:
            break

        J = D_kkt_eq(x, lam, hess_f0, Hf_active, Df_active, A)
        h = np.linalg.solve(J, -r)
        x   = x   + h[:n]
        lam = lam + h[n:n + m_a]
        nu  = nu  + h[n + m_a:]

    return x, lam, nu


def solve_convex(x0, grad_f0, hess_f0, f, Df, Hf, A, b, eps=1e-6, max_iter=100):
    """
    Active-set solver for general convex programs.

    Args:
        x0      : (n,) starting point — does not need to be feasible
        grad_f0 : callable x -> (n,)     gradient of objective
        hess_f0 : callable x -> (n,n)    Hessian of objective
        f       : callable x -> (m,)     inequality constraint values
        Df      : callable x -> (m,n)    Jacobian of inequalities
        Hf      : callable x -> (m,n,n)  Hessians of inequalities
        A       : (p,n) equality constraint matrix
        b       : (p,)  equality constraint RHS
        eps     : tolerance for feasibility / dual sign checks
        max_iter: maximum outer (active-set) iterations

    Returns:
        x   : (m,) optimal primal variable
        lam : (m,) inequality dual variables (shadow prices); 0 for inactive
        nu  : (p,) equality dual variables
    """
    n      = len(x0)
    m      = len(f(x0))
    p      = A.shape[0]
    active = []
    x      = x0.copy()
    nu     = np.zeros(p)

    for _ in range(max_iter):
        # Snapshot so that lambda default-captures the current active list
        act = list(active)

        if act:
            f_act  = lambda x, a=act: f(x)[a]
            Df_act = lambda x, a=act: Df(x)[a]
            Hf_act = lambda x, a=act: Hf(x)[a]
            lam    = np.zeros(len(act))
        else:
            f_act  = lambda x: np.zeros(0)
            Df_act = lambda x: np.zeros((0, n))
            Hf_act = lambda x: np.zeros((0, n, n))
            lam    = np.zeros(0)

        # Solve the equality-constrained subproblem
        x, lam, nu = _solve_inner(x, lam, nu, grad_f0, hess_f0, f_act, Df_act, Hf_act, A, b)

        # Check feasibility of all inequality constraints
        fx       = f(x)
        violated = np.where(fx > eps)[0]

        if len(violated) == 0:
            # Feasible — check active dual variables
            neg_duals = [i for i, li in enumerate(lam) if li < -eps]
            if not neg_duals:
                # All KKT conditions satisfied — optimal
                lam_full = np.zeros(m)
                if act:
                    lam_full[act] = lam
                return x, lam_full, nu
            else:
                # Remove ALL constraints with negative duals
                keep   = [i for i in range(len(act)) if i not in neg_duals]
                active = [act[i] for i in keep]
        else:
            # Add ALL violated constraints (rank-check Jacobian rows to avoid
            # singularity — same check as solver_lpqp.py but using Df(x) instead
            # of C since constraints are nonlinear)
            new_sorted = sorted(list(violated), key=lambda j: float(fx[j]), reverse=True)
            Dfx = Df(x)
            for i in new_sorted:
                if i not in active:
                    rows = np.vstack([A, Dfx[active + [i]]]) if (active or p > 0) else Dfx[[i]]
                    prev = rows[:-1]
                    prev_rank = np.linalg.matrix_rank(prev) if prev.shape[0] > 0 else 0
                    if np.linalg.matrix_rank(rows) > prev_rank:
                        active.append(i)

    raise ValueError(f"solve_convex: active-set method did not converge in {max_iter} iterations")

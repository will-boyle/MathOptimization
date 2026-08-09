import numpy as np
import sympy as sp

'''
Active Set Method for Quadratic and Linear Programs.

Since all Linear Programs are also Quadratic programs, 
just a single QP algo can be used to solve either.

Problem:
    min  (1/2) x^T P x + q^T x
    s.t. Ax  = b     (equalities)
         Cx <= d     (inequalities)

For LPs, P = 0 and the objective reduces to q^T x.

Algorithm:
    1. Solve the KKT system with only the equality constraints active.
    2. If all inequalities are satisfied -> optimal, done.
    3. If any inequalities are violated, add ALL of them to the active set
       at once, sorted by violation magnitude (largest first). Any constraint
       whose row is linearly dependent on those already added is skipped to
       prevent the KKT matrix from becoming singular.
    4. Re-solve. If feasible but some active dual lambda_i < 0, that
       constraint is not genuinely needed at this point -> remove ALL such
       constraints from the active set at once and re-solve.
    5. Repeat until feasible with all active duals >= 0.

Since the objective is quadratic (or linear) and constraints are linear, the
KKT conditions are themselves linear. Each subproblem is therefore a single
direct solve of the saddle-point system:

    [P    A_aug^T] [x]   [-q    ]
    [A_aug  0   ] [v] = [b_aug ]

where A_aug stacks the permanent equality rows and the currently active
inequality rows, and v = [nu (equality duals), lambda (inequality duals)].

At the optimal solution:
    - All active inequality duals lambda_i >= 0  (dual feasibility)
    - All inactive inequality constraints satisfied strictly  Cx < d
    - The active duals are the shadow prices: lambda_i measures how much
      the objective would improve per unit relaxation of constraint i.

LP extension:
    LPs are handled as QPs with P = 0. A small regularization epsilon*I is
    added to P inside _kkt_solve to prevent the KKT matrix from being singular
    when fewer active constraints than variables are present. This is equivalent
    to solving min (epsilon/2)||x||^2 + q^T x, which converges to the LP
    solution as epsilon -> 0.
'''


def parse_qp(f0_str, ineq_strs, eq_strs):
    """
    Parse QP strings into matrix form.

    Returns:
        P    : (n,n)  Hessian of objective
        q    : (n,)   linear term  (grad f0 = Px + q)
        A    : (p,n)  equality constraint matrix
        b    : (p,)   equality RHS
        C    : (m,n)  inequality constraint matrix   (Cx <= d)
        d    : (m,)   inequality RHS
        syms : ordered list of sympy symbols
    """
    f0_expr = sp.sympify(f0_str)

    def _parse_eq(s):
        if '==' in s:
            lhs, rhs = s.split('==', 1)
        elif '=' in s:
            lhs, rhs = s.split('=', 1)
        else:
            return sp.sympify(s)
        return sp.sympify(lhs.strip()) - sp.sympify(rhs.strip())

    ineq_exprs = [sp.sympify(s) for s in ineq_strs]
    eq_exprs   = [_parse_eq(s)  for s in eq_strs]

    all_exprs = [f0_expr] + ineq_exprs + eq_exprs
    syms = sorted(
        set().union(*[e.free_symbols for e in all_exprs]),
        key=lambda s: s.name
    )
    n = len(syms)

    # P is the Hessian of f0 (constant for a QP)
    P = np.array(
        [[float(sp.diff(f0_expr, si, sj)) for sj in syms] for si in syms]
    )

    # q = grad(f0)|_{x=0}  since grad f0 = Px + q  =>  q = grad|_{x=0}
    # c = f0(0)            constant term (dropped by P and q)
    x_zero = {s: 0 for s in syms}
    q = np.array([float(sp.diff(f0_expr, s).subs(x_zero)) for s in syms])
    c = float(f0_expr.subs(x_zero))

    if eq_exprs:
        A_sym, b_sym = sp.linear_eq_to_matrix(eq_exprs, syms)
        A = np.array(A_sym.tolist(), dtype=float)
        b = np.array(b_sym.tolist(), dtype=float).flatten()
    else:
        A = np.zeros((0, n))
        b = np.zeros(0)

    if ineq_exprs:
        C_sym, d_sym = sp.linear_eq_to_matrix(ineq_exprs, syms)
        C = np.array(C_sym.tolist(), dtype=float)
        d = np.array(d_sym.tolist(), dtype=float).flatten()
    else:
        C = np.zeros((0, n))
        d = np.zeros(0)

    return P, q, c, A, b, C, d, syms


def _kkt_solve(P, q, A_aug, b_aug):
    """
    Solve the KKT system for an equality-constrained QP.

        [P    A_aug^T] [x]   [-q    ]
        [A_aug  0   ] [v] = [b_aug ]

    When P=0 (LP), the top-left block is singular unless we have exactly n
    active constraints. A small regularization eps*I is added to prevent this
    without meaningfully changing the solution.
    """
    n = P.shape[0]
    p = A_aug.shape[0]
    P_reg = P if not np.allclose(P, 0) else 1e-10 * np.eye(n)
    K   = np.block([[P_reg, A_aug.T], [A_aug, np.zeros((p, p))]])
    rhs = np.concatenate([-q, b_aug])
    return np.linalg.solve(K, rhs)


def solve_qp(P, q, A, b, C, d, tol=1e-8, max_iter=100):
    """
    Solve a QP via the active set method.

    Returns:
        x      : (n,)  optimal primal variable
        nu     : (p,)  equality constraint duals
        lam    : (m,)  inequality duals (0 for inactive constraints)
        active : list of active inequality indices at the solution
    """
    n    = P.shape[0]
    p_eq = A.shape[0]
    m    = C.shape[0]
    active = []

    for _ in range(max_iter):
        if active:
            A_aug = np.vstack([A, C[active]])
            b_aug = np.concatenate([b, d[active]])
        else:
            A_aug = A
            b_aug = b

        sol = _kkt_solve(P, q, A_aug, b_aug)
        x   = sol[:n]
        v   = sol[n:]

        # Check all inequality constraints
        residuals = C @ x - d if m > 0 else np.zeros(0)
        violated  = np.where(residuals > tol)[0]

        if len(violated) == 0:
            # Feasible — check active duals
            lam_active = v[p_eq:]
            inactive   = [i for i, li in enumerate(lam_active) if li < -tol]

            if len(inactive) == 0:
                # Optimal: all active duals non-negative
                lam = np.zeros(m)
                if active:
                    lam[active] = lam_active
                return x, v[:p_eq], lam, list(active)
            else:
                # Remove ALL constraints with negative duals at once
                keep   = [i for i in range(len(active)) if i not in inactive]
                active = [active[i] for i in keep]
        else:
            # Add ALL violated constraints at once, sorted by violation magnitude.
            # Skip any constraint whose row is linearly dependent on the rows
            # already in the active set — adding it would make the KKT matrix singular.
            new_sorted = sorted(
                [int(i) for i in violated if int(i) not in active],
                key=lambda j: float(residuals[j]),
                reverse=True,
            )
            for i in new_sorted:
                A_cand = np.vstack([A, C[active + [i]]])
                A_prev = A_cand[:-1]
                prev_rank = np.linalg.matrix_rank(A_prev) if A_prev.shape[0] > 0 else 0
                if np.linalg.matrix_rank(A_cand) > prev_rank:
                    active.append(i)

    raise ValueError(f"Active set method did not converge in {max_iter} iterations")

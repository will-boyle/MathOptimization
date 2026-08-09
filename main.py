"""
Convex Optimization Solver CLI
"""

import numpy as np
import sympy as sp
from solver  import parse, phase1, solve
from convex_solver import solve_convex
from solver3 import solve3
from dc_solver import parse_dc, solve_dc
from c2_solver import parse_abb, solve_abb
from lipschitz_solver import parse_lipschitz, solve_lipschitz, estimate_L
from solver_lpqp import parse_qp, solve_qp
from maximin_solver import parse_pd, solve_pd
import matplotlib.pyplot as plt
from neural_net_solver import train, predict_nn

if __name__ == "__main__":
    print("    Welcome to the convex optimization solver!")

    problem_type = input("""    Problem type:
        1. Convex program      (minimize f0(x)         s.t. fi(x) <= 0, Ax = b)
        2. D.C. program        (minimize f0(x)-g0(x)   s.t. fi(x)-gi(x) <= 0)
        3. General nonconvex   (minimize f0(x)         s.t. fi(x) <= 0, lb <= x <= ub)  [requires C2]
        4. Lipschitz           (minimize f0(x)         s.t. fi(x) <= 0, lb <= x <= ub)  [requires Lipschitz only]
        5. Neural network      (fit y = f(x) from training data by gradient descent)
        6. LP / QP             (minimize quadratic or linear f0  s.t. linear Cx <= d, Ax = b)
        7. Dual B&B            (minimize f0(x)         s.t. fi(x) <= 0, lb <= x <= ub)  [alternating primal-dual]
        choice (1-7, default 1)... """).strip()

    # ─────────────────────────────── D.C. BRANCH ───────────────────────────
    if problem_type == "2":
        print("""
    D.C. (Difference of Convex) Programming
    Objective:   minimize f0(x) - g0(x)
    Constraints: fi(x) - gi(x) <= 0  for each i
    Both f0, g0, fi, gi must be convex.
""")
        f0_str = input("    Convex part of objective f0(x)...  example: x1**2 + x2**2\n"
                       "        f0 = ").strip()
        g0_str = input("    Concave part of objective g0(x) (subtracted)...  example: x1\n"
                       "        g0 = ").strip()

        print("""
    Enter D.C. inequality constraints  fi(x) - gi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        dc_ineqs = []
        while True:
            fi_str = input("        fi (convex, or Enter to finish): ").strip()
            if fi_str == "":
                break
            gi_str = input("        gi (concave):                    ").strip()
            dc_ineqs.append((fi_str, gi_str))

        print("""
    Enter bounding box for the variables.
    The initial simplex will cover this box — make it tight for faster convergence.""")
        lb_vals, ub_vals = [], []
        # Infer number of variables by parsing first
        all_strs = [f0_str, g0_str] + [s for pair in dc_ineqs for s in pair]
        all_syms = sorted(
            set().union(*[sp.sympify(s).free_symbols for s in all_strs]),
            key=lambda s: s.name,
        )
        print(f"    Variables detected: {[str(s) for s in all_syms]}")
        for s in all_syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        tol_str = input("    Convergence tolerance (default 1e-4): ").strip()
        tol     = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 500): ").strip()
        max_iter = int(iter_str) if iter_str else 500

        print("\n    Parsing D.C. problem...")
        (f0, g0, grad_f0, grad_g0,
         fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
         syms) = parse_dc(f0_str, g0_str, dc_ineqs)

        print("    Running D.C. branch-and-bound solver...\n")
        x_best, U = solve_dc(
            f0, g0, grad_f0, grad_g0,
            fi_fns, gi_fns, grad_fi_fns, grad_gi_fns,
            lb_vals, ub_vals,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.6f}")
            print(f"\n    Objective f0 - g0 = {U:.6f}")

    # ─────────────────────────────── αBB BRANCH ───────────────────────────
    elif problem_type == "3":
        print("""
    αBB Global Optimizer
    Objective:   minimize f0(x)
    Constraints: fi(x) <= 0  for each i  (optional)
    A bounding box [lb, ub] is required; all global optima must lie inside it.
""")
        f0_str = input("    Objective f0(x)...  example: x1**4 - 4*x1**2\n"
                       "        f0 = ").strip()

        print("""
    Enter inequality constraints  fi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        ineqs = []
        while True:
            fi_str = input("        fi (or Enter to finish): ").strip()
            if fi_str == "":
                break
            ineqs.append(fi_str)

        print("\n    Parsing problem...")
        f0_fn, grad_f0_fn, hess_f0_fn, fi_fns, grad_fi_fns, hess_fi_fns, syms = parse_abb(f0_str, ineqs)
        print(f"    Variables detected: {[str(s) for s in syms]}")

        print("""
    Enter bounding box [lb, ub] for each variable.
    All global minima must lie inside this box.""")
        lb_vals, ub_vals = [], []
        for s in syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        tol_str  = input("    Convergence tolerance (default 1e-4): ").strip()
        tol      = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 1000): ").strip()
        max_iter = int(iter_str) if iter_str else 1000

        print("\n    Running αBB solver...\n")
        x_best, U = solve_abb(
            f0_fn, grad_f0_fn, hess_f0_fn,
            fi_fns, grad_fi_fns, hess_fi_fns,
            lb_vals, ub_vals,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.8f}")
            print(f"\n    Objective f0 = {U:.8f}")

    # ─────────────────────────────── LIPSCHITZ BRANCH ─────────────────────
    elif problem_type == "4":
        print("""
    Lipschitz Global Optimizer
    Objective:   minimize f0(x)              (must be Lipschitz continuous)
    Constraints: fi(x) <= 0  for each i     (optional, also Lipschitz)
    A bounding box [lb, ub] is required.
    Note: Lipschitz does NOT mean all continuous functions qualify.
          f(x) = sqrt(x) near 0 is continuous but not Lipschitz.
""")
        f0_str = input("    Objective f0(x)...  example: Abs(x1) + x2**2\n"
                       "        f0 = ").strip()

        print("""
    Enter inequality constraints  fi(x) <= 0.
    Press Enter with an empty fi to finish.""")
        ineqs = []
        while True:
            fi_str = input("        fi (or Enter to finish): ").strip()
            if fi_str == "":
                break
            ineqs.append(fi_str)

        print("\n    Parsing problem...")
        f0_fn, grad_f0_fn, fi_fns, grad_fi_fns, syms = parse_lipschitz(f0_str, ineqs)
        print(f"    Variables detected: {[str(s) for s in syms]}")

        print("""
    Enter bounding box [lb, ub] for each variable.""")
        lb_vals, ub_vals = [], []
        for s in syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        lb_arr = np.array(lb_vals)
        ub_arr = np.array(ub_vals)

        L_str = input("    Lipschitz constant L (or Enter to estimate from gradient): ").strip()
        if L_str:
            L_f = float(L_str)
            L_fi = None
        else:
            print("    Estimating Lipschitz constants from gradient at box corners...")
            L_f  = estimate_L(grad_f0_fn, lb_arr, ub_arr)
            L_fi = [estimate_L(gfi, lb_arr, ub_arr) for gfi in grad_fi_fns] if grad_fi_fns else None
            print(f"    L_f = {L_f:.4f}" +
                  (f"  L_fi = {[f'{l:.4f}' for l in L_fi]}" if L_fi else ""))

        tol_str  = input("    Convergence tolerance (default 1e-4): ").strip()
        tol      = float(tol_str) if tol_str else 1e-4
        iter_str = input("    Max B&B iterations (default 50000): ").strip()
        max_iter = int(iter_str) if iter_str else 50000

        print("\n    Running Lipschitz B&B solver...\n")
        x_best, U = solve_lipschitz(
            f0_fn, fi_fns, L_f, L_fi,
            lb=lb_arr, ub=ub_arr,
            tol=tol, max_iter=max_iter, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.8f}")
            print(f"\n    Objective f0 = {U:.8f}")

    # ──────────────────────────── NEURAL NET BRANCH ───────────────────────
    elif problem_type == "5":
        print("""
    Neural Network Regression
    Fits a one-hidden-layer ReLU network to training data generated
    from a user-supplied function evaluated at given x values.

    Network:  y_hat(x) = sum_j  v_j * ReLU(w_j * x + b_j)
    Loss:     L = (1/n) * sum_i ( y_hat(x_i) - y_i )^2

    Example:
        function: x**2
        x values: 0 0.5 1 1.5 2
""")
        fn_str = input("    function f(x) = ").strip()
        x_raw  = input("    x values (space-separated): ").strip()

        x_sym = sp.Symbol('x')
        try:
            fn_expr = sp.sympify(fn_str)
            fn      = sp.lambdify(x_sym, fn_expr, 'numpy')
        except Exception as e:
            print(f"    Error parsing function: {e}")
            exit(1)

        x_vals = [float(v) for v in x_raw.split()]
        if len(x_vals) < 2:
            print("    Error: need at least 2 x values.")
            exit(1)

        try:
            y_vals = [float(fn(x)) for x in x_vals]
        except Exception as e:
            print(f"    Error evaluating function: {e}")
            exit(1)

        print(f"\n    {len(x_vals)} training points generated from f(x) = {fn_str}:")

        print("\n    Training...")
        W, b, v, x_shift, x_scale, y_shift, y_scale, losses = train(
            x_vals, y_vals, n_neurons=50, alpha=0.1, n_steps=10000, seed=0, verbose=True)

        print()
        print("    Predictions vs. targets:")
        print(f"    {'x':>10}  {'y_hat':>10}  {'y':>10}  {'error':>10}")
        print(f"    {'----------':>10}  {'----------':>10}  {'----------':>10}  {'----------':>10}")
        for x, y in zip(x_vals, y_vals):
            yhat = predict_nn(x, W, b, v, x_shift, x_scale, y_shift, y_scale)
            print(f"    {x:>10.4f}  {yhat:>10.5f}  {y:>10.5f}  {yhat - y:>+10.5f}")

        print(f"\n    Final loss: {losses[-1][1]:.8f}")

        # Plot
        x_min, x_max = min(x_vals), max(x_vals)
        x_plot = [x_min + (x_max - x_min) * i / 299 for i in range(300)]
        y_fit  = [predict_nn(x, W, b, v, x_shift, x_scale, y_shift, y_scale) for x in x_plot]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.plot(x_plot, y_fit, color='#2A5C8A', linewidth=2.0, label='neural network')
        ax.scatter(x_vals, y_vals, color='#C0392B', zorder=5, s=50, label='training data')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Neural network fit: f(x) = {fn_str}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        steps_log    = [s for s, _ in losses]
        lossvals_log = [l for _, l in losses]
        ax.semilogy(steps_log, lossvals_log, color='#2A5C8A', linewidth=2.0)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training loss')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ─────────────────────────────── LP / QP BRANCH ──────────────────────
    elif problem_type == "6":
        print("""
    LP / QP Solver
    Objective:   minimize  (1/2) x^T P x + q^T x    (P=0 for LP, P>=0 for QP)
    Constraints: Cx <= d   (linear inequalities)
                 Ax = b    (linear equalities)
""")
        f0_str = input("    Objective f0(x)...  example: x1**2 + x2**2  or  2*x1 + x2\n"
                       "        f0 = ").strip()

        print("""    Enter linear inequality constraints  fi(x) <= 0.
    Press Enter with an empty line to finish.""")
        inequalities = []
        while True:
            s = input("        inequality (or Enter to finish): ").strip()
            if s == "":
                break
            inequalities.append(s)

        print("""    Enter linear equality constraints.
    Press Enter with an empty line to finish.""")
        equalities = []
        while True:
            s = input("        equality (or Enter to finish): ").strip()
            if s == "":
                break
            equalities.append(s)

        print("\n    Parsing problem...")
        P, q, c, A, b, C, d, syms = parse_qp(f0_str, inequalities, equalities)
        print("    Variables detected:", [str(s) for s in syms])

        is_lp = np.allclose(P, 0)
        print(f"    Problem detected as: {'LP (linear objective)' if is_lp else 'QP (quadratic objective)'}")

        print("    Solving...")
        x, nu, lam, active = solve_qp(P, q, A, b, C, d)

        obj_val = 0.5 * x @ P @ x + q @ x + c

        print("\n    Solution:")
        for i, s in enumerate(syms):
            print(f"        {s} = {x[i]:.6f}")
        print(f"\n    Objective value: {obj_val:.6f}")

        if inequalities:
            print("\n    Shadow prices (inequality duals):")
            for i, ineq in enumerate(inequalities):
                status = "active" if i in active else "inactive"
                print(f"        {ineq} <= 0 :  lambda = {lam[i]:.6f}  ({status})")

        if equalities:
            print("\n    Shadow prices (equality duals):")
            for i, eq in enumerate(equalities):
                print(f"        {eq} :  nu = {nu[i]:.6f}")

    # ─────────────────────────────── DUAL B&B BRANCH ──────────────────────
    elif problem_type == "7":
        print("""
    Dual B&B Solver
    Objective:   minimize f0(x)
    Constraints: fi(x) <= 0  for each i  (optional)
    A bounding box [lb, ub] is required.
""")
        f0_str = input("    Objective f0(x)...  example: x1**2 + x2**2\n"
                       "        f0 = ").strip()

        print("""    Enter inequality constraints  fi(x) <= 0.
    Press Enter with an empty line to finish.""")
        ineqs = []
        while True:
            fi_str = input("        fi (or Enter to finish): ").strip()
            if fi_str == "":
                break
            ineqs.append(fi_str)

        print("\n    Parsing problem...")
        f0_fn, grad_f0, fi_fns, grad_fi_fns, syms = parse_pd(f0_str, ineqs)
        print(f"    Variables detected: {[str(s) for s in syms]}")

        print("""
    Enter bounding box [lb, ub] for each variable.""")
        lb_vals, ub_vals = [], []
        for s in syms:
            lb_val = float(input(f"        lower bound for {s}: ").strip())
            ub_val = float(input(f"        upper bound for {s}: ").strip())
            lb_vals.append(lb_val)
            ub_vals.append(ub_val)

        eps_str  = input("    Box diameter tolerance (default 1e-3): ").strip()
        eps      = float(eps_str) if eps_str else 1e-3
        box_str  = input("    Max boxes (default 2000): ").strip()
        max_boxes = int(box_str) if box_str else 2000
        lrx_str  = input("    Primal step size lr_x (default 0.02): ").strip()
        lr_x     = float(lrx_str) if lrx_str else 0.02
        lry_str  = input("    Dual step size lr_y (default 0.5): ").strip()
        lr_y     = float(lry_str) if lry_str else 0.5

        print("\n    Running Dual B&B solver...\n")
        x_best, f_best, n_boxes = solve_pd(
            f0_fn, grad_f0, fi_fns, grad_fi_fns,
            lb=lb_vals, ub=ub_vals,
            eps=eps, max_boxes=max_boxes, lr_x=lr_x, lr_y=lr_y, verbose=True,
        )

        print("\n    Solution:")
        if x_best is None:
            print("    No feasible point found in the given bounding box.")
        else:
            for i, s in enumerate(syms):
                print(f"        {s} = {x_best[i]:.8f}")
            print(f"\n    Objective f0 = {f_best:.8f}")

    # ─────────────────────────────── CONVEX BRANCH ─────────────────────────
    else:
        f_0str = input("""    Please enter the objective function...
    example... x1**2 + x2**2
        objective function...""")

        print("""    Please enter inequalities, when done press enter...
    the convention is that the RHS is zero and all inequalities are <=.
    example... x1 + 1""")
        inequalities = []
        while True:
            f_istr = input("        inequality...")
            if f_istr == "":
                break
            inequalities.append(f_istr)

        print("""    Please enter equalities, when done press enter...
    example... x1 - x2 == 0""")
        equalities = []
        while True:
            h_istr = input("        equality...")
            if h_istr == "":
                break
            equalities.append(h_istr)

        print("\n    Parsing problem...")
        grad_f0, hess_f0, f, Df, Hf, A, b, syms = parse(f_0str, inequalities, equalities)

        print("    Variables detected:", [str(s) for s in syms])

        n = len(syms)
        if A.shape[0] > 0:
            x_init = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            x_init = np.zeros(n)

        solver_choice = input("""    Choose solver:
        1. Interior Point      (solver.py  — barrier method, Phase I required)
        2. Active Set          (convex_solver.py — Newton on KKT, tracks active constraints)
        3. Feasible Direction  (solver3.py — LP subproblems, first-order only)
        choice (1, 2, or 3, default 1)... """).strip()

        f0_fn = sp.lambdify(syms, sp.sympify(f_0str), 'numpy')
        lam   = None

        if solver_choice == "2":
            print("    Running Active Set solver...")
            x, lam, nu = solve_convex(x_init, grad_f0=grad_f0, hess_f0=hess_f0,
                                      f=f, Df=Df, Hf=Hf, A=A, b=b)
        elif solver_choice == "3":
            print("    Running Feasible Direction solver...")
            try:
                x = solve3(x_init, lambda v: float(f0_fn(*v)), grad_f0, f, Df, A, b)
            except ValueError as e:
                print(f"\n    Error: {e}")
                exit(1)
        else:
            print("    Running Phase I to find a feasible starting point...")
            try:
                x0 = phase1(f, Df, Hf, A, b, x_init)
            except ValueError as e:
                print(f"\n    Error: {e}")
                exit(1)
            print("    Feasible point found. Running solver...")
            x, lam, nu = solve(x0, mu=10.0, grad_f0=grad_f0, hess_f0=hess_f0,
                               f=f, Df=Df, Hf=Hf, A=A, b=b)

        obj_val = float(f0_fn(*x))

        print("\n    Solution:")
        for i, s in enumerate(syms):
            print(f"        {s} = {x[i]:.6f}")
        print(f"\n    Objective value: {obj_val:.6f}")
        if lam is not None and inequalities:
            print("\n    Shadow prices (inequality dual variables):")
            for i, ineq in enumerate(inequalities):
                print(f"        {ineq} <= 0 :  lambda = {lam[i]:.6f}")

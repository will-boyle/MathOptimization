# A Target-Feasibility Branch-and-Prune Algorithm for Nonconvex Constrained Optimization

**William Boyle**

*Preprint — 2026*

---

## Abstract

We present a branch-and-prune algorithm for nonconvex constrained optimization in which each box is treated as a target-level feasibility problem and solved through an extended Lagrangian saddle-point formulation. The current incumbent objective value $U$ is incorporated directly as an additional inequality constraint $f_0(x) - U \leq 0$, with its own nonnegative multiplier $y_U$. The resulting extended Lagrangian is

$$
L(x,y) = (1+y_U)f_0(x) + \sum_{i=1}^m y_i f_i(x) - y_U U.
$$

For a fixed dual vector, the method minimizes the Lagrangian over the current box; the resulting primal point is then used to update the dual variables by ascent. A box is retained when the oracle returns a feasible point and is pruned when the associated target-feasibility problem is certified infeasible.

The central theoretical observation is that when the original constraints are feasible but the incumbent target is unattainable, compactness and continuity imply a uniform positive objective gap on the feasible portion of the box. Consequently, under the idealized dual-ascent iteration with positive step sizes, the specific multiplier $y_U$ must diverge. This provides a mathematical basis for the pruning mechanism. Separately, continuity and compactness imply that exhaustive subdivision of the box containing a global minimizer produces feasible points whose objective values converge to the global optimum. These two results together establish global convergence of the idealized branch-and-prune procedure under the stated assumptions. The numerical implementation approximates the per-box minimization and detects divergence using finite tolerances and thresholds.

Numerical experiments on a collection of convex and nonconvex test problems demonstrate that the implementation successfully recovers the known global optima, including problems with multiple local minima, active and inactive constraints, multiple global minimizers, and disconnected feasible regions.

---

# 1. Introduction

Branch-and-bound (B&B) is a standard framework for global optimization of nonconvex programs. Classical B&B algorithms typically associate each subproblem with a lower bound and prune a box when its lower bound cannot improve the best known primal value $U$. Lower bounds are often obtained through convex relaxations, interval arithmetic, or other global bounding procedures.

This paper proposes a different mechanism. Instead of computing an independent lower bound for each box, the current incumbent value $U$ is incorporated directly into the optimization problem as an additional target constraint,

$$
f_0(x) \leq U.
$$

The resulting feasibility problem asks whether the current box contains a feasible point whose objective is at least as good as the incumbent. Its Lagrangian contains a multiplier $y_U$ associated with this target constraint. The dual variables are then updated according to constraint violations.

The key observation is simple but useful. If the original constraints remain feasible in a box but no feasible point can satisfy the incumbent target, then continuity and compactness imply that the objective exceeds the target by a uniformly positive amount over the feasible portion of the box. Consequently, repeated dual ascent drives the multiplier $y_U$ associated with the incumbent constraint without bound. Thus target infeasibility can be detected through $y_U$ divergence rather than through an independently computed lower bound.

The global search therefore has two fundamental operations:

1. **branch:** when a candidate KKT point is found, subdivide the current box;
2. **prune:** when the target-feasibility problem is infeasible, discard the box.

The theoretical analysis separates these two mechanisms. First, we establish a one-sided divergence result for target infeasibility conditional on original feasibility. Second, we show that, under the exact-oracle assumptions stated below, continuity guarantees convergence of the incumbent objective value under repeated subdivision.

The numerical implementation uses projected gradient descent for the primal minimization and finite thresholds to detect practical divergence. The mathematical results concern the idealized procedure; the numerical method is then evaluated empirically.

---

# 2. Problem Formulation

Consider

$$
\begin{aligned}
\min_{x\in B}\quad &f_0(x)\\
\text{s.t.}\quad &f_i(x)\leq0,\qquad i=1,\ldots,m,
\end{aligned}
\tag{P}
$$

where

$$
B = [lb,ub] \subset \mathbb{R}^n
$$

is a compact axis-aligned box and $(f_0,f_1,\ldots,f_m)$ are continuous. We assume that the problem has at least one feasible point and that its minimum is attained. Denote the global optimal value by

$$
f^* = \min\{f_0(x) : x \in B,\ f_i(x) \leq 0\}.
$$

Let $U$ denote the best feasible objective value currently known. The target-feasibility problem associated with a box $B' \subseteq B$ is

$$
\begin{aligned}
\text{find } x \in B' \quad \text{s.t.} \quad
&f_i(x) \leq 0, \qquad i = 1, \ldots, m, \\
&f_0(x) - U \leq 0.
\end{aligned}
\tag{T}
$$

Thus a box can contain an improvement over the incumbent only if (T) is feasible.

---

# 3. The Extended Lagrangian

Introduce a nonnegative multiplier $y_U$ for the target constraint $f_0(x) - U \leq 0$. The extended Lagrangian is

$$
L(x,y) = f_0(x) + \sum_{i=1}^m y_i f_i(x) + y_U(f_0(x) - U),
$$

or equivalently,

$$
L(x,y) = (1+y_U)f_0(x) + \sum_{i=1}^m y_i f_i(x) - y_U U,
\tag{1}
$$

where

$$
y = (y_1, \ldots, y_m, y_U) \geq 0.
$$

The associated saddle-point formulation is

$$
\min_{x\in B'} \max_{y\geq0} L(x,y).
$$

The derivatives with respect to the dual variables are exactly the constraint residuals:

$$
\frac{\partial L}{\partial y_i} = f_i(x),
\qquad\text{and}\qquad
\frac{\partial L}{\partial y_U} = f_0(x) - U.
$$

Thus the dual update directly responds to constraint violation.

For fixed $y$, the primal step seeks a minimizer of $L(\cdot,y)$ over the current box. Once this minimization has been performed, the dual variables are updated by projected ascent:

$$
y_j^{k+1} = \max\{0,\ y_j^k + \alpha_k c_j(x^k)\},
\tag{3}
$$

where

$$
c(x) = \bigl(f_1(x),\ldots,f_m(x),\, f_0(x)-U\bigr).
$$

The mathematical analysis below concerns this idealized alternating procedure. The numerical implementation uses projected gradient descent to approximate the primal minimization.

---

# 4. The Branch-and-Prune Algorithm

## 4.1 Per-Box Procedure

For each box $B'$ and finite incumbent $U$, the algorithm repeatedly performs two operations.

**Step 1: Primal minimization.** For fixed $y$, minimize $L(x,y)$ over $x \in B'$.

**Step 2: Dual ascent.** Evaluate the constraint residual vector $c(x)$ and update

$$
y^{k+1} = \left[y^k + \alpha_k c(x^k)\right]_+.
$$

At the abstract level, the per-box procedure is an exact target-feasibility oracle: it either returns a point satisfying the target constraints or certifies that the target problem is infeasible. The extended-Lagrangian iteration studied below is a concrete numerical realization of this oracle, but the present theory establishes only a one-sided divergence result for that realization. In particular, Theorem 1 proves divergence of the incumbent multiplier in the case where the original constraints are feasible but the incumbent target is unattainable. It does not prove that every infeasible box is detected by this multiplier, nor that finite-threshold divergence tests can never falsely prune a feasible target box.

The numerical implementation approximates the oracle using KKT tolerances, iteration limits, and divergence thresholds.

## 4.2 Branching

Whenever a KKT point is obtained, the current point is a feasible primal point. If its objective improves the incumbent, the incumbent is updated.

The current box is then subdivided along its widest dimension, provided its diameter exceeds the prescribed tolerance.

Branching on every retained KKT point is intentional. A local stationary point need not be globally optimal in a nonconvex problem, and different sub-boxes can expose different basins of attraction.

## 4.3 Pruning

A box is pruned when its target-feasibility problem is infeasible.

Importantly, the mathematical pruning rule is **infeasibility of the target problem**, not merely the numerical observation that a finite computer representation has exceeded a threshold. The latter is only the practical implementation of the former.

---

# 5. Theoretical Analysis

## 5.1 KKT Characterization

Suppose the per-box procedure returns a point $x$ together with multipliers $y$ satisfying

$$
\nabla_x L(x, y) = 0
$$

subject to the appropriate first-order conditions at the box boundaries,

$$
c_j(x) \leq 0, \qquad y_j \geq 0, \qquad y_j c_j(x) = 0.
$$

Then $(x, y)$ satisfies the KKT conditions for the target-feasibility problem.

In the convex case, under the usual constraint qualification, these conditions are sufficient for global optimality. In the nonconvex case they identify a stationary/KKT point but do not by themselves establish global optimality.

This distinction is important: the branch-and-prune mechanism is what permits the method to address the nonconvex case.

## 5.2 Dual Divergence of the Incumbent Multiplier

The pruning mechanism uses the multiplier $y_U$ associated specifically with the incumbent constraint

$$
f_0(x) - U \leq 0.
$$

The relevant case is one in which the original constraints remain feasible on the box, but no feasible point can attain the incumbent target.

### Theorem 1 — Divergence of the Incumbent Multiplier

Let $B'$ be a compact box. Suppose the original constraints

$$
f_i(x) \leq 0, \qquad i = 1, \ldots, m
$$

are feasible in $B'$, but the target-feasibility problem

$$
f_i(x) \leq 0, \qquad i = 1, \ldots, m, \qquad f_0(x) - U \leq 0
$$

has no solution in $B'$. Assume $f_0, \ldots, f_m$ are continuous and the dual step sizes satisfy $\alpha_k \geq \underline{\alpha} > 0$. Then the incumbent multiplier in the projected dual-ascent iteration

$$
y_U^{k+1} = \max\{0,\ y_U^k + \alpha_k(f_0(x^k) - U)\}
$$

diverges to $+\infty$, provided the primal procedure returns a feasible point for the original constraints at each dual update.

**Proof.** Let

$$
\mathcal{F}(B') = \{x \in B' : f_i(x) \leq 0,\ i = 1, \ldots, m\}
$$

be the feasible portion of the box. By assumption, $\mathcal{F}(B')$ is nonempty, while no point in it satisfies $f_0(x) \leq U$. Hence

$$
f_0(x) - U > 0 \qquad \text{for every } x \in \mathcal{F}(B').
$$

Because $f_0$ is continuous and $\mathcal{F}(B')$ is a closed subset of the compact box $B'$, $\mathcal{F}(B')$ is compact. Therefore the continuous function $f_0(x) - U$ attains a positive minimum on $\mathcal{F}(B')$. Thus there exists $\delta > 0$ such that

$$
f_0(x) - U \geq \delta \qquad \text{for every } x \in \mathcal{F}(B').
$$

At every dual update, the returned primal point is feasible, so

$$
y_U^{k+1} = \max\{0, y_U^k + \alpha_k(f_0(x^k)-U)\} \geq y_U^k + \alpha_k \delta \geq y_U^k + \underline{\alpha}\delta.
$$

Iterating gives

$$
y_U^k \geq y_U^0 + k\underline{\alpha}\delta,
$$

and therefore

$$
\boxed{y_U^k \to +\infty.}
$$

$\square$

This is the specific divergence result available for the incumbent multiplier. The theorem does not claim that an arbitrary infeasible system causes every dual multiplier, or even the total dual norm, to diverge. In particular, if the original constraints themselves are infeasible in $B'$, then $\mathcal{F}(B') = \emptyset$ and the positive-gap argument above does not apply. Such boxes require the abstract feasibility oracle, or a separate valid infeasibility mechanism; Theorem 1 alone does not certify them through $y_U$.

## 5.3 Pruning Soundness and the Oracle Gap

For the **abstract branch-and-prune algorithm**, pruning is defined by an exact target-feasibility oracle. If

$$
\exists x \in B' : f_i(x) \leq 0\ (i=1,\ldots,m), \qquad f_0(x) \leq U,
$$

then the target problem is feasible and an exact oracle retains the box. If no such point exists, the oracle may prune it. Thus the abstract oracle has, by definition, the two properties needed by the convergence argument:

1. **No false pruning:** every box containing a target-feasible point is retained.
2. **Eventual decision of infeasibility:** every target-infeasible box can be discarded.

These properties are specifications of the exact oracle; they are not consequences of Theorem 1.

The extended-Lagrangian dual iteration is intended as a concrete realization of that oracle, but the theory proved here is only partial. Theorem 1 establishes the implication

$$
\mathcal{F}(B') \neq \emptyset \ \text{ and } \ \min_{x\in\mathcal{F}(B')} f_0(x) > U \quad \Longrightarrow \quad y_U^k \to +\infty,
$$

provided every primal point used in the dual update is feasible for the original constraints. This supplies an asymptotic signal for **target infeasibility conditional on original feasibility**.

Two cases are not resolved by that theorem. First, when $\mathcal{F}(B') = \emptyset$, the incumbent multiplier need not be the multiplier that diverges, so $y_U$ alone is not a proved certificate for boxes whose original constraints are infeasible. Second, when the target problem is feasible, the present nonconvex analysis does not prove that the particular alternating primal-dual dynamics, or a finite numerical divergence threshold, can never exhibit behavior that would be misclassified as infeasibility. In convex problems, standard saddle-point and constraint-qualification theory can supply stronger conclusions, but those conclusions do not automatically extend to the general nonconvex setting considered here.

Accordingly, the logical status of the results is:

$$
\boxed{
\begin{aligned}
&\text{Exact target-feasibility oracle + systematic subdivision} \Rightarrow \text{Theorem 2,} \\
&\text{Theorem 1} \Rightarrow \text{one-sided justification of } y_U \text{ divergence in a specified case,} \\
&\text{finite-threshold numerical oracle} \Rightarrow \text{empirical realization, not a proved exact oracle.}
\end{aligned}
}
$$

Closing this oracle gap for broad nonconvex problems would require additional assumptions or a separate sound feasibility/infeasibility procedure. The global convergence theorem below is therefore intentionally stated conditionally on valid pruning rather than as a convergence theorem for the finite numerical implementation.

## 5.4 Global Convergence of the Branching Procedure

Under an exact target-feasibility oracle, target-infeasible boxes can be discarded while target-feasible boxes are retained. Theorem 1 provides a one-sided dual-dynamical justification for one important infeasible case, but the global result below assumes valid pruning at the oracle level. We now consider the branch containing a global minimizer.

### Theorem 2 — Global Convergence Under Valid Pruning

Assume:

1. $B_0$ is compact;
2. $f_0, f_1, \ldots, f_m$ are continuous on $B_0$;
3. the feasible set is nonempty;
4. a global minimizer $x^*$ exists;
5. the per-box procedure never incorrectly prunes a box containing a point satisfying the target constraints;
6. every non-pruned box is eventually subdivided whenever its diameter exceeds $\varepsilon$;
7. whenever a retained box is target-feasible, the exact oracle returns a target-feasible witness point in that box.

Then, for every $\varepsilon > 0$, the branch containing $x^*$ can be subdivided until its diameter is at most $\varepsilon$. Moreover, the incumbent objective values produced by the algorithm converge to $f^*$ as the subdivision tolerance tends to zero.

**Proof.** At every stage of the branch-and-prune process, there is a box $B_k$ containing $x^*$. Since $x^*$ is globally feasible,

$$
f_i(x^*) \leq 0
$$

for all $i$. Furthermore, $f_0(x^*) = f^*$. The incumbent satisfies $U \geq f^*$ because every incumbent is the objective value of a feasible point. Therefore

$$
f_0(x^*) - U \leq 0.
$$

Thus $x^*$ satisfies every constraint of the target-feasibility problem whenever $U \geq f^*$. By assumption 5, a box containing such a point is never incorrectly pruned.

Consequently, the branch containing $x^*$ survives subdivision.

Repeated subdivision of the widest dimension produces boxes whose diameter tends to zero (assumption 6). Let $B_k$ denote the box containing $x^*$. Then

$$
\operatorname{diam}(B_k) \to 0.
$$

Because $f_0$ is continuous at $x^*$, for every $\eta > 0$ there exists a neighborhood of $x^*$ such that

$$
|f_0(x) - f^*| < \eta
$$

whenever $x$ lies in that neighborhood.

For sufficiently large $k$, $B_k$ lies within this neighborhood. By assumption 7, the exact oracle returns a target-feasible witness point $x_k \in B_k$. Since $B_k$ lies within this neighborhood, that witness satisfies

$$
f_0(x_k) < f^* + \eta.
$$

The incumbent is the best feasible value found, so

$$
f^* \leq U_k \leq f^* + \eta
$$

for sufficiently fine subdivision. Since $\eta > 0$ is arbitrary,

$$
\boxed{U_k \to f^*.}
$$

$\square$

## 5.5 Interpretation of the Global Argument

The preceding theorem separates the roles of the two components of the method.

The **target-feasibility oracle** determines whether a particular box can contain a feasible improvement; the dual dynamics provide the proposed numerical mechanism for approximating this decision. The **branching procedure** guarantees that the region containing a global minimizer is examined at arbitrarily fine resolution.

The argument does not require the objective to be convex.

Convexity would allow a KKT point in the root box to establish global optimality directly. In the nonconvex case, KKT points may correspond to local minima or other stationary solutions. The purpose of branching is to avoid relying on any single local basin.

Thus the global convergence mechanism can be summarized as

$$
\boxed{
\text{valid pruning} + \text{continued subdivision} + \text{continuity} \Longrightarrow U \to f^*.
}
$$

---

# 6. Mathematical Algorithm Versus Numerical Implementation

The preceding analysis describes an idealized algorithm. In particular, it assumes that the primal minimization for fixed $y$ is performed sufficiently accurately and that divergence is determined mathematically rather than by a finite computer threshold.

The actual implementation makes two numerical approximations.

First, the minimization of $L(\cdot,y)$ over a box is approximated by projected gradient descent:

$$
x^{k+1} = \Pi_B\left(x^k - \beta\nabla_x L(x^k, y)\right).
$$

Second, exact divergence cannot be observed numerically. The implementation therefore declares divergence when

$$
\|y\| > \Delta
$$

for a prescribed finite threshold $\Delta$.

These implementation choices should not be confused with the mathematical pruning theorem. Theorem 1 states more narrowly that, when the original constraints are feasible but the incumbent target is unattainable and the primal points used for dual updates remain originally feasible, the incumbent multiplier $y_U$ is unbounded. A finite numerical threshold is only a heuristic detector of this behavior; the theorem does not make that threshold an exact infeasibility certificate.

Likewise, the finite iteration limits used in the implementation mean that a numerical run can terminate without having established either exact KKT convergence or exact dual divergence. Such a termination is treated as a numerical limitation rather than as a mathematical certificate.

The implementation therefore provides an empirical approximation to the exact-oracle algorithm analyzed above. The present theory does not prove that this finite numerical realization is a sound and complete target-feasibility oracle for arbitrary nonconvex continuous problems.

---

# 7. Convex Case

When $f_0$ and all $f_i$ are convex and an appropriate constraint qualification such as Slater's condition holds, KKT conditions are sufficient for global optimality.

Thus, in the convex case, once the per-box solver identifies a KKT point satisfying the target constraints, the resulting primal point is globally optimal for that convex subproblem.

The nonconvex case is different. A KKT point need not be globally optimal. Consequently, the branch-and-prune procedure must continue exploring different regions of the initial box.

The convergence theorem above therefore does not rely on convexity of the objective or constraints. It relies instead on continuity, compactness, valid pruning, and systematic subdivision.

---

# 8. Numerical Implementation

The algorithm is implemented in Python using NumPy for numerical computation and SymPy for symbolic differentiation of user-supplied expression strings.

The implementation uses projected gradient descent for the primal minimization, with configurable step size and iteration limits. Dual ascent uses a configurable positive step size. Numerical divergence is declared when the norm of the dual vector exceeds a user-specified threshold.

The branch-and-prune procedure processes boxes depth-first and branches along the widest dimension. The implementation terminates when the box budget is exhausted or no boxes remain.

A finite incumbent $U$ is required for the target constraint $f_0(x) \leq U$. In a practical implementation, an initial feasible point may therefore be obtained before target-level pruning is activated. Once a finite feasible incumbent is available, the extended Lagrangian is used exactly as defined in (1).

---

# 9. Numerical Experiments

We evaluated the implementation on thirty test problems covering convex and nonconvex objectives, active and inactive inequality constraints, multiple global minimizers, higher-dimensional problems, nonlinear constrained geometries, disconnected feasible regions, thin feasible sets, and transcendental or highly nonconvex objectives.

The algorithm successfully recovered the known global optimum on all thirty test problems.

| # | Problem | True $f^*$ | Found $f^*$ | Error | Boxes | Time | Status |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | $\min (x-2)^2$ | 0.0000 | 0.0000 | $2.19\times10^{-13}$ | 21 | 0.2 s | PASS |
| 2 | $\min x_1^2+x_2^2$ | 0.0000 | 0.0000 | 0.00 | 151 | 1.6 s | PASS |
| 3 | $\min (x_1-1)^2+2(x_2-2)^2$ | 0.0000 | 0.0000 | $2.20\times10^{-13}$ | 41 | 0.4 s | PASS |
| 4 | $\min x^4+x^2-4x$ | -2.1566 | -2.1567 | $5.20\times10^{-5}$ | 27 | 0.2 s | PASS |
| 5 | $\min x^2,\ x\ge2$ | 4.0000 | 4.0000 | $2.74\times10^{-7}$ | 15 | 1.2 s | PASS |
| 6 | $\min x_1^2+x_2^2,\ x_1+x_2\ge1$ | 0.5000 | 0.5000 | $9.97\times10^{-7}$ | 107 | 6.2 s | PASS |
| 7 | $\min x_1^2+x_2^2,\ x_1\ge1,\ x_2\ge1$ | 2.0000 | 2.0000 | $1.95\times10^{-6}$ | 31 | 9.4 s | PASS |
| 8 | $\min x_1^2+x_2^2+x_3^2,\ x_1+x_2+x_3\ge1$ | 0.3333 | 0.3333 | $6.30\times10^{-7}$ | 49 | 2.3 s | PASS |
| 9 | $\min x^4-2x^2$ | -1.0000 | -1.0000 | 0.00 | 63 | 0.4 s | PASS |
| 10 | $\min x^4-3x^2+2x$ | -4.8481 | -4.8481 | $2.38\times10^{-5}$ | 19 | 0.2 s | PASS |
| 11 | $\min (x^2-1)^2$ | 0.0000 | 0.0000 | 0.00 | 87 | 0.5 s | PASS |
| 12 | $\min (x^2-2)^2$ | 0.0000 | 0.0000 | $8.46\times10^{-15}$ | 51 | 0.3 s | PASS |
| 13 | $\min (x_1^2-1)^2+(x_2^2-1)^2$ | 0.0000 | 0.0000 | 0.00 | 479 | 3.3 s | PASS |
| 14 | $\min x_1^4+x_2^4-4x_1^2-4x_2^2$ | -8.0000 | -8.0000 | $1.95\times10^{-14}$ | 151 | 56.3 s | PASS |
| 15 | $\min x^4-2x^2,\ x\ge0.5$ | -1.0000 | -1.0000 | 0.00 | 31 | 0.3 s | PASS |
| 16 | $\min x_1^2+x_2^2,\ (x_1-2)^2+(x_2-2)^2\le1$ | 3.3431 | 3.3431 | $6.67\times10^{-7}$ | 37 | 13.0 s | PASS |
| 17 | $\min (x^2-1)^3$ | -1.0000 | -1.0000 | 0.00 | 25 | 2.1 s | PASS |
| 18 | $\min \sin(x^2)$ | -1.0000 | -1.0000 | $1.55\times10^{-14}$ | 29 | 0.2 s | PASS |
| 19 | $\min x^4-2x^2,\ x\le-0.5$ | -1.0000 | -1.0000 | 0.00 | 31 | 0.2 s | PASS |
| 20 | $\min x_1^2+x_2^4-2x_2^2$ | -1.0000 | -1.0000 | $2.20\times10^{-13}$ | 125 | 1.3 s | PASS |
| 21 | Rastrigin 2D | 0.0000 | 0.0000 | 0.00 | 15 | 24.2 s | PASS |
| 22 | Rosenbrock 2D | 0.0000 | 0.0012 | $1.25\times10^{-3}$ | 3 | 17.2 s | PASS |
| 23 | Ackley 2D | 0.0000 | 0.0000 | $1.26\times10^{-7}$ | 3 | 24.3 s | PASS |
| 24 | 5D $\sum_i(x_i^2-1)^2$, 32 global minima | 0.0000 | 0.0000 | 0.00 | 5000 | 47.5 s | PASS |
| 25 | Thin feasible disk, $(x_1-1)^2+x_2^2\le0.01$ | 0.8100 | 0.8100 | $4.27\times10^{-7}$ | 17 | 15.3 s | PASS |
| 26 | Disconnected feasible region, $x^2\ge4$ | 9.0000 | 9.0000 | $9.71\times10^{-7}$ | 31 | 0.5 s | PASS |
| 27 | Multimodal, $x_1+x_2\ge2.5$ | 0.6328 | 0.6328 | $1.15\times10^{-5}$ | 41 | 19.7 s | PASS |
| 28 | Deceptive $x^2+10\sin(x)^2$ | 0.0000 | 0.0000 | 0.00 | 43 | 12.5 s | PASS |
| 29 | Nonconvex constrained, $(x_1^2+x_2^2-1)^2-0.2x_1,\ x_1x_2\ge0.5$ | -0.1670 | -0.1665 | $4.62\times10^{-4}$ | 3 | 9.0 s | PASS |
| 30 | 10D separable, $\sum_i x_i^4-16x_i^2+5x_i$ | -783.3233 | -783.3233 | 0.00 | 201 | 528.3 s | PASS |

$$
\boxed{30/30\text{ problems passed}}
$$

The results provide empirical evidence that the implementation can realize the branch-and-prune mechanism across a broad range of problem structures. They are not themselves a convergence theorem: the abstract guarantee is the limiting statement $U \to f^*$, while the finite errors and runtimes are properties of this particular numerical implementation and its stopping thresholds.

## 9.1 Code and Reproducibility

The complete Python implementation, numerical parameter settings, and definitions of all thirty test problems used in this study will be made publicly available in an accompanying GitHub repository at [GITHUB REPOSITORY URL]. The repository will include the scripts and test definitions required to reproduce the numerical results reported in Table 1.

---

# 10. Related Work and Discussion

## 10.1 Branch-and-Bound and Deterministic Global Optimization

Branch-and-bound is a standard architecture for deterministic global optimization. Classical geometric branch-and-bound methods associate subregions with bounds or other discarding information and use those quantities to eliminate regions that cannot contain a global solution. Scholz provides a broad treatment of geometric branch-and-bound methods, including Lipschitzian optimization, d.c. programming, interval analysis, and related bounding operations [8]. Scholz also develops a general convergence theory for constrained geometric branch-and-bound methods based on bounding operations [9]. These methods illustrate the central role that computable bounds play in conventional deterministic global optimization. Other representative deterministic global-optimization approaches include $\alpha$BB [1], semidefinite-based finite branch-and-bound for nonconvex quadratic programs [4], and branch-and-bound methods for mixed-integer nonlinear programs [7].

The present framework uses the same basic spatial subdivision philosophy but does not require a lower bound on the objective over each box. Instead, the incumbent $U$ is inserted directly into the subproblem as the target constraint

$$
f_0(x) - U \leq 0.
$$

The question at each node is therefore

$$
\boxed{\text{Does this box contain a feasible point with }f_0(x)\leq U?}
$$

At the abstract level, the answer is supplied by an exact target-feasibility oracle. In the concrete realization studied here, the multiplier $y_U$ provides a practical pruning signal in the case covered by Theorem 1, while the full soundness and completeness of the numerical oracle remain open under the paper's general nonconvex assumptions.

## 10.2 Existing Branch-and-Prune Methods

The term *branch-and-prune* and the alternating branch/prune architecture are established in the literature and are not claimed as novel here. Sotiropoulos and Grapsa presented a branch-and-prune method for univariate global optimization in which interval evaluations of the objective derivative are used for monotonicity tests, function-range enclosures, pruning with the current upper bound, and splitting decisions [10]. Branch-and-prune is also a standard search framework in numerical constraint solving, where branching is alternated with domain-pruning procedures to eliminate portions of the search domain [11].

The present method differs in the source of its pruning information. Rather than using interval derivative information, interval enclosures, or another independently constructed bound to discard a box, it formulates the incumbent test itself as a feasibility problem. The incumbent is therefore not merely compared with a separately computed lower bound; it becomes part of the constraint system being tested. The associated multiplier $y_U$ is the specific quantity used to detect target infeasibility in the extended-Lagrangian realization.

This distinction is important for the generality claim. The proposed abstract framework does not require a Lipschitz constant, interval extension, convex relaxation, differentiability, or another problem-specific lower-bounding construction. Those assumptions can be useful for producing strong finite bounds, but they are not required by the target-feasibility formulation or by the limiting convergence proof presented here.

## 10.3 Lagrangian and Infeasibility Context

Lagrangian relaxation has a long history as a mechanism for moving difficult constraints into the objective through multipliers [6]. Related global-optimization methods have also combined Lagrangian or penalty constructions with global search [3,5]. In convex optimization, divergence or related behavior of primal-dual iterates can sometimes be used to detect infeasibility under specific algorithmic assumptions; Banjac et al. provide such an analysis for ADMM [2]. These results motivate the use of multiplier behavior as an infeasibility signal, but they do not by themselves establish the nonconvex oracle property required here. Theorem 1 is therefore stated directly for the particular incumbent multiplier and under its explicit feasibility assumptions.

## 10.4 Scope of the Contribution

The contribution should therefore be understood narrowly. The paper does not introduce branch-and-prune as a search paradigm, nor does it claim that incumbent-based pruning is itself new. Rather, it introduces and analyzes a particular target-feasibility formulation in which the incumbent is incorporated directly as an explicit constraint and its multiplier supplies a natural infeasibility signal in the case covered by Theorem 1. The main abstract theoretical result is the combination of valid target-feasibility pruning with exhaustive subdivision under weak assumptions; the finite primal-dual implementation is evaluated empirically rather than claimed to satisfy the exact-oracle assumptions in full generality.

The resulting tradeoff is deliberate. Conventional deterministic methods can often provide finite optimality-gap certificates because they construct lower bounds or other quantitative enclosures. The present framework gives up that finite certificate in exchange for a much weaker structural requirement: continuity on a compact domain is sufficient for the limiting result

$$
U_\varepsilon \to f^*.
$$

No particular lower-bound mechanism is required by the abstract algorithm. The numerical experiments demonstrate one empirical realization of the proposed numerical oracle, but the convergence theorem itself is stated at the oracle level.

---

# 11. Limitations and Future Work

The principal limitation of the framework is also a deliberate feature of it. The continuity assumption used in the present convergence theorem is sufficient rather than necessary: the proof only needs a weaker local condition ensuring that sufficiently small neighborhoods of a global minimizer contain points with objective values approaching the optimum. Establishing convergence under broader classes of discontinuous functions is therefore possible in principle, but is outside the theorem proved here. Because no assumptions beyond continuity are used in the abstract convergence argument, the theory provides no finite quantitative bound on the optimality gap. At any finite termination resolution $\varepsilon$, the algorithm does not know how close $U_\varepsilon$ is to $f^*$. The only rigorous statement is the limiting relation

$$
\boxed{U_\varepsilon \to f^* \quad \text{as} \quad \varepsilon \to 0.}
$$

A finite error certificate would require additional information about the functions — for example, a known modulus of continuity, Lipschitz bounds, interval estimates, convexity, or another valid lower-bounding mechanism. Adding such information could strengthen the guarantees, but would necessarily reduce the generality of the framework.

The second limitation concerns the numerical realization of the oracle. The theoretical algorithm assumes an exact target-feasibility oracle, whereas the implementation uses finite iterations and a finite threshold to detect practical dual divergence. These are computational approximations and should not be interpreted as mathematical certificates of infeasibility.

Accordingly, the present theory does not establish convergence of the particular numerical implementation for arbitrary continuous functions. Without additional assumptions on the functions, inner solver, and numerical tolerances, such a result cannot be obtained from the present analysis. Empirical experiments can instead provide evidence about how the implementation behaves in practice.

Future work may therefore investigate the empirical efficiency and scalability of the implementation, as well as stronger finite certificates in settings where additional regularity information is available. Such extensions would complement rather than replace the deliberately general abstract framework presented here.

---

# 12. Conclusion

We have presented a branch-and-prune framework for global optimization based on target-level feasibility. The central construction is the incorporation of the incumbent value $U$ as an explicit objective constraint,

$$
f_0(x) - U \leq 0,
$$

so that each box is asked a direct question: does it contain a feasible point capable of matching or improving the incumbent?

The principal contribution is the generality of this formulation. Continuity is used because it provides a simple, standard sufficient condition for the limiting convergence result, but it is not intrinsic to the branch-and-prune mechanism. The same target-feasibility and subdivision framework can be applied to discontinuous functions whenever an appropriate weaker local condition guarantees that arbitrarily small neighborhoods of a global minimizer contain points approaching the optimal objective value. The abstract branch-and-prune mechanism does not require a computable lower bound or any particular structure from which such a bound must be constructed. Under the assumptions of the convergence theorem, the only structural property required of the objective and constraint functions is continuity on a compact domain. No convexity, differentiability, Lipschitz constant, interval relaxation, or problem-specific lower-bounding construction is required.

The price of this generality is equally important. At no finite stage — even when the algorithm terminates at a prescribed spatial resolution — does the method provide a numerical bound on the distance between the incumbent and the global optimum. Without additional regularity information, box diameter is a resolution measure, not an objective-gap certificate. The rigorous guarantee is instead the asymptotic statement

$$
\boxed{U_\varepsilon \to f^* \quad \text{as} \quad \varepsilon \to 0.}
$$

Thus the framework deliberately exchanges finite quantitative certification for maximal generality while retaining a limiting global-convergence guarantee for continuous problems on compact domains.

The exact target-feasibility oracle is the mathematical abstraction at the heart of the method. The extended Lagrangian and dual-ascent procedure studied in this paper are a proposed numerical realization of that oracle. Theorem 1 justifies the $y_U$ divergence signal in the specific case of target infeasibility with original feasibility, while boxes with infeasible original constraints and the no-false-pruning property of the general nonconvex numerical dynamics are not resolved by the present proof. Practical divergence thresholds should therefore be understood as empirical decision rules rather than exact certificates. These limitations do not alter the conditional convergence result for the abstract exact-oracle branch-and-prune framework.

The numerical experiments provide empirical evidence that the particular implementation can successfully realize this mechanism across a range of test problems. The theoretical framework, however, makes a more fundamental claim: global optimization can be formulated as repeated target-feasibility pruning and subdivision without requiring a finite lower-bound certificate, with continuity alone sufficient for convergence in the limit.

---

# 13. References

[1] C. S. Adjiman, S. Dallwig, C. A. Floudas, and A. Neumaier (1998). A global optimization method, $\alpha$BB, for general twice-differentiable constrained NLPs. *Computers & Chemical Engineering*, 22(9):1137–1158.

[2] G. Banjac, P. Goulart, B. Stellato, and S. Boyd (2019). Infeasibility detection in the alternating direction method of multipliers for convex optimization. *Journal of Optimization Theory and Applications*, 183(2):490–519.

[3] E. G. Birgin, C. A. Floudas, and J. M. Martínez (2010). Global minimization using an augmented Lagrangian method with variable lower-level constraints. *Mathematical Programming*, 125(1):139–162.

[4] S. Burer and D. Vandenbussche (2008). Globally solving box-constrained nonconvex quadratic programs with semidefinite-based finite branch-and-bound. *Mathematical Programming*, 114(2):373–396.

[5] G. Di Pillo, S. Lucidi, and F. Rinaldi (2016). An approach to constrained global optimization based on exact penalty functions. *Journal of Global Optimization*, 54(2):251–260.

[6] M. L. Fisher (1981). The Lagrangian relaxation method for solving integer programming problems. *Management Science*, 27(1):1–18.

[7] M. Tawarmalani and N. V. Sahinidis (2004). Global optimization of mixed-integer nonlinear programs: A theoretical and computational study. *Mathematical Programming*, 99(3):563–591.

[8] D. Scholz (2012). *Deterministic Global Optimization: Geometric Branch-and-bound Methods and their Applications*. Springer. DOI: 10.1007/978-1-4614-1951-8.

[9] D. Scholz (2013). Geometric branch-and-bound methods for constrained global optimization problems. *Journal of Global Optimization*, 57:771–782. DOI: 10.1007/s10898-012-9961-9.

[10] D. G. Sotiropoulos and T. N. Grapsa (2001). A Branch-and-Prune Method for Global Optimization. In *Global Optimization*. DOI: 10.1007/978-1-4757-6484-0_18.

[11] X.-H. Vu, M.-C. Silaghi, D. Sam-Haroud, and B. Faltings (2005). Branch-and-Prune Search Strategies for Numerical Constraint Solving. arXiv:cs/0512045.

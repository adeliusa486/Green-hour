"""Validate the fast separable-QP primitive against a general-purpose solver.

The whole experiment programme rests on solve_sep_qp being exact, so it is
checked against scipy SLSQP on random instances that exercise the deadline
staircase, saturated envelopes, and near-infeasible corners.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import minimize
from gh.core import Feasible, solve_sep_qp, solve_sep_lp


def brute(q, c, X: Feasible):
    T = X.T
    cons = [{"type": "eq", "fun": lambda x: x.sum() - X.E}]
    if X.R is not None and X.R.max() > 0:
        cons.append({"type": "ineq", "fun": lambda x: np.cumsum(x) - X.R})
    x0 = np.full(T, X.E / T)
    x0 = np.minimum(x0, X.u)
    x0 = x0 * X.E / max(x0.sum(), 1e-12)
    res = minimize(lambda x: float(q @ x + c @ (x ** 2)), x0,
                   jac=lambda x: q + 2 * c * x,
                   bounds=[(0.0, ui) for ui in X.u], constraints=cons,
                   method="SLSQP", options={"maxiter": 800, "ftol": 1e-14})
    return res.x, float(q @ res.x + c @ (res.x ** 2))


def random_instance(rng, T=12, with_stair=True):
    u = rng.uniform(0.4, 2.0, T)
    E = rng.uniform(0.25, 0.75) * u.sum()
    R = None
    if with_stair:
        # a feasible staircase: cumulative requirement below what caps allow
        frac = np.sort(rng.uniform(0, 1, T)) ** 2
        R = frac * E * rng.uniform(0.3, 0.9)
        R = np.minimum(R, np.cumsum(u) * 0.95)
        R = np.maximum.accumulate(R)
    q = rng.normal(0, 3, T)
    c = rng.uniform(0.05, 2.0, T)
    return q, c, Feasible(E=E, u=u, R=R)


def main():
    rng = np.random.default_rng(7)
    worst_obj, worst_feas = 0.0, 0.0
    n_better = 0
    N = 400
    for k in range(N):
        q, c, X = random_instance(rng, T=int(rng.integers(4, 20)),
                                  with_stair=(k % 3 != 0))
        x = solve_sep_qp(q, c, X)
        f = float(q @ x + c @ (x ** 2))
        xb, fb = brute(q, c, X)
        assert X.check(x, tol=1e-7), f"case {k}: solver returned infeasible point"
        # our objective must be <= SLSQP's (up to its own tolerance)
        rel = (f - fb) / max(1.0, abs(fb))
        worst_obj = max(worst_obj, rel)
        if f < fb - 1e-9:
            n_better += 1
        worst_feas = max(worst_feas, max(0.0, float(np.max(X.R - np.cumsum(x)))
                                         if X.R is not None else 0.0))
    print(f"solve_sep_qp vs SLSQP over {N} random instances")
    print(f"  worst relative excess over SLSQP : {worst_obj:.3e}")
    print(f"  cases where ours beat SLSQP      : {n_better}")
    print(f"  worst staircase violation        : {worst_feas:.3e}")
    assert worst_obj < 1e-7, "fast solver is not matching the reference"
    assert worst_feas < 1e-7

    # LP path
    rng = np.random.default_rng(11)
    for k in range(60):
        q, c, X = random_instance(rng, T=10, with_stair=(k % 2 == 0))
        x = solve_sep_lp(q, X)
        assert X.check(x, tol=1e-7)
        # compare against QP with vanishing curvature
        xq = solve_sep_qp(q, np.full(X.T, 1e-9), X)
        assert float(q @ x) <= float(q @ xq) + 1e-6
    print("  LP path feasible and optimal on 60 instances")
    print("PASS")


if __name__ == "__main__":
    main()

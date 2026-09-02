"""Is the planner minimiser unique in the PROFILE, or only in the AGGREGATE?

Theorem 5.2 as drafted said C is strictly convex in y, therefore the fixed point
is unique.  That implication does not hold for the full profile x.  For a fixed
slot the Hessian of the quadratic social term with respect to (x_1t,...,x_nt) is

    2 * beta_t * 1 1^T ,

which is rank one -- positive semidefinite, not positive definite.  Contrast
Proposition 4.1, whose potential carries an extra sum_i x_it^2 and so has
Hessian beta_t (I + 1 1^T), which IS positive definite.  That diagonal term is
exactly what C lacks.

So: with lam = 0, any redistribution of load between operators that preserves
the aggregate y and stays feasible attains the same social cost.  This test
constructs one and measures it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.core import Feasible, Instance
from gh.baselines import planner_aggregate, planner_value


def symmetric_instance(n=2, T=4, beta=5.0):
    a = np.array([100.0, 120.0, 140.0, 160.0])[:T]
    m = a * 2.0
    b = np.full(T, beta)
    E = 10.0
    X = [Feasible(E=E, u=np.full(T, 3.0 * E), R=None) for _ in range(n)]
    psi = np.zeros((n, T))
    lam = np.zeros(n)
    return Instance(a=a, m=m, b=b, X=X, psi=psi, lam=lam)


def test_profile_non_uniqueness():
    inst = symmetric_instance()
    y = planner_aggregate(inst)
    Cstar = planner_value(inst)

    # two DIFFERENT feasible profiles with the SAME aggregate
    xA = np.vstack([y / 2.0, y / 2.0])
    half = y / 2.0
    # pick two slots that both carry load, so a swap stays feasible both ways
    carry = np.where(half > 1e-6)[0]
    assert len(carry) >= 2, f"planner solution is a corner (y={y}); raise beta"
    k, j = int(carry[0]), int(carry[1])
    room = inst.X[0].u - half
    d = 0.4 * min(half[k], half[j], room[k], room[j])
    shift = np.zeros_like(y); shift[k], shift[j] = -d, +d
    xB = np.vstack([half + shift, half - shift])

    for X, x in zip(inst.X, xA):
        assert X.check(x, tol=1e-9), "xA infeasible"
    for X, x in zip(inst.X, xB):
        assert X.check(x, tol=1e-9), "xB infeasible"

    assert np.max(np.abs(xA.sum(0) - xB.sum(0))) < 1e-12, "aggregates differ"
    profile_gap = float(np.max(np.abs(xA - xB)))
    cost_gap = abs(inst.social(xA) - inst.social(xB))

    print(f"  aggregate y identical to  {np.max(np.abs(xA.sum(0)-xB.sum(0))):.2e}")
    print(f"  profiles differ by        {profile_gap:.4f}  (max |xA - xB|)")
    print(f"  social cost differs by    {cost_gap:.3e}")
    print(f"  both equal C* to          {max(abs(inst.social(xA)-Cstar), abs(inst.social(xB)-Cstar)):.3e}")

    assert profile_gap > 1e-3, "failed to construct a genuinely different profile"
    assert cost_gap < 1e-9, "costs should be identical"
    print("  => the minimiser is NOT unique in the profile; it is unique in y.")


def test_aggregate_is_unique():
    """The aggregate, by contrast, is pinned: C is strictly convex in y."""
    inst = symmetric_instance()
    y = planner_aggregate(inst)
    C = lambda yy: float(np.sum(inst.m * yy + inst.b * yy ** 2))
    base = C(y)
    worst = 0.0
    rng = np.random.default_rng(0)
    for _ in range(2000):
        d = rng.normal(0, 0.05, len(y))
        d -= d.mean()                       # preserve total energy
        yy = y + d
        if np.any(yy < 0):
            continue
        worst = max(worst, base - C(yy))
    print(f"  best improving aggregate perturbation: {worst:.3e}")
    assert worst < 1e-9
    print("  => the aggregate minimiser is unique.")


if __name__ == "__main__":
    print("test_planner_uniqueness")
    test_profile_non_uniqueness()
    test_aggregate_is_unique()
    print("  ALL PASS")

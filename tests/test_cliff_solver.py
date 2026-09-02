"""Validate the piecewise-quadratic primitive.

The objective is convex but its derivative JUMPS at the headroom kink, so
SLSQP is an unreliable referee here -- it fails to converge on most random
instances.  The primary check is therefore a KKT certificate, which for a
convex separable problem over this polytope is a proof of optimality rather
than a comparison:

    there exists nu such that, for every slot t,
        0 < x_t < u_t   =>  nu in  subdifferential of dE_t at x_t
        x_t = 0         =>  nu <= sup of that subdifferential
        x_t = u_t       =>  nu >= inf of that subdifferential

plus primal feasibility.  A random-perturbation search and (where it does
converge) SLSQP are kept as independent cross-checks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.optimize import minimize
from gh.core import Feasible
from gh.cliff import Cliff, solve_sep_pw

rng = np.random.default_rng(0)


def random_cliff(T):
    m_lo = rng.uniform(50, 300, T)
    b_lo = rng.uniform(0.01, 0.4, T)
    H = rng.uniform(0.2, 3.0, T)
    b_hi = b_lo * rng.uniform(1.5, 12.0, T)
    m_hi = m_lo + 2 * b_lo * H + rng.uniform(0, 200, T)
    return Cliff(H, m_lo, b_lo, m_hi, b_hi)


def random_feasible(T, with_R):
    u = rng.uniform(0.5, 3.0, T)
    E = rng.uniform(0.25, 0.75) * u.sum()
    R = None
    if with_R:
        R = np.sort(rng.uniform(0, 0.5 * E, T))
        R = np.maximum.accumulate(np.minimum(R, E))
    return Feasible(E=E, u=u, R=R)


def subgrad_bounds(cl, x):
    """[lo, hi] of the subdifferential of dE_t at x_t, elementwise."""
    at_kink = np.abs(x - cl.H) < 1e-9
    g = cl.marginal(x)
    lo = np.where(at_kink, cl.m_lo + 2 * cl.b_lo * cl.H, g)
    hi = np.where(at_kink, cl.m_hi, g)
    return lo, hi


def test_marginal_inv():
    worst = 0.0
    for _ in range(400):
        cl = random_cliff(10)
        x = rng.uniform(0, 6, 10)
        worst = max(worst, float(np.max(np.abs(cl.marginal_inv(cl.marginal(x)) - x))))
    print(f"  marginal_inv round trip: worst {worst:.3e}")
    assert worst < 1e-9


def test_kkt(n_inst=300, with_R=False):
    """The certificate.  No staircase: then the whole problem is one
    box+equality block and a single nu must certify it."""
    worst = 0.0
    for _ in range(n_inst):
        T = int(rng.integers(6, 16))
        cl, X = random_cliff(T), random_feasible(T, with_R)
        x = solve_sep_pw(cl, X)
        assert X.check(x, tol=1e-7), "infeasible"
        lo, hi = subgrad_bounds(cl, x)
        free = (x > 1e-9) & (x < X.u - 1e-9)
        if not free.any():
            continue
        nu = 0.5 * (lo[free].max() + hi[free].min())
        # interior slots must admit a COMMON nu
        gap = lo[free].max() - hi[free].min()
        # zero slots: marginal at 0 must be >= nu ; capped slots: <= nu
        z = (x <= 1e-9)
        c = (x >= X.u - 1e-9)
        v = max(gap,
                float(np.max(nu - hi[z])) if z.any() else -np.inf,
                float(np.max(lo[c] - nu)) if c.any() else -np.inf)
        worst = max(worst, v / max(1.0, abs(nu)))
    print(f"  KKT certificate over {n_inst} instances: worst violation "
          f"{worst:.3e}")
    assert worst < 1e-8


def test_no_improving_perturbation(n_inst=150, trials=300):
    """Independent of any optimality theory: try to beat the returned point by
    moving energy between pairs of slots."""
    worst = 0.0
    for _ in range(n_inst):
        T = int(rng.integers(6, 14))
        cl, X = random_cliff(T), random_feasible(T, True)
        x = solve_sep_pw(cl, X)
        f0 = float(cl.value(x).sum())
        for _ in range(trials):
            i, j = rng.integers(0, T, 2)
            if i == j:
                continue
            d = rng.uniform(0, 0.3) * min(x[i], X.u[j] - x[j])
            if d <= 0:
                continue
            z = x.copy(); z[i] -= d; z[j] += d
            if not X.check(z, tol=1e-9):
                continue
            worst = max(worst, (f0 - float(cl.value(z).sum())) / max(1.0, abs(f0)))
    print(f"  best improving perturbation found: {worst:.3e}")
    assert worst < 1e-9


def test_vs_slsqp(n_inst=150):
    """Cross-check where SLSQP converges.  Kept as a sanity check only."""
    worst, ok = 0.0, 0
    for _ in range(n_inst):
        T = int(rng.integers(6, 14))
        cl, X = random_cliff(T), random_feasible(T, False)
        x = solve_sep_pw(cl, X)
        f_ours = float(cl.value(x).sum())
        res = minimize(lambda z: float(cl.value(z).sum()),
                       x0=np.full(T, X.E / T), method="SLSQP",
                       bounds=[(0, ui) for ui in X.u],
                       constraints=[{"type": "eq",
                                     "fun": lambda z: z.sum() - X.E}],
                       options={"maxiter": 2000, "ftol": 1e-14})
        if not res.success:
            continue
        ok += 1
        f_ref = float(cl.value(res.x).sum())
        worst = max(worst, (f_ours - f_ref) / max(1.0, abs(f_ref)))
    print(f"  vs SLSQP on {ok}/{n_inst} converged: worst relative excess "
          f"{worst:.3e}")
    assert worst < 1e-6


if __name__ == "__main__":
    print("test_cliff_solver")
    test_marginal_inv()
    test_kkt(with_R=False)
    test_no_improving_perturbation()
    test_vs_slsqp()
    print("  ALL PASS")

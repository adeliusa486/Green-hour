"""The curtailment cliff: a piecewise dispatch response, and what it does to
the accounting/strategic decomposition.

Motivation (REVIEW2 section 5).  The paper's motivating figure is a *threshold*:
emissions barely respond to added load inside the zero-carbon headroom, then
jump once deferred load crosses it.  The theory replaces that with a
second-order expansion about D_t, which fixes eta_t *before any flexible load is
placed* and therefore cannot represent headroom exhaustion as something the
agents cause.  V9 then showed the strategic share only clears the 15% Plan A
gate at kappa ~ 16 under the smooth model, which is not a plausible calibration.

The question this module answers: does putting the curvature where the physics
actually puts it -- concentrated at the headroom boundary rather than spread
evenly across the horizon -- raise the strategic share?

WHAT IS AND IS NOT CHANGED
--------------------------
Changed: the SOCIAL cost.  The true emissions increment from flexible load y in
slot t is piecewise quadratic,

    dE_t(y) = m_lo*y + b_lo*y^2                                   y <= H
            = m_lo*H + b_lo*H^2 + m_hi*(y-H) + b_hi*(y-H)^2       y >  H

with m_hi >= m_lo + 2*b_lo*H so the marginal is non-decreasing and dE is convex.

NOT changed: the AGENT's cost.  Agents are still charged an affine attributed
factor a_t + b_t*y_t.  This is deliberate and it is the paper's own thesis:
carbon accounting is a crude linear attribution of a realised average, while
physics is the cliff.  The mismatch between the two IS the effect being
measured.  Keeping J_i affine also means every existing result about the
EQUILIBRIUM -- the exact potential, uniqueness, the water-filling local step --
continues to hold unchanged.

WHY WE DO NOT ALSO MAKE THE AGENT COST PIECEWISE.  If the charged factor c_t(y)
is not affine then no exact potential exists: an exact potential needs
d/dx_j [c(y) + x_i c'(y)] to be symmetric in i,j, which forces c'' = 0.  A
piecewise-affine c would therefore cost Proposition 3.1 its proof and require a
Rosen diagonal-strict-concavity argument instead.  That is a genuine theory
change and is left for the paper's Phase 3 proper; it is NOT smuggled in here.
"""

from __future__ import annotations
import numpy as np
from .core import Feasible, Instance


# --------------------------------------------------------------------------
# the piecewise response
# --------------------------------------------------------------------------

class Cliff:
    """Per-slot piecewise-quadratic emissions increment.

    Arrays are per slot: H (headroom), m_lo, b_lo, m_hi, b_hi.
    """

    def __init__(self, H, m_lo, b_lo, m_hi, b_hi):
        self.H = np.asarray(H, float)
        self.m_lo = np.asarray(m_lo, float)
        self.b_lo = np.asarray(b_lo, float)
        self.m_hi = np.asarray(m_hi, float)
        self.b_hi = np.asarray(b_hi, float)
        kink_lo = self.m_lo + 2.0 * self.b_lo * self.H
        if np.any(self.m_hi < kink_lo - 1e-9):
            raise ValueError("non-convex cliff: m_hi below the lower marginal "
                             "at the kink")
        if np.any(self.b_lo <= 0) or np.any(self.b_hi <= 0):
            raise ValueError("cliff needs strictly positive curvature on both "
                             "pieces (uniqueness of the planner solution)")

    def value(self, y):
        """dE_t(y_t), elementwise."""
        y = np.asarray(y, float)
        lo = self.m_lo * y + self.b_lo * y ** 2
        d = np.maximum(y - self.H, 0.0)
        hi = (self.m_lo * self.H + self.b_lo * self.H ** 2
              + self.m_hi * d + self.b_hi * d ** 2)
        return np.where(y <= self.H, lo, hi)

    def marginal(self, y):
        """dE_t'(y_t), elementwise.  Non-decreasing in y by construction."""
        y = np.asarray(y, float)
        lo = self.m_lo + 2.0 * self.b_lo * y
        hi = self.m_hi + 2.0 * self.b_hi * np.maximum(y - self.H, 0.0)
        return np.where(y <= self.H, lo, hi)

    def marginal_inv(self, nu):
        """Generalised inverse of the marginal: the largest x >= 0 whose
        subdifferential of dE at x contains nu.

        The marginal JUMPS at the kink, from m_lo + 2*b_lo*H up to m_hi, so for
        nu strictly inside that gap no x solves marginal(x) = nu and the
        correct value is exactly H.  Getting this wrong -- extrapolating the
        lower branch past H -- makes the solver place load beyond the headroom
        that the true marginal never justifies, and it silently returns
        suboptimal profiles.  (It did; see tests/test_cliff_solver.py.)
        """
        nu = np.asarray(nu, float)
        nu_top_lo = self.m_lo + 2.0 * self.b_lo * self.H   # marginal just below H
        nu_bot_hi = self.m_hi                              # marginal just above H

        x_lo = (nu - self.m_lo) / (2.0 * self.b_lo)
        x_hi = self.H + (nu - self.m_hi) / (2.0 * self.b_hi)

        x = np.where(nu >= nu_bot_hi, x_hi,
                     np.where(nu >= nu_top_lo, self.H, x_lo))
        return np.clip(x, 0.0, None)


# --------------------------------------------------------------------------
# separable minimisation of a piecewise-quadratic over the feasible set
# --------------------------------------------------------------------------

def _waterfill_pw(cl: Cliff, idx, E, u, tol=1e-13, iters=200):
    """min sum_t dE_t(x_t) s.t. sum x = E, 0 <= x_t <= u_t, over slots `idx`.

    The marginal is non-decreasing in x, so x_t(nu) = clip(marginal_inv(nu),
    0, u_t) is non-decreasing in nu and S(nu) = sum_t x_t(nu) is a
    non-decreasing piecewise-linear function.  Bisection on nu is therefore
    exact in the limit and monotone-safe.  Bisection rather than a breakpoint
    sweep because the planner is called far less often than the agent step and
    correctness matters more than the constant.
    """
    sub = Cliff(cl.H[idx], cl.m_lo[idx], cl.b_lo[idx],
                cl.m_hi[idx], cl.b_hi[idx])
    u = np.asarray(u, float)
    tot = u.sum()
    if E <= 1e-15:
        return np.zeros(len(u))
    if E >= tot - 1e-15:
        return u.copy()

    def S(nu):
        return float(np.clip(sub.marginal_inv(nu), 0.0, u).sum())

    lo = float(sub.marginal(np.zeros(len(u))).min()) - 1.0
    hi = float(sub.marginal(u).max()) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if S(mid) < E:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(hi)):
            break
    x = np.clip(sub.marginal_inv(0.5 * (lo + hi)), 0.0, u)

    resid = E - x.sum()
    if abs(resid) > 1e-11 * max(1.0, E):
        room = (u - x) if resid > 0 else x.copy()
        s = room.sum()
        if s > 1e-15:
            x = x + np.sign(resid) * room * (abs(resid) / s)
    return x


def solve_sep_pw(cl: Cliff, X: Feasible):
    """min sum_t dE_t(x_t) over X.

    Same polymatroid decomposition as core.solve_sep_qp: solve the
    box+equality relaxation, find the most-violated prefix of the deadline
    staircase, force it tight, recurse.  The argument only uses that the
    objective is separable and convex, so it carries over unchanged.
    """
    def rec(lo_i, hi_i, E, R):
        n = hi_i - lo_i
        if n == 0:
            return np.zeros(0)
        idx = np.arange(lo_i, hi_i)
        x = _waterfill_pw(cl, idx, E, X.u[lo_i:hi_i])
        if R is None:
            return x
        viol = R - np.cumsum(x)
        k = int(np.argmax(viol))
        if viol[k] <= 1e-11 * max(1.0, E):
            return x
        left = rec(lo_i, lo_i + k + 1, R[k], R[:k + 1] if k > 0 else None)
        right = rec(lo_i + k + 1, hi_i, E - R[k],
                    (R[k + 1:] - R[k]) if k + 1 < n else None)
        return np.concatenate([left, right])

    R = X.R
    if R is not None:
        R = np.maximum(R, 0.0)
        if R.max() <= 1e-12:
            R = None
    return rec(0, X.T, X.E, R)


# --------------------------------------------------------------------------
# social cost and planner under the cliff
# --------------------------------------------------------------------------

def social_cliff(inst: Instance, cl: Cliff, x) -> float:
    """C(x) with the piecewise response.  Penalty term unchanged."""
    y = np.asarray(x).sum(axis=0)
    return float(cl.value(y).sum()) + inst.pen(x)


def planner_cliff(inst: Instance, cl: Cliff) -> float:
    """C(x*) under the cliff.

    With lam = 0 the cost depends on x only through y and the Minkowski sum of
    the feasible sets is a polytope of the same form, so the planner collapses
    to one separable problem in y -- the same reduction core/baselines already
    validate for the quadratic case.
    """
    if not np.allclose(inst.lam, 0.0):
        raise NotImplementedError("cliff planner assumes lam = 0")
    agg = Feasible(E=sum(X.E for X in inst.X),
                   u=sum(X.u for X in inst.X),
                   R=(sum(X.R for X in inst.X)
                      if inst.X[0].R is not None else None))
    y = solve_sep_pw(cl, agg)
    return float(cl.value(y).sum())


# --------------------------------------------------------------------------
# building a cliff that is comparable to a given smooth instance
# --------------------------------------------------------------------------

def cliff_from_instance(inst: Instance, headroom_frac=0.6, jump=6.0,
                        flat=0.05) -> Cliff:
    """A cliff matched to `inst`, so the comparison is like-for-like.

    Same operating-point marginal `m` and the same *total* curvature budget as
    the smooth instance, but redistributed: nearly flat inside a headroom of
    `headroom_frac` of the planner's per-slot load, then `jump` times steeper
    beyond it.  `flat` is the fraction of the smooth curvature retained inside
    the headroom (kept > 0 so the planner solution stays unique).

    The headroom is set from the PLANNER's aggregate under the smooth model,
    which is the natural per-slot scale and does not depend on the equilibrium
    we are about to measure.
    """
    from .baselines import planner_aggregate
    y_star = planner_aggregate(inst)
    H = headroom_frac * np.maximum(y_star, 1e-9)

    b_lo = flat * inst.b
    b_hi = jump * inst.b
    m_lo = inst.m.copy()
    # continuity of the marginal at the kink, then the jump:
    m_hi = m_lo + 2.0 * b_lo * H + (jump - flat) * inst.b * H
    return Cliff(H=H, m_lo=m_lo, b_lo=b_lo, m_hi=m_hi, b_hi=b_hi)

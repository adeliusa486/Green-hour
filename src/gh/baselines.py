"""Baselines.

Beyond the seven in the paper this adds the two that REVIEW.md X1/X2 argue are
essential:

  B0  naive       one-shot best response to the PUBLISHED forecast, with no
                  anticipation of aggregate response.  This is what a deployed
                  day-ahead scheduler actually does; Nash is a strictly more
                  sophisticated model of practice than practice deserves.
  B8a mef_static  agents charged the published MARGINAL factor m_t.
  B8b mef_resp    agents charged the responsive marginal factor m_t + 2 b_t y_t,
                  i.e. the fix the paper's own discussion recommends.

MARL (the paper's B4) is not here: it needs GPU training and is on the user's
side of the plan.
"""

from __future__ import annotations
import numpy as np
from .core import Instance, nash, planner, solve_sep_qp, solve_sep_lp, \
    mef_signal_equilibrium
from .mech import shade


def project(inst: Instance, target):
    """Euclidean projection of a target profile onto each X_i."""
    x = np.zeros((inst.n, inst.T))
    for i in range(inst.n):
        x[i] = solve_sep_qp(-2.0 * target[i], np.ones(inst.T), inst.X[i])
    return x


def carbon_agnostic(inst: Instance):
    """Run on arrival: the closest feasible profile to the raw arrival shape."""
    return project(inst, inst.arrival)


def naive(inst: Instance):
    """B0.  Minimise sum_t x_it * a_t against the published forecast, with no
    congestion term at all.  Linear, so every operator picks the same slots."""
    x = np.zeros((inst.n, inst.T))
    tie = 1e-9 * np.arange(inst.T)          # deterministic tie-break
    for i in range(inst.n):
        x[i] = solve_sep_lp(inst.a + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def mef_static(inst: Instance):
    """B8a.  Same as naive but against a published marginal factor."""
    x = np.zeros((inst.n, inst.T))
    tie = 1e-9 * np.arange(inst.T)
    for i in range(inst.n):
        x[i] = solve_sep_lp(inst.m + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def threshold(inst: Instance, pct=35.0):
    """B3.  Wait-for-a-cleaner-hour: place work only in slots whose published
    intensity is below the pct-th percentile, earliest such slot first."""
    tau = np.percentile(inst.a, pct)
    w = np.maximum(inst.a - tau, 0.0)
    tie = 1e-6 * np.arange(inst.T)
    x = np.zeros((inst.n, inst.T))
    for i in range(inst.n):
        x[i] = solve_sep_lp(w + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def jitter(inst: Instance, sigma=0.10, seed=0, **kw):
    """B5.  Independent per-operator dispersion on the common signal -- the
    cheapest decorrelation heuristic, and the direct empirical probe of
    Theorem 2."""
    rng = np.random.default_rng(seed)
    S = inst.a[None, :] * (1.0 + sigma * rng.standard_normal((inst.n, inst.T)))
    S = np.maximum(S, 1e-6)
    return nash(inst, signal=S, **kw)[0]


def proportional_cap(inst: Instance, declare=None, slack=1.5, **kw):
    """B6.  A disclosure-based coordinator: compute the aggregate optimum, hand
    each operator a per-slot quota pro rata to its declared demand, then let
    operators optimise inside the quota.  `declare` lets an operator inflate its
    declaration, which is the manipulation the paper mentions."""
    from .core import Feasible
    xp = _planner_fast(inst)
    ystar = xp.sum(axis=0)
    E = np.array([Xi.E for Xi in inst.X])
    d = E.copy() if declare is None else np.asarray(declare, float)
    share = d / d.sum()
    inst2 = Instance(a=inst.a, m=inst.m, b=inst.b,
                     X=[Feasible(E=inst.X[i].E,
                                 u=np.minimum(inst.X[i].u,
                                              np.maximum(slack * share[i] * ystar,
                                                         inst.X[i].E / inst.T * 1e-3)),
                                 R=inst.X[i].R) for i in range(inst.n)],
                     psi=inst.psi, lam=inst.lam, arrival=inst.arrival)
    try:
        return nash(inst2, **kw)[0]
    except ValueError:
        return xp                    # quota made the set infeasible


def planner_aggregate(inst: Instance):
    """The planner's optimal AGGREGATE profile y*.

    With no deferral penalty C depends on x only through y, and the Minkowski
    sum of the operators' feasible sets is itself a polytope of the same form
    (these are polymatroid base polytopes; verified exact to machine precision
    in tests/test_planner_reduction.py).  So the planner collapses to a single
    separable QP in y.
    """
    from .core import Feasible
    agg = Feasible(E=sum(X.E for X in inst.X),
                   u=sum(X.u for X in inst.X),
                   R=(sum(X.R for X in inst.X)
                      if inst.X[0].R is not None else None))
    return solve_sep_qp(inst.m, inst.b, agg)


def planner_value(inst: Instance):
    """C(x*), exactly.

    Returned as a VALUE rather than a profile: with lam = 0 every feasible
    decomposition of y* attains it, and constructing one by per-operator
    projection is wrong -- projection does not preserve the aggregate, which
    silently perturbs the normaliser.  (That bug made SHADE appear to beat the
    planner by 5e-5 before it was caught.)
    """
    if np.allclose(inst.lam, 0.0):
        y = planner_aggregate(inst)
        return float(np.sum(inst.m * y + inst.b * y ** 2))
    return inst.social(planner(inst, tol=1e-9)[0])


def _planner_fast(inst: Instance):
    """A feasible planner PROFILE.  Only needed where a profile is required
    (e.g. the quota baseline); ratios should use planner_value."""
    if np.allclose(inst.lam, 0.0):
        y = planner_aggregate(inst)
        # decompose y greedily: give each operator as much as its set allows
        x = np.zeros((inst.n, inst.T))
        rem = y.copy()
        for i in np.argsort([-X.E for X in inst.X]):
            Xi = inst.X[i]
            xi = solve_sep_qp(-2.0 * np.minimum(rem, Xi.u) - 1e-12,
                              np.ones(inst.T), Xi)
            xi = np.minimum(xi, rem)
            deficit = Xi.E - xi.sum()
            if deficit > 1e-9:            # top up where headroom remains
                room = np.minimum(Xi.u - xi, rem - xi)
                room = np.maximum(room, 0.0)
                if room.sum() > 1e-12:
                    xi = xi + room * min(1.0, deficit / room.sum())
            x[i] = xi
            rem = np.maximum(rem - xi, 0.0)
        return x
    return planner(inst, tol=1e-9)[0]


REGISTRY = {
    "carbon_agnostic": lambda inst, **kw: carbon_agnostic(inst),
    "naive":           lambda inst, **kw: naive(inst),
    "threshold":       lambda inst, **kw: threshold(inst),
    "nash":            lambda inst, **kw: nash(inst, **kw)[0],
    "jitter":          lambda inst, **kw: jitter(inst, **kw),
    "mef_static":      lambda inst, **kw: mef_static(inst),
    "mef_responsive":  lambda inst, **kw: mef_signal_equilibrium(inst)[0],
    "prop_cap":        lambda inst, **kw: proportional_cap(inst, **kw),
    "shade":           lambda inst, **kw: shade(inst, **kw)[0],
    "planner":         lambda inst, **kw: _planner_fast(inst),
}

"""Synthetic instance generators.

IMPORTANT / INTEGRITY
---------------------
Nothing in this file is measured.  These are *constructed* instances whose
shapes are chosen to be physically plausible, used for (a) validating the
solver and the theorems and (b) exercising the pipeline end to end.  They are
NOT a substitute for E1's dispatch-response calibration, and any number they
produce is tagged SYNTHETIC in the results files.

The one dial that matters and has no analogue in a real feed is `kappa`, the
congestion strength: the ratio, at the reference flexible share, of the
congestion increment b_t * y_peak to the accounting wedge (m_t - a_t).  Every
headline quantity is reported against a sweep in kappa, because it is the
parameter a reviewer will ask about and the one E1 must eventually pin down.
"""

from __future__ import annotations
import numpy as np
from .core import Feasible, Instance


def diurnal_aef(T, trough_depth=0.52, peak_hour=19.5, trough_hour=13.0,
                base=340.0):
    """A day of published average carbon intensity with a midday solar trough
    and an evening ramp.  Shape only; magnitudes in gCO2/kWh."""
    h = np.arange(T) * 24.0 / T
    trough = np.exp(-0.5 * ((h - trough_hour) / 2.6) ** 2)
    evening = np.exp(-0.5 * ((h - peak_hour) / 2.9) ** 2)
    night = 0.35 * np.exp(-0.5 * ((h - 3.0) / 3.5) ** 2)
    a = base * (1.0 - trough_depth * trough + 0.30 * evening + 0.10 * night)
    return a


def eta_profile(T, eta_max=0.62, eta_min=0.14, trough_hour=13.0):
    """Unpriced share eta_t = 1 - AEF/MEF.  Peaks in the solar trough: that is
    exactly the regime where a low average sits on a thermal margin, which is
    the paper's whole thesis."""
    h = np.arange(T) * 24.0 / T
    w = np.exp(-0.5 * ((h - trough_hour) / 3.4) ** 2)
    return eta_min + (eta_max - eta_min) * w


def make_instance(n=32, T=96, seed=0, kappa=1.0, eta_max=0.62, eta_min=0.14,
                  flex_scale=1.0, cap_frac=0.35, slack_hours=(2.0, 24.0),
                  lam=0.0, split="lognormal", stair=True, label=""):
    """Build one synthetic (region, day) instance.

    n          operators
    T          slots (96 = 15 minutes)
    kappa      congestion strength (see module docstring)
    flex_scale multiplier on aggregate flexible energy (the 1x..16x sweep)
    cap_frac   per-slot envelope as a fraction of an operator's daily energy
    lam        deferral penalty weight (0 = pure carbon objective)
    split      'equal' | 'lognormal' | 'heavytail' -- operator size distribution
    """
    rng = np.random.default_rng(seed)
    a = diurnal_aef(T)
    eta = eta_profile(T, eta_max=eta_max, eta_min=eta_min)
    m = a / (1.0 - eta)

    # operator sizes
    if split == "equal":
        w = np.ones(n)
    elif split == "lognormal":
        w = rng.lognormal(0.0, 0.6, n)
    elif split == "heavytail":
        w = rng.pareto(1.6, n) + 0.3
    else:
        raise ValueError(split)
    w = w / w.sum()

    # aggregate flexible energy, in MWh over the horizon.  The absolute scale is
    # arbitrary and cancels in every ratio we report; flex_scale is what varies.
    Y_total = 1000.0 * flex_scale
    E = w * Y_total

    # curvature: pin kappa at the reference concentration (all load in the
    # cleanest 10% of slots)
    # kappa is pinned at the REFERENCE flexible share, not the scaled one.
    # Normalising against Y_total instead would make b proportional to
    # 1/flex_scale, so the congestion term b*y would be invariant and the
    # flexible-share sweep would cancel itself out -- which is exactly the bug
    # that made an earlier version report identical gaps at 1x and 16x.
    k_concentrate = max(1, int(0.10 * T))
    y_peak = 1000.0 / k_concentrate
    wedge = m - a
    shape = 0.5 + 0.5 * (eta / eta.max())          # steeper where eta is high
    b = kappa * np.mean(wedge) / y_peak * shape

    X, psi, arrival = [], np.zeros((n, T)), np.zeros((n, T))
    for i in range(n):
        u = np.full(T, cap_frac * E[i])
        R = None
        if stair:
            # deadline slack sampled per operator; the staircase requires a
            # fraction of the work done by progressively later slots
            slack = rng.uniform(*slack_hours)
            done_by = np.clip((np.arange(T) * 24.0 / T) / slack, 0, 1)
            R = 0.85 * E[i] * done_by ** 1.7
            R = np.minimum(R, np.cumsum(u) * 0.9)
            R = np.maximum.accumulate(R)
        X.append(Feasible(E=E[i], u=u, R=R))

        # deferral penalty: grows with distance from the operator's arrival peak
        h = np.arange(T) * 24.0 / T
        h0 = rng.uniform(6.0, 18.0)
        psi[i] = np.abs(h - h0) / 24.0
        arr = np.exp(-0.5 * ((h - h0) / 2.0) ** 2)
        arrival[i] = E[i] * arr / arr.sum()

    return Instance(a=a, m=m, b=b, X=X, psi=psi,
                    lam=np.full(n, float(lam)), arrival=arrival,
                    label=label or f"synthetic n={n} kappa={kappa} x{flex_scale}")


def two_slot(n, D, delta, beta, cap=None):
    """The exact two-slot instance behind the corrected Theorem 2 proposition.

    Two slots with a_1 = 0, a_2 = delta (so the gap is delta), common curvature
    beta, aggregate deferrable energy D split equally over n operators, and no
    deferral penalty.  With cap set to D/n every operator must place all of its
    energy in one slot, which is the corner that Assumption 1 describes.
    """
    a = np.array([0.0, float(delta)])
    m = a.copy()                      # no accounting wedge: isolate congestion
    b = np.array([float(beta), float(beta)])
    Ei = D / n
    X = [Feasible(E=Ei, u=np.full(2, Ei if cap is None else cap), R=None)
         for _ in range(n)]
    return Instance(a=a, m=m, b=b, X=X, psi=np.zeros((n, 2)),
                    lam=np.zeros(n), label=f"two-slot n={n}")

"""SHADE, in the corrected form of REVIEW.md B3, plus the two readings of the
published Algorithm 1 that motivated the correction, plus the privacy layer.

Corrected operator step
-----------------------
Two things must both hold for Theorem 3 to go through, and the published
algorithm states neither:

  (i)  the operator RETAINS its own congestion term, writing
       y_t = yhat_{-i,t} + x_it and differentiating it;
  (ii) the adder pi_it is an exogenous CONSTANT carried from round k-1.

Then the perceived marginal is

    a + b*yhat_{-i} + b*x_i   (own effect)
  + (m - a) + b*yhat_{-i}     (adder)
  = m + 2*b*yhat_{-i} + b*x_i

and at a fixed point with exact aggregation yhat_{-i} = y - x_i, so this is
m + 2*b*y - b*x_i + b*x_i = m + 2*b*y = dC/dx_it exactly.  The local problem is
a convex QP with curvature b_t, so the water-filling claim survives.

The two published readings both fail:
  variant "lp"      pi lagged but own congestion dropped -> curvature 0, an LP,
                    and the fixed point under-corrects by b*x_i.
  variant "concave" x_it inside pi treated as the decision variable -> the
                    objective carries -b*x^2 and is concave, so it is not a QP
                    at all and its minimisers are vertices.
"""

from __future__ import annotations
import numpy as np
from .core import Instance, solve_sep_qp, solve_sep_lp


# --------------------------------------------------------------------------
# privacy layer
# --------------------------------------------------------------------------

def zcdp_rho(eps, delta=1e-5):
    """rho such that rho-zCDP implies (eps, delta)-DP, via
    eps = rho + 2*sqrt(rho*log(1/delta))."""
    L = np.log(1.0 / delta)
    # solve r^2 + 2*sqrt(L)*r - eps = 0 in r = sqrt(rho)
    r = (-2 * np.sqrt(L) + np.sqrt(4 * L + 4 * eps)) / 2.0
    return float(max(r, 0.0) ** 2)


def noise_scale(mode, Ebar, eps, K, delta=1e-5):
    """Per-round, per-coordinate noise *standard deviation*, and a sampler.

    'laplace'  basic composition: each round gets eps/K, so the Laplace scale
               is Ebar*K/eps and the sd is sqrt(2)*Ebar*K/eps.  This is what
               Theorem 4 and Figure 2 use (Algorithm 1 inverts it -- B4).
    'gaussian' rho-zCDP composition over K rounds: sd = Ebar*sqrt(K/(2*rho)).
    'none'     exact aggregation.
    """
    if mode == "none" or eps is None or not np.isfinite(eps):
        return 0.0, (lambda rng, T: np.zeros(T))
    if mode == "laplace":
        b = Ebar * K / eps
        return np.sqrt(2.0) * b, (lambda rng, T: rng.laplace(0.0, b, T))
    if mode == "gaussian":
        rho = zcdp_rho(eps, delta)
        s = Ebar * np.sqrt(K / (2.0 * rho))
        return s, (lambda rng, T: rng.normal(0.0, s, T))
    raise ValueError(mode)


# --------------------------------------------------------------------------
# SHADE
# --------------------------------------------------------------------------

def shade(inst: Instance, eps=2.0, K=25, gamma=0.55, theta=None, seed=0,
          dp="laplace", delta=1e-5, variant="corrected",
          no_adder=False, no_own_sub=False, no_damping=False,
          adder_slots=None, return_trace=False, accel="none",
          accel_depth=5, accel_clip=50.0):
    """Run Algorithm 1 (corrected) and return the committed profile.

    Ablation switches mirror Table 4:
      no_adder     operators see the broadcast aggregate but no externality term
      no_own_sub   the adder is not reduced by the operator's own contribution
      no_damping   gamma = 1
      adder_slots  coarsen the adder to blocks of this many slots (hourly = 4)
    """
    rng = np.random.default_rng(seed)
    T, n = inst.T, inst.n
    Ebar = max(Xi.E for Xi in inst.X)
    if no_damping:
        gamma = 1.0
    if theta is None:
        theta = 5e-4 * sum(Xi.E for Xi in inst.X)
    sd, sample = noise_scale(dp, Ebar, eps, K, delta)

    x = inst.feasible_start()
    yhat = x.sum(axis=0).copy()
    prev = yhat.copy()
    hist_y, hist_f = [], []
    rounds = K
    trace = []

    for k in range(1, K + 1):
        for i in range(n):
            y_mi = yhat - x[i] if not no_own_sub else yhat
            if no_adder:
                pi = np.zeros(T)
            else:
                pi = (inst.m - inst.a) + inst.b * y_mi
                if adder_slots and adder_slots > 1:
                    g = adder_slots
                    pad = (-T) % g
                    pp = np.concatenate([pi, np.full(pad, pi[-1])])
                    pi = np.repeat(pp.reshape(-1, g).mean(axis=1), g)[:T]

            if variant == "corrected":
                # own congestion retained -> curvature b, linear coeff below
                q = inst.a + inst.b * (yhat - x[i]) + pi + inst.lam[i] * inst.psi[i]
                x[i] = solve_sep_qp(q, inst.b, inst.X[i])
            elif variant == "lp":
                # published reading: yhat exogenous, own congestion dropped
                q = inst.a + inst.b * yhat + pi + inst.lam[i] * inst.psi[i]
                x[i] = solve_sep_lp(q, inst.X[i])
            else:
                raise ValueError(variant)

        y = x.sum(axis=0)
        noised = y + sample(rng, T)
        f = noised - prev                      # residual G(yhat) - yhat

        hist_y.append(prev.copy())
        hist_f.append(f.copy())
        if len(hist_f) > accel_depth + 1:
            hist_y.pop(0)
            hist_f.pop(0)

        if accel == "anderson" and len(hist_f) >= 2:
            # Anderson acceleration on the clearinghouse update.  A single
            # damping constant cannot converge quickly here because the round
            # map's eigenvalues are spread across slots (each slot has its own
            # b_t) and coupled through each operator's energy constraint, so a
            # per-coordinate step is meaningless; Anderson fits a global
            # least-squares combination of past residuals instead.  It consumes
            # only aggregates the clearinghouse already publishes, so it costs
            # no extra privacy budget and discloses nothing further.
            F = np.stack(hist_f, axis=1)
            Y = np.stack(hist_y, axis=1)
            dF, dY = np.diff(F, axis=1), np.diff(Y, axis=1)
            try:
                coef, *_ = np.linalg.lstsq(dF, f, rcond=None)
                cand = (prev + gamma * f) - (dY + gamma * dF) @ coef
                if not np.all(np.isfinite(cand)):
                    raise np.linalg.LinAlgError
                # safeguard against an over-long extrapolation
                step = cand - prev
                lim = accel_clip * max(float(np.max(np.abs(gamma * f))), theta)
                new = prev + np.clip(step, -lim, lim)
            except np.linalg.LinAlgError:
                new = prev + gamma * f
        else:
            new = prev + gamma * f

        yhat = np.maximum(new, 0.0)
        trace.append(inst.social(x))
        if np.max(np.abs(yhat - prev)) <= theta:
            rounds = k
            prev = yhat
            break
        prev = yhat

    return (x, rounds, trace) if return_trace else (x, rounds)


def settlement(inst: Instance, x_cmt, x_real, yhat, signed=False):
    """Eq. (14).  `signed=False` is the published rule, which charges an
    operator for reducing its load in a dirty slot as well as for increasing it
    (REVIEW.md B6).  `signed=True` is the repair."""
    d = x_real - x_cmt
    price = inst.b * yhat + (inst.m - inst.a)
    return np.sum(price * (d if signed else np.abs(d)), axis=1)

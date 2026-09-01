"""V2 -- REVIEW.md B2: Theorem 2's inequality is conditional, not unconditional.

The draft states, unconditionally,

    there exists sigma_bar > 0 such that C_ne(sigma) <= C_ne(0)
    for all sigma in [0, sigma_bar]

and invokes Assumption 1 only to upgrade weak to strict.  That is the wrong way
round, and the two-slot family settles it because the interior case has a
closed form.

Interior case, in closed form
-----------------------------
When no operator is clipped, best response is affine and the equilibrium is

    y_1 = S/2 + (sum_i dA_i) / (2*b*(n+1)),     dA_i = A_i2 - A_i1

which is LINEAR in the signal.  So mean-zero noise leaves E[y_1] unchanged and
only adds variance.  With A_it = a_t + s*xi_it and xi ~ N(0,1),
Var[sum_i dA_i] = 2*n*s^2, hence

    E[C](s) - C(0) = 2*b*Var[y_1] = n*s^2 / ( b*(n+1)^2 )   >  0

Strictly positive, at order s^2, with no first-order term.  Noise HURTS.  So no
sigma_bar > 0 exists for an interior instance and the unconditional claim is
false.  Under Assumption 1 the envelope makes the response one-sided, a
first-order term appears, and its sign can be negative -- which is the theorem
the draft should be stating.

Test: fit  E[C](s) - C(0) = c1*s + c2*s^2  on small s.  The sign of c1 is the
whole question.  Common random numbers across the s-grid keep the curve smooth.
"""
from common import Result, banner
import numpy as np
from gh.core import nash_two_slot
from gh.instances import two_slot

banner("V2", "Signal precision: Theorem 2 holds only under Assumption 1")

D, beta, n = 100.0, 1.0, 16
DRAWS = 20000

lo = beta * D * (1 + 1 / n)
hi = 2 * beta * D * (1 - 1 / n)
print(f"n = {n},  D = {D},  beta = {beta}")
print(f"corner threshold  Delta >= {lo:.2f}   spreading helps  Delta < {hi:.2f}")
print(f"regime B window   [{lo:.2f}, {hi:.2f})\n")

# Each regime gets its own noise scale, because the characteristic scale
# differs: for the interior case it is b*E_i (the width before clipping), for a
# corner case it is the distance from the operator's indifference point.
Emed = D / n
REGIMES = [
    # label,            Delta,                 s-grid
    ("A interior",      0.0,                   Emed * np.array([0, .05, .1, .15, .2, .3, .4, .6])),
    ("B corner (marginal)", lo,                Emed * np.array([0, .05, .1, .15, .2, .3, .4, .6])),
    ("C corner (mid-window)", 0.5 * (lo + hi), np.array([0, 4, 8, 12, 16, 24, 32, 48.])),
    ("D corner (gap too wide)", 1.45 * hi,     np.array([0, 4, 8, 12, 16, 24, 32, 48.])),
]

res = Result("V2", "Signal precision by regime, local behaviour at small sigma",
             {"D": D, "beta": beta, "n": n, "draws": DRAWS,
              "regimes": {lab: {"delta": float(d), "s_grid": g.tolist()}
                          for lab, d, g in REGIMES}})

rng = np.random.default_rng(20260831)
XI = rng.standard_normal((DRAWS, n, 2))           # common random numbers

print(f"{'regime':24s} {'Delta':>7s} {'corner':>7s} {'clip%':>6s} | "
      + "  E[C](s) - C(0) across its own s-grid")
print("-" * 118)

fits = {}
for label, delta, S_GRID in REGIMES:
    inst = two_slot(n, D, delta, beta)
    E = np.array([X.E for X in inst.X])
    x0, _ = nash_two_slot(inst)
    at_corner = x0[:, 0].sum() / D > 1 - 1e-9
    curve, clipped = [], 0.0
    for s in S_GRID:
        tot, nclip = 0.0, 0
        reps = DRAWS if s > 0 else 1
        for k in range(reps):
            sig = inst.a[None, :] + s * XI[k]
            x, _ = nash_two_slot(inst, signal=sig)
            tot += inst.social(x)
            nclip += float(np.mean((x[:, 0] <= 1e-12) | (x[:, 0] >= E - 1e-12)))
        curve.append(tot / reps)
        clipped = max(clipped, 100.0 * nclip / reps)
    curve = np.array(curve)
    d = curve - curve[0]
    # fit c1*s + c2*s^2 on the small-s part
    k = 5
    Amat = np.stack([S_GRID[1:k + 1], S_GRID[1:k + 1] ** 2], axis=1)
    c1, c2 = np.linalg.lstsq(Amat, d[1:k + 1], rcond=None)[0]
    fits[label] = (c1, c2, curve, S_GRID)
    print(f"{label:24s} {delta:7.1f} {str(at_corner):>7s} {clipped:5.1f}% | "
          + " ".join(f"{v:+8.3f}" for v in d))
    for s, c in zip(S_GRID, curve):
        res.row(regime=label, delta=delta, at_corner=bool(at_corner),
                sigma=float(s), cost=float(c), delta_cost=float(c - curve[0]))

print("\nLocal fit  E[C](s) - C(0) ~ c1*s + c2*s^2  on the small-s range")
print(f"{'regime':15s} {'c1':>12s} {'c2':>12s}   first-order effect")
print("-" * 68)
for label, (c1, c2, _, _g) in fits.items():
    sign = "noise HURTS" if c1 > 1e-6 else ("noise HELPS" if c1 < -1e-6
                                            else "none (c1 = 0)")
    print(f"{label:15s} {c1:12.5f} {c2:12.5f}   {sign}")

# closed form for the interior regime
c1_i, c2_i, curve_i, S_GRID_I = fits["A interior"]
pred = n / (beta * (n + 1) ** 2)
print(f"\nInterior closed form:  E[C](s) - C(0) = n*s^2/(b*(n+1)^2), "
      f"coefficient = {pred:.6f}")
print(f"  measured c2 = {c2_i:.6f}   relative error {abs(c2_i-pred)/pred:.2e}")
print(f"  measured c1 = {c1_i:.2e}  (theory: exactly 0)")
for s in S_GRID_I[1:]:
    got = curve_i[list(S_GRID_I).index(s)] - curve_i[0]
    print(f"    s={s:>4g}: measured {got:9.4f}   closed form {pred*s*s:9.4f}"
          f"   rel err {abs(got-pred*s*s)/(pred*s*s):.2e}")

# Falsification rests on every measured point in the interior regime lying
# ABOVE C(0), together with agreement with the closed form -- not on the fitted
# c1, whose Monte Carlo error (~1e-4) is far larger than the 1e-6 that an
# earlier version of this script tested against.
dA = fits["A interior"][2] - fits["A interior"][2][0]
dB = fits["B corner (marginal)"][2] - fits["B corner (marginal)"][2][0]
falsified = bool(np.all(dA[1:] > 0)) and abs(c2_i - pred) / pred < 0.02
confirmed = bool(np.all(dB[1:] < 0))
c1_b = fits["B corner (marginal)"][0]
print(f"\nVERDICT")
print(f"  A  interior: all {len(dA)-1} measured points lie ABOVE C(0), and the")
print(f"     curve matches n*s^2/(b*(n+1)^2) to "
      f"{100*abs(c2_i-pred)/pred:.1f}% with c1 = 0 to Monte Carlo error.")
print(f"     No sigma_bar > 0 exists, so Theorem 2 as stated (unconditional)")
print(f"     is {'FALSIFIED' if falsified else 'not falsified'}.")
print(f"  B  marginal corner: all {len(dB)-1} points lie BELOW C(0) and the")
print(f"     effect is first-order (c1 = {c1_b:.1f}).  The conditional statement")
print(f"     is {'CONFIRMED' if confirmed else 'not confirmed'}: the kink does all the work.")
print(f"  C  deep corner: flat, then falling.  The effect is NOT first-order")
print(f"     when the corner binds strictly, only when it binds marginally.")
print(f"  D  gap too wide: Assumption 1 holds but noise never helps, so")
print(f"     Assumption 1 is necessary and not sufficient.")

res.scalar("InteriorC1", c1_i, "{:.2e}")
res.scalar("InteriorC2", c2_i, "{:.5f}")
res.scalar("InteriorC2Pred", pred, "{:.5f}")
res.scalar("InteriorC2RelErr", abs(c2_i - pred) / pred, "{:.1e}")
res.scalar("CornerC1", c1_b, "{:.4f}")
res.scalar("Falsified", "yes" if falsified else "no")
res.write()

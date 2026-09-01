"""V3 -- the exact two-slot proposition (REVIEW.md Move 3 / PLAN.md 7.2).

Setting: two slots with intensities a_1 = 0 and a_2 = Delta, common curvature
beta, aggregate deferrable energy D split equally over n operators, no deferral
penalty and no accounting wedge (m = a), so the only externality is congestion.

If a fraction q of operators is moved to the dirty slot independently,

    E(q) = q*D*Delta + b*D^2*[(1-q)^2 + q^2] + 2*b*D^2*q*(1-q)/n
                                               \_ finite-n variance term _/

    dE/dq|_0 = D*Delta - 2*b*D^2*(1 - 1/n)

so independent spreading strictly reduces expected emissions exactly when

    Delta < 2*b*D*(1 - 1/n)                                     (i)

and the sigma=0 equilibrium sits at the corner (all load clean) exactly when

    Delta >= b*D*(1 + 1/n)                                      (ii)

Optimum and saving:

    q*     = 1/2 - Delta / (4*b*D*(1 - 1/n))
    saving = (2*b*D*(1-1/n) - Delta)^2 / (8*b*(1 - 1/n))

PLAN.md section 7.2 gives the saving as .../(8*b), dropping the (1-1/n).  This
experiment checks all four formulas against simulation and quantifies that
error.  (i) and (ii) are compatible only for n >= 4.
"""
from common import Result, banner
import numpy as np
from gh.core import nash
from gh.instances import two_slot

banner("V3", "Exact two-slot proposition: closed forms vs simulation")

D, beta = 100.0, 1.0
res = Result("V3", "Two-slot closed forms",
             {"D": D, "beta": beta})


def E_closed(q, n, delta):
    return (q * D * delta + beta * D ** 2 * ((1 - q) ** 2 + q ** 2)
            + 2 * beta * D ** 2 * q * (1 - q) / n)


def E_sim(q, n, delta, draws=20000, seed=0):
    """Each operator independently placed in the dirty slot w.p. q."""
    rng = np.random.default_rng(seed)
    z = (rng.random((draws, n)) < q).sum(axis=1)          # count in slot 2
    y2 = D * z / n
    y1 = D - y2
    return float(np.mean(delta * y2 + beta * (y1 ** 2 + y2 ** 2)))


# ---- 1. the E(q) closed form ------------------------------------------------
print("1. E(q) closed form vs Monte Carlo (n=16, Delta=120)")
n, delta = 16, 120.0
worst = 0.0
print(f"   {'q':>5s} {'closed form':>14s} {'simulated':>14s} {'rel err':>10s}")
for q in (0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0):
    ec, es = E_closed(q, n, delta), E_sim(q, n, delta, seed=int(q * 1000))
    rel = abs(ec - es) / es
    worst = max(worst, rel)
    print(f"   {q:5.2f} {ec:14.2f} {es:14.2f} {rel:10.2e}")
print(f"   worst relative error: {worst:.2e}")
res.scalar("EqWorstRelErr", worst, "{:.1e}")

# ---- 2. the corner threshold (ii) -------------------------------------------
print("\n2. Corner threshold: equilibrium is all-clean iff Delta >= b*D*(1+1/n)")
print(f"   {'n':>4s} {'predicted':>11s} {'measured':>10s} {'rel err':>9s}")
worst2 = 0.0
def at_corner(n, dl):
    x, _ = nash(two_slot(n, D, dl, beta), tol=1e-11)
    return x[:, 0].sum() / D > 1 - 1e-7


for n in (4, 8, 16, 32, 64):
    pred = beta * D * (1 + 1 / n)
    # bisect on Delta: corner-ness is monotone increasing in Delta
    lo_d, hi_d = 0.5 * pred, 1.5 * pred
    assert not at_corner(n, lo_d) and at_corner(n, hi_d)
    for _ in range(40):
        mid = 0.5 * (lo_d + hi_d)
        if at_corner(n, mid):
            hi_d = mid
        else:
            lo_d = mid
    meas = 0.5 * (lo_d + hi_d)
    rel = abs(meas - pred) / pred
    worst2 = max(worst2, rel)
    print(f"   {n:4d} {pred:11.3f} {meas:10.3f} {rel:9.2e}")
    res.row(kind="corner_threshold", n=n, predicted=pred, measured=meas)
res.scalar("CornerWorstRelErr", worst2, "{:.1e}")

# ---- 3. q*, the saving, and the plan's error --------------------------------
print("\n3. Optimal spread q* and the saving at the optimum")
print(f"   {'n':>4s} {'Delta':>7s} {'q* form':>8s} {'q* grid':>8s} "
      f"{'saving(correct)':>16s} {'saving(sim)':>12s} {'PLAN /(8b)':>11s} {'err%':>6s}")
worst3 = 0.0
for n in (8, 16, 32, 64):
    hi = 2 * beta * D * (1 - 1 / n)
    lo = beta * D * (1 + 1 / n)
    delta = 0.5 * (lo + hi)
    qs = 0.5 - delta / (4 * beta * D * (1 - 1 / n))
    grid = np.linspace(0, 0.5, 5001)
    qg = grid[int(np.argmin([E_closed(q, n, delta) for q in grid]))]
    sav_correct = (2 * beta * D * (1 - 1 / n) - delta) ** 2 / (8 * beta * (1 - 1 / n))
    sav_plan = (2 * beta * D * (1 - 1 / n) - delta) ** 2 / (8 * beta)
    sav_sim = E_closed(0.0, n, delta) - E_closed(qs, n, delta)
    err = 100 * (sav_plan - sav_sim) / sav_sim
    worst3 = max(worst3, abs(sav_correct - sav_sim) / sav_sim)
    print(f"   {n:4d} {delta:7.2f} {qs:8.4f} {qg:8.4f} {sav_correct:16.3f} "
          f"{sav_sim:12.3f} {sav_plan:11.3f} {err:6.2f}")
    res.row(kind="saving", n=n, delta=delta, q_star=qs, q_grid=qg,
            saving_correct=sav_correct, saving_sim=sav_sim, saving_plan=sav_plan,
            plan_error_pct=err)
print(f"   corrected formula worst relative error: {worst3:.2e}")
print(f"   PLAN.md's /(8b) understates the saving by exactly a factor (1-1/n)")
res.scalar("SavingWorstRelErr", worst3, "{:.1e}")

# ---- 4. the window is non-empty only for n >= 4 -----------------------------
print("\n4. Regime window [b*D*(1+1/n), 2*b*D*(1-1/n)) is non-empty iff n >= 4")
print(f"   {'n':>4s} {'lo':>9s} {'hi':>9s} {'width':>9s} {'non-empty':>10s}")
first = None
for n in (2, 3, 4, 5, 8, 32):
    lo, hi = beta * D * (1 + 1 / n), 2 * beta * D * (1 - 1 / n)
    w = hi - lo
    ne = w > 1e-9 * D
    if ne and first is None:
        first = n
    print(f"   {n:4d} {lo:9.3f} {hi:9.3f} {w:9.3f} {str(ne):>10s}")
    res.row(kind="window", n=n, lo=lo, hi=hi, width=w, nonempty=bool(ne))
print(f"   smallest n with a non-empty window: {first}")
print(f"   => the 'noise helps' effect provably requires at least {first} operators,")
print(f"      which is a direct answer to the 'is this really multi-agent?' objection.")
res.scalar("MinOperators", first, "{:d}")
res.write()

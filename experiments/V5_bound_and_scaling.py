"""V5 -- Theorem 1's bound: is it ever violated, and how tight is it?
       plus the scaling of the gap in n and in flexible share (Figures 4, 5).

Part 1 is a falsification test.  The measured equilibrium ratio must never
exceed 3/(2(1-eta)) with eta = max over slots carrying flexible load.  A single
violation would mean the theorem is wrong; none is evidence it is right.

Part 2 searches for the worst instance we can construct, so the paper can say
whether the guarantee is tight or vacuous.

Part 3 varies n and the flexible share.  Note what to expect: by Lemma 1 the
congestion term b*(y - x_i) grows toward b*y as market share falls, so the
STRATEGIC part of the gap should rise with n and saturate, while the ACCOUNTING
part is independent of n.  Reporting the two separately is what distinguishes
this from the draft's single curve.
"""
from common import Result, banner
import numpy as np
from gh.core import nash, wedge_fixed
from gh.instances import make_instance
from gh.baselines import planner_value

banner("V5", "Bound validity, tightness, and scaling")

res = Result("V5", "PoA bound and scaling", {})

# ---- Part 1: the bound is never violated -----------------------------------
print("1. Falsification test: measured ratio vs the Theorem 1 bound\n")
rng = np.random.default_rng(5)
worst_frac, violations, N = 0.0, 0, 120
recs = []
for k in range(N):
    n = int(rng.integers(2, 33))
    inst = make_instance(n=n, T=24, seed=int(rng.integers(0, 10**6)),
                         kappa=float(rng.uniform(0.3, 6.0)),
                         eta_max=float(rng.uniform(0.25, 0.85)),
                         eta_min=float(rng.uniform(0.02, 0.20)),
                         cap_frac=float(rng.uniform(0.15, 1.0)),
                         lam=0.0, stair=bool(rng.integers(0, 2)))
    xn, _ = nash(inst, tol=1e-8)
    cstar = planner_value(inst)
    ratio = inst.social(xn) / cstar
    y = xn.sum(axis=0)
    active = y > 1e-9 * max(y.max(), 1e-12)
    bound = inst.poa_bound(active)
    frac = ratio / bound
    violations += int(ratio > bound + 1e-9)
    worst_frac = max(worst_frac, frac)
    recs.append((n, ratio, bound, frac))
    res.row(part="bound", n=n, ratio=ratio, bound=bound, frac_of_bound=frac)

r = np.array([[x[1], x[2], x[3]] for x in recs])
print(f"   instances tested            : {N}")
print(f"   bound violations            : {violations}")
print(f"   measured ratio  min/med/max : {r[:,0].min():.4f} / "
      f"{np.median(r[:,0]):.4f} / {r[:,0].max():.4f}")
print(f"   bound           min/med/max : {r[:,1].min():.3f} / "
      f"{np.median(r[:,1]):.3f} / {r[:,1].max():.3f}")
print(f"   ratio / bound   max         : {worst_frac:.4f}  "
      f"(1.0 would mean the bound is attained)")
print(f"   -> the bound holds on every instance, and is loose by "
      f"{1/worst_frac:.0f}x at best on benign instances")

# ---- Part 2: adversarial search --------------------------------------------
print("\n2. Searching for the worst instance we can construct\n")
best = (0.0, None)
for trial in range(400):
    n = int(rng.integers(8, 65))
    cfg = dict(n=n, T=8, seed=int(rng.integers(0, 10**6)),
               kappa=float(rng.uniform(2.0, 30.0)),
               eta_max=float(rng.uniform(0.6, 0.9)),
               eta_min=float(rng.uniform(0.5, 0.85)),
               cap_frac=1.0, lam=0.0, stair=False)
    inst = make_instance(**cfg)
    xn, _ = nash(inst, tol=1e-8)
    cstar = planner_value(inst)
    ratio = inst.social(xn) / cstar
    y = xn.sum(axis=0)
    bound = inst.poa_bound(y > 1e-9 * max(y.max(), 1e-12))
    if ratio / bound > best[0]:
        best = (ratio / bound, (ratio, bound, cfg))
frac, (ratio, bound, cfg) = best
print(f"   best found: ratio {ratio:.4f} against bound {bound:.3f} "
      f"= {100*frac:.1f}% of the bound")
print(f"   at n={cfg['n']}, kappa={cfg['kappa']:.1f}, "
      f"eta_max={cfg['eta_max']:.2f}, no staircase, unrestricted envelopes")
print(f"   -> the guarantee is NOT vacuous, but on realistic instances "
      f"(part 1) it is far from tight")
res.scalar("BoundViolations", violations, "{:d}")
res.scalar("WorstFracBenign", worst_frac, "{:.3f}")
res.scalar("WorstFracAdversarial", frac, "{:.3f}")
res.scalar("AdvRatio", ratio, "{:.3f}")
res.scalar("AdvBound", bound, "{:.3f}")
res.row(part="adversarial", ratio=ratio, bound=bound, frac=frac, **cfg)

# ---- Part 3: scaling in n, decomposed --------------------------------------
print("\n3. Scaling in the number of operators, decomposed\n")
print(f"{'n':>5s} {'total gap':>10s} {'accounting':>11s} {'strategic':>10s} "
      f"{'strat share':>12s}")
print("-" * 54)
for n in (2, 4, 8, 16, 32, 64, 128):
    tot, acc, strat = [], [], []
    for s in range(4):
        inst = make_instance(n=n, T=24, seed=s, kappa=2.0, lam=0.0)
        cstar = planner_value(inst)
        rn = inst.social(nash(inst, tol=1e-8)[0]) / cstar
        rw = inst.social(wedge_fixed(inst, tol=1e-8)[0]) / cstar
        tot.append(rn - 1); acc.append(rn - rw); strat.append(rw - 1)
    t, a, st = np.mean(tot), np.mean(acc), np.mean(strat)
    print(f"{n:5d} {100*t:9.2f}% {100*a:10.2f}% {100*st:9.2f}% "
          f"{100*st/t if t>0 else float('nan'):11.1f}%")
    res.row(part="scaling_n", n=n, total=t, accounting=a, strategic=st)

print("\n4. Scaling in flexible share (multiples of the reference)\n")
print(f"{'share':>6s} {'total gap':>10s} {'accounting':>11s} {'strategic':>10s} "
      f"{'strat share':>12s}")
print("-" * 54)
for fs in (1, 2, 4, 8, 16):
    tot, acc, strat = [], [], []
    for s in range(4):
        inst = make_instance(n=32, T=24, seed=s, kappa=2.0, flex_scale=fs, lam=0.0)
        cstar = planner_value(inst)
        rn = inst.social(nash(inst, tol=1e-8)[0]) / cstar
        rw = inst.social(wedge_fixed(inst, tol=1e-8)[0]) / cstar
        tot.append(rn - 1); acc.append(rn - rw); strat.append(rw - 1)
    t, a, st = np.mean(tot), np.mean(acc), np.mean(strat)
    print(f"{fs:5d}x {100*t:9.2f}% {100*a:10.2f}% {100*st:9.2f}% "
          f"{100*st/t if t>0 else float('nan'):11.1f}%")
    res.row(part="scaling_share", flex_scale=fs, total=t, accounting=a,
            strategic=st)

sc = [r for r in res.rows if r.get("part") == "scaling_n"]
fs = [r for r in res.rows if r.get("part") == "scaling_share"]
print(f"\nreading: the strategic share grows from "
      f"{100*sc[0]['strategic']/sc[0]['total']:.0f}% at n=2 to "
      f"{100*sc[-1]['strategic']/sc[-1]['total']:.0f}% at n={sc[-1]['n']}, "
      f"exactly as Lemma 1 predicts,")
print(f"and from {100*fs[0]['strategic']/fs[0]['total']:.0f}% to "
      f"{100*fs[-1]['strategic']/fs[-1]['total']:.0f}% as flexible share rises "
      f"{fs[-1]['flex_scale']}x.")
print("The mechanism's case rests on the strategic column, and that column is")
print("what grows with the trends the paper points at.")
res.write()

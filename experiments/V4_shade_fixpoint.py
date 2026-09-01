"""V4 -- REVIEW.md B3/B4: what Algorithm 1 has to say to make Theorem 3 true,
and what damping constant makes it converge.

Part 1: the variant taxonomy
----------------------------
Write the broadcast aggregate as yhat and the adder as pi_it.  Two independent
binary choices are left implicit by the published Algorithm 1:

  (a) does the operator RETAIN its own congestion term, i.e. write
      y_t = yhat_{-i,t} + x_it and differentiate it?  (curvature b_t vs 0)
  (b) does the adder subtract the operator's own contribution,
      pi = (m-a) + b*(yhat - x_i), or not, pi = (m-a) + b*yhat?

Perceived marginal at a fixed point (where yhat = y), against the social
marginal m + 2*b*y:

  own retained + own subtracted   m + 2*b*y            CORRECT
  own dropped  + NOT subtracted   m + 2*b*y            CORRECT (an LP)
  own dropped  + own subtracted   m + 2*b*y - b*x_i    under-corrects  <- as published
  own retained + NOT subtracted   m + 2*b*y + b*x_i    over-corrects

So there are exactly two correct designs, and Algorithm 1 as literally written
is neither.  Note the consequence for the paper's ablation table: dropping the
own-load subtraction is only an error if the operator retains its own
congestion term.  On the exogenous-yhat reading it is not an error at all --
the subtraction is unnecessary rather than essential.

Part 2: the damping constant
----------------------------
Operators respond in parallel, so the round map on the aggregate has gain of
order -n and the fixed damping gamma = 0.55 that the paper specifies is
unstable for all but the smallest n.  This maps the stability boundary.
"""
from common import Result, banner
import numpy as np
from gh.instances import make_instance
from gh.mech import shade
from gh.baselines import planner_value

banner("V4", "Algorithm 1: correct variants and the stability of gamma")

res = Result("V4", "SHADE variant taxonomy and damping stability", {})

VARIANTS = [
    ("own retained + subtracted",  dict(variant="corrected", no_own_sub=False),
     "m + 2by",          "correct"),
    ("own dropped + NOT subtr.",   dict(variant="lp",        no_own_sub=True),
     "m + 2by",          "correct (LP)"),
    ("own dropped + subtracted",   dict(variant="lp",        no_own_sub=False),
     "m + 2by - b x_i",  "under-corrects  <- as published"),
    ("own retained + NOT subtr.",  dict(variant="corrected", no_own_sub=True),
     "m + 2by + b x_i",  "over-corrects"),
]

print("Part 1: fixed point of each variant, exact aggregation, stable damping\n")
print(f"{'variant':30s} {'FP marginal':18s} {'ratio to planner':>17s}   note")
print("-" * 100)

inst = make_instance(n=8, T=32, seed=3, kappa=1.0, lam=0.0)
cstar = planner_value(inst)
for name, kw, marg, note in VARIANTS:
    x, k = shade(inst, eps=None, dp="none", K=6000, gamma=2.0 / (inst.n + 1),
                 theta=1e-10, **kw)
    r = inst.social(x) / cstar
    print(f"{name:30s} {marg:18s} {r:17.6f}   {note}")
    res.row(part="variants", variant=name, fp_marginal=marg, ratio=r,
            rounds=k, note=note)

print("\nPart 2: stability of the damping constant\n")
GAMMAS = np.array([0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10,
                   0.07, 0.05, 0.03, 0.02, 0.012, 0.008])
NS = [2, 4, 8, 16, 32, 64]
print(f"{'n':>4s} {'gamma=0.55?':>12s} {'max stable gamma':>17s} "
      f"{'2/(n+1)':>9s} {'rounds at 2/(n+1)':>18s} {'ratio':>10s}")
print("-" * 78)

gmax = {}
for n in NS:
    inst = make_instance(n=n, T=32, seed=3, kappa=1.0, lam=0.0)
    cstar = planner_value(inst)
    stable = []
    for g in GAMMAS:
        x, k = shade(inst, eps=None, dp="none", K=3000, gamma=float(g),
                     theta=1e-10, variant="corrected")
        r = inst.social(x) / cstar
        if r < 1.0001 and k < 3000:
            stable.append(float(g))
    g55 = "converges" if 0.55 in stable else "DIVERGES"
    best = max(stable) if stable else float("nan")
    grule = 2.0 / (n + 1)
    x, k = shade(inst, eps=None, dp="none", K=3000, gamma=grule, theta=1e-10,
                 variant="corrected")
    r = inst.social(x) / cstar
    gmax[n] = best
    print(f"{n:4d} {g55:>12s} {best:17.3f} {grule:9.3f} {k:18d} {r:10.6f}")
    res.row(part="stability", n=n, gamma055_converges=(0.55 in stable),
            max_stable_gamma=best, rule_gamma=grule, rounds_at_rule=k, ratio=r)

print("-" * 78)
ns = np.array([k for k in gmax if np.isfinite(gmax[k])], dtype=float)
gs = np.array([gmax[int(k)] for k in ns])
print("the largest stable gamma falls roughly as 1/n:")
for nn, gg in zip(ns, gs):
    print(f"   n={int(nn):3d}: max stable gamma {gg:.3f},  gamma*n = {gg*nn:.2f}")

n_bad = [r["n"] for r in res.rows
         if r.get("part") == "stability" and not r.get("gamma055_converges")]
print(f"\nVERDICT")
print(f"  gamma = 0.55 (the paper's value) fails to converge for n in {n_bad}")
print(f"  the mechanism therefore needs gamma ~ c/n, not a constant.  That")
print(f"  matters beyond convergence: more rounds K means more composed privacy")
print(f"  noise, so the damping rule and the privacy budget are coupled and")
print(f"  cannot be chosen independently as the draft does.")

res.scalar("GammaPaper", 0.55, "{:.2f}")
res.scalar("NFailAtPaperGamma", ", ".join(str(int(v)) for v in n_bad))
res.scalar("MaxStableGammaN32", gmax.get(32, float("nan")), "{:.3f}")
res.write()

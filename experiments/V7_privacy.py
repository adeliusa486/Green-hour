"""V7 -- the privacy sweep, run at the round count the mechanism actually needs.

The draft's Table 4 reports a graceful privacy trade-off (1.078 at eps = 0.5
down to 1.034 at eps = 8) assuming K <= 25 and a median of 7 rounds.  V4 showed
that gamma = 0.55 diverges for n >= 8, and that a stable gamma needs far more
rounds than that.  Since basic composition scales per-round noise as
Ebar*K/eps, the round count and the privacy budget cannot be chosen
independently, and Table 4 has to be recomputed at the true K.

Protocol note.  K is committed in advance, as differential privacy requires:
adapting the number of rounds to the data would itself leak.  So K is measured
once on the noiseless problem at the target tolerance, fixed, and then used both
to run the protocol and to scale the noise.

Also compares basic Laplace composition against zCDP/Gaussian.  At the draft's
stated eps = 2 and K = 7 the Laplace mechanism is actually TIGHTER (the
crossover is near K = 13); it is only because the true K is in the dozens that
zCDP becomes the right choice.
"""
from common import Result, banner
import numpy as np
from gh.instances import make_instance
from gh.mech import shade, noise_scale, zcdp_rho
from gh.baselines import planner_value

banner("V7", "Privacy sweep at the true round count")

T, TOL_FRAC, SEEDS = 32, 0.01, 10
NS = [8, 16, 32]
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, None]

res = Result("V7", "Privacy sweep", {"T": T, "tol_frac": TOL_FRAC,
                                     "seeds": SEEDS, "ns": NS,
                                     "eps": [e if e else "inf" for e in EPS]})

# ---- 1. commit K per n, from the noiseless problem -------------------------
print(f"1. Round count committed in advance (noiseless, tolerance "
      f"{100*TOL_FRAC:g}% of mean per-slot load)\n")
print(f"{'n':>4s} {'gamma':>7s} {'K committed':>12s} {'draft assumes':>14s} "
      f"{'noise inflation':>16s}")
print("-" * 60)
KS = {}
for n in NS:
    inst = make_instance(n=n, T=T, seed=3, kappa=1.0, lam=0.0)
    Ytot = sum(X.E for X in inst.X)
    theta = TOL_FRAC * Ytot / T
    g = 2.0 / (n + 1)
    _, k = shade(inst, eps=None, dp="none", K=4000, gamma=g, theta=theta,
                 variant="corrected")
    KS[n] = k
    print(f"{n:4d} {g:7.3f} {k:12d} {7:14d} {k/7:15.1f}x")
    res.row(part="rounds", n=n, gamma=g, K=k, draft_K=7, inflation=k / 7)

# ---- 2. Laplace vs zCDP at the committed K ---------------------------------
print(f"\n2. Which composition is tighter?  Per-coordinate noise sd, "
      f"Ebar = 1, eps = 2\n")
print(f"{'K':>6s} {'Laplace sd':>12s} {'Gaussian sd':>13s} {'winner':>10s}")
print("-" * 46)
for K in (7, 13, 25, 50, KS[32], 200):
    sl, _ = noise_scale("laplace", 1.0, 2.0, K)
    sg, _ = noise_scale("gaussian", 1.0, 2.0, K)
    print(f"{K:6d} {sl:12.3f} {sg:13.3f} {'Laplace' if sl < sg else 'Gaussian':>10s}")
    res.row(part="composition", K=K, laplace_sd=sl, gaussian_sd=sg)

# ---- 3. the sweep -----------------------------------------------------------
print(f"\n3. Ratio to planner, {SEEDS} seeds, K committed per n\n")
hdr = "  ".join(f"eps={e}" if e else "exact" for e in EPS)
print(f"{'n':>4s} {'K':>5s} {'dp':>9s} | {hdr}")
print("-" * 92)
for n in NS:
    inst = make_instance(n=n, T=T, seed=3, kappa=1.0, lam=0.0)
    cstar = planner_value(inst)
    Ytot = sum(X.E for X in inst.X)
    theta = TOL_FRAC * Ytot / T
    g, K = 2.0 / (n + 1), KS[n]
    for dp in ("laplace", "gaussian"):
        row = []
        for e in EPS:
            vals = []
            for s in range(SEEDS if e else 1):
                x, _ = shade(inst, eps=e, dp=(dp if e else "none"), K=K,
                             gamma=g, theta=theta, seed=s, variant="corrected")
                vals.append(inst.social(x) / cstar)
            row.append(float(np.mean(vals)))
            res.row(part="sweep", n=n, dp=dp, eps=(e if e else "inf"),
                    K=K, ratio=float(np.mean(vals)),
                    sd=float(np.std(vals)) if e else 0.0)
        print(f"{n:4d} {K:5d} {dp:>9s} | "
              + "  ".join(f"{v:7.4f}" for v in row))

sweep = [r for r in res.rows if r.get("part") == "sweep"]
lap32 = {r["eps"]: r["ratio"] for r in sweep if r["n"] == 32 and r["dp"] == "laplace"}
gau32 = {r["eps"]: r["ratio"] for r in sweep if r["n"] == 32 and r["dp"] == "gaussian"}

print("\nVERDICT (n = 32, the draft's default)")
print(f"  exact aggregation                {lap32['inf']:.4f}")
print(f"  eps = 2, Laplace  basic comp.    {lap32[2.0]:.4f}")
print(f"  eps = 2, Gaussian zCDP           {gau32[2.0]:.4f}")
print(f"  draft's Table 4 claims           1.0410 at eps = 2, "
      f"against 1.0310 exact")
better = "Gaussian" if gau32[2.0] < lap32[2.0] else "Laplace"
print(f"  at the true K = {KS[32]}, {better} composition is better")
print(f"\n  The draft's privacy cost of ~1 point assumes K = 7.  At the round")
print(f"  count the mechanism actually needs, the cost is "
      f"{100*(min(lap32[2.0],gau32[2.0]) - lap32['inf']):.1f} points.")

res.scalar("KAtN32", KS[32], "{:d}")
res.scalar("KInflation", KS[32] / 7, "{:.1f}")
res.scalar("ExactRatio", lap32["inf"], "{:.4f}")
res.scalar("LaplaceEps2", lap32[2.0], "{:.4f}")
res.scalar("GaussianEps2", gau32[2.0], "{:.4f}")
res.scalar("BetterComposition", better)
res.write()

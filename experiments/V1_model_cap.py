"""V1 -- REVIEW.md B1: the published emissions model caps eta at 1/2.

This experiment needs no simulation.  It is arithmetic on the model as written
plus the draft's own Table 1, and it is the cheapest way to see that the two
are incompatible.

Model as published (Eq. 2):   E_t(L) = alpha_t*L + beta_t*L^2,  beta_t >= 0
                              AEF_t(L) = E_t(L)/L = alpha_t + beta_t*L
                              MEF_t(L) = E_t'(L)  = alpha_t + 2*beta_t*L

For E_t to be non-negative and increasing on L >= 0 -- which "dispatch is a
merit order" asserts, and which AEF = E/L presupposes -- alpha_t >= 0.  Then

    MEF/AEF = (alpha + 2*beta*D)/(alpha + beta*D)  in  [1, 2]
    eta     = 1 - AEF/MEF                          in  [0, 1/2]
    PoA bd  = 3/(2(1-eta))                         in  [3/2, 3]

Given a measured (AEF, MEF) pair the model's implied intercept is forced:
    beta*D = MEF - AEF        and       alpha = AEF - beta*D = 2*AEF - MEF
so alpha < 0 exactly when MEF > 2*AEF, i.e. exactly when eta > 1/2.
"""
from common import Result, banner
import numpy as np

banner("V1", "Model-implied intercept for the draft's Table 1")

# The draft's Table 1 (itself a placeholder, but the arithmetic is what matters)
TABLE1 = [("CAISO", 181, 433), ("ERCOT", 249, 448), ("PJM", 358, 497),
          ("GB", 122, 391), ("DE", 203, 419)]

res = Result("V1", "Model-implied intercept and the eta cap",
             {"table1": TABLE1}, provenance="ARITHMETIC on the draft's Table 1")

print(f"{'region':8s} {'AEF':>6s} {'MEF':>6s} {'MEF/AEF':>8s} {'eta':>7s} "
      f"{'PoA bd':>7s} {'alpha=2AEF-MEF':>15s}  verdict")
print("-" * 74)
ratios, etas, alphas, n_bad = [], [], [], 0
for name, aef, mef in TABLE1:
    ratio = mef / aef
    eta = 1 - aef / mef
    bd = 1.5 / (1 - eta)
    alpha = 2 * aef - mef
    ok = alpha >= 0
    n_bad += (not ok)
    ratios.append(ratio); etas.append(eta); alphas.append(alpha)
    print(f"{name:8s} {aef:6d} {mef:6d} {ratio:8.3f} {eta:7.3f} {bd:7.2f} "
          f"{alpha:15.0f}  {'ok' if ok else 'IMPOSSIBLE (alpha<0)'}")
    res.row(region=name, aef=aef, mef=mef, ratio=ratio, eta=eta,
            poa_bound=bd, alpha_implied=alpha, feasible=bool(ok))

ratios, etas, alphas = map(np.array, (ratios, etas, alphas))
print("-" * 74)
print(f"{'mean':8s} {np.mean([t[1] for t in TABLE1]):6.0f} "
      f"{np.mean([t[2] for t in TABLE1]):6.0f} {ratios.mean():8.3f} "
      f"{etas.mean():7.3f} {(1.5/(1-etas)).mean():7.2f} {alphas.mean():15.0f}")

print(f"\nRegions outside the model (alpha < 0, equivalently eta > 1/2): "
      f"{n_bad} of {len(TABLE1)}")
print(f"Model's hard ceiling on eta          : 0.500")
print(f"Largest eta claimed in the draft     : {etas.max():.3f}  (GB)")
print(f"Model's hard ceiling on the PoA bound: 3.00")
print(f"Largest bound claimed in the draft   : {(1.5/(1-etas)).max():.2f}  (GB)")

# --- N1: the abstract's 3.26 -------------------------------------------------
mean_ratio = ratios.mean()
mean_bound = (1.5 / (1 - etas)).mean()
ratio_of_means = np.mean([t[2] for t in TABLE1]) / np.mean([t[1] for t in TABLE1])
print("\n" + "-" * 74)
print("N1: what '3.26' actually is")
print(f"  mean of per-region MEF/AEF            : {mean_ratio:.3f}   <- the "
      f"'ratio of marginal to average factors'")
print(f"  MEF/AEF computed from the Mean row    : {ratio_of_means:.3f}")
print(f"  mean of per-region PoA bounds         : {mean_bound:.3f}   <- what "
      f"the abstract calls the ratio")
print(f"  check 1.5 x {mean_ratio:.3f}                    = {1.5*mean_ratio:.3f}")
print(f"  bound evaluated at mean eta           : {1.5/(1-etas.mean()):.3f}   "
      f"<- differs from the mean of bounds (Jensen)")

res.scalar("NumImpossible", n_bad, "{:d}")
res.scalar("EtaCap", 0.5, "{:.2f}")
res.scalar("BoundCap", 3.0, "{:.2f}")
res.scalar("EtaMaxClaimed", etas.max(), "{:.3f}")
res.scalar("BoundMaxClaimed", (1.5 / (1 - etas)).max(), "{:.2f}")
res.scalar("MeanRatio", mean_ratio, "{:.2f}")
res.scalar("MeanBound", mean_bound, "{:.2f}")
res.scalar("BoundAtMeanEta", 1.5 / (1 - etas.mean()), "{:.2f}")

body = "\\begin{tabular}{lrrrrrr}\n\\toprule\n" \
       "Region & $\\aef$ & $\\mef$ & $\\mef/\\aef$ & $\\eta$ & bound & " \
       "implied $\\alpha_t$ \\\\\n\\midrule\n"
for name, aef, mef in TABLE1:
    eta = 1 - aef / mef
    body += (f"{name} & {aef} & {mef} & {mef/aef:.2f} & {eta:.3f} & "
             f"{1.5/(1-eta):.2f} & {2*aef-mef:.0f} \\\\\n")
body += "\\bottomrule\n\\end{tabular}"
res.write(body)

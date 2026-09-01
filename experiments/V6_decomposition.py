"""V6 -- REVIEW.md Move 1: split the equilibrium gap into an accounting
component and a strategic component.

The objection the paper must answer (REVIEW.md P2) is that eta > 0 holds at
n = 1 and with zero flexible load, so the 1/(1-eta) factor is an accounting
error -- documented since Hawkes 2010 and Callaway 2018, both of which the
draft cites -- and not a multi-agent effect.  The answer is to measure the
split rather than argue about it.

Three regimes, differing only in what each agent is charged per unit:

  Nash        a_t + b_t*y_t     perceived marginal  a + b*y + b*x_i
  wedge-fixed m_t + b_t*y_t     perceived marginal  m + b*y + b*x_i
  planner     --                social marginal     m + 2*b*y

Nash -> wedge-fixed removes exactly the accounting wedge (m - a).
wedge-fixed -> planner removes exactly the congestion externality b*(y - x_i).
The two add up to the whole gap, so the split is exhaustive by construction.

Also reported: the two signal-taking baselines the paper's own discussion
recommends but never tests (REVIEW.md X1), and the naive forecast-taker that
models deployed practice better than Nash does (X2).
"""
from common import Result, banner
import numpy as np
from gh.core import nash, wedge_fixed
from gh.instances import make_instance
from gh.baselines import planner_value, naive, mef_static, carbon_agnostic
from gh.core import mef_signal_equilibrium

banner("V6", "Accounting vs strategic decomposition of the equilibrium gap")

N, T, SEEDS = 16, 48, 8
KAPPAS = [0.5, 1.0, 2.0, 4.0]

res = Result("V6", "Gap decomposition", {"n": N, "T": T, "seeds": SEEDS,
                                         "kappas": KAPPAS})

print(f"n = {N}, T = {T}, {SEEDS} instances per kappa\n")
print(f"{'kappa':>6s} {'total gap':>10s} | {'accounting':>11s} {'strategic':>10s} "
      f"| {'acct share':>11s} | {'naive':>8s} {'MEF stat':>9s} {'MEF resp':>9s}")
print("-" * 96)

for kappa in KAPPAS:
    rows = []
    for s in range(SEEDS):
        inst = make_instance(n=N, T=T, seed=s, kappa=kappa, lam=0.0)
        cstar = planner_value(inst)
        c_nash = inst.social(nash(inst, tol=1e-8)[0])
        c_wf = inst.social(wedge_fixed(inst, tol=1e-8)[0])
        c_naive = inst.social(naive(inst))
        c_mefs = inst.social(mef_static(inst))
        c_mefr = inst.social(mef_signal_equilibrium(inst)[0])
        c_agn = inst.social(carbon_agnostic(inst))
        rows.append((c_nash / cstar, c_wf / cstar, c_naive / cstar,
                     c_mefs / cstar, c_mefr / cstar, c_agn / cstar))
    r = np.array(rows)
    nash_r, wf_r, naive_r, mefs_r, mefr_r, agn_r = r.mean(axis=0)
    total = nash_r - 1.0
    acct = nash_r - wf_r
    strat = wf_r - 1.0
    share = acct / total if total > 1e-12 else float("nan")
    print(f"{kappa:6.1f} {100*total:9.2f}% | {100*acct:10.2f}% {100*strat:9.2f}% "
          f"| {100*share:10.1f}% | {naive_r:8.4f} {mefs_r:9.4f} {mefr_r:9.4f}")
    res.row(kappa=kappa, nash=nash_r, wedge_fixed=wf_r, planner=1.0,
            total_gap=total, accounting=acct, strategic=strat,
            accounting_share=share, naive=naive_r, mef_static=mefs_r,
            mef_responsive=mefr_r, carbon_agnostic=agn_r)

print("-" * 96)
print("reading: 'accounting' is what changing the published signal from AEF to")
print("MEF removes; 'strategic' is what is left for a mechanism to fix.")

rr = res.rows
k1 = [x for x in rr if x["kappa"] == 1.0][0]
print(f"\nAt kappa = 1.0 (reference congestion strength):")
print(f"  total equilibrium gap                    {100*k1['total_gap']:6.2f}%")
print(f"  removed by publishing MEF instead of AEF {100*k1['accounting']:6.2f}%"
      f"  ({100*k1['accounting_share']:.0f}% of the gap)")
print(f"  left for a mechanism                     {100*k1['strategic']:6.2f}%"
      f"  ({100*(1-k1['accounting_share']):.0f}% of the gap)")
print(f"\n  naive forecast-taker (deployed practice) {k1['naive']:.4f}"
      f"   vs Nash {k1['nash']:.4f}")
print(f"  -> modelling practice as Nash "
      f"{'UNDERSTATES' if k1['naive'] > k1['nash'] else 'overstates'} the gap")
print(f"  publishing a static MEF forecast         {k1['mef_static']:.4f}")
print(f"  publishing a responsive MEF              {k1['mef_responsive']:.4f}"
      f"   (over-corrects by 2*b*x_i)")

print(f"\nThe accounting share falls as congestion strengthens:")
for row in rr:
    print(f"  kappa={row['kappa']:4.1f}: accounting {100*row['accounting_share']:5.1f}%"
          f"   strategic {100*(1-row['accounting_share']):5.1f}%")

res.scalar("TotalGapPct", 100 * k1["total_gap"], "{:.2f}")
res.scalar("AcctPct", 100 * k1["accounting"], "{:.2f}")
res.scalar("StratPct", 100 * k1["strategic"], "{:.2f}")
res.scalar("AcctShare", 100 * k1["accounting_share"], "{:.0f}")
res.scalar("NaiveRatio", k1["naive"], "{:.4f}")
res.scalar("NashRatio", k1["nash"], "{:.4f}")
res.scalar("MefStaticRatio", k1["mef_static"], "{:.4f}")
res.scalar("MefRespRatio", k1["mef_responsive"], "{:.4f}")
res.write()

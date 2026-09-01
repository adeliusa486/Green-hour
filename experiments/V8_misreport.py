"""V8 (E10) -- REVIEW.md X3: can an operator gain by misreporting its intentions?

SHADE elicits per-slot intentions and computes the broadcast adder from their
aggregate.  Theorem 4 bounds what an operator gains by deviating in its ACTION,
taking the adder as given.  It says nothing about deviating in its REPORT, and
the mechanism has no manipulation analysis at all.  This is the obvious attack.

The attack.  Operator 0 submits a profile concentrated in the slots it most
wants -- an aggressive-looking intention meant to inflate the published
aggregate there so that every other operator sees those slots as expensive and
moves away, leaving them for operator 0.  Total submitted energy is preserved,
so the lie is not detectable by energy accounting alone.

Operator 0 is judged on its own attributed footprint, evaluated at the
*realised* average factor rather than the reported one:

    F_i = sum_t x_it * (a_t + b_t * y_t)

plus, where the settlement rule of Eq. (14) is active, a charge on the gap
between what it committed to and what it actually ran.

The sweep in n is the point.  A report is a claim about 1/n of the aggregate,
so if the attack works anywhere it should work when n is small and the
manipulator's share is large -- which is exactly the regime in which the
market-power term of Theorem 4 is worst.
"""
from common import Result, banner
import numpy as np
from gh.core import solve_sep_qp
from gh.instances import make_instance
from gh.baselines import planner_value

banner("V8", "Strategic misreporting of intentions under SHADE")

T, SEEDS = 32, 4
KAPPA = 1.0
NS = [2, 4, 8, 16, 32]


def run(inst, manip=None, K=200, settle=False):
    """Run SHADE.  `manip` is the index of an operator that submits a profile
    concentrated in its cheapest slots instead of the one it intends to run."""
    n, T_ = inst.n, inst.T
    gamma = 2.0 / (n + 1)
    x = inst.feasible_start()          # what each operator will actually run
    rep = x.copy()                     # what each operator submits
    prev = rep.sum(axis=0).copy()

    for k in range(K):
        yhat = prev
        for i in range(n):
            y_mi = yhat - rep[i]
            pi = (inst.m - inst.a) + inst.b * y_mi
            q = inst.a + inst.b * y_mi + pi + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
            if i == manip:
                # concentrate the *report* in the slots this operator wants
                w = inst.m + 2.0 * inst.b * y_mi
                rep[i] = solve_sep_qp(w * 1e3, np.full(T_, 1e-3), inst.X[i])
            else:
                rep[i] = x[i]
        y = rep.sum(axis=0)
        new = prev + gamma * (y - prev)
        if np.max(np.abs(new - prev)) <= 1e-4 * inst.X[0].E:
            prev = new
            break
        prev = new

    y_true = x.sum(axis=0)
    aef = inst.a + inst.b * y_true
    footprint = (x * aef[None, :]).sum(axis=1)
    if settle and manip is not None:
        price = (inst.m - inst.a) + inst.b * prev
        footprint[manip] += float(np.sum(price * np.abs(x[manip] - rep[manip])))
    return x, footprint, inst.social(x)


res = Result("V8", "Misreporting of intentions",
             {"ns": NS, "T": T, "seeds": SEEDS, "kappa": KAPPA})

print(f"T = {T}, {SEEDS} instances per n.  A POSITIVE gain means the lie pays.")
print()
print(f"{'n':>4s} {'share':>7s} | {'gain':>9s} {'gain+charge':>12s} "
      f"| {'social':>8s} | {'op0 peak/mean':>20s}")
print("-" * 78)

summary = []
for N in NS:
    G, GS, DS, P2 = [], [], [], []
    for s in range(SEEDS):
        inst = make_instance(n=N, T=T, seed=s, kappa=KAPPA, lam=0.0)
        xt, f_t, soc_t = run(inst, manip=None)
        xm, f_m, soc_m = run(inst, manip=0, settle=False)
        _, f_s, _ = run(inst, manip=0, settle=True)
        G.append(100 * (f_t[0] - f_m[0]) / f_t[0])
        GS.append(100 * (f_t[0] - f_s[0]) / f_t[0])
        DS.append(100 * (soc_m - soc_t) / soc_t)
        P2.append((xt[0].max() / xt[0].mean(), xm[0].max() / xm[0].mean()))
    g, gs, ds = np.mean(G), np.mean(GS), np.mean(DS)
    pt, pm = np.mean([a for a, b in P2]), np.mean([b for a, b in P2])
    summary.append((N, g, gs, ds))
    print(f"{N:4d} {100.0/N:6.1f}% | {g:+8.2f}% {gs:+11.2f}% "
          f"| {ds:+7.2f}% | {pt:8.2f} -> {pm:.2f}")
    res.row(n=N, market_share_pct=100.0 / N, gain_pct=g,
            gain_settled_pct=gs, social_cost_pct=ds,
            p2m_truthful=pt, p2m_manip=pm)

print("-" * 78)
best = max(summary, key=lambda r: r[1])
worst_social = max(summary, key=lambda r: abs(r[3]))
print(f"\nlargest gain anywhere in the sweep : {best[1]:+.2f}% at n = {best[0]} "
      f"(market share {100.0/best[0]:.0f}%)")
print(f"largest social effect              : {worst_social[3]:+.2f}% "
      f"at n = {worst_social[0]}")

print(f"\nVERDICT")
if best[1] <= 0.05:
    print(f"  Misreporting never pays, at any market share tested, and it does")
    print(f"  so WITHOUT needing the settlement charge of Eq. (14).")
    print()
    print(f"  The reason is not that the lie fails to move the aggregate --")
    print(f"  it is that the manipulator's own best response is computed from")
    print(f"  an adder built on its own false report.  Inflating the report in")
    print(f"  a slot lowers what the manipulator believes others are placing")
    print(f"  there, so it piles in.  Its attributed footprint, however, is")
    print(f"  charged at the REALISED average factor, which its own")
    print(f"  concentration inflates.  The lie corrupts the liar's information")
    print(f"  more than it moves anyone else's behaviour, and the operator ends")
    print(f"  up concentrated in exactly the slots it made expensive.")
    print()
    print(f"  This is a property of self-applied adders worth stating: because")
    print(f"  each operator subtracts its OWN contribution from the broadcast")
    print(f"  aggregate, a false report is subtracted back out of the")
    print(f"  manipulator's own price but not out of the physical outcome it")
    print(f"  is charged for.")
else:
    print(f"  Misreporting pays up to {best[1]:+.2f}% at n = {best[0]}.  Report")
    print(f"  this and strengthen the mechanism before submission.")

res.scalar("BestGainPct", best[1], "{:.2f}")
res.scalar("BestGainN", best[0], "{:d}")
res.scalar("MaxSocialPct", worst_social[3], "{:.2f}")
res.scalar("Deterred", "yes" if best[1] <= 0.05 else "no")
res.write()

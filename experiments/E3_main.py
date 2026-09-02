"""E3 -- the main comparison, on grid data that is real.

Grid side is measured: per-hour-of-day AEF, MEF and beta for CAISO, ERCOT and
PJM, reconstructed from EIA-930 (Jul-Dec 2024) exactly as in E1.  Workload side
is synthetic: operator sizes, deadline staircases and power envelopes are drawn
from the generator, because the production traces (Azure, Borg, Alibaba) are not
in this repository.  So this is a REAL-GRID / SYNTHETIC-WORKLOAD experiment and
is labelled that way; it is not the full E3 the paper promises, which pairs the
same grid data with those traces.

Every baseline in the paper is run except cooperative MARL, which needs GPU
training:

  carbon-agnostic, naive forecast-taking, static MEF signalling,
  independent greedy (Nash), threshold deferral, randomized jitter,
  proportional cap, SHADE, responsive MEF (oracle), planner.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.core import Feasible, Instance, nash, solve_sep_qp
import gh.baselines as B
from E1_calibrate import load, factors, BAS, EF_GAS_DEFAULT

T = 24
N = 32
SEEDS = 6
FLEX_SHARE = 0.0038
CAP_FRAC = 0.35


def hourly_profile(d, ef_gas=EF_GAS_DEFAULT):
    """Per-hour-of-day AEF, MEF, beta from real data."""
    F = factors(ef_gas)
    gen = np.array([sum(f.values()) for f in d["fuel"]])
    emis = np.array([sum(F[k] * v for k, v in f.items()) for f in d["fuel"]])
    dem, hr = d["dem"], d["hr"]
    ok = (gen > 0) & (dem > 0)
    gen, emis, dem, hr = gen[ok], emis[ok], dem[ok], hr[ok]

    aef = np.zeros(T); mef = np.zeros(T); beta = np.zeros(T)
    for h in range(T):
        m = hr == (h + 1)
        if m.sum() < 30:
            aef[h] = np.median(emis / gen); mef[h] = aef[h] * 1.5; beta[h] = 0.0
            continue
        e, D = emis[m], dem[m]
        aef[h] = float(np.mean(e / gen[m]))
        de, dD = np.diff(e), np.diff(D)
        k = np.abs(dD) > 1e-6
        de, dD = de[k], dD[k]
        Dm = (0.5 * (D[:-1] + D[1:]))[k]
        if len(de) < 20:
            mef[h] = aef[h] * 1.5; beta[h] = 0.0; continue
        mef[h] = float(de @ dD / (dD @ dD))
        X = np.column_stack([dD, dD * (Dm - Dm.mean())])
        c, *_ = np.linalg.lstsq(X, de, rcond=None)
        beta[h] = float(c[1])
    # the model needs MEF >= AEF on the slots that carry load, and beta > 0
    mef = np.maximum(mef, aef * 1.02)
    beta = np.maximum(np.abs(beta), 1e-6)
    return aef, mef, beta, float(np.mean(dem) * T)


def build(aef, mef, beta, daily_MWh, seed):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 0.6, N); w /= w.sum()
    E = w * FLEX_SHARE * daily_MWh
    X, psi, arrival = [], np.zeros((N, T)), np.zeros((N, T))
    for i in range(N):
        u = np.full(T, CAP_FRAC * E[i])
        slack = rng.uniform(2.0, 24.0)
        done = np.clip(np.arange(T) / slack, 0, 1)
        R = np.maximum.accumulate(np.minimum(0.85 * E[i] * done ** 1.7,
                                             np.cumsum(u) * 0.9))
        X.append(Feasible(E=E[i], u=u, R=R))
        h0 = rng.uniform(6, 18)
        psi[i] = np.abs(np.arange(T) - h0) / 24.0
        arr = np.exp(-0.5 * ((np.arange(T) - h0) / 2.0) ** 2)
        arrival[i] = E[i] * arr / arr.sum()
    return Instance(a=aef, m=mef, b=beta, X=X, psi=psi, lam=np.zeros(N),
                    arrival=arrival)


def shade(inst, K=80, gamma=None, tol=1e-10):
    n = inst.n
    gamma = gamma or 1.8 / n
    x = inst.feasible_start(); yhat = x.sum(0).copy(); prev = yhat.copy()
    for _ in range(K):
        for i in range(n):
            ymi = yhat - x[i]
            q = inst.a + inst.b * ymi + ((inst.m - inst.a) + inst.b * ymi)
            x[i] = solve_sep_qp(q + inst.lam[i] * inst.psi[i], inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        if np.max(np.abs(yhat - prev)) <= tol * max(1.0, float(np.max(yhat))):
            break
        prev = yhat
    return x


def responsive_mef(inst, K=80, gamma=None, tol=1e-10):
    """Oracle: agents charged m + 2*b*y, i.e. the true social marginal."""
    n = inst.n
    gamma = gamma or 1.8 / n
    x = inst.feasible_start(); yhat = x.sum(0).copy(); prev = yhat.copy()
    for _ in range(K):
        for i in range(n):
            q = inst.m + 2.0 * inst.b * yhat + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        if np.max(np.abs(yhat - prev)) <= tol * max(1.0, float(np.max(yhat))):
            break
        prev = yhat
    return x


def _ne(inst):
    r = nash(inst)
    return r[0] if isinstance(r, tuple) else r


METHODS = [
    ("Carbon-agnostic",        lambda I, r: B.carbon_agnostic(I)),
    ("Naive forecast-taking",  lambda I, r: B.naive(I)),
    ("Static MEF signalling",  lambda I, r: B.mef_static(I)),
    ("Independent greedy",     lambda I, r: _ne(I)),
    ("Threshold deferral",     lambda I, r: B.threshold(I)),
    ("Randomized jitter",      lambda I, r: B.jitter(I, seed=r)),
    ("Proportional cap",       lambda I, r: B.proportional_cap(I)),
    ("SHADE (ours)",           lambda I, r: shade(I)),
    ("Responsive MEF",         lambda I, r: responsive_mef(I)),
]

if __name__ == "__main__":
    print("loading EIA-930 ...")
    data = load()
    prof = {b: hourly_profile(data[b]) for b in BAS}
    for b in BAS:
        a, m, be, dm = prof[b]
        print(f"  {b}: AEF {a.min():.0f}-{a.max():.0f}, MEF {m.min():.0f}-{m.max():.0f}, "
              f"daily {dm/1e3:.0f} GWh")

    res = {name: {b: [] for b in BAS} for name, _ in METHODS}
    infeasible = {name: 0 for name, _ in METHODS}
    for b in BAS:
        a, m, be, dm = prof[b]
        for s in range(SEEDS):
            inst = build(a, m, be, dm, s)
            Cs = B.planner_value(inst)
            for name, fn in METHODS:
                try:
                    x = fn(inst, s)
                except Exception as e:
                    print(f"    {name}/{b}/{s}: {type(e).__name__} {e}"); continue
                # A ratio below 1 is impossible for a feasible profile, since the
                # planner minimises C.  Check feasibility explicitly rather than
                # trusting the number: proportional_cap silently falls back to an
                # infeasible decomposition when the quota empties a feasible set.
                bad = [i for i, Xi in enumerate(inst.X) if not Xi.check(x[i], tol=1e-6)]
                if bad:
                    infeasible[name] += 1; continue
                res[name][b].append(inst.social(x) / Cs)

    nash_mean = float(np.mean([np.mean(res["Independent greedy"][b]) for b in BAS]))
    print(f"\n{'Method':<24s} " + " ".join(f"{b:>8s}" for b in BAS)
          + f" {'Mean':>8s} {'Gap rm':>8s}")
    print("-" * 68)
    table = {}
    for name, _ in METHODS:
        if infeasible[name]:
            print(f"{name:<24s} INFEASIBLE in {infeasible[name]} runs -- not scored")
            continue
        per = [float(np.mean(res[name][b])) for b in BAS]
        mean = float(np.mean(per))
        gap = ((nash_mean - mean) / (nash_mean - 1.0)) if nash_mean > 1 + 1e-9 else float("nan")
        sd = [float(np.std(res[name][b])) for b in BAS]
        table[name] = dict(per=per, sd=sd, mean=mean, gap=gap)
        g = "n/a" if name == "Carbon-agnostic" else f"{100*gap:7.1f}%"
        print(f"{name:<24s} " + " ".join(f"{v:8.4f}" for v in per)
              + f" {mean:8.4f} {g:>8s}")
    print(f"{'Planner (oracle)':<24s} " + " ".join(f"{1.0:8.4f}" for _ in BAS)
          + f" {1.0:8.4f} {100.0:7.1f}%")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "E3.json")
    json.dump({"note": "REAL GRID (EIA-930 Jul-Dec 2024) + SYNTHETIC WORKLOAD; "
                       "MARL baseline omitted (needs GPU training)",
               "n": N, "T": T, "seeds": SEEDS, "flex_share": FLEX_SHARE,
               "regions": BAS, "table": table}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")

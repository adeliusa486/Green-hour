"""E1 -- dispatch-response calibration from real grid data.

Input: EIA-930 Hourly Grid Monitor, BALANCE file, Jul-Dec 2024 (public, no key).
Hourly demand and net generation by fuel for every US balancing authority.

What this produces, per balancing authority:

    AEF_t   average emission factor  = system emissions / generation
    MEF_t   marginal emission factor = dE/dD, from first differences
    beta_t  response curvature       = d(MEF)/dD
    eta_t   = 1 - AEF/MEF, and the PoA bound 3/(2(1-eta))
    kappa   congestion strength: beta*y_peak / (MEF - AEF) at a reference
            flexible share -- the quantity the Plan A / Plan B gate is defined on

Method.  Emissions are reconstructed from generation by fuel using combustion
factors (below).  MEF is estimated the standard way (Hawkes 2010): regress the
hour-to-hour change in emissions on the hour-to-hour change in demand, binned by
hour-of-day so that diurnal mix shifts do not masquerade as a marginal response.
Curvature is the slope of that regression's residual against demand level, i.e.
a second-order fit of E(D) about the operating point -- which is exactly the
(AEF, MEF, beta) parameterisation the paper uses.

Emission factors, gCO2/kWh of generation (direct combustion, IPCC/EPA ranges):
coal 1000, natural gas 430, petroleum 700, other/unknown 500, biomass 0 (counted
as neutral, per the convention the carbon-intensity feeds use), nuclear, hydro,
wind, solar, geothermal 0.  A sensitivity over gas in [370, 550] is reported,
since gas is what sets the margin in every region here and the result should not
turn on one number.

THIS IS REAL DATA, not a placeholder.  It calibrates AEF/MEF/beta/eta/kappa.
It does NOT complete E3, which additionally needs workload traces and the full
baseline sweep.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSVP = os.path.join(HERE, "..", "..", "data", "EIA930_BALANCE_2024_Jul_Dec.csv")
OUT = os.path.join(HERE, "..", "results", "E1.json")

BAS = ["CISO", "ERCO", "PJM"]        # CAISO, ERCOT, PJM
EF_GAS_DEFAULT = 430.0

FUEL_COL = {                          # column header fragment -> factor key
    "from Coal": "coal",
    "from Natural Gas": "gas",
    "from Nuclear": "nuc",
    "from All Petroleum Products": "oil",
    "from Hydropower Excluding Pumped Storage": "zero",
    "from Solar without Integrated Battery Storage": "zero",
    "from Solar with Integrated Battery Storage": "zero",
    "from Wind without Integrated Battery Storage": "zero",
    "from Wind with Integrated Battery Storage": "zero",
    "from Other Fuel Sources": "other",
    "from Unknown Fuel Sources": "other",
}


def factors(ef_gas):
    return {"coal": 1000.0, "gas": ef_gas, "oil": 700.0,
            "other": 500.0, "nuc": 0.0, "zero": 0.0}


def load():
    """-> {ba: (utc_hour_of_day, demand_MW, {fuelkey: MW})} as arrays."""
    with open(CSVP, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        idx_ba, idx_hr = 0, 2
        idx_dem = head.index("Demand (MW) (Adjusted)")
        cols = {}
        for frag, key in FUEL_COL.items():
            for j, h in enumerate(head):
                if h.startswith("Net Generation (MW)") and h.endswith(frag):
                    cols.setdefault(key, []).append(j)
        data = {b: {"hr": [], "dem": [], "fuel": []} for b in BAS}
        for row in r:
            b = row[idx_ba]
            if b not in data:
                continue
            try:
                dem = float((row[idx_dem] or "").replace(",", ""))
                hr = int(row[idx_hr])
            except ValueError:
                continue
            if not np.isfinite(dem) or dem <= 0:
                continue
            fu = {}
            ok = True
            for key, js in cols.items():
                v = 0.0
                for j in js:
                    s = (row[j] or "").replace(",", "")
                    if s:
                        try:
                            v += float(s)
                        except ValueError:
                            ok = False
                fu[key] = v
            if not ok:
                continue
            data[b]["hr"].append(hr)
            data[b]["dem"].append(dem)
            data[b]["fuel"].append(fu)
    for b in data:
        data[b]["hr"] = np.array(data[b]["hr"])
        data[b]["dem"] = np.array(data[b]["dem"])
    return data


def calibrate(d, ef_gas, flex_share=0.0038, concentrate=0.10, T=24):
    """AEF, MEF, beta, eta and kappa for one balancing authority."""
    F = factors(ef_gas)
    gen = np.array([sum(f.values()) for f in d["fuel"]])
    emis = np.array([sum(F[k] * v for k, v in f.items()) for f in d["fuel"]])  # gCO2/h *1e-3
    dem, hr = d["dem"], d["hr"]
    good = (gen > 0) & (dem > 0)
    gen, emis, dem, hr = gen[good], emis[good], dem[good], hr[good]

    aef_h = emis / gen                                   # gCO2/kWh

    # MEF by first differences within hour-of-day bins
    mefs, betas = [], []
    for h in range(1, T + 1):
        m = hr == h
        if m.sum() < 40:
            continue
        e, D = emis[m], dem[m]
        de, dD = np.diff(e), np.diff(D)
        keep = np.abs(dD) > 1e-6
        de, dD = de[keep], dD[keep]
        Dm = 0.5 * (D[:-1] + D[1:])[keep]
        if len(de) < 30:
            continue
        # MEF = slope of de on dD (through the origin: a zero demand change
        # should imply a zero emissions change)
        mef = float(de @ dD / (dD @ dD))
        # curvature: let the slope vary with the demand level, de ~ (m0+b*Dm)*dD
        X = np.column_stack([dD, dD * (Dm - Dm.mean())])
        coef, *_ = np.linalg.lstsq(X, de, rcond=None)
        mefs.append(mef)
        betas.append(float(coef[1]))                      # gCO2/kWh per MW
    mef_h = np.array(mefs)
    beta_h = np.array(betas)

    # restrict to the hours carbon-aware agents target: the cleanest decile
    k = max(1, int(concentrate * len(aef_h)))
    clean = np.argsort(aef_h)[:k]
    aef_clean = aef_h[clean]

    aef = float(np.mean(aef_clean))
    mef = float(np.median(mef_h))                          # robust across hours
    beta = float(np.median(beta_h))
    eta = 1.0 - aef / mef if mef > 0 else float("nan")
    bound = 1.5 / (1.0 - eta) if eta < 1 else float("inf")

    # kappa: congestion increment vs accounting wedge at the reference share
    daily = float(np.mean(dem) * 24.0)                     # MWh/day
    y_peak = flex_share * daily / max(1, int(concentrate * T))
    kappa = abs(beta) * y_peak / max(mef - aef, 1e-9)

    return dict(aef=aef, mef=mef, beta=beta, eta=eta, bound=bound,
                kappa=float(kappa), load_GWh_d=daily / 1e3,
                aef_all=float(np.mean(aef_h)), n_hours=int(len(dem)))


if __name__ == "__main__":
    if not os.path.exists(CSVP):
        sys.exit(f"missing {CSVP}")
    print("loading EIA-930 ...")
    data = load()
    for b in BAS:
        print(f"  {b}: {len(data[b]['dem'])} hourly records")

    print(f"\n{'BA':>6s} {'load':>8s} {'AEF':>7s} {'MEF':>7s} {'eta':>7s} "
          f"{'bound':>7s} {'beta':>10s} {'kappa':>8s}")
    print("-" * 66)
    res = {}
    for b in BAS:
        c = calibrate(data[b], EF_GAS_DEFAULT)
        res[b] = c
        print(f"{b:>6s} {c['load_GWh_d']:8.0f} {c['aef']:7.1f} {c['mef']:7.1f} "
              f"{c['eta']:7.3f} {c['bound']:7.2f} {c['beta']:10.2e} {c['kappa']:8.4f}")

    print(f"\nsensitivity to the natural-gas factor (gCO2/kWh):")
    print(f"{'BA':>6s} " + " ".join(f"{g:>9d}" for g in (370, 430, 490, 550)))
    print("-" * 50)
    sens = {}
    for b in BAS:
        row = []
        for g in (370, 430, 490, 550):
            c = calibrate(data[b], float(g))
            row.append(c["eta"])
        sens[b] = row
        print(f"{b:>6s} " + " ".join(f"{v:9.3f}" for v in row))

    json.dump({"source": "EIA-930 BALANCE Jul-Dec 2024",
               "ef_gas": EF_GAS_DEFAULT, "regions": res, "eta_sensitivity": sens},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

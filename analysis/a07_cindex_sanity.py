# -*- coding: utf-8 -*-
"""
Reviewer 1 #1: is a test C-index of 0.88 for LC plausible with n = 80?

1. The reviewer's simulation (n = 80, all events, exponential times, LP ~ N(0, sigma^2)) reproduced and extended:
   expected Harrell C for sigma = 1 ... 3, (a) with all events (reviewer's setting) and (b) with censoring that
   mimics our LC test set (about 11% events, follow-up up to ~70 months). Harrell's C depends on the spread
   of the true linear predictor (sigma), not on n; censoring changes which pairs are comparable.
2. The effective spread of each model's prognostic signal in the test set: SD of the test-set linear predictor
   after re-estimating its slope (beta x SD), i.e. the sigma at which the simulation reproduces our C.
3. Censoring-robust discrimination: Uno's IPCW C-index (truncated at 60 months) and the IPCW cumulative/dynamic
   time-dependent AUC at 24 months for the main models.
"""
import os
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from common import ENDPOINTS, MAIN_CONFIGS, RESULTS, load_clinical, load_predictions, cindex, comparable_pairs, cindex_pairs, fit_cox

rng = np.random.default_rng(1)


def harrell(t, e, x):
    I, J = comparable_pairs(t, e)
    return float(cindex_pairs(I, J, x)) if len(I) else np.nan


def simulate(sigma, n=80, censor=False, n_rep=2000, target_event_rate=0.115, follow=(20, 75)):
    out = []
    for _ in range(n_rep):
        x = rng.normal(0, sigma, n)
        if not censor:
            t = -np.log(rng.uniform(size=n)) / (0.1 * np.exp(x))
            e = np.ones(n, int)
        else:
            # baseline hazard chosen so that ~11.5% have the event before administrative censoring
            lam = target_event_rate / 40.0 / np.mean(np.exp(x))
            t_ev = -np.log(rng.uniform(size=n)) / (lam * np.exp(x))
            c = rng.uniform(*follow, n)
            t = np.minimum(t_ev, c)
            e = (t_ev <= c).astype(int)
            if e.sum() < 2:
                continue
        out.append(harrell(t, e, x))
    out = np.array(out)
    return np.median(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


def censor_km(t, e):
    k = KaplanMeierFitter().fit(t, 1 - e)
    return lambda s: np.clip(k.survival_function_at_times(np.atleast_1d(s)).to_numpy(), 1e-8, 1)


def uno_c(t, e, x, tau):
    G = censor_km(t, e)
    num = den = 0.0
    for i in range(len(t)):
        if e[i] != 1 or t[i] >= tau:
            continue
        w = 1.0 / G(t[i])[0] ** 2
        later = t > t[i]
        den += w * later.sum()
        num += w * ((x[i] > x[later]).sum() + 0.5 * (x[i] == x[later]).sum())
    return num / den


def td_auc(t, e, x, horizon):
    G = censor_km(t, e)
    cases = (t <= horizon) & (e == 1)
    controls = t > horizon
    wi = 1.0 / G(t[cases])
    num = sum(w * ((xi > x[controls]).sum() + 0.5 * (xi == x[controls]).sum()) for xi, w in zip(x[cases], wi))
    return num / (wi.sum() * controls.sum())


def main():
    sim = []
    for sigma in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        for cens in (False, True):
            med, lo, hi = simulate(sigma, censor=cens)
            sim.append({'sigma': sigma, 'setting': 'LC-like censoring (~11% events)' if cens else 'all events (reviewer)',
                        'median_C': med, 'p2.5': lo, 'p97.5': hi})
    sim = pd.DataFrame(sim)
    sim.to_csv(os.path.join(RESULTS, 'a07_simulation.csv'), index=False)
    print(sim.round(3).to_string(index=False))

    clin = load_clinical()
    test = clin[clin['is_test']]
    rows = []
    models = ['2D T1_GTV', '2D T2_GTV', '2D T2_GTV_margin', '2D T2_GTV_cmargin', '2D T1_GTV_margin', '2D T1_GTV_cmargin',
              '3D T1_GTV', '3D CT_GTV', '3D PET_GTV']
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        t, e = test[tm].to_numpy(float), test[ev].to_numpy(int)
        risks = {'HNC-PREDICTOR': test[f'hnc_lp_{ep}'].to_numpy()}
        for m in models:
            root, prefix = MAIN_CONFIGS[m]
            risks[m] = load_predictions(root, prefix, ep, clin)['test'].loc[test.index].to_numpy()
        for name, x in risks.items():
            d = pd.DataFrame({'t': t, 'e': e, 'x': x})
            c = fit_cox(d, ['x'], 't', 'e')
            rows.append({'endpoint': ep, 'model': name, 'Harrell_C': cindex(t, e, x),
                         'Uno_C_tau60': uno_c(t, e, x, 60.0), 'tdAUC_24m': td_auc(t, e, x, 24.0),
                         'effective_sigma (beta*SD)': abs(c.params_['x']) * np.std(x),
                         'HR_per_SD': float(np.exp(c.params_['x'] * np.std(x)))})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a07_uno_tdauc_sigma.csv'), index=False)
    pd.set_option('display.width', 200)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

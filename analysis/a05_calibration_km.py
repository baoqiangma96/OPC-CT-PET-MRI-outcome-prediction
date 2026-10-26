# -*- coding: utf-8 -*-
"""
Calibration and risk stratification, re-done (Reviewer 1 #10, #11, #14; Reviewer 2 figures/HR decimals;
Reviewer 3 #10, #12).

Models per endpoint (test set, n = 80):
  * HNC-PREDICTOR, original coefficients: published coefficients kept fixed; only the baseline cumulative
    hazard is estimated in the training set (Breslow, linear predictor as offset).
  * HNC-PREDICTOR, recalibrated: Cox model on the training set with the linear predictor as the only covariate
    (re-estimated slope + baseline); this is what the original manuscript reported for calibration.
  * DL model: Cox on the out-of-fold DL prediction (training set).
  * Hybrid: Cox on HNC-PREDICTOR LP + out-of-fold DL prediction (training set).
Calibration in the test set: calibration slope (Cox regression of outcome on the model's linear predictor),
observed/expected 2-year event probability (Kaplan-Meier vs mean predicted), and the Greenwood-Nam-D'Agostino
(GND) test (3 risk groups for LC/RC with 9/12 events, 4 for OS).
Risk stratification: high vs low risk split at the training-set median (training values are out-of-fold
for DL), log-rank test and Cox hazard ratio (1 decimal); 'not estimable' when a group has no events.
Outputs: results/a05_*.csv, figures/Figure2_revised.png, Figure3_revised.png, FigureA1/A2_revised.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from common import (ENDPOINTS, MAIN_CONFIGS, RESULTS, FIGURES, load_clinical, load_predictions, fit_cox,
                    breslow_baseline, km_risk_at, gnd_test, cindex)

T_EVAL = 24.0
SHORT = {'LC': 'LC', 'RC': 'RC', 'OS': 'OS'}
FIG_MODELS = ['2D T1_GTV', '2D T1_GTV_margin', '2D T1_GTV_cmargin', '2D T2_GTV', '2D T2_GTV_margin', '2D T2_GTV_cmargin']
MAIN_FIG_MODEL = '2D T2_GTV_cmargin'


def build_models(clin, ep):
    """Return dict model -> (train_lp, test_lp, test_risk_2y)."""
    ev, tm, _ = ENDPOINTS[ep]
    test, train = clin[clin['is_test']], clin[~clin['is_test']]
    hnc = f'hnc_lp_{ep}'
    out = {}
    H0 = breslow_baseline(train[tm], train[ev], train[hnc], T_EVAL)
    out['HNC-PREDICTOR (original coefficients)'] = (train[hnc].to_numpy(), test[hnc].to_numpy(),
                                                    1 - np.exp(-H0 * np.exp(test[hnc].to_numpy())))
    c = fit_cox(train, [hnc], tm, ev)
    out['HNC-PREDICTOR (recalibrated)'] = (c.predict_log_partial_hazard(train).to_numpy(),
                                           c.predict_log_partial_hazard(test).to_numpy(),
                                           1 - c.predict_survival_function(test, times=[T_EVAL]).to_numpy()[0])
    for name in FIG_MODELS + ['3D T1_GTV', '3D CT_GTV', '3D PET_GTV']:
        root, prefix = MAIN_CONFIGS[name]
        p = load_predictions(root, prefix, ep, clin)
        tr = train.assign(dl=p['oof_z'].loc[train.index])
        te = test.assign(dl=p['test_z'].loc[test.index])
        c = fit_cox(tr, ['dl'], tm, ev)
        out[name] = (c.predict_log_partial_hazard(tr).to_numpy(), c.predict_log_partial_hazard(te).to_numpy(),
                     1 - c.predict_survival_function(te, times=[T_EVAL]).to_numpy()[0])
        c = fit_cox(tr, [hnc, 'dl'], tm, ev)
        out[f'HNC + {name}'] = (c.predict_log_partial_hazard(tr).to_numpy(), c.predict_log_partial_hazard(te).to_numpy(),
                                1 - c.predict_survival_function(te, times=[T_EVAL]).to_numpy()[0])
    return out


def calib_metrics(test, ep, lp, risk):
    ev, tm, _ = ENDPOINTS[ep]
    d = pd.DataFrame({'t': test[tm].to_numpy(), 'e': test[ev].to_numpy(), 'lp': lp})
    slope_fit = fit_cox(d, ['lp'], 't', 'e')
    slope, slope_lo, slope_hi = (slope_fit.params_['lp'], *slope_fit.confidence_intervals_.loc['lp'].to_numpy())
    obs, var = km_risk_at(d['t'], d['e'], T_EVAL)
    exp_ = float(np.mean(risk))
    groups = 3 if ep in ('LC', 'RC') else 4
    stat, dof, p, tab = gnd_test(d['t'], d['e'], risk, T_EVAL, groups)
    return {'slope': slope, 'slope_lo': slope_lo, 'slope_hi': slope_hi, 'obs_2y': obs, 'exp_2y': exp_,
            'O/E': obs / exp_ if exp_ > 0 else np.nan, 'GND_chi2': stat, 'GND_df': dof, 'GND_p': p,
            'C': cindex(d['t'], d['e'], lp)}, tab


def km_split(test, ep, train_lp, test_lp):
    ev, tm, _ = ENDPOINTS[ep]
    cut = np.median(train_lp)
    hi = test_lp > cut
    t, e = test[tm].to_numpy(), test[ev].to_numpy()
    lr = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
    res = {'cut': cut, 'n_high': int(hi.sum()), 'n_low': int((~hi).sum()), 'ev_high': int(e[hi].sum()),
           'ev_low': int(e[~hi].sum()), 'logrank_p': lr.p_value}
    if e[hi].sum() == 0 or e[~hi].sum() == 0:
        res.update({'HR': np.nan, 'HR_lo': np.nan, 'HR_hi': np.nan})
    else:
        c = fit_cox(pd.DataFrame({'t': t, 'e': e, 'g': hi.astype(int)}), ['g'], 't', 'e')
        res.update({'HR': float(np.exp(c.params_['g'])),
                    'HR_lo': float(np.exp(c.confidence_intervals_.loc['g'].iloc[0])),
                    'HR_hi': float(np.exp(c.confidence_intervals_.loc['g'].iloc[1]))})
    return res, hi


def fmt_p(p):
    if p is None or np.isnan(p):
        return 'n.c.'
    if p < 0.001:
        return 'p < 0.001'
    return f'p = {p:.3f}' if p < 0.01 else f'p = {p:.2f}'


def plot_km(ax, test, ep, hi, res, title):
    ev, tm, _ = ENDPOINTS[ep]
    for mask, lab, col in ((~hi, 'Low risk', '#2b6cb0'), (hi, 'High risk', '#c05621')):
        KaplanMeierFitter().fit(test[tm][mask], test[ev][mask], label=f'{lab} (n={int(mask.sum())})').plot_survival_function(
            ax=ax, ci_show=False, show_censors=True, color=col)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, None)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel(f'{ENDPOINTS[ep][2]}')
    hr = 'HR not estimable\n(no events in low-risk group)' if np.isnan(res['HR']) else \
        f"HR = {res['HR']:.1f} ({res['HR_lo']:.1f}-{res['HR_hi']:.1f})"
    ax.text(0.03, 0.05, f"{hr}\nLog-rank {fmt_p(res['logrank_p'])}", transform=ax.transAxes, fontsize=8)
    ax.legend(loc='lower right', fontsize=7)


def plot_cal(ax, tab, m, title, ep):
    se = np.sqrt(tab['var_km'].clip(lower=0))
    lower = np.minimum(1.96 * se, tab['obs_km'])          # error bars do not go below 0
    upper = np.minimum(1.96 * se, 1 - tab['obs_km'])
    ax.errorbar(tab['pred_mean'], tab['obs_km'], yerr=[lower, upper], fmt='o', color='#2b6cb0', capsize=3)
    top = float(np.nanmax(np.r_[tab['pred_mean'].to_numpy(), (tab['obs_km'] + upper).to_numpy()]))
    lim = min(1.0, max(0.6, np.ceil((top + 0.05) * 10) / 10))
    ax.plot([0, lim], [0, lim], color='grey', ls='--', lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f'Predicted 2-year {ep} event probability')
    ax.set_ylabel(f'Observed (Kaplan-Meier)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    gnd = 'GND test n.c. (events in one group)' if np.isnan(m['GND_p']) else f"GND {fmt_p(m['GND_p'])}"
    ax.text(0.03, 0.97, f"Slope = {m['slope']:.2f} ({m['slope_lo']:.2f} to {m['slope_hi']:.2f})\nO/E = {m['O/E']:.2f}\n{gnd}",
            transform=ax.transAxes, va='top', fontsize=8)


def main():
    clin = load_clinical()
    test = clin[clin['is_test']]
    cal_rows, km_rows, models_all = [], [], {}
    for ep in ('LC', 'RC', 'OS'):
        models_all[ep] = build_models(clin, ep)
        for name, (trlp, telp, risk) in models_all[ep].items():
            m, tab = calib_metrics(test, ep, telp, risk)
            cal_rows.append({'endpoint': ep, 'model': name, **m})
            tab.assign(endpoint=ep, model=name).to_csv(os.path.join(RESULTS, 'a05_gnd_groups.csv'), mode='a', index=False,
                                                       header=not os.path.exists(os.path.join(RESULTS, 'a05_gnd_groups.csv')))
            r, _ = km_split(test, ep, trlp, telp)
            km_rows.append({'endpoint': ep, 'model': name, **r})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(RESULTS, 'a05_calibration_metrics.csv'), index=False)
    km = pd.DataFrame(km_rows)
    km.to_csv(os.path.join(RESULTS, 'a05_km_median_split.csv'), index=False)
    pd.set_option('display.width', 250)
    print(cal.round(3).to_string(index=False))
    print(km.round(3).to_string(index=False))

    # ---- Figure 2 (revised): KM median split, rows = models, columns = endpoints
    rows = ['HNC-PREDICTOR (original coefficients)', MAIN_FIG_MODEL, f'HNC + {MAIN_FIG_MODEL}']
    labels = ['HNC-PREDICTOR', 'T2_GTV_cmargin (2D)', 'HNC-PREDICTOR + T2_GTV_cmargin']
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for j, ep in enumerate(('LC', 'RC', 'OS')):
        for i, (name, lab) in enumerate(zip(rows, labels)):
            trlp, telp, _ = models_all[ep][name]
            r, hi = km_split(test, ep, trlp, telp)
            plot_km(axes[i, j], test, ep, hi, r, f'{ep}: {lab}')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Figure2_revised.png'), dpi=300)
    plt.close(fig)

    # ---- Figure 3 (revised): calibration
    rows3 = ['HNC-PREDICTOR (original coefficients)', 'HNC-PREDICTOR (recalibrated)', MAIN_FIG_MODEL, f'HNC + {MAIN_FIG_MODEL}']
    labels3 = ['HNC-PREDICTOR (original)', 'HNC-PREDICTOR (recalibrated)', 'T2_GTV_cmargin (2D)', 'HNC-PREDICTOR + T2_GTV_cmargin']
    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    for j, ep in enumerate(('LC', 'RC', 'OS')):
        for i, (name, lab) in enumerate(zip(rows3, labels3)):
            trlp, telp, risk = models_all[ep][name]
            m, tab = calib_metrics(test, ep, telp, risk)
            plot_cal(axes[i, j], tab, m, f'{ep}: {lab}', ep)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Figure3_revised.png'), dpi=300)
    plt.close(fig)

    # ---- Figures A.1 / A.2 (revised): all 2D MRI models and their hybrids
    for kind, part in (('KM', 'a'), ('KM', 'b'), ('CAL', 'a'), ('CAL', 'b')):
        names = FIG_MODELS if part == 'a' else [f'HNC + {n}' for n in FIG_MODELS]
        fig, axes = plt.subplots(len(names), 3, figsize=(13, 4.2 * len(names)))
        for i, name in enumerate(names):
            for j, ep in enumerate(('LC', 'RC', 'OS')):
                trlp, telp, risk = models_all[ep][name]
                lab = name.replace('2D ', '').replace('HNC + ', 'HNC-PREDICTOR + ')
                if kind == 'KM':
                    r, hi = km_split(test, ep, trlp, telp)
                    plot_km(axes[i, j], test, ep, hi, r, f'{ep}: {lab}')
                else:
                    m, tab = calib_metrics(test, ep, telp, risk)
                    plot_cal(axes[i, j], tab, m, f'{ep}: {lab}', ep)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, f"FigureA{'1' if kind == 'KM' else '2'}{part}_revised.png"), dpi=150)
        plt.close(fig)
    print('figures written')


if __name__ == '__main__':
    p = os.path.join(RESULTS, 'a05_gnd_groups.csv')
    if os.path.exists(p):
        os.remove(p)
    main()

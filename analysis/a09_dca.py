# -*- coding: utf-8 -*-
"""
Reviewer 3 #12: decision-curve analysis at 2 years on the test set (Vickers et al., BMC Med Inform Decis Mak
2008, survival extension: event probability among 'test-positive' patients from Kaplan-Meier).
Models: HNC-PREDICTOR (original coefficients), recalibrated HNC-PREDICTOR, 2D T2_GTV_cmargin, and the hybrid
HNC-PREDICTOR + 2D T2_GTV_cmargin (predicted 2-year event probabilities from a05_calibration_km.build_models).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from common import ENDPOINTS, RESULTS, FIGURES, load_clinical
from a05_calibration_km import build_models, MAIN_FIG_MODEL

T_EVAL = 24.0
THRESH = {'LC': np.arange(0.02, 0.41, 0.01), 'RC': np.arange(0.02, 0.41, 0.01), 'OS': np.arange(0.02, 0.61, 0.01)}


def km_risk(t, e):
    if len(t) == 0:
        return 0.0
    k = KaplanMeierFitter().fit(t, e)
    return 1 - float(k.survival_function_at_times(T_EVAL).iloc[0])


def net_benefit(t, e, risk, pt):
    pos = risk >= pt
    p_pos = pos.mean()
    if p_pos == 0:
        return 0.0
    r = km_risk(t[pos], e[pos])
    return r * p_pos - (1 - r) * p_pos * pt / (1 - pt)


def main():
    clin = load_clinical()
    test = clin[clin['is_test']]
    names = {'HNC-PREDICTOR (original coefficients)': 'HNC-PREDICTOR',
             'HNC-PREDICTOR (recalibrated)': 'HNC-PREDICTOR (recalibrated)',
             MAIN_FIG_MODEL: 'T2_GTV_cmargin (2D)', f'HNC + {MAIN_FIG_MODEL}': 'HNC-PREDICTOR + T2_GTV_cmargin'}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, ep in zip(axes, ('LC', 'RC', 'OS')):
        ev, tm, lab = ENDPOINTS[ep]
        t, e = test[tm].to_numpy(float), test[ev].to_numpy(int)
        models = build_models(clin, ep)
        r_all = km_risk(t, e)
        for pt in THRESH[ep]:
            rows.append({'endpoint': ep, 'threshold': pt, 'model': 'Treat all', 'NB': r_all - (1 - r_all) * pt / (1 - pt)})
            for k, disp in names.items():
                rows.append({'endpoint': ep, 'threshold': pt, 'model': disp, 'NB': net_benefit(t, e, models[k][2], pt)})
        d = pd.DataFrame([r for r in rows if r['endpoint'] == ep])
        for m, g in d.groupby('model', sort=False):
            ax.plot(g['threshold'], g['NB'], label=m, lw=1.8 if 'HNC-PREDICTOR +' in m else 1.2,
                    ls='--' if m == 'Treat all' else '-')
        ax.axhline(0, color='black', lw=0.8, label='Treat none')
        ax.set_ylim(-0.05, max(0.05, r_all + 0.03))
        ax.set_xlabel(f'Threshold 2-year {ep} event probability')
        ax.set_ylabel('Net benefit')
        ax.set_title(f'{ep}: decision curve (test set)', fontweight='bold')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'FigureA_DCA.png'), dpi=200)
    plt.close(fig)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a09_dca.csv'), index=False)
    # summary: mean net benefit difference hybrid - HNC-PREDICTOR over clinically relevant thresholds
    rng = {'LC': (0.05, 0.25), 'RC': (0.05, 0.25), 'OS': (0.1, 0.4)}
    for ep in ('LC', 'RC', 'OS'):
        d = res[(res['endpoint'] == ep) & res['threshold'].between(*rng[ep])].pivot(index='threshold', columns='model', values='NB')
        diff = d['HNC-PREDICTOR + T2_GTV_cmargin'] - d['HNC-PREDICTOR']
        print(f'{ep}: thresholds {rng[ep]}, mean NB hybrid - HNC-PREDICTOR = {diff.mean():+.4f} '
              f'(range {diff.min():+.4f} to {diff.max():+.4f}); fraction of thresholds hybrid >= HNC: {(diff >= 0).mean():.2f}')


if __name__ == '__main__':
    main()

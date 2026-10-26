# -*- coding: utf-8 -*-
"""
Reviewer 2 (major): comparison with the Ang et al. (NEJM 2010) RPA risk groups, re-done.

Ang groups (RTOG 0129 recursive partitioning, OS):
  low          : HPV+ and <=10 pack-years; or HPV+, >10 pack-years and N0-N2a
  intermediate : HPV+, >10 pack-years and N2b-N3; or HPV-, <=10 pack-years and T2-T3
  high         : HPV-, <=10 pack-years and T4; or HPV-, >10 pack-years
Eligible patients (as in the manuscript): known HPV status, WHO PS < 2, T-stage > 1 (HPV-negative T1 patients
with <=10 pack-years cannot be assigned by the RPA).
Hybrid model groups: HNC-PREDICTOR + 2D MRI model (Cox fitted on the training set with out-of-fold DL
predictions); three groups defined by the tertiles of the hybrid risk in the TRAINING set (no test
information), applied to (a) the eligible TEST patients only (primary, fully independent) and (b) all eligible
patients, using out-of-fold predictions for training patients (secondary). The original Figure 4 used tertiles
of in-sample predictions within the 188 patients themselves.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
from common import ENDPOINTS, MAIN_CONFIGS, RESULTS, FIGURES, load_clinical, load_predictions, fit_cox, cindex

HYBRIDS = ['2D T2_GTV_margin', '2D T2_GTV_cmargin']


def ang_group(r):
    hpv_pos = r['P16_codes'] == 2
    py = r['PACK_YEARS']
    n = r['NSTAD_codes']  # 0 N0, 1 N1, 2 N2a, 3 N2b, 4 N2c, 5 N3 (AJCC 7th)
    t = r['TSTAD_codes']  # 0 T1 ... 3 T4
    if hpv_pos:
        if py <= 10 or n <= 2:
            return 0
        return 1
    if py <= 10:
        if t in (1, 2):
            return 1
        if t == 3:
            return 2
        return np.nan
    return 2


def group_stats(df, gcol, tm, ev):
    out = {'n_low': int((df[gcol] == 0).sum()), 'n_int': int((df[gcol] == 1).sum()), 'n_high': int((df[gcol] == 2).sum())}
    lr = multivariate_logrank_test(df[tm], df[gcol], df[ev])
    out['logrank_p'] = lr.p_value
    out['C_groups'] = cindex(df[tm], df[ev], df[gcol])
    for ref, cmp_, lab in ((0, 2, 'HR_high_vs_low'), (0, 1, 'HR_int_vs_low'), (1, 2, 'HR_high_vs_int')):
        sub = df[df[gcol].isin([ref, cmp_])].assign(g=lambda d: (d[gcol] == cmp_).astype(int))
        if sub.loc[sub['g'] == 0, ev].sum() == 0 or sub.loc[sub['g'] == 1, ev].sum() == 0:
            out[lab] = np.nan
            continue
        c = fit_cox(sub, ['g'], tm, ev)
        out[lab] = float(np.exp(c.params_['g']))
        out[lab + '_ci'] = '{:.1f}-{:.1f}'.format(*np.exp(c.confidence_intervals_.loc['g'].to_numpy()))
    return out


def km_panel(ax, df, gcol, tm, ev, title, stats):
    fitters = []
    for g, lab, col in ((0, 'Low', '#2b6cb0'), (1, 'Intermediate', '#d69e2e'), (2, 'High', '#c53030')):
        m = df[gcol] == g
        if m.sum() == 0:
            continue
        k = KaplanMeierFitter().fit(df.loc[m, tm], df.loc[m, ev], label=f'{lab} risk')
        k.plot_survival_function(ax=ax, ci_show=True, color=col, show_censors=True)
        fitters.append(k)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 84)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Overall survival')
    hr = lambda k: 'n.e.' if pd.isna(stats.get(k)) else f"{stats[k]:.1f} ({stats.get(k + '_ci', '')})"
    lp = 'Log-rank p < 0.001' if stats['logrank_p'] < 0.001 else f"Log-rank p = {stats['logrank_p']:.3f}"
    ax.text(0.02, 0.04, f"HR high vs low: {hr('HR_high_vs_low')}\nHR int vs low: {hr('HR_int_vs_low')}\n"
                        f"{lp}; C = {stats['C_groups']:.2f}",
            transform=ax.transAxes, fontsize=8)
    ax.legend(loc='upper right', fontsize=8)
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk'], fontsize=8)


def main():
    clin = load_clinical()
    ep = 'OS'
    ev, tm, _ = ENDPOINTS[ep]
    clin['ang'] = clin.apply(ang_group, axis=1)
    elig = (clin['P16_codes'] != 1) & (clin['WHO_SCORE_codes'] < 2) & (clin['TSTAD_codes'] > 0) & clin['PACK_YEARS'].notna()
    clin['ang_eligible'] = elig & clin['ang'].notna()
    print('Eligible for Ang comparison:', int(clin['ang_eligible'].sum()),
          '| of which test:', int((clin['ang_eligible'] & clin['is_test']).sum()))
    print(pd.crosstab(clin.loc[clin['ang_eligible'], 'ang'], clin.loc[clin['ang_eligible'], 'CVgroup_event_t1t2']))
    train, test = clin[~clin['is_test']], clin[clin['is_test']]
    hnc = 'hnc_lp_OS'
    # HNC-PREDICTOR alone, tertile cut-offs from training
    q = np.quantile(train[hnc], [1 / 3, 2 / 3])
    clin['hnc3'] = np.digitize(clin[hnc], q)
    rows, panels = [], {}
    for name in HYBRIDS:
        root, prefix = MAIN_CONFIGS[name]
        p = load_predictions(root, prefix, ep, clin)
        tr = train.assign(dl=p['oof_z'].loc[train.index])
        c = fit_cox(tr, [hnc, 'dl'], tm, ev)
        dl_all = pd.concat([p['oof_z'].loc[train.index], p['test_z'].loc[test.index]])
        lp = c.params_[hnc] * clin[hnc] + c.params_['dl'] * dl_all.reindex(clin.index)
        qh = np.quantile(lp.loc[train.index], [1 / 3, 2 / 3])
        clin[f'hyb3_{name}'] = np.digitize(lp, qh)
    for subset, mask in (('test (eligible)', clin['ang_eligible'] & clin['is_test']),
                         ('all eligible (OOF for training patients)', clin['ang_eligible'])):
        d = clin[mask]
        for gname, gcol in [('Ang RPA', 'ang'), ('HNC-PREDICTOR (tertiles)', 'hnc3')] + \
                           [(f'HNC-PREDICTOR + {h[3:]} (tertiles)', f'hyb3_{h}') for h in HYBRIDS]:
            s = group_stats(d, gcol, tm, ev)
            rows.append({'subset': subset, 'model': gname, 'n': len(d), 'events': int(d[ev].sum()), **s})
            panels[(subset, gname)] = (d, gcol, s)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a06_ang_comparison.csv'), index=False)

    # paired bootstrap Delta-C of the 3-level groupings versus Ang (test eligible and all eligible)
    from common import bootstrap_cindex
    drows = []
    for subset, mask in (('test (eligible)', clin['ang_eligible'] & clin['is_test']),
                         ('all eligible (OOF for training patients)', clin['ang_eligible'])):
        d = clin[mask]
        cols = {'Ang RPA': d['ang'], 'HNC-PREDICTOR (tertiles)': d['hnc3']}
        cols.update({f'HNC-PREDICTOR + {h[3:]} (tertiles)': d[f'hyb3_{h}'] for h in HYBRIDS})
        boot = bootstrap_cindex(d[tm], d[ev], {k: v.to_numpy() for k, v in cols.items()}, n_boot=2000, seed=5)
        for k in cols:
            if k == 'Ang RPA':
                continue
            diff = boot[k] - boot['Ang RPA']
            drows.append({'subset': subset, 'model': k, 'dC_vs_Ang': cindex(d[tm], d[ev], cols[k]) - cindex(d[tm], d[ev], cols['Ang RPA']),
                          'lo': np.percentile(diff, 2.5), 'hi': np.percentile(diff, 97.5),
                          'p': min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))})
    dres = pd.DataFrame(drows)
    dres.to_csv(os.path.join(RESULTS, 'a06_deltaC_vs_ang.csv'), index=False)
    print(dres.round(3).to_string(index=False))
    pd.set_option('display.width', 250)
    print(res.round(3).to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    for i, subset in enumerate(('test (eligible)', 'all eligible (OOF for training patients)')):
        for j, gname in enumerate(('Ang RPA', 'HNC-PREDICTOR (tertiles)', 'HNC-PREDICTOR + T2_GTV_margin (tertiles)')):
            d, gcol, s = panels[(subset, gname)]
            km_panel(axes[i, j], d, gcol, tm, ev, f"{gname}\n{subset}, n={len(d)}", s)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'Figure4_revised.png'), dpi=300)
    plt.close(fig)
    clin[['ang', 'ang_eligible', 'hnc3'] + [f'hyb3_{h}' for h in HYBRIDS]].to_csv(os.path.join(RESULTS, 'a06_patient_groups.csv'))
    print('figure written')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Revised Table 1 / Table A.3 (Reviewer 1 #3 and #6, Reviewer 3 #10-#12, Reviewer 2 major comment).

For LC, RC and OS on the 80-patient test set:
  * HNC-PREDICTOR with its published coefficients;
  * every 2D/3D image model: 9-model ensemble (3 folds x 3 seeds) of saved test predictions;
  * hybrid models refitted correctly: Cox(HNC-PREDICTOR linear predictor + out-of-fold DL prediction) on the
    186 training patients, applied to the test set (the original analysis used in-sample DL predictions and
    exp(LP); that version is kept for reference);
  * 95% CIs from 2000 paired bootstrap resamples of test patients and paired bootstrap Delta-C versus
    HNC-PREDICTOR (this replaces the Z-test, whose effective sample size was inflated);
  * Wald test of the DL prediction added to the HNC-PREDICTOR LP in a Cox model fitted on the test set
    (incremental prognostic value in independent data).
"""
import os
import numpy as np
import pandas as pd
from common import (ENDPOINTS, MAIN_CONFIGS, EXTRA_CONFIGS, RESULTS, load_clinical, load_predictions,
                    cindex, bootstrap_cindex, delta_summary, fit_cox)

N_BOOT = 2000


def main():
    clin = load_clinical()
    rows, groups_rows = [], []
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        test = clin[clin['is_test']].copy()
        train = clin[~clin['is_test']].copy()
        hnc_col = f'hnc_lp_{ep}'
        risks = {'HNC-PREDICTOR': test[hnc_col].to_numpy()}
        meta = {}
        configs = {**MAIN_CONFIGS, **EXTRA_CONFIGS}
        for name, (root, prefix) in configs.items():
            p = load_predictions(root, prefix, ep, clin)
            if p is None or p['n_runs'] < 9:
                continue
            dl_test = p['test'].loc[test.index].to_numpy()
            risks[name] = dl_test
            risks[f'{name} [z-ensemble]'] = p['test_z'].loc[test.index].to_numpy()
            # corrected hybrid: LP + out-of-fold DL, both DL inputs on the standardised (z) scale
            dlz_test = p['test_z'].loc[test.index].to_numpy()
            tr = train.assign(dl=p['oof_z'].loc[train.index].to_numpy())
            cph = fit_cox(tr, [hnc_col, 'dl'], tm, ev)
            b = cph.params_
            risks[f'HNC + {name}'] = b[hnc_col] * test[hnc_col].to_numpy() + b['dl'] * dlz_test
            # original (published) hybrid: exp(LP) + in-sample DL
            tr0 = train.assign(clc=np.exp(train[hnc_col]), dl=p['insample'].loc[train.index].to_numpy())
            cph0 = fit_cox(tr0, ['clc', 'dl'], tm, ev)
            b0 = cph0.params_
            risks[f'HNC + {name} [original method]'] = b0['clc'] * np.exp(test[hnc_col].to_numpy()) + b0['dl'] * dl_test
            # incremental value in the independent test set
            te = test.assign(dl=dl_test)
            cte = fit_cox(te, [hnc_col, 'dl'], tm, ev)
            meta[name] = {'n_runs': p['n_runs'],
                          'C_z_ensemble': cindex(test[tm], test[ev], dlz_test),
                          'train_hybrid_coef_dl': b['dl'], 'train_hybrid_p_dl': cph.summary.loc['dl', 'p'],
                          'test_added_value_HR_per_SD': float(np.exp(cte.params_['dl'] * np.std(dl_test))),
                          'test_added_value_p': cte.summary.loc['dl', 'p'],
                          'oof_z_cindex_train': cindex(train[tm], train[ev], p['oof_z'].loc[train.index]),
                          'insample_cindex_train': cindex(train[tm], train[ev], p['insample'].loc[train.index])}

        boot = bootstrap_cindex(test[tm], test[ev], risks, n_boot=N_BOOT)
        for name, r in risks.items():
            c = cindex(test[tm], test[ev], r)
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            row = {'endpoint': ep, 'model': name, 'C': c, 'C_lo': lo, 'C_hi': hi,
                   'n_events_test': int(test[ev].sum())}
            if name != 'HNC-PREDICTOR':
                d, dlo, dhi, dp = delta_summary(boot, name, 'HNC-PREDICTOR')
                row.update({'dC_vs_HNC': c - cindex(test[tm], test[ev], risks['HNC-PREDICTOR']),
                            'dC_lo': dlo, 'dC_hi': dhi, 'dC_p': dp})
            base = name.replace('HNC + ', '').replace(' [original method]', '')
            if base in meta and not name.startswith('HNC + ') and name in meta:
                row.update(meta[base])
            if name.startswith('HNC + ') and not name.endswith('[original method]'):
                # DL model vs its hybrid
                d2, lo2, hi2, p2 = delta_summary(boot, name, base)
                row.update({'dC_vs_DLonly_lo': lo2, 'dC_vs_DLonly_hi': hi2, 'dC_vs_DLonly_p': p2})
            rows.append(row)

        # Modality comparison (Reviewer 2: how was 'improvement 0.12/0.10/0.04' computed?)
        mri3d = [k for k in risks if k.startswith('3D T') and ' ' not in k[3:]]
        ctpet3d = [k for k in risks if (k.startswith('3D CT') or k.startswith('3D PET')) and ' ' not in k[3:]]
        mri2d = [k for k in risks if k.startswith('2D T1_GTV') or k.startswith('2D T2_GTV')]
        mri2d = [k for k in mri2d if '10mm' not in k and 'tot' not in k and 'z-ensemble' not in k]
        comps = {'mean 3D MRI - mean 3D CT/PET': (mri3d, ctpet3d),
                 'mean 2D MRI - mean 3D CT/PET': (mri2d, ctpet3d),
                 'mean 2D MRI - mean 3D MRI': (mri2d, mri3d)}
        for label, (A, B) in comps.items():
            obs = np.mean([cindex(test[tm], test[ev], risks[k]) for k in A]) - np.mean([cindex(test[tm], test[ev], risks[k]) for k in B])
            diff = boot[A].mean(axis=1) - boot[B].mean(axis=1)
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))
            groups_rows.append({'endpoint': ep, 'comparison': label, 'n_A': len(A), 'n_B': len(B),
                                'diff': obs, 'lo': lo, 'hi': hi, 'p': p})
        # 'best vs best' as in the original text
        best_mri = max(mri3d, key=lambda k: cindex(test[tm], test[ev], risks[k]))
        ct = [k for k in ctpet3d if k.startswith('3D CT')]
        best_ct = max(ct, key=lambda k: cindex(test[tm], test[ev], risks[k]))
        d, lo, hi, p = delta_summary(boot, best_mri, best_ct)
        groups_rows.append({'endpoint': ep, 'comparison': f'best 3D MRI ({best_mri}) - best 3D CT ({best_ct})',
                            'n_A': 1, 'n_B': 1,
                            'diff': cindex(test[tm], test[ev], risks[best_mri]) - cindex(test[tm], test[ev], risks[best_ct]),
                            'lo': lo, 'hi': hi, 'p': p})
        print(f'{ep}: done ({len(risks)} models, {len(boot)} valid bootstrap samples)')

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a01_cindex_all_models.csv'), index=False)
    pd.DataFrame(groups_rows).to_csv(os.path.join(RESULTS, 'a01_modality_comparisons.csv'), index=False)

    # Formatted revised Table 1 (wide: LC | RC | OS)
    def fmt(r):
        s = f"{r['C']:.2f} [{r['C_lo']:.2f}-{r['C_hi']:.2f}]"
        return s
    res['C_fmt'] = res.apply(fmt, axis=1)
    res['dC_fmt'] = res.apply(lambda r: '' if pd.isna(r.get('dC_vs_HNC')) else
                              f"{r['dC_vs_HNC']:+.2f} [{r['dC_lo']:+.2f} to {r['dC_hi']:+.2f}], p={r['dC_p']:.2f}", axis=1)
    wide = res.pivot(index='model', columns='endpoint', values='C_fmt')[['LC', 'RC', 'OS']]
    wide_d = res.pivot(index='model', columns='endpoint', values='dC_fmt')[['LC', 'RC', 'OS']]
    order = ['HNC-PREDICTOR'] + [k for k in MAIN_CONFIGS] + [f'HNC + {k}' for k in MAIN_CONFIGS] + \
            [k for k in EXTRA_CONFIGS] + [f'HNC + {k}' for k in EXTRA_CONFIGS] + \
            [f'HNC + {k} [original method]' for k in list(MAIN_CONFIGS) + list(EXTRA_CONFIGS)]
    order = [o for o in order if o in wide.index]
    wide.loc[order].to_csv(os.path.join(RESULTS, 'a01_table1_revised_C.csv'))
    wide_d.loc[order].to_csv(os.path.join(RESULTS, 'a01_table1_revised_deltaC_vs_HNC.csv'))
    print(wide.loc[order].to_string())
    print()
    print(pd.DataFrame(groups_rows).round(3).to_string())


if __name__ == '__main__':
    main()

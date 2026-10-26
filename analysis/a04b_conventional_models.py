# -*- coding: utf-8 -*-
"""
Reviewer 1 #4: conventional imaging baselines and their combination with HNC-PREDICTOR.
  * GTVp volume (CT contour; also the contour mapped to T1/T2 MRI), PET SUVmax, SUVmean, MTV (SUV>=2.5),
    TLG; log-transformed for volume/MTV/TLG.
  * Univariable Cox on the training set -> test C-index [bootstrap 95% CI]; Cox with HNC-PREDICTOR LP +
    feature on the training set -> test C-index, Delta-C versus HNC-PREDICTOR.
  * Does the MRI DL prediction add to GTV volume? Cox(volume + DL out-of-fold) on training -> test C and
    test-set Wald p of DL adjusted for volume; Spearman correlation of DL test prediction with volume.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import (ENDPOINTS, MAIN_CONFIGS, RESULTS, load_clinical, load_predictions, cindex, bootstrap_cindex,
                    delta_summary, fit_cox)

FEATURES = {'GTVp volume (CT)': ('vol_ct_cc', True), 'GTVp volume (T1 MRI)': ('vol_t1_cc', True),
            'GTVp volume (T2 MRI)': ('vol_t2_cc', True), 'SUVmax': ('suvmax', False), 'SUVmean': ('suvmean', False),
            'MTV (SUV>=2.5)': ('mtv25_cc', True), 'TLG': ('tlg25', True)}
DL_FOR_ADJ = ['2D T1_GTV', '2D T2_GTV', '2D T2_GTV_cmargin', '2D T1_GTV_cmargin', '2D T2_GTV_margin', '3D T1_GTV']


def main():
    clin = load_clinical()
    feats = pd.read_csv(os.path.join(RESULTS, 'a04_conventional_features.csv')).set_index('PatientID')
    clin = clin.join(feats, how='left')
    for name, (col, log) in FEATURES.items():
        clin[f'f_{col}'] = np.log1p(clin[col]) if log else clin[col]
    test, train = clin[clin['is_test']], clin[~clin['is_test']]
    desc = clin.groupby('CVgroup_event_t1t2')[[c for c, _ in FEATURES.values()]].median().T
    desc.to_csv(os.path.join(RESULTS, 'a04_feature_medians_by_set.csv'))
    print('Feature medians:\n', desc.round(2).to_string())
    print('Spearman CT vs T1 volume:', round(spearmanr(clin['vol_ct_cc'], clin['vol_t1_cc'])[0], 3),
          '| CT vs T2:', round(spearmanr(clin['vol_ct_cc'], clin['vol_t2_cc'])[0], 3))

    rows, adj_rows = [], []
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        hnc = f'hnc_lp_{ep}'
        risks = {'HNC-PREDICTOR': test[hnc].to_numpy()}
        meta = {}
        for name, (col, _) in FEATURES.items():
            f = f'f_{col}'
            c1 = fit_cox(train, [f], tm, ev)
            risks[name] = c1.params_[f] * test[f].to_numpy()
            c2 = fit_cox(train, [hnc, f], tm, ev)
            risks[f'HNC + {name}'] = c2.params_[hnc] * test[hnc].to_numpy() + c2.params_[f] * test[f].to_numpy()
            meta[name] = {'train_HR_per_unit': float(np.exp(c1.params_[f])), 'train_p': c1.summary.loc[f, 'p']}
        # DL adjusted for volume
        fv = 'f_vol_ct_cc'
        for dl_name in DL_FOR_ADJ:
            root, prefix = MAIN_CONFIGS[dl_name]
            p = load_predictions(root, prefix, ep, clin)
            tr = train.assign(dl=p['oof_z'].loc[train.index])
            te = test.assign(dl=p['test_z'].loc[test.index])
            risks[dl_name] = p['test'].loc[test.index].to_numpy()
            c = fit_cox(tr, [fv, 'dl'], tm, ev)
            risks[f'Volume + {dl_name}'] = c.params_[fv] * te[fv].to_numpy() + c.params_['dl'] * te['dl'].to_numpy()
            ct = fit_cox(te, [fv, 'dl'], tm, ev)
            ct_h = fit_cox(te, [hnc, fv, 'dl'], tm, ev)
            rho = spearmanr(te['dl'], te['vol_ct_cc'])[0]
            adj_rows.append({'endpoint': ep, 'model': dl_name, 'spearman_DL_vs_volume_test': rho,
                             'test_p_DL_adj_volume': ct.summary.loc['dl', 'p'],
                             'test_p_volume_adj_DL': ct.summary.loc[fv, 'p'],
                             'test_p_DL_adj_volume_HNC': ct_h.summary.loc['dl', 'p']})
        boot = bootstrap_cindex(test[tm], test[ev], risks, n_boot=2000, seed=21)
        for name, r in risks.items():
            c = cindex(test[tm], test[ev], r)
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            row = {'endpoint': ep, 'model': name, 'C': c, 'lo': lo, 'hi': hi}
            if name != 'HNC-PREDICTOR':
                _, dlo, dhi, dp = delta_summary(boot, name, 'HNC-PREDICTOR')
                row.update({'dC_vs_HNC': c - cindex(test[tm], test[ev], risks['HNC-PREDICTOR']), 'dC_lo': dlo, 'dC_hi': dhi, 'dC_p': dp})
            if name.startswith('Volume + '):
                _, dlo, dhi, dp = delta_summary(boot, name, 'GTVp volume (CT)')
                row.update({'dC_vs_volume_lo': dlo, 'dC_vs_volume_hi': dhi, 'dC_vs_volume_p': dp})
            row.update(meta.get(name, {}))
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a04_conventional_models.csv'), index=False)
    adj = pd.DataFrame(adj_rows)
    adj.to_csv(os.path.join(RESULTS, 'a04_dl_vs_volume.csv'), index=False)
    pd.set_option('display.width', 250)
    res['fmt'] = res.apply(lambda r: f"{r['C']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]", axis=1)
    wide = res.pivot(index='model', columns='endpoint', values='fmt')[['LC', 'RC', 'OS']]
    wide.to_csv(os.path.join(RESULTS, 'a04_conventional_models_wide.csv'))
    print(wide.to_string())
    print(res[['endpoint', 'model', 'dC_vs_HNC', 'dC_lo', 'dC_hi', 'dC_p', 'dC_vs_volume_p', 'train_HR_per_unit', 'train_p']].round(3).to_string(index=False))
    print(adj.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

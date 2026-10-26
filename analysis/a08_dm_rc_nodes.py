# -*- coding: utf-8 -*-
"""
Editor comment 4 (distant metastasis) and Reviewer 1 #5 (why is RC worse than LC?).

  * Nodal tumour burden: GTVn volume from the clinical CT contours (ct_GTVln, 1 mm grid).
  * DM (MET_code) on the test set: the trained-but-unreported 2D MRI models (GTVp and GTVp+GTVn inputs),
    nodal volume, N-stage and HPV (simple Cox models fitted on the training set).
  * RC on the test set: GTVp-only MRI models vs GTVp+GTVn (GTVtot) MRI models vs GTVn volume vs HNC-PREDICTOR,
    and the combination HNC-PREDICTOR + GTVn volume.
  * How many patients have lymph-node contours mapped to MRI (MRI field of view).
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from common import (ENDPOINTS, ROOT_2D, CONFIGS_2D, CONFIGS_2D_EXTRA, DATA_2D, DATA_3D, RESULTS, load_clinical,
                    load_predictions, cindex, bootstrap_cindex, delta_summary, fit_cox)


def main():
    clin = load_clinical()
    vol_p = os.path.join(RESULTS, 'a08_gtvn_volume.csv')
    if not os.path.exists(vol_p):
        rows = []
        for pid in clin.index:
            img = sitk.ReadImage(os.path.join(DATA_3D, f'{pid}_ct_GTVln.nii.gz'))
            a = sitk.GetArrayFromImage(img) > 0
            rows.append({'PatientID': pid, 'vol_ln_cc': a.sum() * np.prod(img.GetSpacing()) / 1000.0,
                         'ln_on_t1': os.path.exists(os.path.join(DATA_2D, f'{pid}_gtvlntot1.nii.gz')),
                         'ln_on_t2': os.path.exists(os.path.join(DATA_2D, f'{pid}_gtvlntot2.nii.gz'))})
        pd.DataFrame(rows).to_csv(vol_p, index=False)
    vol = pd.read_csv(vol_p).set_index('PatientID')
    clin = clin.join(vol)
    feats = pd.read_csv(os.path.join(RESULTS, 'a04_conventional_features.csv')).set_index('PatientID')
    clin = clin.join(feats[['vol_ct_cc']])
    clin['f_ln'] = np.log1p(clin['vol_ln_cc'])
    clin['f_vol'] = np.log1p(clin['vol_ct_cc'])
    clin['N_ord'] = clin['NSTAD_codes_N01VSN2VSN3']
    clin['HPVpos'] = (clin['P16_codes'] == 2).astype(int)
    print('Patients with GTVn on CT:', int((clin['vol_ln_cc'] > 0).sum()), '/', len(clin),
          '| LN contour present on T1 grid:', int(clin['ln_on_t1'].sum()), '| on T2 grid:', int(clin['ln_on_t2'].sum()))
    test, train = clin[clin['is_test']], clin[~clin['is_test']]

    rows = []
    for ep in ('DM', 'RC'):
        ev, tm, _ = ENDPOINTS[ep]
        risks = {}
        if ep == 'RC':
            risks['HNC-PREDICTOR'] = test['hnc_lp_RC'].to_numpy()
        for name, covs in {'GTVn volume (CT)': ['f_ln'], 'GTVp volume (CT)': ['f_vol'], 'N-stage': ['N_ord'],
                           'N-stage + HPV': ['N_ord', 'HPVpos'], 'GTVp + GTVn volume': ['f_vol', 'f_ln']}.items():
            c = fit_cox(train, covs, tm, ev)
            risks[name] = test[covs].to_numpy() @ c.params_[covs].to_numpy()
        if ep == 'RC':
            c = fit_cox(train, ['hnc_lp_RC', 'f_ln'], tm, ev)
            risks['HNC-PREDICTOR + GTVn volume'] = test[['hnc_lp_RC', 'f_ln']].to_numpy() @ c.params_.to_numpy()
        for name, prefix in {**CONFIGS_2D, **{k: v for k, v in CONFIGS_2D_EXTRA.items() if 'tot' in k}}.items():
            p = load_predictions(ROOT_2D, prefix, ep, clin)
            if p is None or p['n_runs'] < 9:
                continue
            risks[name] = p['test'].loc[test.index].to_numpy()
        boot = bootstrap_cindex(test[tm], test[ev], risks, n_boot=2000, seed=31)
        ref = 'HNC-PREDICTOR' if ep == 'RC' else 'N-stage'
        for name, r in risks.items():
            c = cindex(test[tm], test[ev], r)
            lo, hi = np.percentile(boot[name], [2.5, 97.5])
            row = {'endpoint': ep, 'model': name, 'C': c, 'lo': lo, 'hi': hi, 'events_test': int(test[ev].sum())}
            if name != ref:
                _, dlo, dhi, dp = delta_summary(boot, name, ref)
                row.update({'ref': ref, 'dC': c - cindex(test[tm], test[ev], risks[ref]), 'dC_lo': dlo, 'dC_hi': dhi, 'dC_p': dp})
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'a08_dm_rc_nodes.csv'), index=False)
    pd.set_option('display.width', 220)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Cohort description and subgroup analyses.
  * Reviewer 3 #6: events per cross-validation fold, per endpoint.
  * Reviewer 2 (Materials and Methods): train/test balance by MRI field strength (1.5 T vs 3 T), treatment
    year and RT technique; test-set performance per field-strength subgroup.
  * Reviewer 1 #2: test-set performance in HPV-negative and HPV-positive patients.
  * Reviewer 2 (Appendix): MRI acquisition parameters (Table B.2) recomputed for exactly the 266 patients,
    and the interval between MRI and start of radiotherapy.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from lifelines import CoxPHFitter
from common import (ENDPOINTS, MAIN_CONFIGS, RESULTS, MRI_PARAMS_CSV, load_clinical, load_predictions, cindex,
                    bootstrap_cindex, fit_cox)

KEY_MODELS = ['2D T1_GTV', '2D T1_GTV_margin', '2D T1_GTV_cmargin', '2D T2_GTV', '2D T2_GTV_margin',
              '2D T2_GTV_cmargin', '3D T1_GTV', '3D CT_GTV', '3D PET_GTV']


def mri_params(clin):
    mp = pd.read_csv(MRI_PARAMS_CSV)
    mp['PatientID'] = 'UMCG-' + mp['Patient_ID'].astype(str).str.zfill(7)
    mp = mp[mp['PatientID'].isin(clin.index)].copy()
    name = mp['sequence_name'].str.lower()
    mp['seq'] = np.where(name.str.contains('t1'), 'T1', np.where(name.str.contains('t2'), 'T2', 'other'))
    return mp


def main():
    clin = load_clinical()
    test, train = clin[clin['is_test']], clin[~clin['is_test']]
    out = {}

    # ---- events per fold (R3 #6)
    rows = []
    for ep in ('LC', 'RC', 'OS', 'DM'):
        ev = ENDPOINTS[ep][0]
        col = f'CVgroup_{ev}_t1t2'
        for grp in ('CV1', 'CV2', 'CV3', 'test'):
            g = clin[clin[col] == grp]
            rows.append({'endpoint': ep, 'set': grp, 'n': len(g), 'events': int(g[ev].sum())})
    epf = pd.DataFrame(rows).pivot(index='set', columns='endpoint', values='events')
    epf['n'] = pd.DataFrame(rows).groupby('set')['n'].first()
    epf.to_csv(os.path.join(RESULTS, 'a03_events_per_fold.csv'))
    print('Events per fold:\n', epf.to_string())

    # ---- MRI parameters: field strength per patient, interval MRI -> RT
    mp = mri_params(clin)
    t1 = mp[mp['seq'] == 'T1'].sort_values('scan_date').groupby('PatientID').last()
    t2 = mp[mp['seq'] == 'T2'].sort_values('scan_date').groupby('PatientID').last()
    clin['field_T'] = t1['magnetic_field_strength'].reindex(clin.index)
    clin['field_T2seq'] = t2['magnetic_field_strength'].reindex(clin.index)
    clin['mri_date'] = pd.to_datetime(t1['scan_date'].astype(str), format='%Y%m%d', errors='coerce').reindex(clin.index)
    clin['rt_date'] = pd.to_datetime(clin['RTSTART'], format='%m/%d/%Y', errors='coerce')
    clin['mri_to_rt_days'] = (clin['rt_date'] - clin['mri_date']).dt.days
    print('\nField strength T1 vs T2 agree:', (clin['field_T'] == clin['field_T2seq']).mean().round(3),
          '| missing T1 params:', clin['field_T'].isna().sum(), '| missing T2 params:', clin['field_T2seq'].isna().sum())
    print('MRI-to-RT interval (days): median', clin['mri_to_rt_days'].median(), 'IQR',
          clin['mri_to_rt_days'].quantile([.25, .75]).tolist(), 'min', clin['mri_to_rt_days'].min(),
          'max', clin['mri_to_rt_days'].max(), '| negative (MRI after RT start):', int((clin['mri_to_rt_days'] < 0).sum()))
    print('RT start year range:', clin['RTSTART_Year'].min(), '-', clin['RTSTART_Year'].max(),
          '| MRI date range:', clin['mri_date'].min().date(), '-', clin['mri_date'].max().date())
    odd = clin[(clin['RTSTART_Year'] < 2014) | (clin['mri_to_rt_days'] < 0) | (clin['mri_to_rt_days'] > 180)]
    odd[['RTSTART', 'RTSTART_Year', 'mri_date', 'mri_to_rt_days', 'CVgroup_event_t1t2']].to_csv(
        os.path.join(RESULTS, 'a03_check_dates_outliers.csv'))
    print('Patients with RT year < 2014, MRI after RT, or MRI > 180 days before RT:', len(odd))

    # ---- balance train vs test
    bal = []
    clin['era'] = np.where(clin['RTSTART_Year'] <= 2017, '<=2017', '>=2018')
    for var in ('field_T', 'era', 'RT_modality_codes'):
        tab = pd.crosstab(clin[var], clin['CVgroup_event_t1t2'])
        p = chi2_contingency(tab)[1]
        for lvl in tab.index:
            bal.append({'variable': var, 'level': lvl, 'train': tab.loc[lvl, 'train'], 'test': tab.loc[lvl, 'test'], 'chi2_p': p})
    bal = pd.DataFrame(bal)
    bal.to_csv(os.path.join(RESULTS, 'a03_train_test_balance.csv'), index=False)
    print('\nTrain/test balance:\n', bal.to_string(index=False))

    # ---- Table B.2 for the 266 patients
    used = mp[mp['seq'].isin(['T1', 'T2'])].copy()
    used = used.sort_values('scan_date').groupby(['PatientID', 'seq']).last().reset_index()
    used['date'] = pd.to_datetime(used['scan_date'].astype(str), format='%Y%m%d', errors='coerce')
    tb = []
    for (fs, seq), g in used.groupby(['magnetic_field_strength', 'seq']):
        def rng(c, d=0):
            v = pd.to_numeric(g[c], errors='coerce')
            return f'{v.min():.{d}f}' if v.min() == v.max() else f'{v.min():.{d}f}-{v.max():.{d}f}'
        tb.append({'field_T': fs, 'sequence': seq, 'n_patients': g['PatientID'].nunique(),
                   'dates': f"{g['date'].min():%Y.%m}-{g['date'].max():%Y.%m}",
                   'slice_thickness': rng('slice_thickness', 1), 'spacing_between_slices': rng('spacing_between_slices', 1),
                   'inplane_resolution': ', '.join(sorted(g['pixel_spacing_inplane'].astype(str).str.extract(r'\[([\d.]+)')[0].astype(float).round(2).astype(str).unique())),
                   'TR_ms': rng('Repetition_time'), 'TE_ms': rng('Echo_time'), 'ETL': rng('Echo_train_length'),
                   'flip_angle': rng('Flip_angle'), 'matrix': ', '.join(sorted(g['Acquisition_matrix'].astype(str).unique())),
                   'bandwidth': rng('Pixelbandwidth'), 'averages': rng('Number_of_averages'),
                   'scanner_models': ', '.join(sorted(g['equiment'].astype(str).unique()))})
    tb = pd.DataFrame(tb)
    tb.to_csv(os.path.join(RESULTS, 'a03_tableB2_mri_params_266.csv'), index=False)
    print('\nTable B.2 (266 patients):\n', tb.to_string(index=False))

    # ---- subgroup performance (test set)
    sub_rows = []
    subgroups = {
        'HPV-negative': test['P16_codes'] == 0,
        'HPV-negative or unknown': test['P16_codes'].isin([0, 1]),
        'HPV-positive': test['P16_codes'] == 2,
        '1.5 T MRI': clin.loc[test.index, 'field_T'] == 1.5,
        '3 T MRI': clin.loc[test.index, 'field_T'] == 3.0,
    }
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        hnc = f'hnc_lp_{ep}'
        risks = {'HNC-PREDICTOR': test[hnc]}
        for name in KEY_MODELS:
            root, prefix = MAIN_CONFIGS[name]
            p = load_predictions(root, prefix, ep, clin)
            risks[name] = p['test'].loc[test.index]
            tr = train.assign(dl=p['oof_z'].loc[train.index])
            b = fit_cox(tr, [hnc, 'dl'], tm, ev).params_
            risks[f'HNC + {name}'] = b[hnc] * test[hnc] + b['dl'] * p['test_z'].loc[test.index]
        for sg, mask in subgroups.items():
            mask = mask.to_numpy()
            t, e = test[tm][mask], test[ev][mask]
            if e.sum() < 2:
                for name in risks:
                    sub_rows.append({'endpoint': ep, 'subgroup': sg, 'n': int(mask.sum()), 'events': int(e.sum()), 'model': name})
                continue
            boot = bootstrap_cindex(t, e, {k: v[mask] for k, v in risks.items()}, n_boot=1000, seed=11)
            for name, r in risks.items():
                c = cindex(t, e, r[mask])
                lo, hi = np.percentile(boot[name], [2.5, 97.5])
                sub_rows.append({'endpoint': ep, 'subgroup': sg, 'n': int(mask.sum()), 'events': int(e.sum()),
                                 'model': name, 'C': c, 'lo': lo, 'hi': hi})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(os.path.join(RESULTS, 'a03_subgroup_cindex.csv'), index=False)
    sub['fmt'] = sub.apply(lambda r: '' if pd.isna(r.get('C')) else f"{r['C']:.2f} [{r['lo']:.2f}-{r['hi']:.2f}]", axis=1)
    piv = sub.pivot_table(index='model', columns=['endpoint', 'subgroup'], values='fmt', aggfunc='first')
    piv.to_csv(os.path.join(RESULTS, 'a03_subgroup_cindex_wide.csv'))
    ns = sub.groupby(['endpoint', 'subgroup'])[['n', 'events']].first()
    print('\nSubgroup sizes:\n', ns.to_string())
    pd.set_option('display.width', 300)
    for ep in ('LC', 'RC', 'OS'):
        print(f'\n{ep}:\n', sub[sub['endpoint'] == ep].pivot_table(index='model', columns='subgroup', values='fmt', aggfunc='first').to_string())
    clin[['field_T', 'mri_date', 'mri_to_rt_days', 'era']].to_csv(os.path.join(RESULTS, 'a03_patient_mri_info.csv'))


if __name__ == '__main__':
    main()

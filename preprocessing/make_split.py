"""Train/test split and 3-fold cross-validation groups (as used in the paper).

Patients with both a T1 and a T2 scan are split once into a training (70%) and a test set (30%) with
StratifiedShuffleSplit (seed 42) on the combination of loco-regional control, overall survival and disease-free
survival events. For every endpoint the training set is divided into three folds (StratifiedKFold, shuffled,
seed 42) stratified on that endpoint's event. Patients with a T1 scan only get group 'CV4' and are not used.

Output columns: CVgroup_event_t1t2 (train / test / train_t1only) and CVgroup_<event>_t1t2 (test / CV1-3 / CV4).

    python preprocessing/make_split.py --clinical outcomes.csv --mri_dir mri_zscore --out clinical_split.csv \
        [--hnc_csv hnc_predictor_variables.csv]
"""
import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

EVENTS = ['LR_code', 'RR_code', 'LRR_code', 'MET_code', 'TumorSpecificSurvival_code', 'OS_code', 'DFS_code',
          'RFS_code']
TEST_STRATA = ['LRR_code', 'OS_code', 'DFS_code']
HNC_VARIABLES = ['gender', 'age', 'performance score', 'smoking status', 'pack-years', 'T-stage', 'N-stage',
                 'N-stage-original', 'tumour site', 'HPV status', 'AJCC8th']


def make_split(clinical, with_t1t2, with_t1_only, seed=42, test_size=0.3):
    df = clinical[clinical['PatientID'].isin(with_t1t2)].reset_index(drop=True)
    t1_only = clinical[clinical['PatientID'].isin(with_t1_only)].copy()
    events = [e for e in EVENTS if e in df.columns]
    train_idx, test_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                               .split(X=df, y=df[TEST_STRATA]))
    df['CVgroup_event_t1t2'] = 'train'
    df.loc[test_idx, 'CVgroup_event_t1t2'] = 'test'
    for event in events:
        col = f'CVgroup_{event}_t1t2'
        df.loc[test_idx, col] = 'test'
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (_, cv_idx) in enumerate(skf.split(X=df.iloc[train_idx][[event]], y=df.iloc[train_idx][event])):
            df.loc[train_idx[cv_idx], col] = f'CV{fold + 1}'
        t1_only[col] = 'CV4'
    t1_only['CVgroup_event_t1t2'] = 'train_t1only'
    return pd.concat([df, t1_only], axis=0).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--clinical', required=True, help='CSV with PatientID, <event> and TIME_<...> columns')
    p.add_argument('--mri_dir', required=True, help='folder with <PatientID>_t1.nii.gz / _t2.nii.gz')
    p.add_argument('--out', required=True)
    p.add_argument('--hnc_csv', default=None, help='optional CSV with the HNC-PREDICTOR variables (merged in)')
    opt = p.parse_args()
    files = {os.path.basename(f) for f in glob.glob(os.path.join(glob.escape(opt.mri_dir), '*.nii.gz'))}
    ids = sorted({f.split('_')[0] for f in files})
    with_t1t2 = [i for i in ids if f'{i}_t1.nii.gz' in files and f'{i}_t2.nii.gz' in files]
    with_t1_only = [i for i in ids if f'{i}_t1.nii.gz' in files and f'{i}_t2.nii.gz' not in files]
    df = make_split(pd.read_csv(opt.clinical), with_t1t2, with_t1_only)
    if opt.hnc_csv:
        hnc = pd.read_csv(opt.hnc_csv)
        df = df.drop(columns=[c for c in HNC_VARIABLES if c in df.columns]).merge(
            hnc[['PatientID'] + HNC_VARIABLES], on='PatientID', how='left', validate='one_to_one')
    df.to_csv(opt.out, index=False)
    for event in [e for e in EVENTS if e in df.columns]:
        print(event, df.groupby(f'CVgroup_{event}_t1t2')[event].agg(['size', 'sum']).to_dict('index'))


if __name__ == '__main__':
    main()

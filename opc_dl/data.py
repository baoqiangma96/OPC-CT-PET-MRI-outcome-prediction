"""Clinical/split table, patient sets and MONAI data dictionaries."""
import os

import numpy as np
import pandas as pd
import torch

# endpoint -> (event column, time column) in the clinical CSV (time in months)
ENDPOINTS = {
    'LC': ('LR_code', 'TIME_LR'),
    'RC': ('RR_code', 'TIME_RR'),
    'LRC': ('LRR_code', 'TIME_LRR'),
    'DM': ('MET_code', 'TIME_MET'),
    'TSS': ('TumorSpecificSurvival_code', 'TIME_TumorSpecificSurvival'),
    'OS': ('OS_code', 'TIME_OS'),
    'DFS': ('DFS_code', 'TIME_DFS'),
    'RFS': ('RFS_code', 'TIME_RFS'),
}
EXCLUDED_GROUP = 'CV4'   # patients with a T1 but no T2 scan: never used in this study


def split_column(event_col):
    """Column with 'test', 'CV1', 'CV2', 'CV3' (training folds) or 'CV4' (excluded) for one endpoint."""
    return f'CVgroup_{event_col}_t1t2'


def read_clinical(path):
    return pd.read_csv(path)


def get_sets(clinical, event_col, fold):
    """Patient IDs (CSV order) of the training, validation and test set for one fold, and of all study patients."""
    group = clinical[split_column(event_col)]
    ids = clinical['PatientID']
    test = ids[group == 'test'].tolist()
    val = ids[group == f'CV{fold}'].tolist()
    train = ids[~group.isin(['test', f'CV{fold}', EXCLUDED_GROUP])].tolist()
    study = ids[group != EXCLUDED_GROUP].tolist()
    return train, val, test, study


def build_items(patient_ids, keys, data_path, clinical=None, event_col=None, time_col=None):
    """MONAI data dictionaries: one NIfTI path per key plus, if a clinical table is given, event and time."""
    rows = clinical.set_index('PatientID') if clinical is not None else None
    items = []
    for pid in patient_ids:
        item = {k: os.path.join(data_path, f'{pid}_{k}.nii.gz') for k in keys}
        item['PatientID'] = pid
        if rows is not None:
            item['event'] = torch.tensor(float(rows.loc[pid, event_col]), dtype=torch.float)
            item['time'] = torch.tensor(float(rows.loc[pid, time_col]), dtype=torch.float)
        items.append(item)
    return items


def outcome(clinical, patient_ids, event_col, time_col):
    rows = clinical.set_index('PatientID').loc[patient_ids]
    return rows[time_col].to_numpy(float), rows[event_col].to_numpy(int)


def oversampler(clinical, train_ids, event_col):
    """Draws training patients with probability inversely proportional to the frequency of their event status."""
    labels = clinical.set_index('PatientID').loc[train_ids, event_col].to_numpy().astype(int)
    weights = 1 / np.array([np.count_nonzero(1 - labels), np.count_nonzero(labels)])
    sample_weights = torch.from_numpy(np.array([weights[t] for t in labels]))
    return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

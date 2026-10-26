"""Test-set performance of a trained configuration: the ensemble of its networks (mean output of the 3 folds x
3 seeds) and every single network, with a bootstrap 95% CI of the ensemble C-index.

Example:
    python evaluate.py --results_root runs --split_csv /data/opc/clinical_split.csv \
        --configs 2D_T2_GTV_cmargin 3D_CT_GTV --endpoints LC RC OS

The full statistical analysis of the paper (hybrid models, Delta-C, calibration, figures) is in analysis/.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from opc_dl.data import ENDPOINTS, outcome, read_clinical, split_column
from opc_dl.engine import harrell_c


def bootstrap_ci(time, event, lp, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    n, cs = len(time), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if event[idx].sum() == 0:
            continue
        cs.append(harrell_c(time[idx], event[idx], lp[idx]))
    return np.percentile(cs, [2.5, 97.5])


def evaluate(results_root, clinical, config, endpoint, n_boot=2000):
    event_col, time_col = ENDPOINTS[endpoint]
    test_ids = clinical.loc[clinical[split_column(event_col)] == 'test', 'PatientID'].tolist()
    time, event = outcome(clinical, test_ids, event_col, time_col)
    runs = sorted(glob.glob(os.path.join(glob.escape(os.path.join(results_root, config, endpoint)), 'fold*_sub*',
                                         'predicted_lp.csv')))
    if not runs:
        return []
    preds = pd.concat([pd.read_csv(r).set_index('PatientID')['lp'].rename(os.path.basename(os.path.dirname(r)))
                       for r in runs], axis=1).loc[test_ids]
    ens = preds.mean(axis=1).to_numpy()
    lo, hi = bootstrap_ci(time, event, ens, n_boot)
    rows = [{'config': config, 'endpoint': endpoint, 'model': f'ensemble of {len(runs)}',
             'C': harrell_c(time, event, ens), 'C_lo': lo, 'C_hi': hi, 'n_test': len(test_ids),
             'events': int(event.sum())}]
    rows += [{'config': config, 'endpoint': endpoint, 'model': c, 'C': harrell_c(time, event, preds[c].to_numpy())}
             for c in preds.columns]
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results_root', required=True)
    p.add_argument('--split_csv', required=True)
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--endpoints', nargs='+', default=['LC', 'RC', 'OS'])
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--out', default=None, help='optional CSV')
    opt = p.parse_args()
    clinical = read_clinical(opt.split_csv)
    rows = [r for c in opt.configs for e in opt.endpoints for r in evaluate(opt.results_root, clinical, c, e,
                                                                             opt.n_boot)]
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(res.round(3).to_string(index=False))
    if opt.out:
        res.to_csv(opt.out, index=False)


if __name__ == '__main__':
    main()

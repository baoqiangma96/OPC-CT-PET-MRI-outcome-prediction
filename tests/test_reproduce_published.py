"""Check run by the authors (needs the study data and trained networks, which are not shared): the study's
networks, run through this repository's evaluation pipeline with current PyTorch/MONAI, reproduce the test-set
predictions stored when the models were trained. Results: docs/verification.md.

For every configuration one network (default: fold 2, seed 1, LC) is loaded and applied to the first --n test
patients; the outputs are compared with that network's predicted_lp.csv.

    python tests/test_reproduce_published.py --split_csv clinical_split.csv --data_2d <mri_2d> --data_3d <images_3d>
        --weights <runs root>  [--legacy_2d <MRI_pred_new> --legacy_3d <ctptmri_result>]  --out reproduce.csv
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opc_dl.configs import PAPER_CONFIGS, parse_config  # noqa: E402
from opc_dl.data import ENDPOINTS, build_items, read_clinical, split_column  # noqa: E402
from opc_dl.engine import make_loader, predict  # noqa: E402
from opc_dl.models import build_model  # noqa: E402
from opc_dl.transforms import build_transforms, network_input  # noqa: E402


def run_folder(opt, cfg, endpoint, fold, sub):
    d = os.path.join(opt.weights or '', cfg.name, endpoint, f'fold{fold}_sub{sub}')
    if opt.weights and os.path.isdir(d):
        return d
    legacy = opt.legacy_2d if cfg.dims == 2 else opt.legacy_3d
    return os.path.join(legacy, f'{cfg.legacy_prefix}_{ENDPOINTS[endpoint][0]}_fold{fold}_sub{sub}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split_csv', required=True)
    p.add_argument('--data_2d', required=True)
    p.add_argument('--data_3d', required=True)
    p.add_argument('--weights', default=None, help='runs root in the layout of this repository')
    p.add_argument('--legacy_2d', default='')
    p.add_argument('--legacy_3d', default='')
    p.add_argument('--configs', nargs='+', default=PAPER_CONFIGS)
    p.add_argument('--endpoint', default='LC')
    p.add_argument('--fold', type=int, default=2)
    p.add_argument('--sub', type=int, default=1)
    p.add_argument('--n', type=int, default=8, help='number of test patients per configuration')
    p.add_argument('--tol', type=float, default=1e-3)
    p.add_argument('--out', default=None)
    opt = p.parse_args()

    clinical = read_clinical(opt.split_csv)
    event_col = ENDPOINTS[opt.endpoint][0]
    test_ids = clinical.loc[clinical[split_column(event_col)] == 'test', 'PatientID'].tolist()[:opt.n]
    rows = []
    for name in opt.configs:
        cfg = parse_config(name)
        d = run_folder(opt, cfg, opt.endpoint, opt.fold, opt.sub)
        stored = pd.read_csv(os.path.join(d, 'predicted_lp.csv')).set_index('PatientID')['lp']
        model = build_model(cfg)
        model.load_state_dict(torch.load(os.path.join(d, 'best_metric_model.pth'), map_location='cpu'))
        data_path = opt.data_2d if cfg.dims == 2 else opt.data_3d
        loader = make_loader(build_items(test_ids, cfg.keys, data_path), build_transforms(cfg, False), 4, 0)
        t0 = time.time()
        lp = predict(model, loader, lambda b, dev: network_input(b, cfg, dev), torch.device('cpu'))
        diff = abs(lp - stored.loc[test_ids].to_numpy())
        rows.append({'config': name, 'n': len(test_ids), 'max_abs_diff': diff.max(), 'mean_abs_diff': diff.mean(),
                     'sd_stored_test': stored.std(), 'seconds': round(time.time() - t0, 1)})
        print(f"{name:22s} max |diff| {diff.max():.2e}  (SD of stored outputs {stored.std():.2f})", flush=True)
    res = pd.DataFrame(rows)
    if opt.out:
        res.to_csv(opt.out, index=False)
    bad = res[res['max_abs_diff'] > opt.tol]
    print(f"\n{len(res) - len(bad)}/{len(res)} configurations reproduce the stored predictions within {opt.tol}")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == '__main__':
    main()

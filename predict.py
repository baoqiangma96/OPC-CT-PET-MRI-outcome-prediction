"""Apply trained networks to new patients.

All networks found under <weights_root>/<config>/<endpoint>/fold*_sub*/best_metric_model.pth are applied (nine in
the paper: 3 folds x 3 seeds); <weights_root> is a results folder of train.py or a folder made by
tools/export_weights.py. The CSV contains the per-network outputs, their mean (the ensemble used in the paper)
and, when normalisation.json is present (written by tools/export_weights.py), the mean of the outputs
standardised with the training-set mean and SD of each network (the 'z-ensemble' used for the hybrid models).

Example:
    python predict.py --config 2D_T2_GTV_cmargin --endpoint LC --weights_root runs \
        --data_path /data/new_patients --ids new_patients.csv --out predictions_T2_cmargin_LC.csv

The input images must be pre-processed exactly as in preprocessing/ (z-scored MRI within the head-and-neck mask,
resampling, GTV masks with the same naming), otherwise the outputs are meaningless.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from opc_dl.configs import parse_config
from opc_dl.data import ENDPOINTS, build_items
from opc_dl.engine import make_loader, predict
from opc_dl.models import build_model
from opc_dl.transforms import build_transforms, network_input


def list_runs(weights_root, config, endpoint):
    pattern = os.path.join(glob.escape(os.path.join(weights_root, config, endpoint)), 'fold*_sub*',
                           'best_metric_model.pth')
    return sorted(glob.glob(pattern))


def run_predictions(config, endpoint, weights_root, data_path, ids, device='cpu', batch_size=12, num_workers=0,
                    n_slices=25, crop_centre='as_published'):
    cfg = parse_config(config)
    runs = list_runs(weights_root, config, endpoint)
    if not runs:
        raise FileNotFoundError(f'no best_metric_model.pth under {os.path.join(weights_root, config, endpoint)}')
    loader = make_loader(build_items(ids, cfg.keys, data_path), build_transforms(cfg, False, n_slices, crop_centre),
                         batch_size, num_workers)
    model = build_model(cfg, n_slices).to(device)
    out = pd.DataFrame({'PatientID': ids})
    for path in runs:
        model.load_state_dict(torch.load(path, map_location=device))
        out[os.path.basename(os.path.dirname(path))] = predict(model, loader, lambda b, d: network_input(b, cfg, d),
                                                               device)
    run_cols = [os.path.basename(os.path.dirname(p)) for p in runs]
    out['ensemble_mean'] = out[run_cols].mean(axis=1)
    norm_path = os.path.join(weights_root, config, endpoint, 'normalisation.json')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        z = [(out[c] - norm[c]['mean']) / norm[c]['sd'] for c in run_cols]
        out['ensemble_z'] = np.mean(z, axis=0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--endpoint', required=True, choices=list(ENDPOINTS))
    p.add_argument('--weights_root', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--ids', required=True, help='CSV with a PatientID column')
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--num_workers', type=int, default=0)
    opt = p.parse_args()
    ids = pd.read_csv(opt.ids)['PatientID'].tolist()
    out = run_predictions(opt.config, opt.endpoint, opt.weights_root, opt.data_path, ids, torch.device(opt.device),
                          opt.batch_size, opt.num_workers)
    out.to_csv(opt.out, index=False)
    print(f'{len(out)} patients, {len(out.columns) - 3} networks -> {opt.out}')


if __name__ == '__main__':
    main()

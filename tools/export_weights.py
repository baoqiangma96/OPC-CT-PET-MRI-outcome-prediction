"""Collect trained networks into one compact folder (e.g. to move them to another machine), without any
patient-level data.

Output layout (the layout predict.py reads):
    <out>/<config>/<endpoint>/fold<k>_sub<s>/best_metric_model.pth
    <out>/<config>/<endpoint>/normalisation.json   training-set mean and SD of each network's output (z-ensemble)
    <out>/<config>/model_config.json
    <out>/MANIFEST.csv                             file, bytes, sha256

Sources: a runs folder in this repository's layout (--runs) or the result folders of the original training
scripts (--legacy_2d / --legacy_3d). The split CSV is only used to know which patients were training patients;
predictions and patient IDs are not copied.

Size of the paper's models: 2D ~45 MB and 3D ~133 MB per network; 9 networks x 3 endpoints per configuration.

    python tools/export_weights.py --legacy_2d MRI_pred_new --legacy_3d ctptmri_result --split_csv split.csv \
        --out exported_networks --configs 2D_T2_GTV_cmargin 3D_T1_GTV --endpoints LC RC OS [--dry_run]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opc_dl.configs import PAPER_CONFIGS, parse_config  # noqa: E402
from opc_dl.data import ENDPOINTS  # noqa: E402


def source_dir(opt, cfg, endpoint, fold, sub):
    d = os.path.join(opt.runs or '', cfg.name, endpoint, f'fold{fold}_sub{sub}')
    if opt.runs and os.path.isdir(d):
        return d
    legacy = opt.legacy_2d if cfg.dims == 2 else opt.legacy_3d
    return os.path.join(legacy or '', f'{cfg.legacy_prefix}_{ENDPOINTS[endpoint][0]}_fold{fold}_sub{sub}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default=None)
    p.add_argument('--legacy_2d', default=None)
    p.add_argument('--legacy_3d', default=None)
    p.add_argument('--split_csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--configs', nargs='+', default=PAPER_CONFIGS)
    p.add_argument('--endpoints', nargs='+', default=['LC', 'RC', 'OS'])
    p.add_argument('--dry_run', action='store_true', help='only report what would be copied and its size')
    opt = p.parse_args()
    clinical = pd.read_csv(opt.split_csv)
    train_ids = set(clinical.loc[clinical['CVgroup_event_t1t2'] == 'train', 'PatientID'])
    manifest, total = [], 0
    for name in opt.configs:
        cfg = parse_config(name)
        for ep in opt.endpoints:
            norm = {}
            for fold in (1, 2, 3):
                for sub in (1, 2, 3):
                    src = source_dir(opt, cfg, ep, fold, sub)
                    w = os.path.join(src, 'best_metric_model.pth')
                    if not os.path.exists(w):
                        continue
                    run = f'fold{fold}_sub{sub}'
                    total += os.path.getsize(w)
                    lp = pd.read_csv(os.path.join(src, 'predicted_lp_all.csv'))
                    lp = lp.loc[lp['PatientID'].isin(train_ids), 'lp']
                    norm[run] = {'mean': float(lp.mean()), 'sd': float(lp.std()), 'n_training_patients': int(len(lp))}
                    if opt.dry_run:
                        continue
                    dst = os.path.join(opt.out, name, ep, run)
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy2(w, os.path.join(dst, 'best_metric_model.pth'))
                    rel = os.path.relpath(os.path.join(dst, 'best_metric_model.pth'), opt.out)
                    manifest.append({'file': rel.replace(os.sep, '/'), 'bytes': os.path.getsize(w),
                                     'sha256': sha256(os.path.join(dst, 'best_metric_model.pth'))})
            print(f'{name} {ep}: {len(norm)} networks')
            if norm and not opt.dry_run:
                with open(os.path.join(opt.out, name, ep, 'normalisation.json'), 'w') as f:
                    json.dump(norm, f, indent=2)
        if not opt.dry_run and os.path.isdir(os.path.join(opt.out, name)):
            with open(os.path.join(opt.out, name, 'model_config.json'), 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
    print(f'total {total / 1e9:.1f} GB')
    if not opt.dry_run:
        pd.DataFrame(manifest).to_csv(os.path.join(opt.out, 'MANIFEST.csv'), index=False)


if __name__ == '__main__':
    main()

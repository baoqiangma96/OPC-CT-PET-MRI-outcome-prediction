"""Train one model (one configuration, endpoint, fold and seed) and predict the test set and all study patients.

Example (2D T2 MRI, GTV with cuboid margin, local control, fold 1, first seed):
    python train.py --config 2D_T2_GTV_cmargin --endpoint LC --fold 1 --sub 1 \
        --data_path /data/opc/mri_2d --split_csv /data/opc/clinical_split.csv --results_root runs

The paper trained every configuration for 3 folds x 3 seeds (--sub 1/2/3 = seed 42/0/1) and averaged the nine
networks (see scripts/train_all.sh). Output: <results_root>/<config>/<endpoint>/fold<k>_sub<s>/ with
best_metric_model.pth, predicted_lp.csv (test set), predicted_lp_all.csv (all study patients), training_log.csv
and config.json.
"""
import argparse
import os
import platform
import sys

import monai
import torch

from opc_dl import __version__
from opc_dl.configs import parse_config
from opc_dl.data import ENDPOINTS, build_items, get_sets, oversampler, outcome, read_clinical
from opc_dl.engine import (SEEDS, Logger, harrell_c, make_loader, predict, raise_open_file_limit, save_predictions,
                           seed_everything, train, write_json)
from opc_dl.losses import CoxLoss
from opc_dl.models import build_model
from opc_dl.transforms import build_transforms, network_input


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='e.g. 2D_T1_GTV, 2D_T2_GTV_cmargin, 3D_CT_GTV_margin')
    p.add_argument('--endpoint', default='LC', choices=list(ENDPOINTS))
    p.add_argument('--fold', type=int, default=1, choices=[1, 2, 3], help='validation fold')
    p.add_argument('--sub', type=int, default=1, choices=[1, 2, 3], help='repetition; sets the seed (42, 0, 1)')
    p.add_argument('--data_path', required=True, help='folder with <PatientID>_<key>.nii.gz')
    p.add_argument('--split_csv', required=True, help='clinical table with outcomes and split columns')
    p.add_argument('--results_root', default='runs')
    # optimisation (defaults = paper)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=2e-4)
    p.add_argument('--n_epochs', type=int, default=400, help='epochs 1 .. n_epochs-1, as in the original code')
    p.add_argument('--esn', type=int, default=25, help='early-stopping patience')
    p.add_argument('--no_oversample', dest='oversample', action='store_false',
                   help='disable the event-balanced sampler (the paper used it)')
    p.add_argument('--loss', default='as_published', choices=['as_published', 'cox'],
                   help="'as_published' reproduces the paper; 'cox' is the corrected loss (docs/known_issues.md)")
    p.add_argument('--crop_centre', default='as_published', choices=['as_published', 'bbox'],
                   help='2D crop centre; see docs/known_issues.md')
    p.add_argument('--n_slices', type=int, default=25, help='2D: number of axial slices (channels)')
    # running
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--no_monitor', dest='monitor', action='store_false',
                   help='do not log the training/test C-index every epoch (logging only, not used for selection)')
    p.add_argument('--checkpoint', action='store_true', help='save last_checkpoint.pth every epoch (for --resume)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--predict_only', action='store_true', help='skip training; use best_metric_model.pth in run dir')
    p.add_argument('--wandb_project', default=None, help='also log to Weights & Biases')
    return p.parse_args(argv)


def main(argv=None):
    opt = parse_args(argv)
    raise_open_file_limit()
    seed = SEEDS[opt.sub]
    seed_everything(seed)
    cfg = parse_config(opt.config)
    event_col, time_col = ENDPOINTS[opt.endpoint]
    device = torch.device(opt.device)
    run_dir = os.path.join(opt.results_root, cfg.name, opt.endpoint, f'fold{opt.fold}_sub{opt.sub}')
    os.makedirs(run_dir, exist_ok=True)

    clinical = read_clinical(opt.split_csv)
    train_ids, val_ids, test_ids, study_ids = get_sets(clinical, event_col, opt.fold)
    items = {name: build_items(ids, cfg.keys, opt.data_path, clinical, event_col, time_col)
             for name, ids in (('train', train_ids), ('val', val_ids), ('test', test_ids), ('study', study_ids))}
    print(f'{cfg.name} {opt.endpoint} fold {opt.fold} seed {seed}: '
          f'{len(train_ids)} train / {len(val_ids)} validation / {len(test_ids)} test patients')

    train_tf = build_transforms(cfg, train=True, n_slices=opt.n_slices, centre=opt.crop_centre)
    eval_tf = build_transforms(cfg, train=False, n_slices=opt.n_slices, centre=opt.crop_centre)
    bs, nw = opt.batch_size, opt.num_workers
    sampler = oversampler(clinical, train_ids, event_col) if opt.oversample else None
    loaders = {'train': make_loader(items['train'], train_tf, bs, nw, sampler=sampler, shuffle=sampler is None),
               'train_eval': make_loader(items['train'], eval_tf, bs, nw),
               'val': make_loader(items['val'], eval_tf, bs, nw),
               'test': make_loader(items['test'], eval_tf, bs, nw)}
    sets = {'clinical': clinical,
            'train_outcome': outcome(clinical, train_ids, event_col, time_col),
            'val_outcome': outcome(clinical, val_ids, event_col, time_col),
            'test_outcome': outcome(clinical, test_ids, event_col, time_col)}

    model = build_model(cfg, opt.n_slices).to(device)
    to_input = lambda batch, dev: network_input(batch, cfg, dev)
    config = {**vars(opt), 'seed': seed, 'model_config': cfg.__dict__, 'n_train': len(train_ids),
              'n_val': len(val_ids), 'n_test': len(test_ids), 'opc_dl': __version__, 'python': sys.version,
              'torch': torch.__version__, 'monai': monai.__version__, 'platform': platform.platform()}
    write_json(config, os.path.join(run_dir, 'config.json'))

    summary = {}
    if not opt.predict_only:
        logger = Logger(os.path.join(run_dir, 'training_log.csv'), opt.wandb_project, config, append=opt.resume)
        criterion = CoxLoss(opt.loss)
        summary = train(model, loaders, sets, to_input, criterion, opt, run_dir, device, logger)
    else:
        logger = None

    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_metric_model.pth'), map_location=device))
    lp_test = predict(model, loaders['test'], to_input, device)
    summary['test_cindex'] = harrell_c(*sets['test_outcome'], lp_test)
    save_predictions(test_ids, lp_test, os.path.join(run_dir, 'predicted_lp.csv'))
    study_loader = make_loader(items['study'], eval_tf, bs, nw)
    save_predictions(study_ids, predict(model, study_loader, to_input, device),
                     os.path.join(run_dir, 'predicted_lp_all.csv'))
    write_json(summary, os.path.join(run_dir, 'summary.json'))
    if logger:
        logger.finish(summary)
    print(f"test C-index of this network: {summary['test_cindex']:.4f}  ->  {run_dir}")
    return summary


if __name__ == '__main__':
    main()

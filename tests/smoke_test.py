"""End-to-end test on synthetic data (no study data needed; CPU, about 20-30 minutes).

Runs the whole chain: synthetic images -> preprocessing (z-score, resampling, margins, CT/PET bounding box and
crop) -> train/test split -> training of 2D and 3D models (a few epochs) -> evaluate.py -> export_weights.py ->
predict.py, and checks the outputs.

    python tests/smoke_test.py [work_dir]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'tests'))
from make_synthetic_data import make_dataset  # noqa: E402

CONFIGS = ['2D_T1_GTV', '2D_T2_GTV_margin', '2D_T1_GTV_cmargin', '3D_CT_GTV', '3D_PET_GTV_margin',
           '3D_T2_GTV_cmargin']
ENSEMBLE_CONFIG = '2D_T2_GTV_cmargin'


def run(*args):
    print('>>', ' '.join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO)


def main(work):
    t0 = time.time()
    syn = os.path.join(work, 'synthetic')
    outcomes = make_dataset(syn)
    raw_mri, raw_3d = os.path.join(syn, 'raw_mri'), os.path.join(syn, 'raw_3d')
    mri_2d, crop_3d, images_3d = (os.path.join(work, d) for d in ('mri_2d', 'crop_3d', 'images_3d'))

    # ---- preprocessing: 2D MRI
    run('preprocessing/mri_02_hn_mask_zscore.py', '--in_dir', raw_mri, '--out_dir', mri_2d, '--suffixes', 't1', 't2')
    for f in glob.glob(os.path.join(raw_mri, '*_gtvpttot*.nii.gz')):
        shutil.copy(f, mri_2d)
    run('preprocessing/mri_03_resample.py', '--in_dir', mri_2d, '--out_dir', mri_2d, '--images', 't1', 't2',
        '--masks', 'gtvpttot1', 'gtvpttot2')
    run('preprocessing/mri_04_gtv_margin.py', '--dir', mri_2d, '--masks', 'gtvpttot1r', 'gtvpttot2r', '--preset', '5mm')
    # ---- preprocessing: 3D (synthetic 64 mm box instead of 192 mm)
    ids_csv, bb_csv = os.path.join(work, 'ids.csv'), os.path.join(work, 'bb_box.csv')
    pd.read_csv(outcomes)[['PatientID']].to_csv(ids_csv, index=False)
    run('preprocessing/ctpet_01_bounding_box.py', '--pet_dir', raw_3d, '--ids', ids_csv, '--out', bb_csv, '--size', 64)
    run('preprocessing/ctpet_02_resample_crop.py', '--in_dir', raw_3d, '--bb', bb_csv, '--out_dir', crop_3d,
        '--images', 'ct', 'pt', 'T1dr', 'T2dr', '--masks', 'ct_GTVpt')
    run('preprocessing/mri_02_hn_mask_zscore.py', '--in_dir', crop_3d, '--out_dir', images_3d,
        '--suffixes', 'T1dr', 'T2dr')
    for k in ('ct', 'pt', 'ct_GTVpt'):
        for f in glob.glob(os.path.join(crop_3d, f'*_{k}.nii.gz')):
            shutil.copy(f, images_3d)
    # ---- split
    split = os.path.join(work, 'clinical_split.csv')
    run('preprocessing/make_split.py', '--clinical', outcomes, '--mri_dir', raw_mri, '--out', split)
    clinical = pd.read_csv(split)
    n_test = int((clinical['CVgroup_LR_code_t1t2'] == 'test').sum())
    n_study = int((clinical['CVgroup_LR_code_t1t2'] != 'CV4').sum())

    # ---- training
    runs = os.path.join(work, 'runs')
    common = ['--endpoint', 'LC', '--split_csv', split, '--results_root', runs, '--n_epochs', 5,
              '--batch_size', 4, '--num_workers', 0, '--device', 'cpu']
    data = lambda c: mri_2d if c.startswith('2D') else images_3d
    for cfg in CONFIGS:
        run('train.py', '--config', cfg, '--data_path', data(cfg), '--fold', 1, '--sub', 1, *common)
    run('train.py', '--config', ENSEMBLE_CONFIG, '--data_path', mri_2d, '--fold', 1, '--sub', 1, *common)
    run('train.py', '--config', ENSEMBLE_CONFIG, '--data_path', mri_2d, '--fold', 2, '--sub', 1, '--loss', 'cox',
        *common)
    run('train.py', '--config', ENSEMBLE_CONFIG, '--data_path', mri_2d, '--fold', 3, '--sub', 2,
        '--crop_centre', 'bbox', '--checkpoint', *common)
    run('train.py', '--config', ENSEMBLE_CONFIG, '--data_path', mri_2d, '--fold', 3, '--sub', 2, '--resume',
        '--crop_centre', 'bbox', '--checkpoint', *[a if a != 5 else 6 for a in common])
    run('train.py', '--config', ENSEMBLE_CONFIG, '--data_path', mri_2d, '--fold', 3, '--sub', 2, '--predict_only',
        '--crop_centre', 'bbox', *common)

    for cfg in CONFIGS + [ENSEMBLE_CONFIG]:
        for d in glob.glob(os.path.join(glob.escape(os.path.join(runs, cfg, 'LC')), 'fold*_sub*')):
            for f in ('best_metric_model.pth', 'predicted_lp.csv', 'predicted_lp_all.csv', 'config.json'):
                assert os.path.exists(os.path.join(d, f)), (d, f)
            assert len(pd.read_csv(os.path.join(d, 'predicted_lp.csv'))) == n_test
            assert len(pd.read_csv(os.path.join(d, 'predicted_lp_all.csv'))) == n_study
    log = pd.read_csv(os.path.join(runs, ENSEMBLE_CONFIG, 'LC', 'fold3_sub2', 'training_log.csv'))
    assert log['epoch'].tolist() == [1, 2, 3, 4, 5], log['epoch'].tolist()       # 4 epochs + 1 after --resume

    # ---- evaluation, export, inference
    run('evaluate.py', '--results_root', runs, '--split_csv', split, '--configs', *CONFIGS, '--endpoints', 'LC',
        '--n_boot', 200, '--out', os.path.join(work, 'evaluate.csv'))
    weights = os.path.join(work, 'weights')
    run('tools/export_weights.py', '--runs', runs, '--split_csv', split, '--out', weights,
        '--configs', ENSEMBLE_CONFIG, '--endpoints', 'LC')
    norm = json.load(open(os.path.join(weights, ENSEMBLE_CONFIG, 'LC', 'normalisation.json')))
    assert len(norm) == 3, norm
    test_ids = os.path.join(work, 'test_ids.csv')
    clinical.loc[clinical['CVgroup_LR_code_t1t2'] == 'test', ['PatientID']].to_csv(test_ids, index=False)
    pred = os.path.join(work, 'predict.csv')
    run('predict.py', '--config', ENSEMBLE_CONFIG, '--endpoint', 'LC', '--weights_root', weights,
        '--data_path', mri_2d, '--ids', test_ids, '--out', pred, '--device', 'cpu')
    p = pd.read_csv(pred)
    assert len(p) == n_test and {'ensemble_mean', 'ensemble_z'} <= set(p.columns), p.columns
    # predict.py must give the same outputs as the predictions written at the end of training (fold 1)
    stored = pd.read_csv(os.path.join(runs, ENSEMBLE_CONFIG, 'LC', 'fold1_sub1', 'predicted_lp.csv'))
    diff = (p.set_index('PatientID')['fold1_sub1'] - stored.set_index('PatientID')['lp']).abs().max()
    assert diff < 1e-5, diff
    print(f'\nSMOKE TEST PASSED in {(time.time() - t0) / 60:.1f} min ({work})')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='opc_smoke_'))

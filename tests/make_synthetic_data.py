"""Synthetic stand-in for the (non-shareable) study data: random 'head' volumes with a tumour, GTV masks and a
clinical table in the same format and with the same file names as the real data. Only meant to exercise the code.

Writes into <root>/:
    raw_mri/      SYN-xxxx_t1.nii.gz, _t2.nii.gz (raw intensities, own grid) + _gtvpttot1/2.nii.gz
    raw_3d/       SYN-xxxx_ct, _pt, _T1dr, _T2dr, _ct_GTVpt (1.5 mm grid covering the head and the brain)
    outcomes.csv  outcomes of all patients (no split yet)
"""
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk

EVENT_PAIRS = [('LR_code', 'TIME_LR'), ('RR_code', 'TIME_RR'), ('LRR_code', 'TIME_LRR'), ('MET_code', 'TIME_MET'),
               ('TumorSpecificSurvival_code', 'TIME_TumorSpecificSurvival'), ('OS_code', 'TIME_OS'),
               ('DFS_code', 'TIME_DFS'), ('RFS_code', 'TIME_RFS')]


def _write(arr_xyz, spacing, path, dtype=np.float32):
    img = sitk.GetImageFromArray(np.transpose(arr_xyz, (2, 1, 0)).astype(dtype))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, path)


def _ellipsoid(shape, centre, radii):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum(((g[i] - centre[i]) / radii[i]) ** 2 for i in range(3)) <= 1


def make_mri(rng, shape, spacing, tumour_mm):
    head = _ellipsoid(shape, [s / 2 for s in shape], [0.4 * shape[0], 0.35 * shape[1], 0.6 * shape[2]])
    c = [shape[0] / 2 + rng.uniform(-15, 15) / spacing[0], shape[1] / 2 + rng.uniform(-15, 15) / spacing[1],
         shape[2] / 2 + rng.uniform(-2, 2)]
    gtv = _ellipsoid(shape, c, [tumour_mm / sp for sp in spacing])
    img = rng.normal(2, 1, shape) + head * rng.normal(150, 25, shape) + gtv * 80
    return np.clip(img, 0, None), gtv


def make_3d(rng, n=96, spacing=1.5, tumour_mm=9.0):
    shape = (n, n, n)
    brain_c = (n / 2, n / 2, 0.82 * n)                    # brain in the upper third of the scan
    brain = _ellipsoid(shape, brain_c, (18, 22, 10))
    head = _ellipsoid(shape, (n / 2, n / 2, n / 2), (0.4 * n, 0.4 * n, 0.55 * n))
    # GTV inside the box that ctpet_01_bounding_box.py places below the brain (64 mm box, 30 mm shift)
    c = (n / 2 + rng.uniform(-4, 4), n / 2 - 20 / spacing * 1.5 + rng.uniform(-4, 4), 0.62 * n + rng.uniform(-3, 3))
    gtv = _ellipsoid(shape, c, [tumour_mm / spacing] * 3)
    ct = -1000 + head * (1000 + rng.normal(40, 30, shape)) + gtv * 30
    pt = np.clip(rng.normal(0.5, 0.2, shape) + head * 1.0 + brain * 10 + gtv * rng.uniform(5, 15), 0, None)
    t1 = np.clip(rng.normal(2, 1, shape) + head * rng.normal(150, 25, shape) + gtv * 60, 0, None)
    t2 = np.clip(rng.normal(2, 1, shape) + head * rng.normal(120, 25, shape) + gtv * 90, 0, None)
    return {'ct': ct, 'pt': pt, 'T1dr': t1, 'T2dr': t2}, gtv


def make_dataset(root, n_patients=30, n_t1_only=4, seed=0):
    rng = np.random.default_rng(seed)
    for sub in ('raw_mri', 'raw_3d'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rows = []
    for i in range(n_patients):
        pid = f'SYN-{i + 1:04d}'
        tumour_mm = rng.uniform(6, 14)
        risk = (tumour_mm - 10) / 3 + rng.normal(0, 1)
        sequences = [('t1', (220, 220, 24), (0.8, 0.8, 4.0))]
        if i < n_patients - n_t1_only:
            sequences.append(('t2', (240, 240, 26), (0.7, 0.7, 3.5)))
        for seq, shape, spacing in sequences:
            img, gtv = make_mri(rng, shape, spacing, tumour_mm)
            _write(img, spacing, os.path.join(root, 'raw_mri', f'{pid}_{seq}.nii.gz'))
            _write(gtv, spacing, os.path.join(root, 'raw_mri', f'{pid}_gtvpttot{seq[1]}.nii.gz'), np.uint8)
        images, gtv = make_3d(rng, tumour_mm=tumour_mm * 0.8)
        for k, v in images.items():
            _write(v, (1.5, 1.5, 1.5), os.path.join(root, 'raw_3d', f'{pid}_{k}.nii.gz'))
        _write(gtv, (1.5, 1.5, 1.5), os.path.join(root, 'raw_3d', f'{pid}_ct_GTVpt.nii.gz'), np.uint8)
        event = int(risk + rng.normal(0, 0.5) > 0.3)
        row = {'PatientID': pid}
        for ev, tm in EVENT_PAIRS:
            row[ev] = event
            row[tm] = float(np.round(rng.uniform(3, 30) if event else rng.uniform(24, 80), 1))
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(root, 'outcomes.csv'), index=False)
    return os.path.join(root, 'outcomes.csv')


if __name__ == '__main__':
    import sys
    print(make_dataset(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data'))

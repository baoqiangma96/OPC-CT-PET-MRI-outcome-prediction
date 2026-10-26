# -*- coding: utf-8 -*-
"""
How much of the GTV (and GTV_margin) mask was inside the 2D input crop?

The 2D crop (320 x 320 x 25 voxels at 0.5 x 0.5 x 3 mm) was meant to be centred on the GTV in the slice with
the largest GTV area. Because of torch.nonzero semantics (docs/known_issues.md #2), the in-plane centre was
computed from the first two non-zero voxels of that slice instead of the bounding-box centre; the slice range
(25 slices around the largest-area slice) was as intended. This script reports, for all 266 patients, the
fraction of the mask volume inside the crop as used in training, and inside a crop centred as intended.
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from common import DATA_2D, RESULTS, load_clinical
from opc_dl.transforms import CROP_2D, _window, largest_area_slice_centre


def load_xyz(path):
    """Array in (x, y, z) order, as MONAI LoadImaged returns it."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).transpose(2, 1, 0).astype(np.float32)


def crop_largest_area_center(img, gtv, size=(*CROP_2D, 25), torch_semantics=True):
    c = largest_area_slice_centre(torch.from_numpy(gtv), 'as_published' if torch_semantics else 'bbox')
    sl = tuple(slice(*_window(c[d], size[d], gtv.shape[d])) for d in range(3))
    return img[sl], gtv[sl]


def main():
    clin = load_clinical()
    rows = []
    for pid in clin.index:
        for seq in ('1', '2'):
            for kind, suffix in (('GTV', f'gtvpttot{seq}r'), ('GTV_margin', f'gtvpttot{seq}r_5mm')):
                g = load_xyz(os.path.join(DATA_2D, f'{pid}_{suffix}.nii.gz')) > 0
                g = g.astype(np.float32)
                full = g.sum()
                _, used = crop_largest_area_center(g, g, torch_semantics=True)
                _, intended = crop_largest_area_center(g, g, torch_semantics=False)
                # slice-direction coverage alone (intended in-plane centre, 25 slices)
                rows.append({'PatientID': pid, 'seq': f'T{seq}', 'mask': kind, 'is_test': bool(clin.loc[pid, 'is_test']),
                             'frac_inside_used_crop': used.sum() / full, 'frac_inside_intended_crop': intended.sum() / full})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'a12_crop_coverage.csv'), index=False)
    summ = df.groupby(['seq', 'mask']).agg(
        n=('PatientID', 'size'),
        median_used=('frac_inside_used_crop', 'median'),
        min_used=('frac_inside_used_crop', 'min'),
        n_lt_100=('frac_inside_used_crop', lambda x: int((x < 0.999).sum())),
        n_lt_90=('frac_inside_used_crop', lambda x: int((x < 0.9).sum())),
        n_lt_75=('frac_inside_used_crop', lambda x: int((x < 0.75).sum())),
        min_intended=('frac_inside_intended_crop', 'min'),
        n_lt_100_intended=('frac_inside_intended_crop', lambda x: int((x < 0.999).sum())))
    summ.to_csv(os.path.join(RESULTS, 'a12_crop_coverage_summary.csv'))
    pd.set_option('display.width', 200)
    print(summ.round(3).to_string())
    t = df[df['is_test']]
    print('\nTest set only:')
    print(t.groupby(['seq', 'mask'])['frac_inside_used_crop'].agg(['median', 'min', lambda x: int((x < 0.999).sum())]).round(3).to_string())


if __name__ == '__main__':
    main()

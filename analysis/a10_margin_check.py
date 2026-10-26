# -*- coding: utf-8 -*-
"""
Measure the real margin of the 2D GTV_margin masks (gtvpt*r_5mm / _10mm) relative to the resampled GTV
(gtvpt*r), per side and per axis, at 0.5 x 0.5 x 3 mm. The masks were made by
Data preparation/add_margin_togtv.py with a box structuring element ([2,10,10] / [3,20,20] voxels in z,y,x).
Also reports the margin of the 3D pipeline for comparison (SimpleITK ball dilation, radius 5 voxels at 1 mm).
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from common import DATA_2D, RESULTS, load_clinical


def bbox(a):
    nz = np.nonzero(a)
    return [(int(x.min()), int(x.max())) for x in nz]


def measure(base, dil, spacing_zyx):
    """Per-side growth in voxels and mm along z, y, x (from bounding boxes) plus the maximum in-plane
    distance (mm) from the base mask to the dilated boundary."""
    bb0, bb1 = bbox(base), bbox(dil)
    out = {}
    for ax, name in enumerate('zyx'):
        lo = bb0[ax][0] - bb1[ax][0]
        hi = bb1[ax][1] - bb0[ax][1]
        out[f'{name}_low_vox'], out[f'{name}_high_vox'] = lo, hi
        out[f'{name}_low_mm'], out[f'{name}_high_mm'] = lo * spacing_zyx[ax], hi * spacing_zyx[ax]
    # in-plane distance map per slice (only slices containing the base mask)
    dmax = 0.0
    for z in np.unique(np.nonzero(base)[0]):
        dist = ndimage.distance_transform_edt(base[z] == 0, sampling=spacing_zyx[1:])
        ring = (dil[z] > 0) & (base[z] == 0)
        if ring.any():
            dmax = max(dmax, float(dist[ring].max()))
    out['inplane_max_dist_mm'] = dmax
    out['vol_ratio'] = dil.sum() / base.sum()
    return out


def main():
    clin = load_clinical()
    ids = list(clin.index[:12])
    rows = []
    for pid in ids:
        for seq in ('1', '2'):
            base_p = os.path.join(DATA_2D, f'{pid}_gtvpttot{seq}r.nii.gz')
            img = sitk.ReadImage(base_p)
            base = sitk.GetArrayFromImage(img) > 0
            sp = img.GetSpacing()[::-1]  # z, y, x
            for m in ('5mm', '10mm'):
                dil = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(DATA_2D, f'{pid}_gtvpttot{seq}r_{m}.nii.gz'))) > 0
                r = measure(base, dil, sp)
                r.update({'PatientID': pid, 'seq': f'T{seq}', 'nominal': m})
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'a10_margin_check.csv'), index=False)
    cols = ['z_low_mm', 'z_high_mm', 'y_low_mm', 'y_high_mm', 'x_low_mm', 'x_high_mm', 'inplane_max_dist_mm', 'vol_ratio']
    print(df.groupby('nominal')[cols].agg(['median', 'min', 'max']).T.round(2).to_string())

    # 3D pipeline: ball dilation radius 5 voxels at 1 mm on one CT GTV, for comparison
    from common import DATA_3D
    pid = ids[0]
    g = sitk.ReadImage(os.path.join(DATA_3D, f'{pid}_ct_GTVpt.nii.gz'))
    gi = sitk.Cast(g > 0, sitk.sitkInt32)
    f = sitk.BinaryDilateImageFilter()
    f.SetKernelType(sitk.sitkBall)
    f.SetKernelRadius([5, 5, 5])
    d = sitk.GetArrayFromImage(f.Execute(gi)) > 0
    b = sitk.GetArrayFromImage(gi) > 0
    r = measure(b, d, g.GetSpacing()[::-1])
    print('\n3D pipeline (ball radius 5 voxels, 1 mm):', {k: round(v, 2) for k, v in r.items() if k.endswith('_mm') or k == 'vol_ratio'})


if __name__ == '__main__':
    main()

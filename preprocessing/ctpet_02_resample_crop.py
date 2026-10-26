"""Step 2 (3D): crop CT, PET (and the GTV masks) to the bounding box and resample to 1 x 1 x 1 mm.

Images: B-spline interpolation; masks: nearest neighbour. Output names follow the training code:
<PatientID>_ct.nii.gz, _pt.nii.gz, _ct_GTVpt.nii.gz (primary GTV from the planning CT). PET is expected in SUV.
The T1/T2 scans for the 3D models (_T1dr, _T2dr) were deformably registered to the planning CT and resampled onto
the same grid; that registration step is not part of this repository (see preprocessing/README.md).

    python preprocessing/ctpet_02_resample_crop.py --in_dir nii --bb bb_box.csv --out_dir data_3d \
        --images ct pt --masks ct_GTVpt
"""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk


def crop_resample(img, bb, spacing=(1.0, 1.0, 1.0), is_label=False):
    lo, hi = np.array([bb['x1'], bb['y1'], bb['z1']]), np.array([bb['x2'], bb['y2'], bb['z2']])
    f = sitk.ResampleImageFilter()
    f.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    f.SetOutputSpacing(spacing)
    f.SetOutputOrigin(lo.tolist())
    f.SetSize([int(k) for k in np.round((hi - lo) / np.array(spacing))])
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return f.Execute(img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_dir', required=True)
    p.add_argument('--bb', required=True, help='CSV from ctpet_01_bounding_box.py')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--images', nargs='*', default=['ct', 'pt'])
    p.add_argument('--masks', nargs='*', default=['ct_GTVpt'])
    opt = p.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    for _, bb in pd.read_csv(opt.bb).iterrows():
        pid = bb['PatientID']
        for suffix, is_label in [(s, False) for s in opt.images] + [(s, True) for s in opt.masks]:
            src = os.path.join(opt.in_dir, f'{pid}_{suffix}.nii.gz')
            out = crop_resample(sitk.ReadImage(src), bb, is_label=is_label)
            sitk.WriteImage(out, os.path.join(opt.out_dir, f'{pid}_{suffix}.nii.gz'))
        print(pid)


if __name__ == '__main__':
    main()

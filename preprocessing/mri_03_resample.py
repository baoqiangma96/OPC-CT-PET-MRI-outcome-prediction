"""Step 3 (MRI, 2D margin models): resample z-scored images and GTV masks to 0.5 x 0.5 x 3 mm.

Images: linear interpolation, outside value = image minimum; masks: nearest neighbour, binarised. The output name
gets an 'r' appended (PAT-0001_t1 -> PAT-0001_t1r, _gtvpttot1 -> _gtvpttot1r). The 2D GTV-only models
resample on the fly instead (MONAI Spacingd, same spacing).

    python preprocessing/mri_03_resample.py --in_dir mri_zscore --out_dir mri_2d --images t1 t2 \
        --masks gtvpttot1 gtvpttot2 gtvtottot1 gtvtottot2
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk

SPACING = (0.5, 0.5, 3.0)


def resample(img, spacing=SPACING, is_label=False):
    size = [int(np.round(sz * sp / out)) for sz, sp, out in zip(img.GetSize(), img.GetSpacing(), spacing)]
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(spacing)
    f.SetSize(size)
    f.SetOutputDirection(img.GetDirection())
    f.SetOutputOrigin(img.GetOrigin())
    f.SetTransform(sitk.Transform())
    if is_label:
        f.SetInterpolator(sitk.sitkNearestNeighbor)
        f.SetDefaultPixelValue(0)
    else:
        f.SetInterpolator(sitk.sitkLinear)
        f.SetDefaultPixelValue(float(np.min(sitk.GetArrayFromImage(img))))
    out = f.Execute(img)
    if is_label:
        out[out > 0] = 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--images', nargs='*', default=['t1', 't2'])
    p.add_argument('--masks', nargs='*', default=['gtvpttot1', 'gtvpttot2'])
    opt = p.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    for suffix, is_label in [(s, False) for s in opt.images] + [(s, True) for s in opt.masks]:
        for path in sorted(glob.glob(os.path.join(glob.escape(opt.in_dir), f'*_{suffix}.nii.gz'))):
            name = os.path.basename(path)[:-len('.nii.gz')] + 'r.nii.gz'
            sitk.WriteImage(resample(sitk.ReadImage(path), is_label=is_label), os.path.join(opt.out_dir, name))
            print(name)


if __name__ == '__main__':
    main()

"""Step 2 (MRI): head-and-neck mask and z-score normalisation within that mask.

Mask = voxels above a fixed intensity threshold (20 on the raw scanner values), largest connected component, holes
filled. Every image is then z-scored with the mean and SD of the voxels inside its mask (the background ends up
negative). Used for the 2D inputs (T1/T2 on their own grid) and for the T1/T2 scans registered to the planning CT
(3D inputs, T1dr/T2dr).

    python preprocessing/mri_02_hn_mask_zscore.py --in_dir mri_nii --out_dir mri_zscore --suffixes t1 t2
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def head_neck_mask(array, threshold=20):
    labelled, n = ndimage.label(array > threshold)
    sizes = ndimage.sum(array > threshold, labelled, range(n + 1))
    return ndimage.binary_fill_holes(labelled == np.argmax(sizes))


def zscore_in_mask(img, threshold=20):
    array = sitk.GetArrayFromImage(img)
    mask = head_neck_mask(array, threshold)
    values = array[mask]
    z = sitk.GetImageFromArray((array - np.mean(values)) / np.std(values))
    z.CopyInformation(img)
    m = sitk.GetImageFromArray(mask.astype(np.uint8))
    m.CopyInformation(img)
    return z, m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--suffixes', nargs='+', default=['t1', 't2'], help='files <PatientID>_<suffix>.nii.gz')
    p.add_argument('--threshold', type=float, default=20)
    p.add_argument('--mask_dir', default=None, help='optional folder for the head-and-neck masks')
    opt = p.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    if opt.mask_dir:
        os.makedirs(opt.mask_dir, exist_ok=True)
    for suffix in opt.suffixes:
        for path in sorted(glob.glob(os.path.join(glob.escape(opt.in_dir), f'*_{suffix}.nii.gz'))):
            z, m = zscore_in_mask(sitk.ReadImage(path), opt.threshold)
            name = os.path.basename(path)
            sitk.WriteImage(z, os.path.join(opt.out_dir, name))
            if opt.mask_dir:
                sitk.WriteImage(m, os.path.join(opt.mask_dir, name))
            print(name)


if __name__ == '__main__':
    main()

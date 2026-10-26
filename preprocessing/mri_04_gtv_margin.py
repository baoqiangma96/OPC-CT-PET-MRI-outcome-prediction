"""Step 4 (MRI, 2D margin models): dilate the resampled GTV masks with a box-shaped structuring element.

The structuring element is given in voxels as (slices, rows, columns) on the 0.5 x 0.5 x 3 mm grid. The study used
    '_5mm'  : (2, 10, 10)  -> 5 mm wide in-plane, i.e. 2.0-2.5 mm on each side, plus one 3-mm slice on one side
    '_10mm' : (3, 20, 20)  -> 4.5-5.0 mm on each side in-plane and one slice (3 mm) on each side
The suffixes are the file names used by the training code; the true margins are the ones above (see
docs/known_issues.md). The _cmargin ROI (box around each component) is built during training.

    python preprocessing/mri_04_gtv_margin.py --dir mri_2d --masks gtvpttot1r gtvpttot2r --preset 5mm
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

PRESETS = {'5mm': (2, 10, 10), '10mm': (3, 20, 20)}


def dilate(img, kernel):
    out = sitk.GetImageFromArray(binary_dilation(sitk.GetArrayFromImage(img), structure=np.ones(kernel)).astype(int))
    out.CopyInformation(img)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True, help='folder with the resampled masks; output is written next to them')
    p.add_argument('--masks', nargs='+', default=['gtvpttot1r', 'gtvpttot2r'])
    p.add_argument('--preset', choices=list(PRESETS), default='5mm')
    opt = p.parse_args()
    kernel = PRESETS[opt.preset]
    for suffix in opt.masks:
        for path in sorted(glob.glob(os.path.join(glob.escape(opt.dir), f'*_{suffix}.nii.gz'))):
            name = os.path.basename(path)[:-len('.nii.gz')] + f'_{opt.preset}.nii.gz'
            sitk.WriteImage(dilate(sitk.ReadImage(path), kernel), os.path.join(opt.dir, name))
            print(name)


if __name__ == '__main__':
    main()

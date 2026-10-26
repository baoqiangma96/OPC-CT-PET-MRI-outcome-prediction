"""Step 1 (3D): head-and-neck bounding box per patient, found automatically on the PET scan.

Adapted from the HECKTOR 2021 bounding-box code (https://github.com/voreille/hecktor) as used in our earlier TransRP
work: the brain is detected as the largest high-uptake blob in the upper third of the smoothed PET and a box of
--size mm is placed below it. Output: CSV with PatientID, x1, x2, y1, y2, z1, z2 in physical coordinates (mm).
The 3D models of the paper used a 192 mm box (192^3 voxels of 1 mm) with a PET threshold of 3, 4 or 5 per
patient. Check every box visually, and re-run a patient with another --threshold if it is misplaced.

    python preprocessing/ctpet_01_bounding_box.py --pet_dir nii --ids ids.csv --out bb_box.csv --size 192
"""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, label


def bbox_auto(pet, size_mm=192, threshold=3.0):
    arr = np.transpose(sitk.GetArrayFromImage(pet), (2, 1, 0))     # (x, y, z)
    spacing, origin = pet.GetSpacing(), pet.GetOrigin()
    shape = [s / sp for s, sp in zip((size_mm,) * 3, spacing)]
    smooth = gaussian_filter(arr, sigma=3)
    for th in (threshold, 0.1):                                         # as in the original: retry with a low threshold
        labelled, _ = label(smooth > th)
        counts = np.bincount(labelled[:, :, arr.shape[2] * 2 // 3:].flat)[1:]
        if counts.size and counts.max() > 0:
            break
    else:
        raise RuntimeError('no high-uptake region found in the upper third of the PET')
    brain = labelled == np.argmax(counts) + 1
    z = np.min(np.argwhere(np.sum(brain, axis=(0, 1))))
    y1, y2 = np.min(np.argwhere(np.sum(brain, axis=(0, 2)))), np.max(np.argwhere(np.sum(brain, axis=(0, 2))))
    x1, x2 = np.min(np.argwhere(np.sum(brain, axis=(1, 2)))), np.max(np.argwhere(np.sum(brain, axis=(1, 2))))

    zshift = 30 // spacing[2]
    if z - (shape[2] - zshift) < 0:
        zbb = (0, shape[2])
    elif z + zshift > arr.shape[2]:
        zbb = (arr.shape[2] - shape[2], arr.shape[2])
    else:
        zbb = (z - (shape[2] - zshift), z + zshift)
    yshift = 30 // spacing[1]
    yc = int((y2 + y1) / 2 - yshift - int(shape[1] / 2))
    if yc < 0:
        ybb = (0, shape[1])
    elif yc > arr.shape[1]:
        ybb = (arr.shape[1] - shape[1], arr.shape[1])
    else:
        ybb = ((y2 + y1) / 2 - yshift - shape[1] / 2, (y2 + y1) / 2 - yshift + shape[1] / 2)
    xc = int((x2 + x1) / 2 - int(shape[0] / 2))
    if xc < 0:
        xbb = (0, shape[0])
    elif xc > arr.shape[0]:
        xbb = (arr.shape[0] - shape[0], arr.shape[0])
    else:
        xbb = ((x2 + x1) / 2 - shape[0] / 2, (x2 + x1) / 2 + shape[0] / 2)
    return [np.asarray(b) * spacing[i] + origin[i] for i, b in enumerate((xbb, ybb, zbb))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pet_dir', required=True, help='folder with <PatientID>_pt.nii.gz')
    p.add_argument('--ids', required=True, help='CSV with a PatientID column')
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=float, default=192, help='box edge in mm')
    p.add_argument('--threshold', type=float, default=3.0, help='PET threshold for the brain detection')
    opt = p.parse_args()
    rows = []
    for pid in pd.read_csv(opt.ids)['PatientID']:
        (x1, x2), (y1, y2), (z1, z2) = bbox_auto(sitk.ReadImage(os.path.join(opt.pet_dir, f'{pid}_pt.nii.gz')),
                                                 opt.size, opt.threshold)
        rows.append({'PatientID': pid, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'z1': z1, 'z2': z2,
                     'th': opt.threshold})
        print(pid)
    pd.DataFrame(rows).to_csv(opt.out, index=False)


if __name__ == '__main__':
    main()

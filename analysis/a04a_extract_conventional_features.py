# -*- coding: utf-8 -*-
"""
Extract conventional image features for Reviewer 1 #4 (all 266 study patients):
  * GTVp volume (cc) from the clinical CT contour (ct_GTVpt, 1 mm isotropic grid used by the 3D models);
  * GTVp volume (cc) as mapped to the T1 and T2 MRI (native MRI grid);
  * PET (SUV) within the CT GTVp: SUVmax, SUVmean, MTV (SUV >= 2.5 and >= 41% SUVmax), TLG (MTV2.5 x SUVmean
    within MTV2.5). PET is the 1 mm resampled image used by the 3D models (interpolated, so SUVmax is
    slightly smoothed relative to the native PET).
Output: results/a04_conventional_features.csv
"""
import os
import time
import numpy as np
import pandas as pd
import SimpleITK as sitk
from common import DATA_2D, DATA_3D, RESULTS, load_clinical


def read(p):
    img = sitk.ReadImage(p)
    return sitk.GetArrayFromImage(img), np.prod(img.GetSpacing())


def main():
    clin = load_clinical()
    out_p = os.path.join(RESULTS, 'a04_conventional_features.csv')
    done = set(pd.read_csv(out_p)['PatientID']) if os.path.exists(out_p) else set()
    rows = []
    t0 = time.time()
    for i, pid in enumerate(clin.index):
        if pid in done:
            continue
        r = {'PatientID': pid}
        gtv, vox = read(os.path.join(DATA_3D, f'{pid}_ct_GTVpt.nii.gz'))
        m = gtv > 0
        r['vol_ct_cc'] = m.sum() * vox / 1000.0
        pet, _ = read(os.path.join(DATA_3D, f'{pid}_pt.nii.gz'))
        s = pet[m]
        r['suvmax'] = float(s.max())
        r['suvmean'] = float(s.mean())
        m25 = s >= 2.5
        r['mtv25_cc'] = m25.sum() * vox / 1000.0
        r['mtv41_cc'] = (s >= 0.41 * s.max()).sum() * vox / 1000.0
        r['tlg25'] = r['mtv25_cc'] * (float(s[m25].mean()) if m25.any() else 0.0)
        for seq in ('1', '2'):
            g, v = read(os.path.join(DATA_2D, f'{pid}_gtvpttot{seq}.nii.gz'))
            r[f'vol_t{seq}_cc'] = (g > 0).sum() * v / 1000.0
        rows.append(r)
        if (i + 1) % 20 == 0:
            df = pd.DataFrame(rows)
            if os.path.exists(out_p):
                df = pd.concat([pd.read_csv(out_p), df], ignore_index=True)
            df.to_csv(out_p, index=False)
            rows = []
            print(f'{i + 1}/{len(clin)} patients, {time.time() - t0:.0f} s', flush=True)
    if rows:
        df = pd.DataFrame(rows)
        if os.path.exists(out_p):
            df = pd.concat([pd.read_csv(out_p), df], ignore_index=True)
        df.to_csv(out_p, index=False)
    print('done', flush=True)


if __name__ == '__main__':
    main()

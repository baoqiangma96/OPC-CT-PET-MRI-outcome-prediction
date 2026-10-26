"""Step 0 (MRI): acquisition parameters of every T1/T2 series, read from the first DICOM file (Table B.2).

Same input layout as mri_01_dicom_to_nifti.py: <dicom_root>/<patient>/<study>/<series>/. Writes one row per series;
analysis/a03_subgroups_cohort.py summarises it for the study patients. Requires pydicom.

    python preprocessing/mri_00_header_parameters.py --dicom_root DICOM --out mri_parameters.csv
"""
import argparse
import os

import pandas as pd
import pydicom

TAGS = {'scan_date': (0x0008, 0x0022), 'equiment': (0x0008, 0x0070), 'magnetic_field_strength': (0x0018, 0x0087),
        'slice_thickness': (0x0018, 0x0050), 'spacing_between_slices': (0x0018, 0x0088),
        'pixel_spacing_inplane': (0x0028, 0x0030), 'Repetition_time': (0x0018, 0x0080),
        'Echo_time': (0x0018, 0x0081), 'Echo_train_length': (0x0018, 0x0091), 'Flip_angle': (0x0018, 0x1314),
        'Acquisition_matrix': (0x0018, 0x1310), 'Pixelbandwidth': (0x0018, 0x0095),
        'Number_of_averages': (0x0018, 0x0083)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dicom_root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--first_file', default='IM1.DCM', help='name of the first DICOM file in a series folder')
    opt = p.parse_args()
    rows = []
    for patient in sorted(os.listdir(opt.dicom_root)):
        study = os.path.join(opt.dicom_root, patient, sorted(os.listdir(os.path.join(opt.dicom_root, patient)))[0])
        for series in sorted(os.listdir(study)):
            ds = pydicom.dcmread(os.path.join(study, series, opt.first_file), stop_before_pixels=True)
            row = {'Patient_ID': patient, 'sequence_name': series}
            row.update({k: ds[tag].value if tag in ds else None for k, tag in TAGS.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(opt.out)


if __name__ == '__main__':
    main()

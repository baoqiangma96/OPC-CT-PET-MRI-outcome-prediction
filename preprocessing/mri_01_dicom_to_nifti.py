"""Step 1 (MRI): convert the pre-treatment T1 and T2 DICOM series to NIfTI.

Expected input layout: <dicom_root>/<patient>/<study>/<series>/*.dcm with one T1 and one T2 series per patient
(the sequence was recognised from 't1' / 't2' in the converted file name). Output: <out_dir>/<prefix><patient>_t1.nii.gz
and _t2.nii.gz. In the study MRIcron's dcm2nii was used; dcm2niix (https://github.com/rordenlab/dcm2niix) gives the
same images.

    python preprocessing/mri_01_dicom_to_nifti.py --dicom_root DICOM --out_dir mri_nii --converter dcm2niix
"""
import argparse
import os
import shutil
import subprocess
import tempfile


def convert_series(converter, series_dir, tmp_dir):
    if os.path.basename(converter).lower().startswith('dcm2niix'):
        cmd = [converter, '-z', 'y', '-f', '%p_%s', '-o', tmp_dir, series_dir]
    else:   # dcm2nii writes next to the DICOM files
        cmd = [converter, '-o', tmp_dir, series_dir]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dicom_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--converter', default='dcm2niix', help='path to dcm2niix or dcm2nii')
    p.add_argument('--prefix', default='UMCG-', help='prefix of the patient ID in the output names')
    opt = p.parse_args()
    os.makedirs(opt.out_dir, exist_ok=True)
    for patient in sorted(os.listdir(opt.dicom_root)):
        study = os.path.join(opt.dicom_root, patient, sorted(os.listdir(os.path.join(opt.dicom_root, patient)))[0])
        for series in sorted(os.listdir(study)):
            with tempfile.TemporaryDirectory() as tmp:
                convert_series(opt.converter, os.path.join(study, series), tmp)
                for f in os.listdir(tmp):
                    if not f.endswith('.nii.gz'):
                        continue
                    seq = 't1' if 't1' in f.lower() else 't2' if 't2' in f.lower() else None
                    dest = os.path.join(opt.out_dir, f'{opt.prefix}{patient}_{seq}.nii.gz')
                    if seq and not os.path.exists(dest):
                        shutil.move(os.path.join(tmp, f), dest)
        print(patient, 'done')


if __name__ == '__main__':
    main()

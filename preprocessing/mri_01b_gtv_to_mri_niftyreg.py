"""Step 1b (MRI): map the planning-CT GTV masks onto the T1 and T2 grids (Appendix B.3).

The planning CT is registered to the T1 scan with NiftyReg reg_aladin (rigid followed by affine;
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg). The GTV masks are then resampled onto the T1 and T2 grids
with that transform (nearest neighbour). Check every result visually. In the study, patients with a poor
alignment of the tumour region were registered rigidly with manual adjustment in Mirada RTx instead.

Input: <ct_dir>/<ID>_ct.nii.gz and <ID>_ct_<mask>.nii.gz (masks on the CT grid), <mri_dir>/<ID>_t1/_t2.nii.gz.
Output: <out_dir>/<ID>_<name>1.nii.gz (T1 grid) and <ID>_<name>2.nii.gz (T2 grid), e.g. gtvpttot1 / gtvpttot2.

    python preprocessing/mri_01b_gtv_to_mri_niftyreg.py --ct_dir ct_nii --mri_dir mri_nii --out_dir mri_gtv \
        --niftyreg_bin /opt/niftyreg/bin --masks GTVpt:gtvpttot GTVtot:gtvtottot
"""
import argparse
import glob
import os
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ct_dir', required=True)
    p.add_argument('--mri_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--niftyreg_bin', default='', help='folder with reg_aladin and reg_resample')
    p.add_argument('--masks', nargs='+', default=['GTVpt:gtvpttot'], help='<CT mask suffix>:<output stem>')
    p.add_argument('--rigid_only', action='store_true')
    opt = p.parse_args()
    aladin = os.path.join(opt.niftyreg_bin, 'reg_aladin')
    resample = os.path.join(opt.niftyreg_bin, 'reg_resample')
    os.makedirs(opt.out_dir, exist_ok=True)
    for t1 in sorted(glob.glob(os.path.join(glob.escape(opt.mri_dir), '*_t1.nii.gz'))):
        pid = os.path.basename(t1)[:-len('_t1.nii.gz')]
        ct = os.path.join(opt.ct_dir, f'{pid}_ct.nii.gz')
        with tempfile.TemporaryDirectory() as tmp:
            aff = os.path.join(tmp, 'ct_to_t1.txt')
            cmd = [aladin, '-ref', t1, '-flo', ct, '-res', os.path.join(tmp, 'ct_on_t1.nii.gz'), '-aff', aff]
            subprocess.run(cmd + (['-rigOnly'] if opt.rigid_only else []), check=True, capture_output=True)
            for seq in ('1', '2'):
                ref = os.path.join(opt.mri_dir, f'{pid}_t{seq}.nii.gz')
                if not os.path.exists(ref):
                    continue
                for spec in opt.masks:
                    src, stem = spec.split(':')
                    subprocess.run([resample, '-ref', ref, '-flo', os.path.join(opt.ct_dir, f'{pid}_ct_{src}.nii.gz'),
                                    '-res', os.path.join(opt.out_dir, f'{pid}_{stem}{seq}.nii.gz'), '-trans', aff,
                                    '-inter', '0'], check=True, capture_output=True)
        print(pid)


if __name__ == '__main__':
    main()

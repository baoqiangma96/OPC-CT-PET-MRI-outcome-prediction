# Preprocessing

The scripts are listed in the order they were applied. Every script has `--help`. The parameters are the ones
used in the study. All steps except those that need DICOM data or NiftyReg (`mri_00`, `mri_01`, `mri_01b`) are
run on synthetic data by `tests/smoke_test.py`.

## MRI (2D models and MRI inputs of the 3D models)

| step | script | what it does |
|---|---|---|
| 0 | `mri_00_header_parameters.py` | acquisition parameters per series (Table B.2) |
| 1 | `mri_01_dicom_to_nifti.py` | T1 and T2 DICOM -> NIfTI (`<ID>_t1`, `<ID>_t2`) |
| 1b | `mri_01b_gtv_to_mri_niftyreg.py` | planning-CT GTV masks mapped onto the T1 and T2 grids (NiftyReg rigid + affine CT-to-T1). Every case was checked visually; poorly aligned cases were registered rigidly with manual adjustment in Mirada RTx (not scriptable) |
| 2 | `mri_02_hn_mask_zscore.py` | head-and-neck mask (threshold 20, largest component, holes filled) and z-score within it |
| 3 | `mri_03_resample.py` | images and masks to 0.5 x 0.5 x 3 mm (`t1r`, `gtvpttot1r`, ...) for the margin models |
| 4 | `mri_04_gtv_margin.py` | box dilation of the resampled GTV (`_5mm`, `_10mm`; real margins in docs/known_issues.md) |

The 2D GTV-only models read `t1`/`t2` and `gtvpttot1`/`gtvpttot2` and resample during training (MONAI
`Spacingd`, bilinear for images and nearest neighbour for masks). The crop of 320 x 320 x 25 voxels and the box
ROI (`_cmargin`) are built during training (`opc_dl/transforms.py`).

## CT and PET (3D models)

| step | script | what it does |
|---|---|---|
| 1 | `ctpet_01_bounding_box.py` | head-and-neck box from the PET (HECKTOR-style brain detection), 192 mm |
| 2 | `ctpet_02_resample_crop.py` | CT, PET and GTV masks cropped to the box and resampled to 1 mm (B-spline / nearest) |
| – | *not included* | T1 and T2 registered to the cropped planning CT with SimpleITK (rigid followed by deformable registration, see the paper's appendix) and resampled onto the same 1 mm grid (`T1dr`, `T2dr`), then z-scored with `mri_02_hn_mask_zscore.py --suffixes T1dr T2dr` |

Steps 1–2 follow the preprocessing of our earlier CT/PET work (TransRP). In this study every box measured
192 mm in each direction, and the PET threshold of step 1 (`--threshold`) was set per patient to 3, 4 or 5.

## Train/test split

`make_split.py`: 30% test set (StratifiedShuffleSplit, seed 42, stratified on LRC, OS and DFS events) and three
cross-validation folds per endpoint (StratifiedKFold, seed 42). Patients without a T2 scan get group `CV4` and are
not used.

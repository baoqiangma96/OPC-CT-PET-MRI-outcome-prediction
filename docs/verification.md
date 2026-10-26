# Verification

Checks run in October 2026 with Python 3.11, PyTorch 2.14.1 (CPU) and MONAI 1.6.1, against the original training
code (2024, PyTorch 1.12.1, CUDA 11.7) and its saved outputs. Sections 1 and 5 can be run by anyone. Sections 2–4
need the study data and trained networks, which are not shared, and were run by the authors.

## 1. Refactored code equals the original code (`tests/test_units.py`, no data needed)

* The 2D crop and box ROI (40 random masks), the 3D margin and box ROI (10 random masks, margins 0 and 5 mm)
  and the loss (20 random batches) give results identical to verbatim copies of the original functions
  (`tests/original_code.py`).
* The published loss does not depend on which patients had the event, only on how many (docs/known_issues.md #1).

## 2. The study's networks reproduce their stored predictions (`tests/test_reproduce_published.py`)

Each trained network was loaded into this repository's evaluation pipeline and applied to 8 test patients. The
outputs were compared with the `predicted_lp.csv` written by the original code when the network was trained.

| networks (one per configuration) | configurations | max abs difference |
|---|---|---|
| LC, fold 2, seed 42 | all 18 of the paper | ≤ 1.1e-6 |
| OS, fold 1, seed 1 | all 18 | ≤ 2.0e-6 |
| RC, fold 3, seed 0 | 2D (6) / 3D (12) | ≤ 7e-7 / ≤ 7.4e-4 |
| LC, fold 2, seed 42 | 8 supplementary 2D (10 mm, GTVtot) | ≤ 1.9e-6 |

The network outputs have an SD of 0.03–1.4. The 3D RC networks show differences of about 1e-4. Those jobs ran on
an A100 GPU, where PyTorch uses TF32 arithmetic for convolutions; the original test and all-patient passes of
these networks also differ by up to 4e-4.

This check also established which box ROI each model was trained with (docs/known_issues.md #3). The main 2D
`_cmargin` models are only reproduced by the single-box variant (difference 4e-7). The per-component box in the
final script gives differences up to 0.4.

## 3. Ensemble result of the paper

The nine networks of the best LC model (2D T2_GTV_cmargin) were exported with `tools/export_weights.py` and
applied with `predict.py` to all 80 test patients. The ensemble C-index is 0.876 (paper: 0.88); the z-standardised
ensemble gives 0.884.

## 4. Split and analyses

* `preprocessing/make_split.py`, run on the original outcome table and MRI folder, regenerates all nine split
  columns of the study's split for all 333 patients, in the same order.
* All 14 analysis scripts ran on the study data. Their 37 output tables are numerically identical (maximum
  difference 0) to the results computed for the manuscript. The only difference is that this repository omits
  two unreported T1+T2 two-path models. `a00_cohort_tables.py` reproduces Tables A.1/A.2 as produced by the
  original `clinical_statistics.py`.

## 5. Whole pipeline on synthetic data (`tests/smoke_test.py`, ~20 min on CPU)

Synthetic images → z-scoring, resampling, margins, PET bounding box and CT/PET crop → split → training of
2D_T1_GTV, 2D_T2_GTV_margin, 2D_T1_GTV_cmargin, 3D_CT_GTV, 3D_PET_GTV_margin and 3D_T2_GTV_cmargin. Then three
folds of 2D_T2_GTV_cmargin (including `--loss cox`, `--crop_centre bbox`, `--resume` and `--predict_only`),
`evaluate.py`, `tools/export_weights.py` and `predict.py`. The test checks that every output exists, has the
right size, and that `predict.py` reproduces the predictions written by `train.py`.

Not repeated here: training on the full study data (GPU), and the steps that need DICOM data or NiftyReg
(`mri_00`, `mri_01`, `mri_01b`).

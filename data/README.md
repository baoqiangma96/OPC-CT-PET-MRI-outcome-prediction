# Data format

The imaging and clinical data of the study cannot be shared (patient privacy). This folder describes the format
the code expects, so that the pipeline can be run on other data. `tests/make_synthetic_data.py` writes a small
synthetic dataset in exactly this format.

**Never commit patient data, patient IDs or per-patient predictions to the repository.** `.gitignore` excludes
`data/*` except this README.

## Images: `<data_path>/<PatientID>_<key>.nii.gz`

2D models (one folder, `--data_path` / `OPC_DATA_2D`), MRI z-scored within the head-and-neck mask:

| key | content | used by |
|---|---|---|
| `t1`, `t2` | axial T1 / coronal T2 on their own grid | `2D_T*_GTV` (resampled on the fly to 0.5 x 0.5 x 3 mm) |
| `gtvpttot1`, `gtvpttot2` | primary GTV on the T1 / T2 grid | `2D_T*_GTV` |
| `t1r`, `t2r` | `t1` / `t2` resampled to 0.5 x 0.5 x 3 mm | `_margin`, `_cmargin` |
| `gtvpttot1r_5mm`, `gtvpttot2r_5mm` | dilated GTV on that grid (see docs/known_issues.md #4) | `_margin`, `_cmargin` |
| `gtvpttot*r_10mm` | larger dilation | `_cmargin_10mm` (Table A.4) |
| `gtvtottot*`, `gtvtottot*r_5mm` | primary + nodal GTV | `2D_T*_GTVtot*` (supplementary analyses) |

3D models (`OPC_DATA_3D`): 192 x 192 x 192 mm head-and-neck box at 1 mm, all on the planning-CT grid.

| key | content |
|---|---|
| `ct` | planning CT (HU), clipped to [-200, 200] and scaled to [0, 1] during training |
| `pt` | FDG-PET (SUV), clipped to [0, 25] and scaled to [0, 1] |
| `T1dr`, `T2dr` | T1 / T2 deformably registered to the CT, z-scored within the head-and-neck mask |
| `ct_GTVpt` | primary GTV (clinical contour on the planning CT) |
| `ct_GTVln` | nodal GTV (analysis a08 only) |

## Clinical table (`--split_csv` / `OPC_SPLIT_CSV`)

One row per patient. Required for training:

* `PatientID`
* event (0/1) and time (months) for each endpoint: `LR_code`/`TIME_LR` (local control), `RR_code`/`TIME_RR`
  (regional control), `OS_code`/`TIME_OS` (overall survival); optional `LRR_code`, `MET_code`, `DFS_code`,
  `RFS_code`, `TumorSpecificSurvival_code` with their `TIME_*` columns
* `CVgroup_<event>_t1t2` for each endpoint: `test`, `CV1`, `CV2`, `CV3` (training folds) or `CV4` (not used);
  `CVgroup_event_t1t2`: `train` / `test` / `train_t1only`. These are written by `preprocessing/make_split.py`.

Additionally used by the analyses (`analysis/`):

* HNC-PREDICTOR variables (`OPC_HNC_CSV`, may be the same file): `age` (0: <=55, 1: 56-65, 2: 66-75, 3: >75),
  `performance score` (0, 1, >=2), `pack-years` (0: <=5, 1: 5-25, 2: 26-50, 3: >50), `T-stage` (0-3 = T1-T4),
  `AJCC8th` (0-4 = I, II, III, IVa, IVb), `HPV status` (0 = positive), `tumour site` (2 = oropharynx), plus
  `gender`, `smoking status`, `N-stage`, `N-stage-original`
* cohort description: `AGE`, `GESLACHT_codes` (1 male), `PACK_YEARS`, `TSTAD_codes`, `NSTAD_codes` (0 N0,
  1 N1, 2 N2a, 3 N2b, 4 N2c, 5 N3), `NSTAD_codes_N01VSN2VSN3`, `P16_codes` (2 positive, 1 unknown, 0 negative),
  `P16_codes_newexperi`, `WHO_SCORE_codes`, `RT_modality_codes`, `RTSTART` (mm/dd/yyyy), `RTSTART_Year`
* `OPC_MRI_PARAMS_CSV`: output of `preprocessing/mri_00_header_parameters.py` (a03 only)

## Trained networks (`--results_root` / `OPC_RUNS`)

    <root>/<config>/<endpoint>/fold<k>_sub<s>/best_metric_model.pth
                                              predicted_lp.csv       (test set; written by train.py)
                                              predicted_lp_all.csv   (all study patients)

`predicted_lp*.csv` contain patient IDs; `tools/export_weights.py` copies the networks without them.

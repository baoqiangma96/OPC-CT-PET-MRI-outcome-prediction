# Analyses: tables and figures of the paper

These scripts produce the tables and figures of the paper. The numbering below is that of the manuscript version
under review; several analyses (for example Tables A.5–A.16 and Figure A.3) were added during peer review and are
not in the preprint. The scripts use the saved predictions of the trained networks (`predicted_lp.csv`,
`predicted_lp_all.csv`) and the clinical table; no network is retrained. Set the paths (see `common.py`), then run
the scripts from this folder:

```bash
export OPC_SPLIT_CSV=/data/opc/clinical_split.csv   # Windows PowerShell: $env:OPC_SPLIT_CSV = "..."
export OPC_RUNS=/data/opc/runs                      # <config>/<endpoint>/fold<k>_sub<s>/predicted_lp*.csv
export OPC_DATA_2D=/data/opc/mri_2d OPC_DATA_3D=/data/opc/images_3d OPC_OUTPUT=/data/opc/outputs
cd analysis
python a00_cohort_tables.py
python a01_table1_cindex.py
...                                                 # any order, but a04a before a04b and a08
```

The result folders written by the original training scripts can be read directly by setting `OPC_LEGACY_2D` and
`OPC_LEGACY_3D` instead of `OPC_RUNS`.

| script | paper item | content |
|---|---|---|
| `a00_cohort_tables.py` | Tables A.1, A.2 | patient characteristics and events, training vs test set |
| `a01_table1_cindex.py` | Tables 1, A.3, A.4, A.6, A.8, A.16 | C-index with bootstrap CIs, ΔC vs HNC-PREDICTOR, hybrid models refitted with out-of-fold predictions, modality comparison |
| `a02_selection_multiplicity.py` | Table A.7 | selection on validation C-index, family-wise permutation test over 18 models, per-network variability |
| `a03_subgroups_cohort.py` | Tables A.5, A.13, B.2 | events per fold, train/test balance, HPV and field-strength subgroups, MRI parameters |
| `a04a_extract_conventional_features.py` | – | GTV volume, SUVmax/SUVmean/MTV/TLG from the NIfTI files |
| `a04b_conventional_models.py` | Tables A.9, A.10 | conventional baselines; DL predictions versus GTV volume |
| `a05_calibration_km.py` | Figures 2, 3, A.1, A.2; Table A.11 | Kaplan-Meier risk groups (training-set median cut-off), calibration slope, O/E, GND test |
| `a06_ang_figure4.py` | Figure 4, Table A.12 | Ang RPA risk groups versus HNC-PREDICTOR and hybrid tertiles |
| `a07_cindex_sanity.py` | Table A.15 | simulation of the C-index under censoring, Uno's C, time-dependent AUC |
| `a08_dm_rc_nodes.py` | Table A.14 | distant metastasis; nodal volume and regional control |
| `a09_dca.py` | Figure A.3 | decision curves |
| `a10_margin_check.py` | Methods | measured size of the 2D and 3D margins |
| `a12_crop_coverage.py` | Methods | fraction of the GTV inside the 2D crop |
| `a13_original_method_ci.py` | Table 1 (submitted) | CIs with the original bootstrap code, to trace the numbers of the submitted version |

Statistics in brief: Harrell's C on the 80 test patients; 95% CIs and ΔC from 2,000 paired bootstrap samples of
test patients. The ensemble is the mean of the nine networks (3 folds x 3 seeds). Hybrid models are Cox models
fitted on the 186 training patients, with the HNC-PREDICTOR linear predictor (published coefficients) and the
out-of-fold DL prediction standardised per network (z-ensemble).

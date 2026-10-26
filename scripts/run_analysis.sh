#!/bin/bash
# Run all analyses (tables and figures). Set OPC_SPLIT_CSV, OPC_RUNS (or OPC_LEGACY_2D/_3D), OPC_DATA_2D,
# OPC_DATA_3D, OPC_MRI_PARAMS_CSV and OPC_OUTPUT first (see analysis/common.py).
set -e
cd "$(dirname "$0")/../analysis"
for s in a00_cohort_tables a01_table1_cindex a02_selection_multiplicity a03_subgroups_cohort \
         a04a_extract_conventional_features a04b_conventional_models a05_calibration_km a06_ang_figure4 \
         a07_cindex_sanity a08_dm_rc_nodes a09_dca a10_margin_check a12_crop_coverage a13_original_method_ci; do
    echo "== $s"
    python "$s.py" > "${OPC_OUTPUT:-../outputs}/results/$s.log"
done

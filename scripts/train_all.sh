#!/bin/bash
# Train every network of one configuration and endpoint: 3 folds x 3 seeds (as in the paper), one GPU job each.
#
#   sbatch --array=0-8 scripts/train_all.sh 2D_T2_GTV_cmargin LC
#
# Configurations of the paper: 2D_{T1,T2}_GTV{,_margin,_cmargin} and 3D_{CT,PET,T1,T2}_GTV{,_margin,_cmargin};
# endpoints LC, RC, OS. Edit the paths below. The original jobs ran on the Habrok cluster (University of
# Groningen) on one V100 GPU (3D: A100) with 8-10 CPU workers and 50-100 GB memory.
#
#SBATCH --job-name=opc_dl
#SBATCH --time=23:59:59
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/%x_%A_%a.log

CONFIG=${1:?config, e.g. 2D_T2_GTV_cmargin}
ENDPOINT=${2:-LC}
SPLIT_CSV=/path/to/clinical_split.csv
RESULTS=/path/to/runs
if [[ $CONFIG == 2D_* ]]; then DATA=/path/to/mri_2d; else DATA=/path/to/images_3d; fi

TASK=${SLURM_ARRAY_TASK_ID:-0}
FOLD=$(( TASK / 3 + 1 ))
SUB=$(( TASK % 3 + 1 ))

python train.py --config "$CONFIG" --endpoint "$ENDPOINT" --fold "$FOLD" --sub "$SUB" \
    --data_path "$DATA" --split_csv "$SPLIT_CSV" --results_root "$RESULTS"

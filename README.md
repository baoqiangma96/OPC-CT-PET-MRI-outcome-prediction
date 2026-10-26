# CT, PET and MRI deep learning models for outcome prediction in oropharyngeal cancer

Code accompanying the manuscript

> Ma B, Guo J, van Dijk LV, Langendijk JA, van Ooijen PMA, Both S, Sijtsema NM. Comparation of CT/PET/MRI in deep
> learning models for outcome prediction in oropharyngeal cancer patients. Preprint, Research Square, 2025.
> https://doi.org/10.21203/rs.3.rs-8163685/v1

**Status: the manuscript is under peer review.** The code corresponds to the version under review. Some analyses
in `analysis/` were added during the review and are not in the preprint. This README will be updated with the
journal reference after publication.

The study trains 2D and 3D ResNet18 survival models on pre-treatment CT, FDG-PET, T1- and T2-weighted MRI of 266
patients with oropharyngeal cancer (186 training with three-fold cross-validation, 80 test). It predicts local
control (LC), regional control (RC) and overall survival (OS), and compares the models with the clinical
HNC-PREDICTOR model and with hybrid (clinical + imaging) models. Each image is restricted to the primary tumour
(GTV), the GTV with a margin (GTV_margin) or a box around it (GTV_cmargin).

**Data and trained models are not shared.** The patient data cannot be made public (patient privacy), and the
trained networks are not released. The repository contains the full code to preprocess images, train and test
the models, and reproduce the statistical analyses on comparable data. `data/README.md` describes the expected
input format, and `tests/` runs the whole pipeline on synthetic data.

## Repository

```
train.py              train one network (2D or 3D) and write its test/all-patient predictions
evaluate.py           test-set C-index of the 9-network ensembles (with bootstrap CI)
predict.py            apply trained networks to new, pre-processed patients
opc_dl/               configurations, data handling, transforms, loss, training loop
preprocessing/        MRI and CT/PET preprocessing, train/test split            (preprocessing/README.md)
analysis/             tables and figures of the paper from saved predictions    (analysis/README.md)
tools/export_weights.py   collect trained networks into one folder (no patient data)
scripts/              SLURM job for the 3 folds x 3 seeds; run all analyses
tests/                unit tests, synthetic end-to-end test, check against the study's networks (authors only)
data/README.md        expected file names and CSV columns
docs/known_issues.md  behaviour of the original code that is kept for reproducibility (read this)
docs/verification.md  what has been checked, with results
```

## Installation

```bash
git clone https://github.com/baoqiangma96/OPC-CT-PET-MRI-outcome-prediction.git
cd OPC-CT-PET-MRI-outcome-prediction
conda env create -f environment.yml      # or: pip install -r requirements.txt  (Python 3.11)
conda activate opc-dl
python tests/test_units.py               # seconds, no data needed
python tests/smoke_test.py               # synthetic data, whole pipeline, ~20 min on CPU
```

The study networks were trained with PyTorch 1.12.1 (CUDA 11.7) and MONAI 1.x on one NVIDIA V100 or A100 GPU.
This repository was tested with PyTorch 2.14 and MONAI 1.6 (see requirements.txt), and gives the same network
outputs as the original code (docs/verification.md).

## Model configurations

| paper name | `--config` | input |
|---|---|---|
| 2D T1_GTV / T2_GTV | `2D_T1_GTV`, `2D_T2_GTV` | 25 axial slices (0.5 x 0.5 x 3 mm, 320 x 320 crop) inside the GTV |
| 2D T1_GTV_margin / T2_GTV_margin | `2D_T1_GTV_margin`, `2D_T2_GTV_margin` | inside the dilated GTV (2.0–2.5 mm in-plane) |
| 2D T1_GTV_cmargin / T2_GTV_cmargin | `2D_T1_GTV_cmargin`, `2D_T2_GTV_cmargin` | inside one box around the whole dilated GTV |
| 3D CT/PET/T1/T2_GTV | `3D_CT_GTV`, `3D_PET_GTV`, `3D_T1_GTV`, `3D_T2_GTV` | 192 mm cube resized to 96³ (2 mm), inside the GTV |
| 3D ..._GTV_margin | `3D_CT_GTV_margin`, ... | inside the GTV + 5 mm |
| 3D ..._GTV_cmargin | `3D_CT_GTV_cmargin`, ... | inside a box around each component of GTV + 5 mm |
| supplementary | `2D_T1_GTV_cmargin_10mm`, `2D_T2_GTVtot`, ... | larger 2D margin; primary + nodal GTV |

All networks are MONAI ResNet18 models with one output (log-risk). They are trained with SGD (lr 2e-4, momentum
0.9, weight decay 2e-4, batch 12), an event-balanced sampler, flips, affine and elastic augmentation, and at most
400 epochs. Early stopping uses the validation C-index (patience 25). The ensemble prediction is the mean of nine
networks: 3 folds x 3 seeds (42, 0, 1).

## Running

**1. Preprocessing**: see [preprocessing/README.md](preprocessing/README.md). This step makes the images and
masks listed in [data/README.md](data/README.md) and the split CSV (`preprocessing/make_split.py`).

**2. Training** one network:

```bash
python train.py --config 2D_T2_GTV_cmargin --endpoint LC --fold 1 --sub 1 \
    --data_path /data/opc/mri_2d --split_csv /data/opc/clinical_split.csv --results_root runs
```

To train all nine networks of a configuration, run `sbatch --array=0-8 scripts/train_all.sh 2D_T2_GTV_cmargin LC`.
Each run folder `runs/<config>/<endpoint>/fold<k>_sub<s>/` gets `best_metric_model.pth`, `predicted_lp.csv`
(test set), `predicted_lp_all.csv` (all study patients), `training_log.csv` and `config.json`. The defaults
reproduce the study. `--loss cox` and `--crop_centre bbox` switch to the corrected loss and crop centre
(docs/known_issues.md).

**3. Testing**: test-set C-index of the ensembles:

```bash
python evaluate.py --results_root runs --split_csv /data/opc/clinical_split.csv \
    --configs 2D_T2_GTV_cmargin 3D_T1_GTV --endpoints LC RC OS
```

**4. Trained networks on new data**: apply the nine networks of a configuration to new, identically
pre-processed patients:

```bash
python predict.py --config 2D_T2_GTV_cmargin --endpoint LC --weights_root runs \
    --data_path /data/new/mri_2d --ids new_patients.csv --out predictions.csv
```

**5. Tables and figures of the paper**: [analysis/README.md](analysis/README.md) (`scripts/run_analysis.sh`).

## Reproducibility

* Unit tests compare the refactored transforms and loss with the original code (`tests/test_units.py`).
* With the study's trained networks, this code reproduces the stored test-set predictions of all 18 model
  configurations (`tests/test_reproduce_published.py`, run by the authors). The maximum absolute difference is
  2e-6, or 7e-4 for the 3D networks trained on an A100. The ensemble of the best LC model (2D T2_GTV_cmargin)
  gives C = 0.876 on the test set.
* `preprocessing/make_split.py` regenerates the study's train/test split and folds exactly, and the analysis
  scripts reproduce the paper's tables.
* GPU training is not bit-wise deterministic, so retrained networks differ slightly from the study's networks.

Read [docs/known_issues.md](docs/known_issues.md) before re-using the training code. Among other things, the loss
the study networks were trained with does not use the event indicator as intended.

## Citation, licence, contact

If you use this code, please cite the preprint above (see `CITATION.cff`); the journal reference will follow
after publication. The code is released under the MIT licence (`LICENSE`). Questions and problems: please open
a [GitHub issue](https://github.com/baoqiangma96/OPC-CT-PET-MRI-outcome-prediction/issues).

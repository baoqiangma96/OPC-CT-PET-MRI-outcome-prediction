# Known issues in the original training code

The code in this repository reproduces the models of the study exactly. That includes the behaviour below, which
was found while preparing the code for release. The defaults keep the original behaviour so that the study's
networks and numbers can be reproduced. Options with the corrected behaviour are available for new models.

## 1. Survival loss: event status only enters through the number of events

`opc_dl/losses.py`, `--loss as_published` (default) versus `--loss cox`.

In the original `losses.py` the event vector has shape `(B,)` and the network output has shape `(B, 1)`. The
product `(risk - log_risk_set) * event` therefore broadcasts to a `B x B` matrix. The loss becomes

    -[n_events / (n_events + 0.1)] * sum over ALL patients i of (risk_i - log mean_{j: t_j >= t_i} exp(risk_j))

so every patient in the batch is treated as if they had an event at their own follow-up time. Censoring still
enters through the risk sets, but not through the event indicator. The intended Cox partial likelihood sums over
patients with an event only. The loss stays a ranking loss on observed times, so the models still learned from
the data, and the C-index evaluation is unaffected. However, the training objective differs from the standard
Cox partial likelihood. `tests/test_units.py::test_loss_as_published_ignores_event_status`
demonstrates the behaviour. Use `--loss cox` for new models.

## 2. 2D crop centre

`opc_dl/transforms.py`, `--crop_centre as_published` (default) versus `--crop_centre bbox`.

The 25-slice crop is centred on the axial slice with the largest GTV area (correct). The in-plane centre was
computed as `(nz[0].min() + nz[0].max()) / 2` with `nz = mask[:, :, z].nonzero()`. `torch.nonzero` returns one
row per voxel, so `nz[0]` and `nz[1]` are the first and second non-zero voxels, not the x and y columns. The crop
centre is therefore not the centre of the GTV bounding box. With a 320 x 320 crop (160 mm) no GTV voxel fell
outside the crop in-plane in the test patients. Along z, however, the 25 slices (75 mm) miss part of the GTV in
33 of 266 patients for T1 (at least 77% of the GTV kept) and in 9 for T2 (at least 94% kept).

## 3. Box ROI (`_cmargin`): one box in 2D, one box per component in 3D

* **Main 2D models** (`2D_T1_GTV_cmargin`, `2D_T2_GTV_cmargin`; `single_box`): **one** box around all
  voxels of the dilated GTV inside the 25-slice crop. It is written into the mask, so the ROI is the box
  `[min, max)` plus the mask voxels on its upper faces. The script `main_t*_2D_gtvvolume_5mmbb.py` was later
  changed to one box per connected component. The changed version does not reproduce the stored predictions of
  these models. The earlier version, still in the file as a commented-out block, reproduces them exactly
  (`docs/verification.md`). When the dilated GTV has several components inside the crop, the 2D box therefore
  also covers the tissue between them.
* **Supplementary 2D models** trained later with the changed script (`_cmargin_10mm`, `GTVtot_cmargin`) and **all
  3D models** (`component_boxes`): one box per connected component, `[min, max)` on every axis, so the last row,
  column and slice of each box are missing.

Each configuration uses the variant it was trained with (`boxes` in `opc_dl/configs.py`).

## 4. 2D margins are smaller than their file names

`preprocessing/mri_04_gtv_margin.py`. The 2D "5 mm" masks (`*_5mm`) were made with a 2 x 10 x 10 voxel box on
the 0.5 x 0.5 x 3 mm grid. That is a margin of 2.0–2.5 mm in-plane, plus one slice on one side. The "10 mm" masks
used 3 x 20 x 20 voxels: 4.5–5.0 mm in-plane and one slice on each side. The 3D margin is a true 5 mm ball (1 mm
grid). `analysis/a10_margin_check.py` measures the margins on the masks.

## 5. Smaller points

* The training-set and test-set C-index were logged every epoch (W&B in the original code, `training_log.csv`
  here). They were never used for checkpoint selection, which uses the validation fold only. Disable the logging
  with `--no_monitor`. Note that it changes the random-number sequence, so a run without it is not identical.
* GPU training is not bit-wise reproducible (cuDNN). Retrained networks differ from the study's networks, but
  inference with trained networks is reproducible (see `docs/verification.md`).
* The original scripts hard-coded the HPC paths and the W&B account, wrote an example input image every batch,
  and selected the study patients with a fixed row slice (`[:-67]`). These were removed or replaced by the split
  column (`CV4` = excluded).

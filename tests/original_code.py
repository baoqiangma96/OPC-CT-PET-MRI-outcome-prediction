"""Verbatim excerpts of the original training code (MRI_umcg_new / ct_pt_mri, 2024), used only by the tests to
check that the refactored code in opc_dl/ computes exactly the same thing. Only device handling (.cuda()) and the
removed `scipy.ndimage.measurements` import path were changed."""
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import label
from torch import nn


# ---- MRI_umcg_new/main_t*_2D_gtvvolume_5mmbb.py ----------------------------------------------------------------
def get_largestArea_slice_center(gtv):
    gtv = gtv[0]
    area_slices = [torch.sum(gtv[:, :, z]) for z in range(gtv.shape[2])]
    slice_idx = torch.argmax(torch.tensor(area_slices))
    center = (gtv[:, :, slice_idx].nonzero()[0].min() + gtv[:, :, slice_idx].nonzero()[0].max()) / 2, \
             (gtv[:, :, slice_idx].nonzero()[1].min() + gtv[:, :, slice_idx].nonzero()[1].max()) / 2, slice_idx
    return center


class CropByLargestAreaCenter:
    """As in the script file today. Its per-component box (bb=True) reproduces the supplementary 2D _cmargin models
    (10 mm, GTVtot), but NOT the 2D _cmargin models of the paper: those are only reproduced by the earlier,
    commented-out box code (CropByLargestAreaCenterAsTrained below); see docs/verification.md."""

    def __init__(self, crop_size, bb=False, keys=("image", "gtv"), allow_missing_keys=False):
        self.crop_size = crop_size
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.bb = bb

    def __call__(self, data):
        gtv = data[self.keys[-1]]
        center = get_largestArea_slice_center(gtv)
        center = int(center[0].numpy()), int(center[1].numpy()), int(center[2].numpy())
        zmin = max(0, center[0] - self.crop_size[0] // 2)
        zmax = min(gtv.shape[1], zmin + self.crop_size[0])
        zmin = max(0, zmax - self.crop_size[0])
        ymin = max(0, center[1] - self.crop_size[1] // 2)
        ymax = min(gtv.shape[2], ymin + self.crop_size[1])
        ymin = max(0, ymax - self.crop_size[1])
        xmin = max(0, center[2] - self.crop_size[2] // 2)
        xmax = min(gtv.shape[3], xmin + self.crop_size[2])
        xmin = max(0, xmax - self.crop_size[2])
        for key in self.keys:
            if key in data:
                data[key] = data[key][:, zmin:zmax, ymin:ymax, xmin:xmax]
        if self.bb == True:
            gtv = data[self.keys[-1]]
            labeled_gtv, num_tumors = label(gtv)
            bounding_boxes = []
            for tumor_label in range(1, num_tumors + 1):
                tumor_mask = labeled_gtv == tumor_label
                tumor_mask = torch.tensor(tumor_mask.astype(int))
                bounding_boxes.append((tumor_mask.nonzero()[:, 1].min(), tumor_mask.nonzero()[:, 1].max(),
                                       tumor_mask.nonzero()[:, 2].min(), tumor_mask.nonzero()[:, 2].max(),
                                       tumor_mask.nonzero()[:, 3].min(), tumor_mask.nonzero()[:, 3].max()))
            new_gtv = torch.zeros_like(gtv)
            for zmin, zmax, ymin, ymax, xmin, xmax in bounding_boxes:
                new_gtv[:, zmin:zmax, ymin:ymax, xmin:xmax] = 1
            data[self.keys[-1]] = new_gtv
        return data


class CropByLargestAreaCenterAsTrained(CropByLargestAreaCenter):
    """Same crop; box = the block that is commented out in the script file ("using bounding box around tumor
    instead of gtv tumor"), which reproduces the stored predictions of the published 2D _cmargin models."""

    def __call__(self, data):
        bb, self.bb = self.bb, False
        data = super().__call__(data)
        self.bb = bb
        if self.bb == True:
            gtv = data[self.keys[-1]]
            gtv[:, gtv.nonzero()[:, 1].min(): gtv.nonzero()[:, 1].max(), gtv.nonzero()[:, 2].min(): gtv.nonzero()[:, 2].max(),
                gtv.nonzero()[:, 3].min(): gtv.nonzero()[:, 3].max()] = 1
            data[self.keys[-1]] = gtv
        return data


# ---- ct_pt_mri/main.py: AddMarginToGTV.__call__ / add_margin ---------------------------------------------------
def add_margin_to_gtv(gtv, margin_size, use_bb):
    if margin_size != 0:
        gtv_image = sitk.GetImageFromArray(gtv)
        spacing = [1.0, 1.0, 1.0]
        margin_size_voxels = [int(margin_size / s) for s in spacing]
        gtv_image = sitk.Cast(gtv_image, sitk.sitkInt32)
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelType(sitk.sitkBall)
        dilate_filter.SetKernelRadius(margin_size_voxels)
        dilated_image = dilate_filter.Execute(gtv_image)
        gtv = torch.tensor(sitk.GetArrayFromImage(dilated_image))
    if use_bb == True:
        labeled_gtv, num_tumors = label(gtv)
        bounding_boxes = []
        for tumor_label in range(1, num_tumors + 1):
            tumor_mask = labeled_gtv == tumor_label
            tumor_mask = torch.tensor(tumor_mask.astype(int))
            bounding_boxes.append((tumor_mask.nonzero()[:, 0].min(), tumor_mask.nonzero()[:, 0].max(),
                                   tumor_mask.nonzero()[:, 1].min(), tumor_mask.nonzero()[:, 1].max(),
                                   tumor_mask.nonzero()[:, 2].min(), tumor_mask.nonzero()[:, 2].max()))
        new_gtv = torch.zeros_like(gtv)
        for zmin, zmax, ymin, ymax, xmin, xmax in bounding_boxes:
            new_gtv[zmin:zmax, ymin:ymax, xmin:xmax] = 1
        gtv = new_gtv
    return gtv


# ---- losses.py (both folders) -----------------------------------------------------------------------------------
class Regularization(object):
    def __init__(self, order, weight_decay):
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        return self.weight_decay * reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.L2_reg = 0.00005
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        y = y.unsqueeze(1)
        mask[(y.T - y) > 0] = 0
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / (torch.sum(mask, dim=0) + 0.1)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / (torch.sum(e) + 0.1)
        l2_norm = self.reg(model)
        return neg_log_loss + l2_norm, neg_log_loss, l2_norm

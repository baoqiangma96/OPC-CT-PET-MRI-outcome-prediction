"""Image transforms of the 2D and 3D pipelines.

These reproduce the transforms used to train the study's models (MONAI 1.x, PyTorch 1.12); with them the trained
networks reproduce their stored predictions (docs/verification.md). Details of the original code that are kept
on purpose (docs/known_issues.md):
  * 2D crop centre (centre='as_published'): the in-plane centre comes from the first two non-zero voxels of the
    largest-area slice, not from the bounding box of the GTV in that slice (use centre='bbox' for new models);
  * box ROI (_cmargin): the 2D paper models use ONE box around all voxels of the dilated GTV inside the crop,
    written into the mask (single_box); the 3D and the supplementary 2D models one box per connected component
    (component_boxes). Both boxes stop one voxel short of the last row/column/slice of the mask.
"""
import numpy as np
import SimpleITK as sitk
import torch
from monai.data import MetaTensor
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged, MapTransform, Rand3DElasticd, RandAffined,
                              RandFlipd, Resized, ScaleIntensityRanged, Spacingd)
from scipy.ndimage import label

CROP_2D = (320, 320)            # in-plane crop (voxels of 0.5 mm = 160 mm)
SPACING_2D = (0.5, 0.5, 3.0)    # mm
SIZE_3D = (96, 96, 96)          # 192 mm cube at 1 mm resized to 2 mm voxels
INTENSITY_3D = {'ct': (-200, 200), 'pt': (0, 25)}   # clipped and scaled to [0, 1]; MRI is z-scored beforehand


def _plain(x):
    return x.as_tensor() if isinstance(x, MetaTensor) else torch.as_tensor(x)


def single_box(mask):
    """2D paper _cmargin models: set the bounding box [min, max) of all mask voxels to 1 inside the mask, i.e. the
    box plus the mask voxels on its upper faces (last three axes are spatial)."""
    out = mask.clone()
    nz = _plain(mask).nonzero()[:, -3:]
    lo, hi = nz.min(dim=0).values.tolist(), nz.max(dim=0).values.tolist()
    out[..., lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return out


def component_boxes(mask):
    """Replace every connected component of a binary mask by its bounding box (last three axes are spatial)."""
    labelled, n = label(np.asarray(mask))
    out = torch.zeros_like(mask)
    for k in range(1, n + 1):
        idx = np.nonzero(labelled == k)[-3:]
        lo = [int(i.min()) for i in idx]
        hi = [int(i.max()) for i in idx]
        # as published: the upper bound is exclusive, so the box misses the last voxel of the component
        out[..., lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return out


def largest_area_slice_centre(mask, centre='as_published'):
    """mask: (X, Y, Z). Returns the in-plane centre and the index of the axial slice with the largest mask area."""
    mask = _plain(mask)
    z = int(torch.argmax(torch.tensor([torch.sum(mask[:, :, k]) for k in range(mask.shape[2])])))
    nz = mask[:, :, z].nonzero()           # (N, 2), one row per non-zero voxel
    if centre == 'as_published':
        cx, cy = (nz[0].min() + nz[0].max()) / 2, (nz[1].min() + nz[1].max()) / 2
    elif centre == 'bbox':
        cx, cy = (nz[:, 0].min() + nz[:, 0].max()) / 2, (nz[:, 1].min() + nz[:, 1].max()) / 2
    else:
        raise ValueError(centre)
    return int(cx), int(cy), z


def _window(c, size, length):
    lo = max(0, c - size // 2)
    hi = min(length, lo + size)
    return max(0, hi - size), hi


class CropLargestAreaSliced(MapTransform):
    """2D pipeline: crop (320, 320, n_slices) around the largest-area slice of the mask; then, for the _cmargin ROI,
    boxes='single' or 'components' (see single_box / component_boxes). Arrays are (C, X, Y, Z)."""

    def __init__(self, keys, mask_key, n_slices, boxes='', centre='as_published'):
        super().__init__(keys)
        if boxes not in ('', 'single', 'components'):
            raise ValueError(boxes)
        self.mask_key, self.size, self.boxes, self.centre = mask_key, (*CROP_2D, n_slices), boxes, centre

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key]
        c = largest_area_slice_centre(mask[0], self.centre)
        win = [_window(c[i], self.size[i], mask.shape[i + 1]) for i in range(3)]
        for key in self.key_iterator(d):
            d[key] = d[key][:, win[0][0]:win[0][1], win[1][0]:win[1][1], win[2][0]:win[2][1]]
        if self.boxes == 'single':
            d[self.mask_key] = single_box(d[self.mask_key])
        elif self.boxes == 'components':
            d[self.mask_key] = component_boxes(d[self.mask_key])
        return d


class AddMarginToGTVd(MapTransform):
    """3D pipeline: dilate the GTV with a ball of margin_mm (the 3D data are at 1 mm) and optionally replace it by
    boxes around its connected components. Applied before the channel dimension is added."""

    def __init__(self, keys, margin_mm=0, boxes=False):
        super().__init__(keys)
        self.margin_mm, self.boxes = margin_mm, boxes

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = d[key]
            if self.margin_mm:
                img = sitk.Cast(sitk.GetImageFromArray(np.asarray(mask)), sitk.sitkInt32)
                dilate = sitk.BinaryDilateImageFilter()
                dilate.SetKernelType(sitk.sitkBall)
                dilate.SetKernelRadius([int(self.margin_mm / 1.0)] * 3)
                arr = torch.tensor(sitk.GetArrayFromImage(dilate.Execute(img)))
                mask = MetaTensor(arr, meta=dict(mask.meta)) if isinstance(mask, MetaTensor) else arr
            if self.boxes:
                mask = component_boxes(mask)
            d[key] = mask
        return d


def _augmentation(keys, modes):
    return [RandFlipd(keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys, prob=0.5, spatial_axis=2),
            RandAffined(keys, prob=0.5, translate_range=(7, 7, 7), rotate_range=(np.pi / 24,) * 3,
                        scale_range=(0.07, 0.07, 0.07), padding_mode='border', mode=modes),
            Rand3DElasticd(keys, prob=0.2, sigma_range=(5, 8), magnitude_range=(100, 200),
                           translate_range=(7, 7, 7), rotate_range=(np.pi / 24,) * 3,
                           scale_range=(0.07, 0.07, 0.07), padding_mode='border', mode=modes)]


def build_transforms(cfg, train, n_slices=25, centre='as_published'):
    """Training (with augmentation) or evaluation transforms for a ModelConfig."""
    keys, modes = cfg.keys, ('bilinear', 'nearest')
    t = [LoadImaged(keys)]
    if cfg.dims == 2:
        t.append(EnsureChannelFirstd(keys, channel_dim='no_channel'))
        if cfg.resample_2d:
            t.append(Spacingd(keys, pixdim=SPACING_2D, mode=modes))
        t.append(CropLargestAreaSliced(keys, cfg.mask_key, n_slices, boxes=cfg.boxes, centre=centre))
    else:
        t += [AddMarginToGTVd(cfg.mask_key, cfg.margin_mm, cfg.boxes),
              EnsureChannelFirstd(keys, channel_dim='no_channel'),
              Resized(keys, spatial_size=SIZE_3D, mode=('trilinear', 'nearest'), align_corners=(True, None))]
        if cfg.image_key in INTENSITY_3D:
            a, b = INTENSITY_3D[cfg.image_key]
            t.append(ScaleIntensityRanged(cfg.image_key, a_min=a, a_max=b, b_min=0, b_max=1, clip=True))
    if train:
        t += _augmentation(keys, modes)
    return Compose(t)


def network_input(batch, cfg, device):
    """Image restricted to the ROI. 2D: (B, Z, Y, X), the 25 slices as channels; 3D: (B, 1, X, Y, Z)."""
    x = batch[cfg.image_key].to(device) * batch[cfg.mask_key].to(device)
    if cfg.dims == 2:
        x = torch.transpose(x[:, 0], 1, 3)
    return x.as_tensor() if isinstance(x, MetaTensor) else x

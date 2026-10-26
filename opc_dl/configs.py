"""Model configurations of the paper.

A configuration name fixes the input image, the tumour mask, the region of interest and the network:

    2D_<T1|T2>_<GTV|GTVtot>[_margin|_cmargin][_10mm]     2D ResNet18, 25 axial MRI slices as channels
    3D_<CT|PET|T1|T2>_GTV[_margin|_cmargin]              3D ResNet18, 96^3 volume

ROI: GTV = image inside the GTV; _margin = image inside the dilated GTV; _cmargin = image inside a box ("cuboid
margin") around the dilated GTV. The box variant differs between the published models, because the original
2D script was changed between training runs (docs/known_issues.md #3):
    2D_T1/T2_GTV_cmargin (paper models)   one box around all of the dilated GTV inside the crop ('single')
    2D ..._cmargin_10mm, 2D ..._GTVtot_cmargin, all 3D _cmargin   one box per connected component ('components')

Input files are <data_path>/<PatientID>_<key>.nii.gz with the keys returned by parse_config (see data/README.md).
"""
import re
from dataclasses import dataclass

# 2D: sequence -> (image key at the original resolution, image key resampled to 0.5 x 0.5 x 3 mm)
MRI_2D = {'T1': ('t1', 't1r'), 'T2': ('t2', 't2r')}
# 3D: modality -> image key (CT/PET resampled to 1 mm; T1/T2 deformably registered to the planning CT)
IMAGES_3D = {'CT': 'ct', 'PET': 'pt', 'T1': 'T1dr', 'T2': 'T2dr'}
MASK_3D = 'ct_GTVpt'

PAPER_CONFIGS = ([f'2D_{s}_GTV{r}' for s in ('T1', 'T2') for r in ('', '_margin', '_cmargin')] +
                 [f'3D_{m}_GTV{r}' for m in ('CT', 'PET', 'T1', 'T2') for r in ('', '_margin', '_cmargin')])


@dataclass(frozen=True)
class ModelConfig:
    name: str
    dims: int               # 2 or 3
    image_key: str
    mask_key: str
    resample_2d: bool       # 2D only: resample image and mask to 0.5 x 0.5 x 3 mm on the fly
    margin_mm: int          # 3D only: ball dilation of the GTV (mm, data at 1 mm)
    boxes: str              # '' (no box), 'single' or 'components' (see above)
    legacy_prefix: str      # run-folder prefix used by the original training scripts

    @property
    def keys(self):
        return [self.image_key, self.mask_key]


def parse_config(name):
    m = re.fullmatch(r'2D_(T1|T2)_(GTV|GTVtot)(_margin|_cmargin)?(_10mm)?', name)
    if m:
        seq, gtv, roi, ten = m.group(1), m.group(2), m.group(3) or '', m.group(4)
        if ten and not roi:
            raise ValueError(f'{name}: _10mm needs _margin or _cmargin')
        n = seq[1]
        stem = f"gtv{'pt' if gtv == 'GTV' else 'tot'}tot{n}"
        if roi:   # pre-resampled image and pre-computed dilated mask (preprocessing/mri_04_gtv_margin.py)
            image, mask, resample = MRI_2D[seq][1], f"{stem}r_{'10mm' if ten else '5mm'}", False
        else:     # original resolution, resampled on the fly
            image, mask, resample = MRI_2D[seq][0], stem, True
        legacy = {'': 'gtvvolume_', '_margin': 'gtvvolume_5mm_', '_cmargin': 'gtvvolume_5mmbb_'}[roi]
        legacy += f"ResNet18_2D_input_['{image}', '{mask}']_slice_num_25_OS_True"
        boxes = '' if roi != '_cmargin' else 'single' if (gtv == 'GTV' and not ten) else 'components'
        return ModelConfig(name, 2, image, mask, resample, 0, boxes, legacy)
    m = re.fullmatch(r'3D_(CT|PET|T1|T2)_GTV(_margin|_cmargin)?', name)
    if m:
        image, roi = IMAGES_3D[m.group(1)], m.group(2) or ''
        margin = 5 if roi else 0
        legacy = (f"ResNet18_input_['{image}', '{MASK_3D}']_inputtype_1_margin{margin}"
                  f"_usebb_{roi == '_cmargin'}_OS_True")
        return ModelConfig(name, 3, image, MASK_3D, False, margin, 'components' if roi else '', legacy)
    raise ValueError(f'Unknown configuration {name!r}; examples: {", ".join(PAPER_CONFIGS[:3])}, 3D_CT_GTV')

"""Unit tests that need no data: the refactored transforms and loss equal the original code.

    python tests/test_units.py        (or: python -m pytest tests/test_units.py)
"""
import os
import sys

import numpy as np
import torch
from monai.data import MetaTensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import original_code as orig  # noqa: E402
from opc_dl.losses import CoxLoss  # noqa: E402
from opc_dl.transforms import AddMarginToGTVd, CropLargestAreaSliced, component_boxes  # noqa: E402


def random_mask(rng, shape, n_blobs=2):
    m = np.zeros(shape, np.float32)
    for _ in range(n_blobs):
        c = [rng.integers(5, s - 5) for s in shape]
        r = [rng.integers(2, max(3, s // 6)) for s in shape]
        g = np.ogrid[tuple(slice(0, s) for s in shape)]
        m[sum(((g[i] - c[i]) / r[i]) ** 2 for i in range(3)) <= 1] = 1
    return m


def test_crop_and_boxes_2d():
    rng = np.random.default_rng(0)
    for trial in range(40):
        shape = (int(rng.integers(330, 420)), int(rng.integers(330, 420)), int(rng.integers(26, 40)))
        mask = random_mask(rng, shape, n_blobs=int(rng.integers(1, 4)))
        img = rng.normal(size=shape).astype(np.float32)
        for original, bb, boxes in ((orig.CropByLargestAreaCenterAsTrained, False, ''),
                                    (orig.CropByLargestAreaCenterAsTrained, True, 'single'),
                                    (orig.CropByLargestAreaCenter, True, 'components')):
            d0 = {'t1r': MetaTensor(img[None].copy()), 'gtv': MetaTensor(mask[None].copy())}
            d1 = {'t1r': MetaTensor(img[None].copy()), 'gtv': MetaTensor(mask[None].copy())}
            a = original((320, 320, 25), bb=bb, keys=['t1r', 't2r', 'dose', 'gtv'], allow_missing_keys=True)(d0)
            b = CropLargestAreaSliced(['t1r', 'gtv'], 'gtv', 25, boxes=boxes)(d1)
            for k in ('t1r', 'gtv'):
                assert a[k].shape == b[k].shape, (trial, k, a[k].shape, b[k].shape)
                assert torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])), (trial, k, boxes)


def test_2d_box_differs_from_per_component_box():
    """The per-component box in today's script file differs from the box the 2D models were trained with when the
    dilated GTV has several components inside the crop."""
    mask = np.zeros((340, 340, 30), np.float32)
    mask[100:120, 100:120, 10:14] = 1
    mask[200:215, 180:200, 12:15] = 1
    img = np.ones_like(mask)
    kw = dict(crop_size=(320, 320, 25), bb=True, keys=['t1r', 'gtv'])
    trained = orig.CropByLargestAreaCenterAsTrained(**kw)({'t1r': MetaTensor(img[None]), 'gtv': MetaTensor(mask[None])})
    script = orig.CropByLargestAreaCenter(**kw)({'t1r': MetaTensor(img[None]), 'gtv': MetaTensor(mask[None])})
    assert trained['gtv'].sum() > script['gtv'].sum()


def test_margin_and_boxes_3d():
    rng = np.random.default_rng(1)
    for trial in range(10):
        mask = random_mask(rng, (60, 64, 56), n_blobs=int(rng.integers(1, 4)))
        for margin, boxes in ((0, ''), (5, ''), (5, 'components')):
            a = orig.add_margin_to_gtv(MetaTensor(mask.copy()), margin, bool(boxes))
            b = AddMarginToGTVd('gtv', margin, boxes)({'gtv': MetaTensor(mask.copy())})['gtv']
            assert torch.equal(torch.as_tensor(a).float(), torch.as_tensor(b).float()), (trial, margin, boxes)


def test_component_boxes_excludes_last_voxel():
    m = torch.zeros(1, 10, 10, 10)
    m[0, 2:6, 3:7, 4:8] = 1                      # one 4 x 4 x 4 cube
    box = component_boxes(m)
    assert int(box.sum()) == 3 * 3 * 3           # as published: 3 x 3 x 3, not 4 x 4 x 4


def test_loss_as_published_equals_original():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    for _ in range(20):
        n = 12
        risk = torch.randn(n, 1)
        time = torch.rand(n) * 60
        event = (torch.rand(n) < 0.3).float()
        a = orig.NegativeLogLikelihood()(risk, time, event, model)
        b = CoxLoss('as_published')(risk, time, event, model)
        for x, y in zip(a, b):
            assert torch.allclose(x, y, atol=1e-6)


def test_loss_as_published_ignores_event_status():
    """The published loss treats every patient as an event: changing WHICH patients had the event does not change
    it (only the number of events enters, via the factor n_events / (n_events + 0.1))."""
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    risk, time = torch.randn(10, 1), torch.rand(10) * 60
    e1 = torch.tensor([1, 0, 0, 1, 0, 0, 0, 1, 0, 0]).float()
    e2 = torch.tensor([0, 1, 1, 0, 0, 0, 1, 0, 0, 0]).float()    # same number of events, other patients
    pub = CoxLoss('as_published')
    cox = CoxLoss('cox')
    assert torch.allclose(pub(risk, time, e1, model)[1], pub(risk, time, e2, model)[1])
    assert not torch.allclose(cox(risk, time, e1, model)[1], cox(risk, time, e2, model)[1])
    all_events = torch.ones(10)
    factor = e1.sum() / (e1.sum() + 0.1) * (all_events.sum() + 0.1)
    assert torch.allclose(pub(risk, time, e1, model)[1], factor * cox(risk, time, all_events, model)[1], atol=1e-6)


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok', name)

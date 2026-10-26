"""Survival losses.

'as_published' is the loss the published models were trained with (original losses.py). Because the event vector
has shape (B,) and the network output (B, 1), the product (risk - log_risk_set) * event broadcasts to a B x B
matrix: every patient in the batch contributes as if it had an event, scaled by n_events / (n_events + 0.1).
Censoring therefore only enters through the risk sets. 'cox' is the intended Cox negative partial log-likelihood:
only patients with an event contribute (the risk-set sum is divided by the risk-set size, which only adds a
constant). See docs/known_issues.md and tests/test_losses.py.
"""
import torch
from torch import nn


class L2Penalty:
    """weight_decay * sum of the L2 norms (not squared) of all weight tensors, as in the original code."""

    def __init__(self, weight_decay=5e-5):
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg = reg + torch.norm(w, p=2)
        return self.weight_decay * reg


class CoxLoss(nn.Module):
    def __init__(self, kind='as_published', l2=5e-5):
        super().__init__()
        if kind not in ('as_published', 'cox'):
            raise ValueError(kind)
        self.kind, self.reg = kind, L2Penalty(l2)

    def forward(self, risk, time, event, model):
        """risk: (B, 1) network output; time, event: (B,). Returns (total loss, survival term, L2 term)."""
        time = time.to(risk.device).float()
        event = event.to(risk.device).float()
        mask = torch.ones(time.shape[0], time.shape[0], device=risk.device)
        t = time.unsqueeze(1)
        mask[(t.T - t) > 0] = 0                       # mask[i, j] = 1 if patient i is in the risk set of j
        log_risk_set = torch.log(torch.sum(torch.exp(risk) * mask, dim=0) / (torch.sum(mask, dim=0) + 0.1))
        log_risk_set = log_risk_set.reshape(-1, 1)
        if self.kind == 'as_published':
            nll = -torch.sum((risk - log_risk_set) * event) / (torch.sum(event) + 0.1)          # (B,1)*(B,) -> (B,B)
        else:
            nll = -torch.sum((risk - log_risk_set) * event.reshape(-1, 1)) / (torch.sum(event) + 0.1)
        l2 = self.reg(model)
        return nll + l2, nll, l2

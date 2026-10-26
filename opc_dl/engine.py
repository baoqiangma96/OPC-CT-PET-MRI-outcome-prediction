"""Training with early stopping, inference and evaluation (shared by the 2D and 3D models)."""
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index
from monai.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

SEEDS = {1: 42, 2: 0, 3: 1}   # --sub -> random seed (three repetitions per fold)


def raise_open_file_limit():
    """Many DataLoader workers on Linux can hit the default open-file limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    except (ImportError, ValueError):
        pass


def seed_everything(seed):
    from monai.utils import set_determinism
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def harrell_c(time, event, lp):
    """Harrell's C-index; lp is the network output (higher = higher risk). NaN outputs count as 0."""
    score = -np.asarray(lp, float)
    score[np.isnan(score)] = 0
    return concordance_index(time, score, event)


def make_loader(items, transform, batch_size, num_workers, sampler=None, shuffle=False):
    return DataLoader(Dataset(data=items, transform=transform), batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, loader, to_input, device):
    model.eval()
    out = [model(to_input(batch, device)).cpu() for batch in loader]
    return torch.cat(out).numpy()[:, 0]


def evaluate_loss(model, loader, to_input, criterion, device):
    """Mean over batches of the survival term of the loss (without L2), and the outputs."""
    model.eval()
    total, steps, outs = 0.0, 0, []
    with torch.no_grad():
        for batch in loader:
            out = model(to_input(batch, device))
            total += criterion(out, batch['time'], batch['event'], model)[1].item()
            steps += 1
            outs.append(out.cpu())
    return total / steps, torch.cat(outs).numpy()[:, 0]


class Logger:
    """Per-epoch CSV log; optionally mirrored to Weights & Biases."""

    def __init__(self, path, wandb_project=None, config=None, append=False):
        self.path, self.rows, self.wandb = path, [], None
        if append and os.path.exists(path):
            self.rows = pd.read_csv(path).to_dict('records')
        if wandb_project:
            import wandb
            self.wandb = wandb.init(project=wandb_project, config=config)

    def log(self, row):
        self.rows.append(row)
        with open(self.path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in self.rows for k in r)))
            w.writeheader()
            w.writerows(self.rows)
        if self.wandb:
            self.wandb.log(row)

    def finish(self, summary):
        if self.wandb:
            self.wandb.summary.update(summary)
            self.wandb.finish()


def train(model, loaders, sets, to_input, criterion, opt, run_dir, device, logger):
    """Training as in the paper: SGD, early stopping on the validation C-index.

    From epoch 4 onwards a model is kept when its validation C-index is the best so far AND its validation loss is
    below the best loss + 1.5; otherwise the patience counter (opt.esn) decreases, and training stops when it drops
    below zero. With opt.monitor the training-set and test-set C-index are also logged every epoch (as in the
    original scripts); they are never used for model selection.
    """
    clinical = sets['clinical']
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[200, 300], gamma=0.2)
    best_c, best_loss, best_epoch, patience = -1, 100000, -1, opt.esn
    start = 1
    ckpt_path = os.path.join(run_dir, 'last_checkpoint.pth')
    if opt.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        best_c, best_loss, best_epoch, patience = ck['best_c'], ck['best_loss'], ck['best_epoch'], ck['patience']
        start = ck['epoch'] + 1
        logger.rows = [r for r in logger.rows if r['epoch'] < start]
        print(f'resumed from epoch {ck["epoch"]}')
    val_time, val_event = sets['val_outcome']

    for epoch in range(start, opt.n_epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in loaders['train']:
            steps += 1
            optimizer.zero_grad()
            out = model(to_input(batch, device))
            loss = criterion(out, batch['time'], batch['event'], model)[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        row = {'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'], 'train_loss': epoch_loss / steps}
        if opt.monitor:
            lp = predict(model, loaders['train_eval'], to_input, device)
            row['train_cindex'] = harrell_c(*sets['train_outcome'], lp)
        scheduler.step()

        val_loss, val_lp = evaluate_loss(model, loaders['val'], to_input, criterion, device)
        val_c = harrell_c(val_time, val_event, val_lp)
        row.update({'val_loss': val_loss, 'val_cindex': val_c, 'saved': 0})
        if epoch > 3:
            if val_c > best_c and val_loss < best_loss + 1.5:
                best_c, best_loss, best_epoch, patience = val_c, val_loss, epoch, opt.esn
                torch.save(model.state_dict(), os.path.join(run_dir, 'best_metric_model.pth'))
                row['saved'] = 1
            else:
                patience -= 1
        row['patience'] = patience
        if opt.monitor:
            lp = predict(model, loaders['test'], to_input, device)
            row['test_cindex'] = harrell_c(*sets['test_outcome'], lp)
        logger.log(row)
        print(f"epoch {epoch}: train loss {row['train_loss']:.4f}, val loss {val_loss:.4f}, val C {val_c:.4f}"
              f"{' (saved)' if row['saved'] else ''}")
        if opt.checkpoint:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_c': float(best_c), 'best_loss': float(best_loss),
                        'best_epoch': int(best_epoch), 'patience': int(patience)}, ckpt_path)
        if patience < 0:
            print(f'early stopping at epoch {epoch}')
            break
    print(f'best validation C-index {best_c:.4f} at epoch {best_epoch}')
    return {'best_val_cindex': best_c, 'best_val_loss': best_loss, 'best_epoch': best_epoch}


def save_predictions(ids, lp, path):
    pd.DataFrame({'PatientID': ids, 'lp': lp}).to_csv(path)


def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)

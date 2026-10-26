# -*- coding: utf-8 -*-
"""
Shared data loading and statistics for the analyses of the paper (tables, figures, revision analyses).

Everything here works from saved model predictions (predicted_lp.csv / predicted_lp_all.csv of every network),
the clinical split CSV and, for a few scripts, the NIfTI inputs; no network is retrained.

Paths are read from environment variables (defaults in brackets, relative to the repository):
    OPC_SPLIT_CSV       clinical table with outcomes and split columns   [data/clinical_split.csv]
    OPC_HNC_CSV         table with the HNC-PREDICTOR variables            [same as OPC_SPLIT_CSV]
    OPC_MRI_PARAMS_CSV  MRI acquisition parameters (a03 only)             [data/mri_parameters.csv]
    OPC_DATA_2D         2D MRI inputs (a04a, a08, a10, a12)               [data/mri_2d]
    OPC_DATA_3D         3D inputs (a04a, a08, a10)                        [data/images_3d]
    OPC_RUNS            trained networks: <OPC_RUNS>/<config>/<endpoint>/fold<k>_sub<s>/   [runs]
    OPC_LEGACY_2D/_3D   optional: result folders of the original training scripts (flat folder names)
    OPC_OUTPUT          output folder; results/ and figures/ are created in it              [outputs]
"""
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from opc_dl.configs import parse_config  # noqa: E402


def _env(name, default):
    return os.environ.get(name) or default


SPLIT_CSV = _env('OPC_SPLIT_CSV', os.path.join(REPO, 'data', 'clinical_split.csv'))
HNC_CSV = _env('OPC_HNC_CSV', SPLIT_CSV)
MRI_PARAMS_CSV = _env('OPC_MRI_PARAMS_CSV', os.path.join(REPO, 'data', 'mri_parameters.csv'))
DATA_2D = _env('OPC_DATA_2D', os.path.join(REPO, 'data', 'mri_2d'))
DATA_3D = _env('OPC_DATA_3D', os.path.join(REPO, 'data', 'images_3d'))
RUNS = _env('OPC_RUNS', os.path.join(REPO, 'runs'))
LEGACY = {2: os.environ.get('OPC_LEGACY_2D', ''), 3: os.environ.get('OPC_LEGACY_3D', '')}
OUTPUT = _env('OPC_OUTPUT', os.path.join(REPO, 'outputs'))
RESULTS = os.path.join(OUTPUT, 'results')
FIGURES = os.path.join(OUTPUT, 'figures')
ROOT_2D = ROOT_3D = RUNS
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# endpoint key -> (event column, time column, label)
ENDPOINTS = {
    'LC': ('LR_code', 'TIME_LR', 'Local control'),
    'RC': ('RR_code', 'TIME_RR', 'Regional control'),
    'OS': ('OS_code', 'TIME_OS', 'Overall survival'),
    'DM': ('MET_code', 'TIME_MET', 'Distant metastasis-free survival'),
}
EVENT_TO_ENDPOINT = {v[0]: k for k, v in ENDPOINTS.items()}

# Published HNC-PREDICTOR coefficients (van Dijk et al., Eur J Cancer 2023), as used in the original analysis.
HNC_COEFS = {
    'OS': {'performance score 1': 0.469, 'performance score >=2': 0.781,
           'AJCC8th II': 0.117, 'AJCC8th III': 0.679, 'AJCC8th IVa': 0.793, 'AJCC8th IVb': 1.509,
           'pack-years 5-25': 0.267, 'pack-years 26-50': 0.499, 'pack-years >50': 0.867,
           'age 56-65': 0.085, 'age 65-75': 0.400, 'age >75': 0.753},
    'LC': {'T-stage 2': 1.432, 'T-stage 3': 1.473, 'T-stage 4': 1.613,
           'HPV status positive': -0.694,
           'performance score 1': 0.421, 'performance score >=2': 0.801,
           'pack-years 5-25': -0.039, 'pack-years 26-50': 0.294, 'pack-years >50': 0.496},
    'RC': {'AJCC8th II': 0.442, 'AJCC8th III': 0.984, 'AJCC8th IVa': 1.567, 'AJCC8th IVb': 2.565,
           'performance score 1': 0.573, 'performance score >=2': 0.793,
           'tumour site OPC': -0.648},
}

# Model configurations: display name -> configuration name (opc_dl/configs.py)
CONFIGS_2D = {f'2D {s}_GTV{r}': f'2D_{s}_GTV{r}' for s in ('T1', 'T2') for r in ('', '_margin', '_cmargin')}
CONFIGS_2D_EXTRA = {
    '2D T1_GTV_cmargin(10mm)': '2D_T1_GTV_cmargin_10mm',
    '2D T2_GTV_cmargin(10mm)': '2D_T2_GTV_cmargin_10mm',
    **{f'2D {s}_GTVtot{r}': f'2D_{s}_GTVtot{r}' for s in ('T1', 'T2') for r in ('', '_margin', '_cmargin')},
}
CONFIGS_3D = {f'3D {m}_GTV{r}': f'3D_{m}_GTV{r}' for m in ('CT', 'PET', 'T1', 'T2') for r in ('', '_margin', '_cmargin')}

MAIN_CONFIGS = {**{k: (ROOT_3D, v) for k, v in CONFIGS_3D.items()}, **{k: (ROOT_2D, v) for k, v in CONFIGS_2D.items()}}
EXTRA_CONFIGS = {k: (ROOT_2D, v) for k, v in CONFIGS_2D_EXTRA.items()}


def load_clinical():
    """266 study patients (train + test) with outcomes, split columns and HNC-PREDICTOR covariates."""
    df = pd.read_csv(SPLIT_CSV)
    df = df[df['CVgroup_event_t1t2'].isin(['train', 'test'])].copy()
    hnc = pd.read_csv(HNC_CSV)
    keep = ['PatientID', 'gender', 'age', 'performance score', 'smoking status', 'pack-years', 'T-stage',
            'N-stage', 'N-stage-original', 'tumour site', 'HPV status', 'AJCC8th']
    hnc = hnc[keep].rename(columns={'age': 'age_cat'})
    df = df.drop(columns=[c for c in keep[1:] if c in df.columns])
    df = df.merge(hnc, on='PatientID', how='left', validate='one_to_one')
    d = df
    for k, v in {'age 56-65': 1, 'age 65-75': 2, 'age >75': 3}.items():
        d[k] = (d['age_cat'] == v).astype(int)
    for k, v in {'performance score 1': 1, 'performance score >=2': 2}.items():
        d[k] = (d['performance score'] == v).astype(int)
    for k, v in {'pack-years 5-25': 1, 'pack-years 26-50': 2, 'pack-years >50': 3}.items():
        d[k] = (d['pack-years'] == v).astype(int)
    for k, v in {'T-stage 2': 1, 'T-stage 3': 2, 'T-stage 4': 3}.items():
        d[k] = (d['T-stage'] == v).astype(int)
    d['tumour site OPC'] = (d['tumour site'] == 2).astype(int)
    d['HPV status positive'] = (d['HPV status'] == 0).astype(int)
    for k, v in {'AJCC8th II': 1, 'AJCC8th III': 2, 'AJCC8th IVa': 3, 'AJCC8th IVb': 4}.items():
        d[k] = (d['AJCC8th'] == v).astype(int)
    for ep, coefs in HNC_COEFS.items():
        d[f'hnc_lp_{ep}'] = d[list(coefs)].dot(pd.Series(coefs))
    d['is_test'] = d['CVgroup_event_t1t2'] == 'test'
    return d.set_index('PatientID', drop=False)


def run_dir(root, config, event_col, fold, sub):
    """Run folder of one network: <root>/<config>/<endpoint>/fold<k>_sub<s>; if absent and OPC_LEGACY_2D/3D is
    set, the folder written by the original training scripts."""
    d = os.path.join(root, config, EVENT_TO_ENDPOINT.get(event_col, event_col), f'fold{fold}_sub{sub}')
    cfg = parse_config(config)
    if not os.path.isdir(d) and LEGACY[cfg.dims]:
        d = os.path.join(LEGACY[cfg.dims], f'{cfg.legacy_prefix}_{event_col}_fold{fold}_sub{sub}')
    return d


def load_predictions(root, prefix, ep, clin):
    """Return dict with 'test' (ensemble mean of the 9 runs on the test set), 'oof' (out-of-fold mean for
    training patients), 'insample' (mean of all 9 runs for training patients, as in the original analysis)
    and per-run test predictions. Returns None if no run exists."""
    event_col = ENDPOINTS[ep][0]
    cv_col = f'CVgroup_{event_col}_t1t2'
    runs_test, runs_all, folds = [], [], []
    for fold in (1, 2, 3):
        for sub in (1, 2, 3):
            d = run_dir(root, prefix, event_col, fold, sub)
            p_test, p_all = os.path.join(d, 'predicted_lp.csv'), os.path.join(d, 'predicted_lp_all.csv')
            if not (os.path.exists(p_test) and os.path.exists(p_all)):
                continue
            runs_test.append(pd.read_csv(p_test).set_index('PatientID')['lp'])
            runs_all.append(pd.read_csv(p_all).set_index('PatientID')['lp'])
            folds.append(fold)
    if not runs_test:
        return None
    test_ids = clin.index[clin['is_test']]
    train_ids = clin.index[~clin['is_test']]
    test_mat = pd.concat(runs_test, axis=1).loc[test_ids]
    all_mat = pd.concat(runs_all, axis=1)
    # The two inference passes agree to <= 4e-4 (GPU non-determinism in some 3D runs).
    assert np.allclose(all_mat.loc[test_ids].to_numpy(), test_mat.to_numpy(), atol=1e-3), 'test predictions differ'
    # Each network's output has its own offset and scale; standardise every run with the mean/SD of its
    # predictions on the training patients (no test information) so that runs, out-of-fold and test
    # predictions are on a common scale before averaging.
    mu = all_mat.loc[train_ids].mean(axis=0)
    sd = all_mat.loc[train_ids].std(axis=0)
    z_mat = (all_mat - mu) / sd
    folds = np.array(folds)

    def oof_from(mat):
        o = pd.Series(np.nan, index=train_ids)
        for k in (1, 2, 3):
            ids = train_ids[clin.loc[train_ids, cv_col] == f'CV{k}']
            cols = np.where(folds == k)[0]
            if len(cols):
                o.loc[ids] = mat.loc[ids].iloc[:, cols].mean(axis=1)
        return o

    return {'test': test_mat.mean(axis=1), 'test_runs': test_mat, 'oof': oof_from(all_mat),
            'insample': all_mat.loc[train_ids].mean(axis=1),
            'test_z': z_mat.loc[test_ids].mean(axis=1), 'oof_z': oof_from(z_mat),
            'n_runs': len(runs_test), 'folds': folds, 'all_mat': all_mat}


def cindex(time, event, risk):
    """Harrell's C (lifelines), risk = higher means earlier event."""
    return concordance_index(np.asarray(time), -np.asarray(risk), np.asarray(event))


def comparable_pairs(time, event):
    """Index arrays (i, j) of comparable pairs under lifelines' rules: i had the event and either
    T_i < T_j, or T_i == T_j and j is censored."""
    t, e = np.asarray(time, float), np.asarray(event, int)
    ti, tj = t[:, None], t[None, :]
    ok = (e[:, None] == 1) & ((ti < tj) | ((ti == tj) & (e[None, :] == 0)))
    np.fill_diagonal(ok, False)
    return np.nonzero(ok)


def cindex_pairs(I, J, risk):
    """Vectorised Harrell's C from precomputed comparable pairs; risk may be (n,) or (n, m)."""
    r = np.asarray(risk, float)
    ri, rj = r[I], r[J]
    return ((ri > rj) + 0.5 * (ri == rj)).mean(axis=0)


def bootstrap_cindex(time, event, risks, n_boot=2000, seed=2026):
    """Paired bootstrap over test patients. risks: dict name -> array. Returns DataFrame of bootstrap C."""
    rng = np.random.default_rng(seed)
    t, e = np.asarray(time, float), np.asarray(event, int)
    names = list(risks)
    R = np.column_stack([np.asarray(risks[k], float) for k in names])
    n = len(t)
    out = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if e[idx].sum() == 0:
            continue
        I, J = comparable_pairs(t[idx], e[idx])
        if len(I) == 0:
            continue
        out[b] = cindex_pairs(I, J, R[idx])
    return pd.DataFrame(out, columns=names).dropna()


def ci_str(x, lo, hi, d=2):
    return f'{x:.{d}f} [{lo:.{d}f}-{hi:.{d}f}]'


def delta_summary(boot, a, b):
    """Paired bootstrap difference C_a - C_b: 95% percentile CI and two-sided bootstrap p-value."""
    diff = boot[a] - boot[b]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return diff.mean(), lo, hi, min(p, 1.0)


def fit_cox(df, covs, time_col, event_col, penalizer=0.0):
    """Unpenalised Cox fit. lifelines' Newton-Raphson can diverge when a covariate has extreme values
    (e.g. one patient with a DL prediction ~10 SD from the mean), so several step sizes are tried and the
    solution with the highest partial log-likelihood is kept."""
    best = None
    for step in (None, 0.5, 0.2, 0.05):
        cph = CoxPHFitter(penalizer=penalizer)
        try:
            opts = {} if step is None else {'step_size': step, 'max_steps': 2000}
            cph.fit(df[covs + [time_col, event_col]], duration_col=time_col, event_col=event_col, fit_options=opts)
        except Exception:
            continue
        if best is None or cph.log_likelihood_ > best.log_likelihood_ + 1e-6:
            best = cph
    return best


def breslow_baseline(time, event, lp, t_eval):
    """Breslow cumulative baseline hazard at t_eval for a fixed linear predictor (offset model)."""
    t, e, w = np.asarray(time, float), np.asarray(event, int), np.exp(np.asarray(lp, float))
    H = 0.0
    for ti in np.unique(t[(e == 1) & (t <= t_eval)]):
        d = ((t == ti) & (e == 1)).sum()
        H += d / w[t >= ti].sum()
    return H


def km_risk_at(time, event, t_eval):
    kmf = KaplanMeierFitter().fit(time, event)
    s = float(kmf.survival_function_at_times(t_eval).iloc[0])
    # Greenwood variance of S(t)
    tab = kmf.event_table
    tab = tab[tab.index <= t_eval]
    var_term = (tab['observed'] / (tab['at_risk'] * (tab['at_risk'] - tab['observed']))).replace([np.inf], np.nan).sum()
    var = s * s * var_term
    return 1 - s, var


def gnd_test(time, event, pred_risk, t_eval=24.0, n_groups=4):
    """Greenwood-Nam-D'Agostino calibration test (Demler et al., Stat Med 2015) at t_eval.
    Returns chi2, df, p, and the per-group table."""
    from scipy.stats import chi2
    t, e, p = np.asarray(time, float), np.asarray(event, int), np.asarray(pred_risk, float)
    groups = pd.qcut(pd.Series(p).rank(method='first'), n_groups, labels=False).to_numpy()
    rows = []
    for g in range(n_groups):
        m = groups == g
        obs, var = km_risk_at(t[m], e[m], t_eval)
        rows.append({'group': g + 1, 'n': int(m.sum()), 'events': int(e[m].sum()),
                     'pred_mean': p[m].mean(), 'obs_km': obs, 'var_km': var})
    tab = pd.DataFrame(rows)
    valid = tab['var_km'] > 0
    stat = (((tab['obs_km'] - tab['pred_mean']) ** 2) / tab['var_km'])[valid].sum()
    dof = int(valid.sum()) - 1
    pval = 1 - chi2.cdf(stat, dof) if dof > 0 else np.nan
    return stat, dof, pval, tab

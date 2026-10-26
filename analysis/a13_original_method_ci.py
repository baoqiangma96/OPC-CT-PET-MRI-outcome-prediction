# -*- coding: utf-8 -*-
"""
C-index 95% CIs with exactly the procedure of the submitted manuscript (conf_cindex in
MRI_umcg_new/clinical_model_km_cali_HNpredictor.py: 1000 bootstrap samples, numpy RandomState(42),
indices = randint(0, n - 1, n), lifelines concordance_index, samples without events skipped).
Used so that unchanged rows of Table 1 / Table A.3 / Table A.4 keep their published values and only the
corrected hybrid models get new numbers. Also checks that the published values are reproduced.
"""
import os
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from common import (ENDPOINTS, MAIN_CONFIGS, EXTRA_CONFIGS, RESULTS, load_clinical, load_predictions, fit_cox)


def conf_cindex_original(pred, t, e, bootstrap=1000, seed=42, confint=0.95):
    rng = np.random.RandomState(seed)
    pred, t, e = np.asarray(pred), np.asarray(t), np.asarray(e)
    scores = []
    for _ in range(bootstrap):
        idx = rng.randint(0, len(pred) - 1, len(pred))
        if len(np.unique(t[idx])) < 2:
            continue
        try:
            scores.append(concordance_index(t[idx], pred[idx], e[idx]))
        except Exception:
            continue
    s = np.sort(np.array(scores))
    lo = s[int((1 - confint) / 2 * len(s))]
    hi = s[int((1 - (1 - confint) / 2) * len(s))]
    return lo, concordance_index(t, pred, e), hi


def main():
    clin = load_clinical()
    test, train = clin[clin['is_test']], clin[~clin['is_test']]
    rows = []
    configs = {**MAIN_CONFIGS, **{k: v for k, v in EXTRA_CONFIGS.items() if '10mm' in k}}
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        hnc = f'hnc_lp_{ep}'
        t, e = test[tm].to_numpy(), test[ev].to_numpy()
        risks = {'HNC-PREDICTOR': test[hnc].to_numpy()}
        for name, (root, prefix) in configs.items():
            p = load_predictions(root, prefix, ep, clin)
            risks[name] = p['test'].loc[test.index].to_numpy()
            tr = train.assign(dl=p['oof_z'].loc[train.index])
            b = fit_cox(tr, [hnc, 'dl'], tm, ev).params_
            risks[f'HNC + {name}'] = b[hnc] * test[hnc].to_numpy() + b['dl'] * p['test_z'].loc[test.index].to_numpy()
        for name, r in risks.items():
            lo, c, hi = conf_cindex_original(-r, t, e)
            rows.append({'endpoint': ep, 'model': name, 'C': c, 'lo': lo, 'hi': hi,
                         'fmt': f'{c:.2f} [{lo:.2f}-{hi:.2f}]'})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'a13_cindex_original_ci_method.csv'), index=False)
    wide = df.pivot(index='model', columns='endpoint', values='fmt')[['LC', 'RC', 'OS']]
    pd.set_option('display.width', 200)
    print(wide.to_string())


if __name__ == '__main__':
    main()

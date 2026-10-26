# -*- coding: utf-8 -*-
"""
Model selection and multiplicity (Reviewer 2 major comment; Reviewer 3 #8 and #9; Reviewer 1 #1).

1. Validation-based selection: for every model, the cross-validation C-index computed on the held-out fold
   of each of the three folds (out-of-fold predictions, averaged over the 3 seeds; mean over folds). The model
   with the highest validation C-index is selected per endpoint and per family, and its test C-index is
   reported. This is the selection a pre-specified, test-blind procedure would have made.
   Note: the validation folds were also used for early stopping, so validation C-indices are optimistic in
   absolute terms; they are used here only to rank models.
2. Permutation test with family-wise correction: test-set outcomes are permuted against the matrix of all
   models' test predictions (10,000 permutations, preserving the correlation between models). The null
   distribution of the maximum C-index over all candidate models gives a selection-adjusted p-value for the
   best observed model; per-model permutation p-values are reported as well.
3. Per-run variability of the test C-index (9 runs per model).
"""
import os
import numpy as np
import pandas as pd
from common import (ENDPOINTS, MAIN_CONFIGS, EXTRA_CONFIGS, RESULTS, load_clinical, load_predictions,
                    cindex, comparable_pairs, cindex_pairs)

N_PERM = 10000


def main():
    clin = load_clinical()
    test = clin[clin['is_test']]
    train = clin[~clin['is_test']]
    sel_rows, perm_rows, run_rows = [], [], []
    configs = {**MAIN_CONFIGS, **{k: v for k, v in EXTRA_CONFIGS.items() if 'tot' not in k}}
    for ep in ('LC', 'RC', 'OS'):
        ev, tm, _ = ENDPOINTS[ep]
        cv_col = f'CVgroup_{ev}_t1t2'
        preds, val_c = {}, {}
        for name, (root, prefix) in configs.items():
            p = load_predictions(root, prefix, ep, clin)
            if p is None or p['n_runs'] < 9:
                continue
            preds[name] = p['test'].loc[test.index].to_numpy()
            per_fold = []
            for k in (1, 2, 3):
                ids = train.index[train[cv_col] == f'CV{k}']
                per_fold.append(cindex(train.loc[ids, tm], train.loc[ids, ev], p['oof'].loc[ids]))
            val_c[name] = (np.mean(per_fold), per_fold)
            runs = [cindex(test[tm], test[ev], p['test_runs'].iloc[:, j]) for j in range(p['test_runs'].shape[1])]
            run_rows.append({'endpoint': ep, 'model': name, 'test_C_ensemble': cindex(test[tm], test[ev], preds[name]),
                             'run_C_min': np.min(runs), 'run_C_median': np.median(runs), 'run_C_max': np.max(runs),
                             'n_runs_C_lt_0.5': int(np.sum(np.array(runs) < 0.5))})
        for name in preds:
            sel_rows.append({'endpoint': ep, 'model': name, 'val_C_mean': val_c[name][0],
                             'val_C_fold1': val_c[name][1][0], 'val_C_fold2': val_c[name][1][1],
                             'val_C_fold3': val_c[name][1][2],
                             'test_C': cindex(test[tm], test[ev], preds[name])})

        # permutation null for max C over the main candidate set (Table 1 models)
        names = [n for n in preds if n in MAIN_CONFIGS]
        P = np.column_stack([preds[n] for n in names])
        I, J = comparable_pairs(test[tm].to_numpy(), test[ev].to_numpy())
        obs = cindex_pairs(I, J, P)
        rng = np.random.default_rng(7)
        max_null = np.empty(N_PERM)
        exceed = np.zeros(len(names))
        for b in range(N_PERM):
            perm = rng.permutation(len(P))
            c = cindex_pairs(I, J, P[perm])
            max_null[b] = c.max()
            exceed += c >= obs
        best = int(np.argmax(obs))
        perm_rows.append({'endpoint': ep, 'n_models': len(names), 'best_model': names[best], 'best_C': obs[best],
                          'familywise_p_best': (np.sum(max_null >= obs[best]) + 1) / (N_PERM + 1),
                          'null_max_C_median': np.median(max_null), 'null_max_C_95pct': np.percentile(max_null, 95)})
        for n, o, e in zip(names, obs, exceed):
            perm_rows.append({'endpoint': ep, 'model': n, 'C': o, 'unadjusted_perm_p': (e + 1) / (N_PERM + 1),
                              'familywise_p': (np.sum(max_null >= o) + 1) / (N_PERM + 1)})
        print(f'{ep}: permutation done')

    sel = pd.DataFrame(sel_rows)
    sel.to_csv(os.path.join(RESULTS, 'a02_validation_vs_test.csv'), index=False)
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(os.path.join(RESULTS, 'a02_permutation_familywise.csv'), index=False)
    pd.DataFrame(run_rows).to_csv(os.path.join(RESULTS, 'a02_per_run_variability.csv'), index=False)

    # selection summary: best by validation, overall and per family
    fam = lambda m: ('2D MRI' if m.startswith('2D') else '3D MRI' if m[3:5] in ('T1', 'T2') else '3D ' + m.split()[1].split('_')[0])
    sel['family'] = sel['model'].map(fam)
    summ = []
    for ep, g in sel[sel['model'].isin(MAIN_CONFIGS)].groupby('endpoint'):
        b = g.loc[g['val_C_mean'].idxmax()]
        bt = g.loc[g['test_C'].idxmax()]
        summ.append({'endpoint': ep, 'family': 'all', 'selected_by_validation': b['model'], 'val_C': b['val_C_mean'],
                     'its_test_C': b['test_C'], 'best_by_test': bt['model'], 'best_test_C': bt['test_C'],
                     'spearman_val_vs_test': g['val_C_mean'].corr(g['test_C'], method='spearman')})
        for f, gf in g.groupby('family'):
            b = gf.loc[gf['val_C_mean'].idxmax()]
            summ.append({'endpoint': ep, 'family': f, 'selected_by_validation': b['model'], 'val_C': b['val_C_mean'],
                         'its_test_C': b['test_C'], 'family_mean_test_C': gf['test_C'].mean(),
                         'family_mean_val_C': gf['val_C_mean'].mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RESULTS, 'a02_selection_summary.csv'), index=False)
    pd.set_option('display.width', 250)
    print(summ.round(3).to_string(index=False))
    print(perm[perm['best_model'].notna()].round(4).to_string(index=False))
    print(pd.DataFrame(run_rows).round(2).to_string(index=False))


if __name__ == '__main__':
    main()

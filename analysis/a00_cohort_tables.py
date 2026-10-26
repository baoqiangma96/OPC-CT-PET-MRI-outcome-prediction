# -*- coding: utf-8 -*-
"""
Tables A.1 and A.2: patient characteristics and outcomes of the training (n = 186) and test set (n = 80).

Continuous variables: mean (SD), z-test; categorical: n (%), chi-square test; outcomes: number of events and the
median time to event among patients with the event. Output: results/a00_table_a1.csv, results/a00_table_a2.csv.
"""
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.weightstats as st

from common import HNC_CSV, RESULTS

CATEGORICAL = {   # variable -> (column, {level label: codes})
    'Gender': ('GESLACHT_codes', {'male': [1], 'female': [0]}),
    'WHO performance status': ('WHO_SCORE_codes', {'0': [0], '1': [1], '2': [2], '3': [3]}),
    'HPV status': ('P16_codes_newexperi', {'positive': [0], 'unknown': [1], 'negative': [2]}),
    'T-stage': ('TSTAD_codes', {'T1': [0], 'T2': [1], 'T3': [2], 'T4': [3]}),
    'N-stage': ('NSTAD_codes', {'N0': [0], 'N1': [1], 'N2': [2, 3, 4], 'N3': [5]}),
    'AJCC 8th edition': ('AJCC8th', {'I': [0], 'II': [1], 'III': [2], 'IVa': [3], 'IVb': [4]}),
}
# The original analysis tested WHO PS 0/1/2 only (one patient with PS 3).
CHI2_LEVELS = {'WHO performance status': ['0', '1', '2']}
OUTCOMES = {'LC': ('LR_code', 'TIME_LR'), 'RC': ('RR_code', 'TIME_RR'), 'LRC': ('LRR_code', 'TIME_LRR'),
            'DMFS': ('MET_code', 'TIME_MET'), 'OS': ('OS_code', 'TIME_OS')}


def main():
    data = pd.read_csv(HNC_CSV)
    sets = {'train': data[data['CVgroup_event_t1t2'] == 'train'], 'test': data[data['CVgroup_event_t1t2'] == 'test']}
    rows = []
    for label, col in (('Age (years)', 'AGE'), ('Pack-years', 'PACK_YEARS')):
        p = st.ztest(sets['train'][col], sets['test'][col])[1]
        rows.append({'variable': label, 'level': 'mean (SD)', 'p': p,
                     **{s: f'{d[col].mean():.1f} ({d[col].std():.1f})' for s, d in sets.items()}})
    for label, (col, levels) in CATEGORICAL.items():
        counts = {s: {lvl: int(d[col].isin(codes).sum()) for lvl, codes in levels.items()} for s, d in sets.items()}
        used = CHI2_LEVELS.get(label, list(levels))
        p = stats.chi2_contingency(np.array([[counts[s][lvl] for lvl in used] for s in sets]))[1]
        for i, lvl in enumerate(levels):
            rows.append({'variable': label, 'level': lvl, 'p': p if i == 0 else np.nan,
                         **{s: f'{counts[s][lvl]} ({100 * counts[s][lvl] / len(d):.1f}%)' for s, d in sets.items()}})
    t1 = pd.DataFrame(rows)[['variable', 'level', 'train', 'test', 'p']]
    t2 = pd.DataFrame([{'endpoint': ep, **{f'{s}_events': int(d[ev].sum()) for s, d in sets.items()},
                        **{f'{s}_median_time_to_event': d.loc[d[ev] == 1, tm].median() for s, d in sets.items()}}
                       for ep, (ev, tm) in OUTCOMES.items()] +
                      [{'endpoint': 'follow-up (median TIME_OS)',
                        **{f'{s}_median_time_to_event': d['TIME_OS'].median() for s, d in sets.items()}}])
    t1.to_csv(os.path.join(RESULTS, 'a00_table_a1.csv'), index=False)
    t2.to_csv(os.path.join(RESULTS, 'a00_table_a2.csv'), index=False)
    pd.set_option('display.width', 200)
    print(t1.to_string(index=False))
    print(t2.to_string(index=False))


if __name__ == '__main__':
    main()

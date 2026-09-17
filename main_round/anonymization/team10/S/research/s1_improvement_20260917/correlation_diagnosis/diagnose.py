"""Read-only diagnosis; writes aggregate statistics only, never synthesized rows."""
from pathlib import Path
from itertools import combinations
import json
import hashlib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[3]
S1 = TEAM / 'S' / 'S1'
DATA = TEAM / 'participant_data'
FEATURES = ['age', 'sex', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG', 'smoking']
NUMERIC = [x for x in FEATURES if x not in ['sex', 'smoking']]
PAIRS = list(combinations(FEATURES, 2))
cfg = json.loads((S1 / 'config' / 'aim_main_team10_S1.json').read_text(encoding='utf-8'))
schema = json.loads((DATA / 'schema.json').read_text(encoding='utf-8'))
paths = {'B': DATA / 'B.csv', 'C': S1 / 'C.csv', 'A_bg': DATA / 'A_bg.csv'}
frames = {name: pd.read_csv(path) for name, path in paths.items()}

def rare_mask(frame):
    spec = schema['rare_detector']
    diff = frame[spec['columns']].to_numpy(dtype=float) - np.asarray(spec['mu'])
    dist = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', diff, np.linalg.pinv(spec['sigma']), diff), 0, None))
    return dist >= spec['band']['lo']

def corr_summary(corrs):
    pairs = []
    for x, y in PAIRS:
        r = {'pair': [x, y], **{name: float(corr.loc[x, y]) for name, corr in corrs.items()}}
        r.update(abs_C_minus_B=abs(r['C']-r['B']), abs_A_bg_minus_B=abs(r['A_bg']-r['B']), sign_reversal_C_B=bool(r['C'] * r['B'] < 0))
        pairs.append(r)
    pairs.sort(key=lambda r: r['abs_C_minus_B'], reverse=True)
    d = np.array([r['abs_C_minus_B'] for r in pairs])
    public_d = np.array([r['abs_A_bg_minus_B'] for r in pairs])
    return {'pairs': pairs, 'mean_abs_C_B': float(d.mean()), 'q90_abs_C_B': float(np.quantile(d, .9)), 'mean_abs_A_bg_B': float(public_d.mean()), 'q90_abs_A_bg_B': float(np.quantile(public_d, .9)), 'max_abs_A_bg_B': float(public_d.max()), 'sign_reversal_count_C_B': sum(r['sign_reversal_C_B'] for r in pairs)}

raw = corr_summary({name: frame[FEATURES].corr(method='spearman') for name, frame in frames.items()})
binned_frames = {}
for name, frame in frames.items():
    binned = frame[FEATURES].copy()
    for col, edges in cfg['numeric_bin_edges'].items():
        binned[col] = np.searchsorted(np.asarray(edges[1:]), frame[col].to_numpy(), side='left')
    binned_frames[name] = binned
binned = corr_summary({name: frame.corr(method='spearman') for name, frame in binned_frames.items()})
nonrare_frames = {name: frame.loc[~rare_mask(frame)] for name, frame in frames.items()}
nonrare = corr_summary({name: frame[FEATURES].corr(method='spearman') for name, frame in nonrare_frames.items()})

categories = {
    'numeric_numeric': lambda x,y: x in NUMERIC and y in NUMERIC,
    'sex_numeric': lambda x,y: 'sex' in (x,y) and any(c in NUMERIC for c in [x,y]),
    'smoking_numeric': lambda x,y: 'smoking' in (x,y) and any(c in NUMERIC for c in [x,y]),
    'sex_smoking': lambda x,y: set([x,y]) == set(['sex', 'smoking']),
}
by_category = {}
for cat, choose in categories.items():
    rows = [r for r in raw['pairs'] if choose(*r['pair'])]
    by_category[cat] = {'n_pairs':len(rows), 'mean_abs_C_B':float(np.mean([r['abs_C_minus_B'] for r in rows])), 'mean_abs_A_bg_B':float(np.mean([r['abs_A_bg_minus_B'] for r in rows])), 'mean_abs_C':float(np.mean([abs(r['C']) for r in rows])), 'mean_abs_B':float(np.mean([abs(r['B']) for r in rows]))}

row_index_corr = {}
for name, frame in frames.items():
    z = frame[FEATURES].copy()
    z['row_index'] = np.arange(len(z))
    row_index_corr[name] = {col:float(v) for col,v in z.corr(method='spearman')['row_index'].drop('row_index').items()}

workload = {frozenset(q['columns']): {'weight':q['weight'],'fixed':q.get('fixed',False)} for q in cfg['workload']['queries']}
for group in cfg['workload']['pairwise_groups']:
    for pair in combinations(group['columns'],2):
        workload[frozenset(pair)] = {'weight':group['weight'],'fixed':False}
workload_status = [{'pair':list(pair), 'status':workload.get(frozenset(pair),'absent')} for pair in PAIRS]

subgroup_correlations = {}
for grouper in ['sex','smoking']:
    subgroup_correlations[grouper] = {}
    for value in [0,1]:
        feature_cols = [f for f in FEATURES if f != grouper]
        sub = {}
        for name, frame in frames.items():
            ff = frame.loc[frame[grouper] == value]
            cc = ff[feature_cols].corr(method='spearman')
            sub[name] = {'n':len(ff),'numeric_pair_mean_abs':float(np.mean([abs(cc.loc[x,y]) for x,y in combinations(NUMERIC,2)])), 'pairwise':{f'{x}:{y}':float(cc.loc[x,y]) for x,y in combinations(NUMERIC,2)}}
        subgroup_correlations[grouper][str(value)] = sub

bin_counts = {name:{col:frame[col].value_counts().sort_index().to_dict() for col in NUMERIC} for name,frame in binned_frames.items()}
report = {'description':'Read-only aggregate diagnostic. Uses private B for local evaluation, not a DP release; no model generation or submission files created.', 'input_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()}, 'rows':{name:len(frame) for name,frame in frames.items()}, 'raw_spearman':raw, 'binned_spearman':binned, 'nonrare_spearman':nonrare, 'nonrare_rows':{name:len(frame) for name,frame in nonrare_frames.items()}, 'pair_categories':by_category, 'row_index_spearman':row_index_corr, 'workload_status':workload_status, 'subgroup_correlations':subgroup_correlations, 'bin_counts':bin_counts}
(HERE/'aggregate_diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':report['rows'], 'raw':{k:v for k,v in raw.items() if k!='pairs'}, 'top10':raw['pairs'][:10], 'bin':{k:v for k,v in binned.items() if k!='pairs'}, 'nonrare':{k:v for k,v in nonrare.items() if k!='pairs'}, 'categories':by_category,'row_index':row_index_corr,'absent_workload':[r['pair'] for r in workload_status if r['status']=='absent']},ensure_ascii=False,indent=2))

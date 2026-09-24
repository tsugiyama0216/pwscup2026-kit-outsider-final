"""Aggregate-only audit of frozen, authorized competition releases.

Run from any working directory. All outputs stay alongside this script.
Never reads token.txt or ZIPs. See PROTOCOL.md before interpreting results.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path

for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var]='1'

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
R_DIR = HERE.parent
DATA = R_DIR.parent / 'participant_data'
SEED = 2026091807
BOOTSTRAPS = 400
NUM = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
CAT = ['sex', 'prefecture', 'smoking']
COLS = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG', 'time', 'onset', 'death']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [max(0., center-radius), min(1., center+radius)]


def calibration_threshold(scores, alpha=.01):
    """Strict exceedance; n+1 rule is conservative under exchangeability."""
    scores = np.asarray(scores, float)
    if not len(scores):
        return float('inf')
    rank = math.ceil((1-alpha)*(len(scores)+1))
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')


def bootstrap_auc(members, controls, seed=SEED):
    if min(len(members), len(controls)) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(len(members)), np.zeros(len(controls))]
    vals = []
    for _ in range(BOOTSTRAPS):
        sm = rng.choice(members, len(members), replace=True)
        sc = rng.choice(controls, len(controls), replace=True)
        vals.append(roc_auc_score(labels, np.r_[sm, sc]))
    return np.quantile(vals, [.025, .975]).tolist()


def mia_metrics(members, controls, calibration):
    m, c, cal = [np.asarray(x, float) for x in (members, controls, calibration)]
    threshold = calibration_threshold(cal)
    if not len(m) or not len(c):
        return {'n_members':len(m), 'n_controls':len(c), 'n_calibration':len(cal), 'unavailable':True}
    tp = int((m > threshold).sum())
    fp = int((c > threshold).sum())
    return {
        'n_members':len(m), 'n_controls':len(c), 'n_calibration':len(cal),
        'auc':roc_auc_score(np.r_[np.ones(len(m)),np.zeros(len(c))], np.r_[m,c]),
        'auc_bootstrap_95':bootstrap_auc(m,c),
        'threshold':threshold, 'threshold_infinite':not np.isfinite(threshold),
        'calibration_fpr':float(np.mean(cal > threshold)) if len(cal) else None,
        'tpr':tp/len(m), 'tpr_wilson_95':wilson(tp,len(m)),
        'fpr':fp/len(c), 'fpr_wilson_95':wilson(fp,len(c)),
        'advantage_tpr_minus_fpr':tp/len(m)-fp/len(c),
    }


def paired_bootstrap_mean(values, seed=SEED):
    values = np.asarray(values, float)
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(seed)
    out = [rng.choice(values,len(values),replace=True).mean() for _ in range(BOOTSTRAPS)]
    return np.quantile(out,[.025,.975]).tolist()


def aia_metrics(truth, prediction, baseline):
    y, p, q = [np.asarray(x, float) for x in (truth,prediction,baseline)]
    if not len(y):
        return {'n':0}
    hit = np.abs(y-p) <= .5
    baseline_hit = np.abs(y-q) <= .5
    ten_hit = np.abs(y-10.) <= .5
    difference = hit.astype(float)-baseline_hit.astype(float)
    return {
        'n':len(y), 'accuracy_half_year':hit.mean(), 'accuracy_wilson_95':wilson(hit.sum(),len(y)),
        'mae_years':np.abs(y-p).mean(), 'public_baseline_accuracy':baseline_hit.mean(),
        'public_baseline_mae_years':np.abs(y-q).mean(),
        'accuracy_gain_over_public':difference.mean(),
        'gain_bootstrap_95':paired_bootstrap_mean(difference),
        'always_ten_accuracy':ten_hit.mean(),
    }


def rare_mask(frame, schema):
    det=schema['rare_detector']
    x=frame[det['columns']].to_numpy(float)-np.asarray(det['mu'])
    r=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',x,np.linalg.pinv(det['sigma']),x)))
    return r >= det['band']['lo']


def prepare(frame, columns):
    out=frame[columns].copy()
    for col in ('TG','ALT'):
        if col in out:
            out[col]=np.log(out[col].to_numpy(float))
    for col in CAT:
        if col in out:
            out[col]=out[col].astype(str)
    return out


def transformer(reference, columns):
    cats=[c for c in columns if c in CAT]
    nums=[c for c in columns if c not in CAT]
    proc=ColumnTransformer([
        ('numeric', StandardScaler(), nums),
        ('categories',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats),
    ],sparse_threshold=0.)
    proc.fit(prepare(reference,columns))
    return proc


def transform(proc,frame,columns):
    return np.asarray(proc.transform(prepare(frame,columns)),float)


def row_hashes(frame):
    return pd.util.hash_pandas_object(frame[COLS],index=False).to_numpy()


def feature_summary(frame, schema):
    output={'n':len(frame),'detected_rare':int(rare_mask(frame,schema).sum()),'duplicate_full_rows':int(frame[COLS].duplicated().sum())}
    output['numeric']={}
    for c in NUM+['time']:
        x=frame[c].to_numpy(float)
        output['numeric'][c]={'mean':np.mean(x),'sd':np.std(x,ddof=1),'quantiles_01_05_50_95_99':np.quantile(x,[.01,.05,.5,.95,.99])}
    output['event_counts']={k:int(frame[k].sum()) for k in ['onset','death']}
    output['event_counts']['censored']=int(((frame.onset==0)&(frame.death==0)).sum())
    output['prefecture_histogram']={str(k):int(v) for k,v in frame.prefecture.value_counts().sort_index().items()}
    output['sex_smoking_histogram']={str(k):int(v) for k,v in frame.groupby(['sex','smoking']).size().items()}
    return output


def distribution_difference(target,synth):
    if min(len(target),len(synth))<2:
        return {'n_target':len(target),'n_synthetic':len(synth),'unavailable':True}
    cols={}
    for c in NUM+['time']:
        a,b=target[c].to_numpy(float),synth[c].to_numpy(float)
        scale=max(float(np.std(a,ddof=1)),1e-12)
        cols[c]={'ks_distance':ks_2samp(a,b).statistic,'wasserstein_in_target_sd':wasserstein_distance(a,b)/scale,'mean_shift_in_target_sd':(b.mean()-a.mean())/scale}
    ca=target[NUM].corr(method='spearman').to_numpy()
    cb=synth[NUM].corr(method='spearman').to_numpy()
    off=np.triu_indices(len(NUM),1)
    return {'n_target':len(target),'n_synthetic':len(synth),'numeric':cols,'spearman_mean_absolute_difference':np.nanmean(np.abs(ca[off]-cb[off])),'onset_rate_difference':synth.onset.mean()-target.onset.mean(),'death_rate_difference':synth.death.mean()-target.death.mean()}


def strata(frame, rare, true_rare=None):
    masks={'overall':np.ones(len(frame),bool),'detected_rare':np.asarray(rare,bool),'detected_nonrare':~np.asarray(rare,bool),
           'censored':((frame.onset==0)&(frame.death==0)).to_numpy(),'onset':frame.onset.to_numpy()==1,'death':frame.death.to_numpy()==1,
           'any_event':((frame.onset==1)|(frame.death==1)).to_numpy()}
    masks['detected_rare_event']=masks['detected_rare']&masks['any_event']
    if true_rare is not None:
        masks['true_rare']=np.asarray(true_rare,bool)
        masks['true_rare_event']=masks['true_rare']&masks['any_event']
    return masks


def equal_columns(a,b):
    if len(a)!=len(b):
        return []
    return [c for c in COLS if np.array_equal(a[c].to_numpy(),b[c].to_numpy())]


def main():
    schema=json.loads((DATA/'schema.json').read_text(encoding='utf-8'))
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    b=read(DATA/'B.csv')
    bs=read(DATA/'B_self.csv')
    bg=read(DATA/'A_bg.csv')
    if b.record_id.duplicated().any() or bs.record_id.duplicated().any():
        raise ValueError('Nonunique member ID')
    joined=b[['record_id']].merge(bs[['record_id','is_rare']],on='record_id',how='left',validate='one_to_one')
    if joined.is_rare.isna().any():
        raise ValueError('Missing true rare labels')
    truth=joined.is_rare.to_numpy(bool)
    id_overlap=bg.record_id.isin(b.record_id).to_numpy()
    full_overlap=np.isin(row_hashes(bg),row_hashes(b))
    # record_id namespaces recur across datasets: an ID match is not a person match.
    exclude=full_overlap
    public=bg.loc[~exclude].reset_index(drop=True)
    if len(public)<400:
        raise ValueError('Insufficient public controls after overlap removal')
    rng=np.random.default_rng(SEED)
    order=rng.permutation(len(public))
    nref=len(public)//2
    ncal=(len(public)-nref)//2
    reference=public.iloc[order[:nref]].reset_index(drop=True)
    calibration=public.iloc[order[nref:nref+ncal]].reset_index(drop=True)
    controls=public.iloc[order[nref+ncal:]].reset_index(drop=True)
    datasets={'members':b,'calibration':calibration,'controls':controls}
    rare={name:rare_mask(frame,schema) for name,frame in datasets.items()}
    releases={f'R{i}':read(R_DIR/f'R{i}/C.csv') for i in range(1,10)}
    report={
        'title':'R1–R9 fixed-release privacy audit', 'date':'2026-09-18','protocol':'PROTOCOL.md',
        'not_official':True,'seed':SEED,'bootstrap_resamples':BOOTSTRAPS,
        'methods':{'aia_qi_columns':schema['aia_qi_columns'],'mia_full_columns':COLS,'continuous_transform':'log(TG), log(ALT); other numeric values unchanged; StandardScaler fit on public reference','categorical_transform':'one-hot sex, prefecture, smoking with unseen public-reference categories ignored','mia_classifier':'HistGradientBoostingClassifier max_iter100, max_leaf_nodes15, min_samples_leaf20, l2_regularization1, learning_rate.05, equal total class weights','aia_regression':'ExtraTrees 200 trees, min leaf10, max_features.8; KNN k5 distance weights','uncertainty':'400 resamples conditional on fixed release, model and calibrated threshold; Wilson95 for event proportions'},
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'hashes':{str(p.relative_to(R_DIR.parent)):digest(p) for p in [DATA/'B.csv',DATA/'B_self.csv',DATA/'A_bg.csv',DATA/'schema.json']+[R_DIR/f'R{i}/C.csv' for i in range(1,10)]},
        'input_counts':{'B':len(b),'B_true_rare':int(truth.sum()),'B_detected_rare':int(rare['members'].sum()),'B_true_and_detected_rare':int((truth&rare['members']).sum()),'A_bg':len(bg),'shared_record_ids_NOT_person_matches':int(id_overlap.sum()),'excluded_id_overlap':0,'excluded_exact_full_overlap':int(full_overlap.sum()),'public_reference':len(reference),'public_calibration':len(calibration),'public_controls':len(controls),'calibration_detected_rare':int(rare['calibration'].sum()),'control_detected_rare':int(rare['controls'].sum())},
        'limitations':[
            'Public A_bg controls can differ from B by cohort distribution and are not a true exchangeable main-round private holdout.',
            'AUC is B-versus-public discriminability; it does not identify membership leakage separately from population shift.',
            'Bootstrap intervals hold release, fitted model and calibration threshold fixed and omit training, calibration and cohort-shift uncertainty.',
            'True rare B versus detected rare controls uses different labels and is diagnostic only.',
            'Repeated version tuning on utility references and multiple audit attacks are not corrected by these confidence intervals.',
            'Saved releases differ in multiple factors. Comparisons do not estimate causal effects of isolated changes.',
            'Zero observed attacks, no exact matches, and high utility never constitute DP or privacy certification.',
        ],
        'source_distributions':{'members':feature_summary(b,schema),'public_reference':feature_summary(reference,schema),'public_controls':feature_summary(controls,schema)},
        'public_vs_members_distribution':distribution_difference(b,controls),
        'versions':{},'lineage':{}
    }
    lineage={'R2':'R1','R3':'R2','R4':'R3','R6':'R4','R7':'R6','R8':'R7','R9':'R8'}
    for v,parent in lineage.items():
        a,c=releases[parent],releases[v]
        report['lineage'][v]={'parent':parent,'same_index_exact_columns':equal_columns(a,c),'same_full_row_hash_set':bool(set(row_hashes(a))==set(row_hashes(c))),'n_same_index_full_rows':int(np.all(a[COLS].to_numpy()==c[COLS].to_numpy(),axis=1).sum())}
    report['lineage']['R5']={'parent':None,'note':'Independent CART synthesis; no saved R1–R4 CSV used.'}
    all_proc=transformer(reference,COLS)
    qi=schema['aia_qi_columns']
    qi_proc=transformer(reference,qi)
    matrices={name:transform(all_proc,f,COLS) for name,f in datasets.items()}
    qmatrices={name:transform(qi_proc,f,qi) for name,f in datasets.items()}
    refmat=transform(all_proc,reference,COLS)
    refqi=transform(qi_proc,reference,qi)
    ref_nn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(refmat)
    public_distances={name:ref_nn.kneighbors(x,return_distance=True)[0][:,0] for name,x in matrices.items()}
    ref_ett=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=10,max_features=.8,random_state=SEED,n_jobs=2).fit(refqi,reference.time)
    ref_knn=KNeighborsRegressor(n_neighbors=5,weights='distance',n_jobs=1).fit(refqi,reference.time)
    baselines={model:{name:m.predict(x) for name,x in qmatrices.items() if name!='calibration'} for model,m in [('extra_trees',ref_ett),('knn5',ref_knn)]}
    for version,c in releases.items():
        print(f'Auditing {version}',flush=True)
        cr=rare_mask(c,schema)
        x=transform(all_proc,c,COLS)
        xqi=transform(qi_proc,c,qi)
        nn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(x)
        qnn=NearestNeighbors(n_neighbors=1,n_jobs=1).fit(xqi)
        dist={name:nn.kneighbors(arr,return_distance=True)[0][:,0] for name,arr in matrices.items()}
        qdist={name:qnn.kneighbors(arr,return_distance=True)[0][:,0] for name,arr in qmatrices.items()}
        # Equal class weight prevents public/synthetic row-count imbalance determining prior.
        train=np.vstack([x,refmat])
        labels=np.r_[np.ones(len(x)),np.zeros(len(refmat))]
        weights=np.r_[np.full(len(x),.5/len(x)),np.full(len(refmat),.5/len(refmat))]*len(train)
        clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,random_state=SEED).fit(train,labels,sample_weight=weights)
        scores={
            'nn_full':{n:-d for n,d in dist.items()},
            'nn_qi':{n:-d for n,d in qdist.items()},
            'nn_ratio_full':{n:np.log((public_distances[n]+1e-9)/(d+1e-9)) for n,d in dist.items()},
            'synthetic_public_classifier':{n:clf.predict_proba(arr)[:,1] for n,arr in matrices.items()},
        }
        audit={'distribution':feature_summary(c,schema),'distribution_comparison':{'overall_to_B':distribution_difference(b,c),'detected_rare_to_B_detected_rare':distribution_difference(b.loc[rare['members']],c.loc[cr]),'detected_rare_to_B_true_rare_DIFFERENT_LABELS':distribution_difference(b.loc[truth],c.loc[cr])},
            'identical_full_rows_with_B':int(np.isin(row_hashes(c),row_hashes(b)).sum()),'identical_full_rows_with_A_bg':int(np.isin(row_hashes(c),row_hashes(bg)).sum()),'mia':{},'aia':{}}
        audit['exact_B_sex_smoking_histogram']=audit['distribution']['sex_smoking_histogram']==report['source_distributions']['members']['sex_smoking_histogram']
        audit['exact_B_prefecture_histogram']=audit['distribution']['prefecture_histogram']==report['source_distributions']['members']['prefecture_histogram']
        for name,s in scores.items():
            audit['mia'][name]={
                'overall':mia_metrics(s['members'],s['controls'],s['calibration']),
                'detected_rare':mia_metrics(s['members'][rare['members']],s['controls'][rare['controls']],s['calibration'][rare['calibration']]),
                'detected_nonrare':mia_metrics(s['members'][~rare['members']],s['controls'][~rare['controls']],s['calibration'][~rare['calibration']]),
                'true_rare_members_vs_detected_rare_controls_DIFFERENT_LABELS':mia_metrics(s['members'][truth],s['controls'][rare['controls']],s['calibration'][rare['calibration']]),
            }
        models={'extra_trees':ExtraTreesRegressor(n_estimators=200,min_samples_leaf=10,max_features=.8,random_state=SEED,n_jobs=2),'knn5':KNeighborsRegressor(n_neighbors=5,weights='distance',n_jobs=1)}
        for name,model in models.items():
            model.fit(xqi,c.time)
            result={}
            predictions={n:model.predict(arr) for n,arr in qmatrices.items() if n!='calibration'}
            for population,frame in [('members',b),('controls',controls)]:
                masks=strata(frame,rare[population],truth if population=='members' else None)
                result[population]={key:aia_metrics(frame.time.to_numpy()[mask],predictions[population][mask],baselines[name][population][mask]) for key,mask in masks.items()}
            # A member-control contrast of gains: useful diagnostic, still confounded by cohort shift.
            contrasts={}
            for key in ['overall','censored','any_event','onset','death','detected_rare','detected_rare_event']:
                mm=strata(b,rare['members'])[key]
                cm=strata(controls,rare['controls'])[key]
                yy=b.time.to_numpy();zz=controls.time.to_numpy()
                mg=(np.abs(yy-predictions['members'])<=.5).astype(float)-(np.abs(yy-baselines[name]['members'])<=.5).astype(float)
                cg=(np.abs(zz-predictions['controls'])<=.5).astype(float)-(np.abs(zz-baselines[name]['controls'])<=.5).astype(float)
                a,d=mg[mm],cg[cm]
                if len(a) and len(d):
                    br=np.random.default_rng(SEED)
                    draws=[br.choice(a,len(a),replace=True).mean()-br.choice(d,len(d),replace=True).mean() for _ in range(BOOTSTRAPS)]
                    contrasts[key]={'member_gain_minus_control_gain':a.mean()-d.mean(),'bootstrap_95':np.quantile(draws,[.025,.975]),'n_members':len(a),'n_controls':len(d)}
            result['gain_difference_diagnostics']=contrasts
            audit['aia'][name]=result
        audit['nearest_distance_summary']={n:{'median':np.median(d),'q01':np.quantile(d,.01),'q99':np.quantile(d,.99)} for n,d in dist.items() if n!='calibration'}
        report['versions'][version]=audit
    cats=sorted(set(b.prefecture)|set(releases['R9'].prefecture))
    counts_equal=b.prefecture.value_counts().reindex(cats,fill_value=0).equals(releases['R9'].prefecture.value_counts().reindex(cats,fill_value=0))
    report['structural_findings']={
        'R9_exact_B_prefecture_histogram':counts_equal,
        'source':'R9/categorical.py:calibrate_prefecture',
        'interpretation':'The release deterministically reveals B prefecture counts. Adjacent datasets with one changed prefecture have different exact outputs for this histogram, so this exact-count release mechanism cannot satisfy finite-epsilon DP with delta<1 under that adjacency. This is not an empirical person-level identification rate.',
        'fixed_seed':'Fixed seeds support reproducibility and do not provide privacy randomization guarantees.',
        'co_lineage':'R3/R4/R6/R7/R8/R9 refine previously saved releases. Multiple releases share information and cannot be assumed independent privacy mechanisms.',
    }
    report=clean_json(report)
    (HERE/'audit_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_summary(report)
    print('Wrote aggregate-only audit_results.json and FINDINGS.md',flush=True)


def write_summary(r):
    n=r['input_counts']
    lines=['# R1–R9 privacy audit: fixed saved releases','',
           '**Exploratory local audit, not official scores or privacy certification.** Public A_bg controls differ from a true exchangeable private holdout. Member/public differences can reflect cohort shift. Versions are not causal ablations.','',
           f"B has {n['B']} rows ({n['B_true_rare']} true rare; {n['B_detected_rare']} detected rare; intersection {n['B_true_and_detected_rare']}). Public splits: reference {n['public_reference']}, calibration {n['public_calibration']}, evaluation {n['public_controls']}; rare calibration {n['calibration_detected_rare']}, rare evaluation {n['control_detected_rare']}. Excluded overlaps: ID {n['excluded_id_overlap']}, full record {n['excluded_exact_full_overlap']}.",'',
           '## Overall membership diagnostics','',
           'Each cell gives AUC [bootstrap 95% interval]; higher scores were specified in advance as more member-like. The synthetic/public classifier measures distribution similarity.','',
           '| Version | NN full | NN relative to public reference | Synthetic/public classifier | NN TPR / observed control FPR at calibration target 1% |','|---|---|---|---|---|']
    for v,a in r['versions'].items():
        vals=[]
        for attack in ['nn_full','nn_ratio_full','synthetic_public_classifier']:
            m=a['mia'][attack]['overall'];ci=m['auc_bootstrap_95']
            vals.append(f"{m['auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        m=a['mia']['nn_full']['overall']
        lines.append(f"| {v} | {' | '.join(vals)} | {m['tpr']:.3%} / {m['fpr']:.3%} |")
    lines += ['','## Detected-rare membership diagnostics','','Thresholds are calibrated within the detected-rare subgroup; limited control counts make low-FPR estimates imprecise. True-rare/member comparisons against detector-defined controls are kept only in JSON and explicitly label-mismatched.','',
              '| Version | NN full AUC | Relative NN AUC | Classifier AUC | Relative NN TPR / observed FPR |','|---|---|---|---|---|']
    for v,a in r['versions'].items():
        vals=[f"{a['mia'][k]['detected_rare']['auc']:.3f}" for k in ['nn_full','nn_ratio_full','synthetic_public_classifier']]
        m=a['mia']['nn_ratio_full']['detected_rare']
        rate='N/A: insufficient calibration controls' if m['threshold_infinite'] else f"{m['tpr']:.3%} / {m['fpr']:.3%}"
        lines.append(f"| {v} | {' | '.join(vals)} | {rate} |")
    lines += ['','## Time attribute inference: nonlinear ExtraTrees','','Accuracy means prediction within ±0.5 year using only the official QIs. Public baseline is trained only on the separate public reference split. Censoring dominates overall accuracy, so event-only results are essential. The last column is the member-versus-control contrast of accuracy gains over public baselines; it remains confounded by different cohort difficulty.','',
              '| Version | B overall accuracy | B event-only accuracy | B event-only public baseline | Control event-only accuracy | Event-only gain contrast [bootstrap 95%] |','|---|---|---|---|---|---|']
    for v,a in r['versions'].items():
        m=a['aia']['extra_trees'];allm=m['members']['overall'];ev=m['members']['any_event'];co=m['controls']['any_event'];g=m['gain_difference_diagnostics']['any_event'];ci=g['bootstrap_95']
        lines.append(f"| {v} | {allm['accuracy_half_year']:.3%} | {ev['accuracy_half_year']:.3%} | {ev['public_baseline_accuracy']:.3%} | {co['accuracy_half_year']:.3%} | {g['member_gain_minus_control_gain']:.3%} [{ci[0]:.3%}, {ci[1]:.3%}] |")
    if 'extra_trees_interval_mode' in r['versions']['R9']['aia']:
        lines += ['','## AIA sensitivity aligned to the interval-hit objective','','Mean regression is not optimal for interval-hit accuracy. This nonlinear classifier estimates time-bin probabilities and chooses the most likely ±0.5-year interval, with a distinct censoring atom. The same procedure is used for the separate public baseline. Parameters were fixed without selecting on B attack success.','',
                  '| Version | B overall accuracy | Always-10 B baseline | B event accuracy | Public-only B event accuracy | Event gain contrast [bootstrap 95%] |','|---|---|---|---|---|---|']
        for v,a in r['versions'].items():
            mm=a['aia']['extra_trees_interval_mode'];aa=mm['members']['overall'];ev=mm['members']['any_event'];g=mm['gain_difference_diagnostics']['any_event'];ci=g['bootstrap_95']
            lines.append(f"| {v} | {aa['accuracy_half_year']:.3%} | {aa['always_ten_accuracy']:.3%} | {ev['accuracy_half_year']:.3%} | {ev['public_baseline_accuracy']:.3%} | {g['member_gain_minus_control_gain']:.3%} [{ci[0]:.3%}, {ci[1]:.3%}] |")
    lines += ['','## Structural observations','',
              f"- R9 prefecture counts exactly equal B: **{r['structural_findings']['R9_exact_B_prefecture_histogram']}**. Code targets the exact B histogram. This leaks that aggregate and excludes a finite-epsilon DP claim with delta<1 under a neighboring change to prefecture. It does not by itself quantify reidentification.",
              '- R3 → R4 → R6 → R7 → R8 → R9 uses shared saved synthetic rows. Deterministic refinements and multiple-release information must be evaluated jointly; versions are not independent draws.',
              '- Exact B sex-by-smoking counts are preserved in R1/R2/R3/R4/R6/R7/R8/R9. This also releases that confidential aggregate without noise. R5 does not match those counts.',
              '- No output row matching B does not establish anonymity. All counts and distances are aggregated in the JSON.',
              '', '## Interpretation limits','']+[f'- {x}' for x in r['limitations']]
    lines+=['','## Reproduction','','Run `python -m unittest discover -s . -p test_audit.py -v`, then `python audit.py` and `python aia_mode.py` in this directory. Libraries and file hashes are recorded in `audit_results.json`. Source data and frozen releases are required; no token files are read. Full metrics include QI nearest-neighbor attack, 5-neighbor AIA, nonlinear interval-mode AIA, per-stratum intervals, distribution diagnostics, and deterministic lineage.','']
    (HERE/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()

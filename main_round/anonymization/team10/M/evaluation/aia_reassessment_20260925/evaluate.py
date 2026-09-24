"""Fixed-candidate reevaluation with allowed AIA QIs and paired proxy challenges.

This is a local diagnostic, not an official score. No candidate is generated.
Prior DOMIAS permutations are reused only after confidence/hash/statistic checks.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, importlib.util, json, sys, time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TEAM = HERE.parents[2]
OLD = HERE.parent / 'domias_main_round_20260924'
CONFIRM = TEAM / 'M/M1_v5/confirmation'
spec = importlib.util.spec_from_file_location('prior_evaluation', OLD / 'evaluate.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
MODELS = ['M1_v3', 'M1_v4', 'M1_v5']
SEEDS = [20260924, 20260925, 20260926, 20261001, 20261002, 20261003,
         20261011, 20261012, 20261013]
QUOTAS = {(False, 0): 138, (False, 1): 12, (True, 0): 9, (True, 1): 4}
VIEWS = [('all', QI[:5], ['TG', 'ALT']),
         ('stable', ['age', 'SBP', 'HDL'], []),
         ('clinical_allowed', ['SBP', 'TG', 'HDL', 'ALT'], ['TG', 'ALT'])]
PHI = 0.026887

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, (int, np.integer)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x

def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def keys(frame, rare):
    # The rulebook does not specify the age-band width here: five years is a proxy assumption.
    return list(zip(frame.prefecture.astype(str), frame.sex.astype(int),
                    ((frame.age - 40) // 5).astype(int), frame.onset.astype(int), rare.astype(bool)))

def paired_plan(b, a, br, ar, seed):
    bk, ak = keys(b, br), keys(a, ar)
    donors = defaultdict(list)
    for i, key in enumerate(ak): donors[key].append(i)
    eligible = np.array([key in donors for key in bk])
    rng = np.random.default_rng(seed)
    for attempt in range(10000):
        chosen = np.concatenate([rng.choice(np.flatnonzero(eligible & (br == r) &
                     (b.onset.to_numpy() == o)), n, replace=False) for (r, o), n in QUOTAS.items()])
        needed = Counter(bk[i] for i in chosen)
        if all(n <= len(donors[k]) for k, n in needed.items()): break
    else: raise RuntimeError('Matching capacity failed')
    chosen = rng.permutation(chosen)
    available = {k: list(rng.choice(donors[k], n, replace=False)) for k, n in needed.items()}
    controls = np.array([available[bk[i]].pop() for i in chosen], dtype=int)
    validate_plan(b, a, br, ar, chosen, controls)
    return {'seed': seed, 'attempts': attempt+1, 'member_indices': chosen.tolist(),
            'control_indices': controls.tolist(), 'eligible_B_count': int(eligible.sum()),
            'excluded_B_count': int((~eligible).sum()),
            'eligibility_by_stratum': [{'rare':r,'onset':o,'available_B':int(((br==r)&(b.onset.to_numpy()==o)).sum()),
               'eligible_B':int((eligible&(br==r)&(b.onset.to_numpy()==o)).sum()),'target_count':n}
               for (r,o),n in QUOTAS.items()],
            'scope': 'B targets, A_bg controls; private Pi identity unknown; age bands five years; public rare detector surrogate'}

def validate_plan(b, a, br, ar, mi, ci):
    mi, ci = np.asarray(mi), np.asarray(ci)
    assert len(mi) == len(ci) == 163
    assert len(set(mi)) == len(set(ci)) == 163
    bk, ak = keys(b, br), keys(a, ar)
    assert all(bk[m] == ak[c] for m, c in zip(mi, ci))
    for (r,o),n in QUOTAS.items():
        assert int(((br[mi]==r)&(b.onset.to_numpy()[mi]==o)).sum()) == n
        assert int(((ar[ci]==r)&(a.onset.to_numpy()[ci]==o)).sum()) == n

def predict(c, query, background, cfg):
    # Enforce the disclosure boundary before any attack, rather than merely documenting it.
    q = query.loc[:, QI].copy()
    bg = background.loc[:, QI].copy()
    assert list(q.columns) == QI and set(q.columns).isdisjoint({'BMI','FPG','time','onset','is_rare','record_id'})
    _, official = ev.aia_predict(c, q, bg, cfg)
    predictions = []
    for _, columns, logs in VIEWS:
        center, scale = ev.local.fit_view(bg, tuple(columns), tuple(logs))
        cx = ev.local.transform_view(c, tuple(columns), tuple(logs), center, scale)
        qx = ev.local.transform_view(q, tuple(columns), tuple(logs), center, scale)
        _, idx = ev.local.grouped_nn(c, cx, q, qx)
        predictions.append(c.time.to_numpy()[idx])
    result = {'official_qi_nearest': official,
              'allowed_qi_multiview': np.median(np.column_stack(predictions), axis=1)}
    assert all(v.shape == (len(q),) and np.isfinite(v).all() for v in result.values())
    return result

def aia_result(pred, b, a, br, ar, mi, ci, cfg):
    result = ev.score_aia(None, pred[np.asarray(mi)], b.onset.to_numpy()[mi], b.time.to_numpy()[mi],
              None, pred[len(b)+np.asarray(ci)], a.onset.to_numpy()[ci], a.time.to_numpy()[ci], cfg,
              member_rare_mask=br[mi], control_rare_mask=ar[ci])
    return {'risk_before_phi': float(result.score), 'breakdown': safe(result.breakdown)}

def combine(risks, mia, utility, phi=0):
    aia = float(np.clip(max(risks)-phi, 0, 1))
    anon = 1-max(mia, aia)
    return aia, anon, min(utility, anon)

def checked_mia(conf, model, seed, cfg):
    label = 'M1_v4' if model == 'M1_v3' else model
    plan_path = CONFIRM / f'cohorts_{seed}.json'
    path = CONFIRM / label / f'domias_{seed}.json'
    plan, result = read(plan_path), read(path)
    digest = hashlib.sha256(); raw = []
    for p in plan['attackers']:
        values = conf[p['pool_indices']]; y=np.array(p['truth']); r=np.array(p['rare'], bool)
        digest.update(np.ascontiguousarray(np.column_stack([values,y,r]),dtype='<f8').tobytes())
        raw.append(ev.score_mia(values,y,cfg,rare_mask=r).score)
    assert result['permutation_seed'] == int.from_bytes(digest.digest()[:8], 'little')
    assert np.isclose(result['beta_raw'], np.mean(raw), atol=1e-14, rtol=0)
    assert result['permutations'] == 20000
    null_path = CONFIRM / label / f'null_{seed}.npy'
    if not null_path.exists(): null_path = TEAM / 'M/M1_v4/confirmation' / f'null_{seed}.npy'
    if not null_path.exists(): null_path = OLD / 'results/M1_v3' / f'null_{seed}.npy'
    null = np.load(null_path, allow_pickle=False)
    assert null.shape == (20000,) and np.isfinite(null).all()
    assert np.isclose(result['chance95'],np.quantile(null,.95),atol=1e-14,rtol=0)
    beta=max(result['beta_raw']-result['chance95']+.01,.01)
    power=float(np.clip((beta/.01-1)/11,0,1))
    assert abs(power-result['a_MIA']) < 1e-12
    return {**{k: result[k] for k in ['beta_raw','chance95','beta','a_MIA']},
            'source': str(path.relative_to(TEAM)).replace('\\','/'), 'source_sha256':sha(path),
            'cohorts_sha256':sha(plan_path), 'null_sha256':sha(null_path),
            'reused_after_verification':True, 'permutations':20000}

def main():
    started=time.monotonic(); cfg, env=ev.utility.environment()
    cfg.setdefault('aia',{}).update({'qi_continuous': QI[:5], 'block_keys':['sex','prefecture'],'k_nn_onset':5})
    inputs={str(p.relative_to(TEAM)).replace('\\','/'):sha(p)
            for p in [*(TEAM/'M'/m/'C.csv' for m in MODELS),
                      *(TEAM/'participant_data'/f for f in ['B.csv','B_self.csv','A_bg.csv','schema.json','utility_ref.json']),
                      TEAM/'M/M1_v3/submission/team10_M1_v3_defense.zip']}
    downloads=Path.home()/'Downloads'
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(), 'models':MODELS, 'seeds':SEEDS,
       'primary_seed':SEEDS[0], 'input_sha256':inputs, 'code_sha256':sha(__file__), 'environment':env,
       'source_context_sha256':{f:sha(downloads/f) for f in ['PWSCup_team10_context_ja.md','PWSCup_team10_candidates.csv']},
       'QI':QI,'views':VIEWS,'age_band_width':5,'targets':163,'controls':163,'rare_per_side':13,
       'onset_per_side':16,'rare_onset_per_side':4,'provided_phi':PHI,
       'phi_status':'User-provided, not validated for these pseudo-challenges; scenario only',
       'aia_aggregation':'max across two attacks and all/SE-adjusted rare, then optionally subtract phi once',
       'mia_policy':'Recompute confidences and observed statistics, validate cached permutation hash and 95% null quantile, reuse 20000 permutations',
       'candidate_tuning':False,'official_score':False,
       'limitations':['AIA targets sampled from B, not known Pi; controls sampled from public A instead of outside A and all cohorts',
        'Five-year age-band assumption; exact matching excludes some B records; public rare labels are detector surrogates',
        'AIA samples and preexisting MIA samples are separate proxies, not a joint reconstruction of all shared/private blocks',
        'Nine seeds share the same underlying data; seed variation is not a confidence interval or independent holdout',
        'Candidates were previously optimized using these data and six MIA seeds; two AIA attacks do not cover all possible attacks']}
    if (HERE/'protocol.json').exists():
        prior=read(HERE/'protocol.json'); protocol['created_utc']=prior['created_utc']
        if protocol != prior: raise RuntimeError('Frozen protocol changed; use a new output directory')
    save(HERE/'protocol.json',protocol)
    b=pd.read_csv(TEAM/'participant_data/B_self.csv'); a=pd.read_csv(TEAM/'participant_data/A_bg.csv')
    schema=read(TEAM/'participant_data/schema.json'); br=b.is_rare.to_numpy(bool); ar=ev.local.rare_mask(a,schema)
    pool=pd.concat([b.drop(columns='is_rare'),a],ignore_index=True)
    plans={s:paired_plan(b,a,br,ar,s) for s in SEEDS}
    for seed,plan in plans.items(): save(HERE/f'private/plans_{seed}.json',plan)
    rows=[];attack_rows=[];diagnostics=[]; audit=[]
    for model in MODELS:
        cpath=TEAM/'M'/model/'C.csv'; c=pd.read_csv(cpath)
        utility=ev.utility.evaluate_csv(cpath); save(HERE/f'results/{model}/utility.json',utility)
        u=utility['score']['utility']
        conf=ev.overlap_mia(c,a,pool,cfg)
        source_model='M1_v4' if model=='M1_v3' else model
        assert np.array_equal(conf,np.load(CONFIRM/f'{source_model}_domias_confidence.npy',allow_pickle=False))
        predictions=predict(c,pool,a,cfg)
        np.savez_compressed(HERE/f'private/{model}_aia_predictions.npz',**predictions)
        diag={name:aia_result(pred,b,a,br,ar,np.arange(len(b)),np.arange(len(a)),cfg)
              for name,pred in predictions.items()}
        diagnostics.append({'model':model,'all_B_all_A_QI_only':max(x['risk_before_phi'] for x in diag.values()),
                            **{name:res['risk_before_phi'] for name,res in diag.items()}})
        save(HERE/f'results/{model}/all_B_all_A_QI_only.json',diag)
        for seed,plan in plans.items():
            mi=np.array(plan['member_indices']); ci=np.array(plan['control_indices'])
            attacks={name:aia_result(pred,b,a,br,ar,mi,ci,cfg) for name,pred in predictions.items()}
            mia=checked_mia(conf,model,seed,cfg)
            aia,anon,obj=combine([x['risk_before_phi'] for x in attacks.values()],mia['a_MIA'],u)
            adjusted,anon_phi,obj_phi=combine([x['risk_before_phi'] for x in attacks.values()],mia['a_MIA'],u,PHI)
            row={'model':model,'seed':seed,'U':u,**{k:mia[k] for k in ['beta_raw','chance95','beta','a_MIA']},
                 'AIA_before_phi':aia,'Anon_no_phi_proxy':anon,'Objective_no_phi_proxy':obj,
                 'AIA_phi_scenario':adjusted,'Anon_phi_scenario':anon_phi,'Objective_phi_scenario':obj_phi}
            rows.append(row); audit.append({'model':model,'seed':seed,**mia})
            save(HERE/f'results/{model}/aia_{seed}.json',{'attacks':attacks,'summary':row})
            for name,res in attacks.items():
                attack_rows.append({'model':model,'seed':seed,'attack':name,
                                    'risk_before_phi':res['risk_before_phi'],**res['breakdown']})
        print(json.dumps({'model':model,'U':u,'primary':rows[-len(SEEDS)],'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    table=pd.DataFrame(rows);table.to_csv(HERE/'comparison_all_seeds.csv',index=False)
    table[table.seed==SEEDS[0]].to_csv(HERE/'comparison_primary.csv',index=False)
    pd.DataFrame(attack_rows).to_csv(HERE/'aia_attack_details.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(HERE/'all_B_all_A_QI_only.csv',index=False)
    summary=[]
    for model,group in table.groupby('model',sort=False):
        r={'model':model,'U':group.U.iloc[0]}
        for col in ['a_MIA','AIA_before_phi','Anon_no_phi_proxy','Objective_no_phi_proxy','Anon_phi_scenario','Objective_phi_scenario']:
            for stat in ['min','median','max']: r[col+'_'+stat]=float(getattr(group[col],stat)())
        summary.append(r)
    pd.DataFrame(summary).to_csv(HERE/'sensitivity.csv',index=False)
    assert all(sha(TEAM/path)==h for path,h in inputs.items())
    save(HERE/'mia_cache_verification.json',audit)
    save(HERE/'completion.json',{'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),
         'candidate_count':len(MODELS),'seed_count':len(SEEDS),'comparison_rows':len(rows),'aia_attack_rows':len(attack_rows),
         'inputs_unchanged':True,'candidate_tuning':False,'submission_zip_unchanged':True,
         'B_record_id_equals_row_index':bool(np.array_equal(b.record_id.to_numpy(),np.arange(len(b)))),
         'mia_confidences_recomputed_and_equal':True,'mia_cached_permutations_validated':True,
         'elapsed_seconds':time.monotonic()-started,'protocol_sha256':sha(HERE/'protocol.json'),
         'result_sha256':{f:sha(HERE/f) for f in ['comparison_all_seeds.csv','comparison_primary.csv','sensitivity.csv','aia_attack_details.csv']}})

if __name__=='__main__': main()

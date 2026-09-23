"""End-to-end PUBLIC check of one selected plan, then freeze configuration."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,yaml
HERE=Path(__file__).resolve().parent;CORE=HERE.parent;TEAM=CORE.parents[1];KIT=CORE.parents[5]/'official-kit'
sys.path[:0]=[str(KIT/'starter/src')]
import pipeline20
import dp_copula as dc
from pwscup2026.kit.selfscore import self_score
from pwscup2026.scoring.utility import UtilityReference
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    best=read(HERE/'public_target_study.json')['summaries'][0]
    h=[r for r in read(HERE/'public_histogram_study.json')['summaries'] if r['histogram_epsilon']==int(best['plan']['histogram_epsilon'])]
    radial=max(h,key=lambda r:(min(r['mean_gen_floor0'],r['mean_rare_proxy']),-r['radial']))['radial']
    plan={**best['plan'],'radial_strength':radial,'postprocess_seed':20260918,'iterations':300,'candidates':4096,'trials':12}
    # At most one full PUBLIC check per cohort, neither score changes this plan.
    data=TEAM/'participant_data';all_public=pd.read_csv(data/'A_bg.csv');schema=read(data/'schema.json');det=schema['rare_detector']
    cfg=yaml.safe_load((KIT/'starter/src/pwscup2026/kit/kit_config.yaml').read_text(encoding='utf-8'))
    rows=[]
    for seed in (2026091811,2026091827):
        order=np.random.default_rng(seed).permutation(len(all_public))
        proxy=all_public.iloc[order[:1049]].sort_values('record_id').reset_index(drop=True)
        public=all_public.iloc[order[1049:]].reset_index(drop=True)
        model=pipeline20.fit_once(proxy,public,schema,plan)
        candidate,trace=pipeline20.construct(model)
        ref=UtilityReference(mu=np.asarray(det['mu']),sigma=np.asarray(det['sigma']),band=det['band'],
            b_is_rare=dc._radii(proxy,det)>=float(det['band']['lo']),a_bg=public)
        score=self_score(candidate,proxy,ref,cfg)
        row={'public_cohort_seed':seed,'utility_public_proxy':score['utility'],'facets_public_proxy':score['facets'],'trace':trace}
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
    study={'source':'PUBLIC A_bg only','complete':True,'plan':plan,'rows':rows,
           'selection_rule':'Best mean target IO across six public simulations; then best min(mean_gen_floor0,mean_rare_proxy) radial for that histogram budget. End-to-end checks do not alter selection.',
           'limitations':['Two public cohorts; proxy rare labels and KM floor zero; not actual local utility or guarantee.',
                          'Selected among24 plans using144 public target simulations; optimistic selection bias possible.'],
           'selection_evidence_sha256':{n:sha(HERE/n) for n in ('public_target_study.json','public_histogram_study.json')},
           'public_input_sha256':{n:sha(data/n) for n in ('A_bg.csv','schema.json')}}
    (HERE/'public_final_check.json').write_text(json.dumps(study,ensure_ascii=False,indent=2),encoding='utf-8')
    config={'version':1,'standalone_epsilon_exact':'20','additional_epsilon_exact':'20','previous_epsilon_exact':'414',
            'composed_epsilon_exact':'434','delta':0,'public_n':1049,'plan':plan,
            'public_selection_sha256':sha(HERE/'public_final_check.json')}
    (HERE/'config20.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()

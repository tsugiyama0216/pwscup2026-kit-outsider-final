"""Matched cached-world development; freeze before confirmation."""
import argparse
import runtime as rt
from model import variants,MIA_OPTIONS,AIA_OPTIONS
import pandas as pd
from sugiyama_v2.io import load_world,prediction_arguments
from sugiyama_v2.scoring import score_world
from sugiyama_v2.runner import per_target_diagnostics
def main():
    stage=argparse.ArgumentParser();stage.add_argument('stage',choices=['development','confirmation']);args=stage.parse_args()
    out=rt.HERE/'evaluation'/args.stage
    if out.exists():raise ValueError('Preserve existing evaluation')
    out.mkdir(parents=True)
    rt.preserve()
    _,_,_,cfg,_=rt.load('_rv4_common',rt.ATTACK/'Attack_R/v1/common.py').load_actual()[0].load_runtime()
    seeds=[2026100711,2026100712] if args.stage=='development' else [2026100721,2026100722]
    selection=None if args.stage=='development' else rt.read(rt.HERE/'selection.json')
    rows=[];details=[]
    for seed in seeds:
        saved=rt.MOCK/'local-output/R-v2'/args.stage/str(seed)
        public,private,manifest,before=load_world(saved/'case')
        baseline={n:pd.read_csv(saved/'predictions'/c/'T'/(n+'.csv'),float_precision='round_trip') for n,c in [('F_mia','mia'),('F_aia','aia')]}
        v=variants(**prediction_arguments(public,cfg),baseline=baseline)
        for component,options in [('mia',MIA_OPTIONS),('aia',AIA_OPTIONS)]:
            names=options if selection is None else list(dict.fromkeys(['Rv2',selection[component]]))
            for name in names:
                fm=v['mia'][name if component=='mia' else 'Rv2'];fa=v['aia'][name if component=='aia' else 'Rv2']
                result=score_world(fm,fa,public,saved/'scoring_truth',cfg)
                rows.append({'seed':seed,'component':component,'option':name,'power':result[component+'_power']})
                rt.save(out/f'{seed}-{component}-{name}.json',result)
                d=per_target_diagnostics(fm,fa,public,saved/'scoring_truth',cfg,manifest['metadata'],result)
                d['component']=component;d['option']=name;d['seed']=seed;details+=d.to_dict(orient='records')
                print(seed,component,name,result[component+'_power'],flush=True)
        for n,c in [('F_mia','mia'),('F_aia','aia')]:
            for name,frame in v[c].items():
                p=out/str(seed)/c/name;p.mkdir(parents=True);(p/(n+'.csv')).write_bytes(rt.csv(frame))
    frame=pd.DataFrame(rows);frame.to_csv(out/'summary.csv',index=False);pd.DataFrame(details).to_csv(out/'by_target.csv',index=False)
    if selection is None:
        selection={}
        for c in ['mia','aia']:
            table=frame[frame.component==c].pivot(index='seed',columns='option',values='power');gain=table.subtract(table.Rv2,axis=0)
            eligible=[n for n in table if (gain[n]>=-.01).all() and (n=='Rv2' or gain[n].mean()>.001)]
            selection[c]=max(eligible,key=lambda n:table[n].mean())
        selection['source_sha256']={n:rt.sha(rt.HERE/n) for n in ['model.py','plugin.py','runtime.py']}
        selection['confirmation_not_started']=True;selection['basis']='2 previously used development worlds; mean gain > .001, no world loss > .01'
        rt.save(rt.HERE/'selection.json',selection)
    result={'stage':args.stage,'selected':selection,'means':frame.groupby(['component','option']).power.mean().to_dict()}
    means={c:frame[frame.component==c].groupby('option').power.mean().to_dict() for c in ['mia','aia']}
    rt.save(out/'result.json',{'completed':True,'means':means,'Rv2':means['mia']['Rv2']+means['aia']['Rv2'],'Rv4':sum(means[c][selection[c]] for c in means),'selection':selection,'official_score':None})
    rt.preserve();print('Completed',args.stage,selection,flush=True)
if __name__=='__main__':main()

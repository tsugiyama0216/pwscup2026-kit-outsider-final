"""Summarize completed, frozen Rv12 checks; never selects or modifies predictions."""
from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    path=HERE/'comparison.json'
    if path.exists():raise ValueError('Preserve completed report')
    selected=json.loads((HERE/'selection.json').read_text())
    for name,digest in selected['source_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen source changed')
    rows=[];evidence={};sensitivity={}
    for role,seeds in [('development',(61,62)),('confirmation',(81,82))]:
        all_targets=[]
        for seed in seeds:
            base=ROOT/'main_round/mock_test/local-output'/f'rv12-{role}'
            folder=base/(f'scores-all-{seed}' if role=='development' else f'scores-{seed}')
            result=json.loads((folder/'result.json').read_text())
            if not result['evaluation_role'].startswith(role):raise ValueError('Wrong evaluation role')
            if datetime.fromisoformat(result['created_utc'])<=datetime.fromisoformat(selected['created_utc']) and role=='confirmation':raise ValueError('Confirmation predates freeze')
            for name,digest in result['output_sha256'].items():
                if sha(folder/name)!=digest:raise ValueError('Report changed')
            summary=pd.read_csv(folder/'summary.csv',float_precision='round_trip').set_index('model')
            candidate='retain' if role=='development' else 'Rv12'
            for model,key in [('Rv2','Rv2'),('Rv12',candidate)]:
                record=summary.loc[key].to_dict()
                rows.append({'phase':role,'seed':result['seed'],'model':model,**{k:v for k,v in record.items() if k in ['stable_index','official_attack_power','mean_mia_all','mean_mia_power','mean_aia_power','mean_member_hit_rate','mean_control_hit_rate','delta_vs_Rv2','official_delta_vs_Rv2']}})
            detail=pd.read_csv(folder/'by_target.csv',float_precision='round_trip')
            x=detail.loc[detail.model.eq(candidate)].set_index('target_k').delta_stable_index_vs_Rv2
            all_targets.append(x)
            destination=HERE/'evidence'/f'{role}-{seed}'
            if role=='confirmation':
                destination.mkdir(parents=True,exist_ok=False)
                for name in ('result.json','summary.csv','by_target.csv'):shutil.copyfile(folder/name,destination/name)
            evidence[f'{role}-{seed}']={n:sha(folder/n) for n in ('result.json','summary.csv','by_target.csv')}
        paired=pd.concat(all_targets,axis=1).mean(axis=1)
        individual=[float(x.drop(k).mean()) for x in all_targets for k in x.index]
        paired_deletions=[float(paired.drop(k).mean()) for k in paired.index]
        sensitivity[role]={'paired_leave_one_target_min':min(paired_deletions),'paired_leave_one_target_max':max(paired_deletions),'individual_leave_one_target_min':min(individual),'individual_leave_one_target_max':max(individual)}
    frame=pd.DataFrame(rows)
    means=frame.groupby(['phase','model'],sort=False).mean(numeric_only=True).drop(columns=['seed']).reset_index()
    frame.to_csv(HERE/'comparison_by_case.csv',index=False,float_format='%.17g')
    means.to_csv(HERE/'comparison.csv',index=False,float_format='%.17g')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':sha(HERE/'selection.json'),'same_algorithm_as_Rv8':True,'all_selected_sources_unchanged':True,'no_post_confirmation_tuning':True,'means':means.to_dict('records'),'cases':rows,'target_deletion_sensitivity':sensitivity,'source_reports_sha256':evidence,'official_score_prediction':False,'limitations':['New seeds share the same donor population and generator assumptions; not an independent population.','Historical fitted models retain earlier training exposure.','This is Rv8-equivalent, not an improvement over Rv8.','No official Rv12 score has been obtained.']}
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(means.to_string(index=False))

if __name__=='__main__':main()

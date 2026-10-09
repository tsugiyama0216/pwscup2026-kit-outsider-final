"""Apply the predeclared AIA gate, recording rejected hypotheses and exact sources."""
from datetime import datetime,timezone
import json
from pathlib import Path
import hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
DEV=ROOT/'main_round/mock_test/local-output/rv12-development'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    output=HERE/'selection.json'
    if output.exists():
        raise ValueError('Preserve frozen selection')
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8-sig'))
    summaries=[]
    details=[]
    inputs={}
    for seed in (61,62):
        folder=DEV/f'scores-all-{seed}'
        record=json.loads((folder/'result.json').read_text(encoding='utf8'))
        if record['seed']!=2026100900+seed or record['evaluation_role']!='development':
            # Earlier records used the equally explicit long-form role.
            if record['seed']!=2026100900+seed or not record['evaluation_role'].startswith('development'):
                raise ValueError('Wrong development case')
        for name,digest in record['output_sha256'].items():
            if sha(folder/name)!=digest:
                raise ValueError('Development report changed')
        inputs[str(seed)]={name:sha(folder/name) for name in ('result.json','summary.csv','by_target.csv')}
        summaries.append(pd.read_csv(folder/'summary.csv',float_precision='round_trip').set_index('model'))
        details.append(pd.read_csv(folder/'by_target.csv',float_precision='round_trip'))
    candidates=[]
    for name in protocol['aia_candidates']:
        case=[]
        for i in range(2):
            row=summaries[i].loc[name]
            target=details[i].loc[details[i].model.eq(name)].set_index('target_k')
            delta=target.delta_aia_power_vs_Rv2
            case.append({'seed':2026100961+i,'aia_delta':float(row.aia_delta_vs_Rv2),'stable_index':float(row.stable_index),'official_sum':float(row.official_attack_power),'member_hit_delta':int(row.member_hit_delta),'control_hit_delta':int(row.control_hit_delta),'aia_leave_one_target_out_min':min(float(delta.drop(k).mean()) for k in delta.index)})
        eligible=name!='retain' and min(x['aia_delta'] for x in case)>=.001 and sum(x['aia_delta'] for x in case)/2>=.003 and min(x['aia_leave_one_target_out_min'] for x in case)>=0
        suppressed=sum(sum(v['suppressed'] for v in json.loads((DEV/f'inference-all-{s}'/name/'diagnostics.json').read_text())['targets'].values()) for s in (61,62))
        candidates.append({'name':name,'eligible':eligible,'cases':case,'mean_aia_delta':sum(x['aia_delta'] for x in case)/2,'suppressed':suppressed})
    eligible=[c for c in candidates if c['eligible']]
    selected=max(eligible,key=lambda c:(min(x['aia_delta'] for x in c['cases']),-c['suppressed']))['name'] if eligible else 'retain'
    sources=[p for p in HERE.iterdir() if p.suffix in ('.py','.ps1')]+[HERE/'protocol.json',HERE/'generation_protocol.json',HERE/'generation_freeze.json']
    selection={'created_utc':datetime.now(timezone.utc).isoformat(),'name':'Rv12','mia':'Rv8 specificity strength 0.35 cap 7','aia':selected,'same_algorithm_as_Rv8':selected=='retain','reason':'No AIA modification passed the predeclared component gate; preserve baseline AIA.' if selected=='retain' else 'Passed predeclared AIA component gate.','candidates':candidates,'development_reports_sha256':inputs,'source_sha256':{p.name:sha(p) for p in sources},'confirmation_seeds':[2026100981,2026100982],'confirmation_results_seen':False,'official_submission_performed':False,'github_push_performed':False}
    output.write_text(json.dumps(selection,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'aia':selected,'selection_sha256':sha(output),'same_as_Rv8':selected=='retain'}))

if __name__=='__main__':main()

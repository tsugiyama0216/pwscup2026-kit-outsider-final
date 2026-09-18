"""Collect all research batches and freeze the eligible maximum-utility setting."""
import json
from evaluation import HERE

BATCHES=['rejected_initial_grid.json','bounded_summary.json','balanced_summary.json',
         'projected_summary.json','majority_summary.json','outcomes_summary.json',
         'guarded_summary.json','tail_summary.json','radius_summary.json']
DEFAULTS={'covariance_weight':0.,'information_strength':1.,'preserve_strength':0.,
          'majority_strength':0.,'outcome_steps':0,'significance_guard':False,
          'tail_strength':0.,'preserve_tail_radius':False}

if __name__=='__main__':
    candidates=[]
    for batch in BATCHES:
        data=json.loads((HERE/batch).read_text(encoding='utf-8'))
        for row in data['candidates']:
            candidates.append({**DEFAULTS,**row,'batch':batch})
    baseline=json.loads((HERE/'R6_user_reported_result.json').read_text())['utility_facets']
    eligible=[r for r in candidates if r.get('eligible') and all(r['facets'][k]>=v-1e-12 for k,v in baseline.items())]
    selected=max(eligible,key=lambda r:r['utility'])
    result={'selection':'maximum utility with no regression in any R6 facet; candidates adaptively specified in fixed batches',
            'candidate_count':len(candidates),'candidates':candidates,'selected':selected}
    with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({'candidate_count':len(candidates),'selected':selected}))

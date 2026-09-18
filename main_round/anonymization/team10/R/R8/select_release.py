"""Select the best evaluated candidate with no R7 facet regression."""
import json
from evaluation import HERE,digest

BATCHES=('initial','selective','tail','group','stable','quantile','bounds','combined','shape')

def main():
    feedback=json.loads((HERE/'R7_user_reported_result.json').read_text(encoding='utf-8'))['utility_facets']
    candidates=[]
    for batch in BATCHES:
        path=HERE/(batch+'_summary.json')
        data=json.loads(path.read_text(encoding='utf-8'))
        for i,candidate in enumerate(data['candidates']):
            item={**candidate,'batch':batch,'candidate_index':i,'batch_sha256':digest(path)}
            item['eligible']=item['utility']>min(feedback.values())+1e-9 and all(item['facets'][k]>=v-1e-12 for k,v in feedback.items())
            candidates.append(item)
    eligible=[c for c in candidates if c['eligible']]
    if not eligible:
        raise RuntimeError('No candidate passed the non-regression gate')
    selected=max(eligible,key=lambda c:c['utility'])
    result={'selection':'Maximum official local utility; every facet >= R7; improvement > 1e-9',
            'adaptive_batches':True,'independent_holdout':False,'seed_or_row_order_search':False,
            'note':'Canonical eligibility is recomputed here; the first batch originally counted an identity round-off difference as eligible.',
            'candidate_count':len(candidates),'selected':selected,'candidates':candidates}
    with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'count':len(candidates),'selected':selected}))

if __name__=='__main__': main()

"""Select an evaluated improvement with no regression from R8 official feedback."""
import json
from evaluation import HERE,digest
BATCHES=('initial','constraints','age')

def main():
    baseline=json.loads((HERE/'R8_user_reported_result.json').read_text())['utility_facets']
    candidates=[]
    for batch in BATCHES:
        path=HERE/(batch+'_summary.json')
        data=json.loads(path.read_text(encoding='utf-8'))
        for i,c in enumerate(data['candidates']):
            item={k:c[k] for k in ['settings','utility','facets','csv_sha256']}
            item.update(batch=batch,index=i,batch_sha256=digest(path))
            item['eligible']=c['utility']>min(baseline.values())+1e-9 and all(c['facets'][k]>=v-1e-12 for k,v in baseline.items())
            candidates.append(item)
    eligible=[c for c in candidates if c['eligible']]
    if not eligible: raise RuntimeError('No improved candidate')
    selected=max(eligible,key=lambda c:c['utility'])
    result={'selection':'Maximum official local utility subject to no R8 facet regression and >1e-9 improvement',
            'candidate_count':len(candidates),'adaptive_batches':True,'independent_holdout':False,
            'random_seed_or_row_order_search':False,'selected':selected,'candidates':candidates}
    with (HERE/'experiment_summary.json').open('w',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'count':len(candidates),'selected':selected}))

if __name__=='__main__': main()

"""Select the highest-U fully nonregressing candidate across completed batches."""
import json,hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    selected_path=HERE/'experiment_summary.json'
    if selected_path.exists(): raise RuntimeError('Refusing to overwrite fixed selection')
    candidates=[]; summaries={}
    for name,count in [('initial',10),('joint',4),('locked',2)]:
        path=HERE/f'{name}_summary.json'
        raw=path.read_bytes(); report=json.loads(raw)
        if len(report['candidates'])!=count: raise RuntimeError('Incomplete batch: '+name)
        summaries[path.name]=hashlib.sha256(raw).hexdigest()
        for candidate in report['candidates']:
            candidates.append(dict(candidate,source_summary=path.name))
    eligible=[c for c in candidates if c['eligible']]
    if not eligible: raise RuntimeError('No candidate improved all required utility criteria')
    chosen=max(eligible,key=lambda c:(c['score']['utility'],sum(c['score']['facets'].values()),-c['index']))
    result=dict(selected=chosen,candidate_count=len(candidates),eligible_count=len(eligible),
                selection='Highest utility subject to all facets >= R9; sum of facets breaks ties; no AUC criterion.',
                adaptive_between_batches=True,independent_holdout=False,summary_sha256=summaries)
    selected_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(settings=chosen['settings'],utility=chosen['score']['utility'],facets=chosen['score']['facets'],
                          candidate_count=len(candidates),eligible_count=len(eligible))),flush=True)

if __name__=='__main__': main()

"""Freeze and create exactly one deterministic joint14 DP postprocessing output.

There is deliberately no private-data path and no evaluation call in this CLI.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; CORE=HERE.parent
sys.path[:0]=[str(HERE),str(CORE)]
import dp_copula as dc
from pipeline import construct

read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    study=read(HERE/'public_pipeline_study.json')
    config=read(HERE/'joint14_config.json')
    if not study['complete'] or len(study['rows'])!=24 or config['settings']!=study['chosen_settings']:
        raise ValueError('Complete public-only selection study and matching settings required')
    if any(sha(CORE/name)!=h for name,h in study['code_sha256'].items()):
        raise ValueError('Code changed after public selection; refusing to label a new design as frozen')
    models=[]; parents=[]
    ledger=[json.loads(line) for line in (CORE/'release_ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    completed={x['epsilon_exact'] for x in ledger if x['event']=='complete'}
    for epsilon in (1,3,10):
        path=CORE/f'releases/eps{epsilon}'; manifest=read(path/'manifest.json'); model=read(path/'model.json')
        if str(epsilon) not in completed or manifest['model_sha256']!=sha(path/'model.json'):
            raise ValueError('Parent model not verified against completed original release')
        if manifest['accounting']!=model['accounting'] or Fraction(model['accounting']['epsilon_exact'])!=epsilon:
            raise ValueError('Parent privacy metadata mismatch')
        models.append(model)
        parents.append({'epsilon_exact':str(epsilon),'model_sha256':manifest['model_sha256'],
                        'manifest_sha256':sha(path/'manifest.json')})
    output=HERE/'joint14'
    if output.exists():
        raise FileExistsError('Joint candidate already exists; never overwrite or select a new seed using private scores')
    freeze={'parent_epsilon_exact':'14','additional_epsilon_exact':'0','delta':0,
        'parents':parents,'settings':config['settings'],'code_sha256':study['code_sha256'],
        'runner_sha256':sha(Path(__file__)),'public_selection_sha256':sha(HERE/'public_pipeline_study.json'),
        'config_sha256':sha(HERE/'joint14_config.json'),
        'release_rule':'Generate exactly this public-selected candidate, before any private scoring; preserve regardless of subsequent diagnostic score'}
    output.mkdir()
    (output/'freeze.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
    generated,model,original=construct(models,config['settings'])
    predictor_cols=[c for c in original.columns if c not in ['time','onset','death']]
    pd.testing.assert_frame_equal(generated[predictor_cols],original[predictor_cols],check_exact=True)
    schema=model['histogram_model']['schema']; detector=schema['rare_detector']
    strata=dc._radii(original,detector)>=float(detector['band']['lo'])
    outcome=['time','onset','death']
    for value in (False,True):
        expected=original.loc[strata==value,outcome].sort_values(outcome).reset_index(drop=True)
        observed=generated.loc[strata==value,outcome].sort_values(outcome).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,observed,check_exact=True)
    if len(generated)!=1049 or list(generated.columns)!=schema['submitted_c_columns'] or generated.isna().any().any():
        raise ValueError('Synthetic format invariant failed')
    generated.to_csv(output/'C.csv',index=False,encoding='utf-8',lineterminator='\n')
    (output/'model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    manifest={**freeze,'C_sha256':sha(output/'C.csv'),'model_sha256':sha(output/'model.json'),
        'public_n':1049,'predictor_rows_preserved_exactly':True,'outcome_triples_preserved_per_rare_stratum':True,
        'private_queries':0,'private_scores_used_for_selection':False,
        'privacy_scope':'Joint postprocessing of the existing epsilon1+3+10 releases; total still14, not epsilon10'}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'candidate':str(output/'C.csv'),'parent_epsilon':14,'additional_epsilon':0,
                     'C_sha256':manifest['C_sha256'],'optimizer_residual':model['optimizer_trace']['selected_coefficient_l2_error']}))

if __name__=='__main__':main()

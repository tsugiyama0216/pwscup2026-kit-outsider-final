"""Predeclared one-coefficient ordinal hypothesis diagnostic.

This never changes official scoring, generation, predictions or submitted values.
Each fold fits only three development-model pairs, then checks the omitted model
against those three on BOTH reserved confirmation cases. It is retrospective
validation on known reference models, not unknown-model performance estimation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TIE = Fraction(1, 10**12)
PROTOCOL = {
    'version':'ordinal-one-coefficient-lomo-v1',
    'frozen_utc':'2026-10-09T04:48:22.717714+00:00',
    'model_count':4,
    'development_seeds':[2026100941],
    'confirmation_seeds':[2026100951,2026100952],
    'indicator':'alpha * mean(mia_power) + (1-alpha) * mean(aia_power_after_floor)',
    'component_aggregation':'Equal weight over the same 22 targets, using existing official scorer components after AIA floor correction.',
    'fit':'Leave one model out; fit exactly 3 development models using all 3 strict pairwise inequalities from their known official relative order. Do not pass the omitted model to fit_alpha.',
    'alpha_domain':[0,1],
    'interval':'Intersect pairwise linear strict inequalities exactly using rational representations of the decimal input values.',
    'selection':'If 0.5 lies strictly inside the feasible interval use 0.5; otherwise choose the midpoint between the nearest interval boundary and the interval midpoint. Never choose an open boundary.',
    'impossible':'Return impossible for empty feasible interval. Preserve and report that fold; no fallback coefficient or feature.',
    'numerical_margin':'A fitted development difference at or below 1e-12 is numerically unsupported; confirmation absolute difference at or below 1e-12 is a tie and never a hit.',
    'confirmation':'For each fold score omitted-versus-each-training-model on both confirmation cases separately: 6 relations. Preserve both cases, no pooled rescue or case selection.',
    'controls':'Report raw MIA+AIA and alpha=0.5 relations alongside calibrated indicator; no official score replacement.',
    'forbidden':['No four-model all-fit coefficient','No confirmation-based alpha selection','No target weights','No nonlinear features','No generation or scorer changes'],
    'interpretation':'Known-reference retrospective hypothesis diagnostic only; overlapping folds/pairs are not independent observations and successful folds do not establish unknown-model accuracy or hidden official weights.',
}


def protocol_hash():
    return hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _fraction(value):
    if isinstance(value,bool):
        raise ValueError('Boolean is not a score')
    if isinstance(value,Fraction):
        return value
    try:
        if not math.isfinite(float(value)):
            raise ValueError('Nonfinite score')
        return Fraction(str(value))
    except (TypeError,ValueError,ZeroDivisionError) as error:
        raise ValueError('Expected finite numeric score') from error


def _components(rows):
    result={}
    for model,row in rows.items():
        if not isinstance(model,str) or not model:
            raise ValueError('Model names must be nonempty strings')
        mia,aia=_fraction(row['mia']),_fraction(row['aia'])
        if not (0<=mia<=1 and 0<=aia<=1):
            raise ValueError('Corrected component powers must be in [0,1]')
        result[model]={'mia':mia,'aia':aia}
    return result


def _order(order,models):
    order=list(order)
    if len(order)!=len(set(order)) or set(order)!=set(models):
        raise ValueError('A complete strict official order with the same unique model set is required')
    return order


def feasible_interval(constraints):
    """Intersect c+s*alpha>0 with [0,1], preserving strict bounds exactly."""
    low,high=Fraction(0),Fraction(1)
    low_open=high_open=False
    for constraint in constraints:
        c,s=_fraction(constraint['intercept']),_fraction(constraint['slope'])
        if s==0:
            if c<=0:
                return {'feasible':False,'reason':'constant_nonstrict_or_reversed_pair'}
            continue
        boundary=-c/s
        if s>0:
            if boundary>low:
                low,low_open=boundary,True
            elif boundary==low:
                low_open=True
        else:
            if boundary<high:
                high,high_open=boundary,True
            elif boundary==high:
                high_open=True
    if low>high or (low==high and (low_open or high_open)):
        return {'feasible':False,'reason':'empty_strict_interval'}
    return {'feasible':True,'lower':low,'upper':high,'lower_open':low_open,'upper_open':high_open}


def select_alpha(interval):
    if not interval['feasible']:
        return None
    low,high=interval['lower'],interval['upper']
    half=Fraction(1,2)
    if low<half<high:
        return half
    middle=(low+high)/2
    nearest=low if half<=low else high
    selected=(nearest+middle)/2
    if not (low<selected<high):
        raise ValueError('No strict interior selection is available')
    return selected


def indicator(row,alpha):
    return alpha*row['mia']+(1-alpha)*row['aia']


def fit_alpha(training,official_training_order):
    """This API intentionally accepts only the three fitted models."""
    if len(training)!=3:
        raise ValueError('Fit exactly three models; a four-model fit is prohibited')
    training=_components(training)
    order=_order(official_training_order,training)
    constraints=[]
    for better,worse in combinations(order,2):
        a,b=training[better],training[worse]
        c=a['aia']-b['aia']
        slope=a['mia']-b['mia']-c
        constraints.append({'better':better,'worse':worse,'intercept':c,'slope':slope})
    interval=feasible_interval(constraints)
    alpha=select_alpha(interval)
    margins=[] if alpha is None else [x['intercept']+x['slope']*alpha for x in constraints]
    if margins and not all(x>0 for x in margins):
        raise AssertionError('Selected alpha violates a strict fitted pair')
    supported=alpha is not None and all(x>TIE for x in margins)
    return {'training_models':order,'constraints':constraints,'interval':interval,'alpha':alpha,
        'development_margins':margins,'numerically_supported':supported,
        'status':'impossible' if alpha is None else ('feasible' if supported else 'insufficient_numerical_margin')}


def compare_heldout(rows,heldout,training_models,official_order,alpha):
    rows=_components(rows)
    position={name:i for i,name in enumerate(_order(official_order,rows))}
    pairs=[]
    for other in training_models:
        expected=1 if position[heldout]<position[other] else -1
        delta=indicator(rows[heldout],alpha)-indicator(rows[other],alpha)
        observed=1 if delta>TIE else -1 if delta < -TIE else 0
        pairs.append({'heldout':heldout,'other':other,'expected_direction':expected,
            'observed_direction':observed,'delta_heldout_minus_other':delta,
            'tie':observed==0,'agrees':observed==expected and observed!=0})
    return {'pairs':pairs,'correct':sum(x['agrees'] for x in pairs),'total':len(pairs),
        'all_correct':bool(pairs) and all(x['agrees'] for x in pairs)}


def diagnose(development,confirmations,official_order):
    """Fit on development only; each confirmation is evaluated without refitting."""
    development=_components(development)
    if len(development)!=PROTOCOL['model_count'] or len(confirmations)!=2:
        raise ValueError('Expected four reference models and exactly two confirmation cases')
    order=_order(official_order,development)
    confirmation_rows=[]
    for rows in confirmations:
        normalized=_components(rows)
        _order(order,normalized)
        confirmation_rows.append(normalized)
    folds=[]
    for heldout in order:
        training_names=[name for name in order if name!=heldout]
        fit=fit_alpha({name:development[name] for name in training_names},training_names)
        checks=[]
        for index,rows in enumerate(confirmation_rows):
            baseline=compare_heldout(rows,heldout,training_names,order,Fraction(1,2))
            check=None if fit['alpha'] is None else compare_heldout(rows,heldout,training_names,order,fit['alpha'])
            # Raw official formula is MIA+AIA. At alpha=.5 ranks coincide, but
            # report raw gaps and apply the SAME absolute tie tolerance separately.
            raw_pairs=[]
            for p in baseline['pairs']:
                p=dict(p); p['delta_heldout_minus_other']*=2
                delta=p['delta_heldout_minus_other']
                p['observed_direction']=1 if delta>TIE else -1 if delta < -TIE else 0
                p['tie']=p['observed_direction']==0
                p['agrees']=p['observed_direction']==p['expected_direction'] and not p['tie']
                raw_pairs.append(p)
            checks.append({'confirmation_index':index,'calibrated':check,'alpha_half':baseline,
                'raw_mia_plus_aia':{'pairs':raw_pairs,'correct':sum(p['agrees'] for p in raw_pairs),'total':3,
                    'all_correct':all(p['agrees'] for p in raw_pairs)}})
        folds.append({'heldout_model':heldout,'fit':fit,'confirmation_checks':checks,
            'confirmation_correct':sum(x['calibrated']['correct'] if x['calibrated'] else 0 for x in checks),
            'confirmation_total':6,'passes_both_confirmations':fit['numerically_supported'] and all(x['calibrated']['all_correct'] for x in checks)})
    return {'format':PROTOCOL['version'],'protocol':PROTOCOL,'protocol_sha256':protocol_hash(),
        'official_relative_order':order,'official_scores_changed':False,'unknown_model_accuracy_claim':False,
        'all_folds_retained':True,'folds':folds,'passed_folds':sum(x['passes_both_confirmations'] for x in folds),
        'all_four_folds_pass':all(x['passes_both_confirmations'] for x in folds),
        'confirmation_pair_hits':sum(x['confirmation_correct'] for x in folds),'confirmation_pair_count':24,
        'pair_count_note':'24 overlapping checks, not 24 independent observations; impossible folds count as failures.'}


def _jsonable(value):
    if isinstance(value,Fraction):
        return {'decimal':float(value),'exact':str(value)}
    if isinstance(value,dict):
        return {key:_jsonable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new_output(output,report):
    output=Path(output).resolve()
    base=(ROOT/'main_round/mock_test/local-output').resolve()
    if output.exists() or output==base or not output.is_relative_to(base):
        raise ValueError('Output must be a new child of mock_test/local-output')
    output.mkdir(parents=True)
    (output/'ordinal_calibration.json').write_text(json.dumps(_jsonable(report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return output


def from_panels(paths,official_order,generation_protocol):
    # Deferred import: pure interval tests/freeze never open prediction or score files.
    from rank_validation.joint_report import load_panel
    protocol_path=Path(generation_protocol)
    generation=json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    for key in ('development_seeds','confirmation_seeds'):
        if generation[key]!=PROTOCOL[key]:
            raise ValueError('Generation seeds differ from the preregistered ordinal protocol')
    reserved={PROTOCOL['development_seeds'][0]:'development',**{s:'confirmation' for s in PROTOCOL['confirmation_seeds']}}
    rows_by_seed={}; provenance=[]
    for path in paths:
        state,rows,_=load_panel(path)
        meta=state['case_metadata']; seed=meta['seed']
        if seed not in reserved or seed in rows_by_seed or meta['reserved_seed_role']!=reserved[seed]:
            raise ValueError('Duplicate, unexpected or incorrectly labeled scenario')
        if meta['protocol_sha256']!=_sha(protocol_path):
            raise ValueError('Case generation protocol mismatch')
        rows_by_seed[seed]={r['model']:{'mia':r['mia'],'aia':r['aia']} for r in rows}
        provenance.append({'panel':str(Path(path).resolve()),'seed':seed,'case_sha256':state['case_snapshot']['manifest'],
            'generation_protocol_sha256':meta['protocol_sha256']})
    if set(rows_by_seed)!=set(reserved):
        raise ValueError('All three reserved cases are required')
    result=diagnose(rows_by_seed[PROTOCOL['development_seeds'][0]],
        [rows_by_seed[s] for s in PROTOCOL['confirmation_seeds']],official_order)
    result['inputs']=provenance
    result['created_utc']=datetime.now(timezone.utc).isoformat()
    result['source_sha256']=_sha(__file__)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true',help='Save procedure/source hashes without reading any model results')
    parser.add_argument('--panel',type=Path,action='append')
    parser.add_argument('--official-order',nargs=4,help='Known strict reference order, best first; no numeric official scores needed')
    parser.add_argument('--generation-protocol',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.freeze:
        if args.panel or args.official_order or args.generation_protocol:
            parser.error('freeze must not receive score or official-order inputs')
        report={'protocol':PROTOCOL,'protocol_sha256':protocol_hash(),'source_sha256':_sha(__file__),
            'freeze_record_utc':datetime.now(timezone.utc).isoformat(),'model_results_read':False}
    else:
        if not args.panel or len(args.panel)!=3 or not args.official_order or not args.generation_protocol:
            parser.error('diagnosis requires three panels, official order and generation protocol')
        report=from_panels(args.panel,args.official_order,args.generation_protocol)
    output=_write_new_output(args.output,report)
    print(str(output/'ordinal_calibration.json'))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

"""One fixed extra-budget measurement after public-only epsilon selection.

No scoring import, raw coefficient output, or private retry is allowed here.
"""
from pathlib import Path
import sys,json,hashlib
from fractions import Fraction
import pandas as pd

HERE=Path(__file__).resolve().parent; CORE=HERE.parent; V2=CORE/'postprocessing_v2'
TEAM=CORE.parents[1]
sys.path[:0]=[str(CORE),str(V2)]
import dp_copula as dc
from release import _atomic_json,_exclusive_run,_now,_canonical,ReleaseBlocked
from dp_cox_target import build_public_target,measure_target
from target_calibration import calibrate_target


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve_and_measure(folder,freeze,producer,*,ledger_dir):
    """Crash-safe at-most-one sensitive producer invocation for this release."""
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    ledger_dir=Path(ledger_dir); ledger_dir.mkdir(parents=True,exist_ok=True)
    frozen=folder/'freeze.json'; reservation=folder/'reservation.json'
    durable_reservation=ledger_dir/'reservation.json'
    model_path=folder/'model.json'; completed=folder/'measurement_complete.json'
    if frozen.exists() and read(frozen)!=freeze:
        raise ReleaseBlocked('Freeze changed; moving files does not restore spent budget')
    if not frozen.exists(): _atomic_json(frozen,freeze,exclusive=True)
    if durable_reservation.exists():
        if not (reservation.exists() and model_path.exists() and completed.exists()):
            raise ReleaseBlocked('Reserved budget is spent; no saved complete model, do not retry private fit')
        if read(durable_reservation)!=read(reservation) or read(reservation)['freeze_sha256']!=sha(frozen):
            raise ReleaseBlocked('Reservation identity mismatch')
        record=read(completed)
        if record['freeze_sha256']!=sha(frozen) or record['model_sha256']!=sha(model_path):
            raise ReleaseBlocked('Saved model hash mismatch')
        return read(model_path),True
    if reservation.exists() or model_path.exists() or completed.exists():
        raise ReleaseBlocked('Unexpected output without reservation')
    record={'reserved_utc':_now(),'freeze_sha256':sha(frozen),
        'additional_epsilon_exact':freeze['additional_epsilon_exact'],
        'composed_epsilon_exact':freeze['composed_epsilon_exact'],'delta':0,
        'failure_is_spent':True,'output_directory':str(folder.resolve())}
    _atomic_json(durable_reservation,record,exclusive=True)
    _atomic_json(reservation,record,exclusive=True)
    # The reservation is durable BEFORE this callback can read private rows.
    model=producer()
    _atomic_json(model_path,model,exclusive=True)
    _atomic_json(completed,{'completed_utc':_now(),'freeze_sha256':sha(frozen),
        'model_sha256':sha(model_path)},exclusive=True)
    return model,False


def check_public_selection(config):
    study=read(HERE/'public_budget_study.json')
    if not study['complete'] or sha(HERE/'public_budget_study.json')!=config['public_selection_sha256']:
        raise ReleaseBlocked('Public study missing or altered')
    epsilon=Fraction(config['additional_epsilon_exact'])
    eligible=[r for r in study['summaries'] if r['n']==6 and r['mean_U_spec_public']>=.95 and r['minimum_U_spec_public']>=.92]
    if not eligible or epsilon!=eligible[0]['additional_epsilon'] or epsilon!=study['selected_additional_epsilon']:
        raise ReleaseBlocked('Budget does not match first qualifying public plan')
    if config['plan']!=study['plan'] or Fraction(config['previous_epsilon_exact'])!=14:
        raise ReleaseBlocked('Changed public settings or previous accounting')
    if Fraction(config['composed_epsilon_exact'])!=14+epsilon or config['delta']!=0 or config['public_n']!=1049:
        raise ReleaseBlocked('Incorrect composition or public size')
    for name,h in study['code_sha256'].items():
        if sha(CORE/name)!=h: raise ReleaseBlocked('Public-selected source changed')
    return study,epsilon


def assert_preservation(before,after,detector):
    outcomes=['time','onset','death']; predictors=[c for c in before.columns if c not in outcomes]
    pd.testing.assert_frame_equal(before[predictors],after[predictors],check_exact=True)
    strata=dc._radii(before,detector)>=float(detector['band']['lo'])
    for group in (False,True):
        a=before.loc[strata==group,outcomes].sort_values(outcomes).reset_index(drop=True)
        b=after.loc[strata==group,outcomes].sort_values(outcomes).reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True)


def run():
    config=read(HERE/'expanded_config.json'); study,epsilon=check_public_selection(config)
    data=TEAM/'participant_data'; output=HERE/'release'; base=V2/'joint14'
    original_events=[json.loads(x) for x in (CORE/'release_ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    reserved=[r for r in original_events if r['event']=='reserved']
    if len(reserved)!=3 or sum(Fraction(r['accounting']['epsilon_exact']) for r in reserved)!=14:
        raise ReleaseBlocked('Original ledger is inconsistent')
    base_manifest=read(base/'manifest.json')
    if base_manifest['parent_epsilon_exact']!='14' or base_manifest['additional_epsilon_exact']!='0':
        raise ReleaseBlocked('Base accounting mismatch')
    if sha(base/'C.csv')!=base_manifest['C_sha256'] or sha(base/'model.json')!=base_manifest['model_sha256']:
        raise ReleaseBlocked('Base output changed')
    for name,h in study['public_input_sha256'].items():
        if sha(data/name)!=h: raise ReleaseBlocked('Public input changed')
    sources={**study['code_sha256'],'expanded_budget/expanded_release.py':sha(Path(__file__)),
             'release.py':sha(CORE/'release.py')}
    freeze={'format':'S_Copula_expanded_budget_v1','additional_epsilon_exact':str(epsilon),
            'previous_epsilon_exact':'14','composed_epsilon_exact':str(14+epsilon),'delta':0,
            'public_n':1049,'data_directory':str(data.resolve()),'output_directory':str(output.resolve()),
            'config_sha256':sha(HERE/'expanded_config.json'),'source_sha256':sources,
            'public_input_sha256':study['public_input_sha256'],
            'public_selection_sha256':sha(HERE/'public_budget_study.json'),
            'base_manifest_sha256':sha(base/'manifest.json'),'base_C_sha256':sha(base/'C.csv'),
            'original_ledger_sha256':sha(CORE/'release_ledger/events.jsonl'),
            'selection_rule':'One public-selected budget and fixed candidate; preserve regardless of private diagnostics'}
    with _exclusive_run(HERE/'release_ledger'):
        public=pd.read_csv(data/'A_bg.csv'); schema=read(data/'schema.json')
        categories=sorted(public.prefecture.unique())
        defaults={c:float(public[c].median()) for c in dc.NUMERIC}
        clean_public=dc._prepare(public,schema,categories,defaults)
        public_target=build_public_target(clean_public,public_n=1049)
        plan=config['plan']
        def producer():
            private=pd.read_csv(data/'B.csv')
            prepared=dc._prepare(private,schema,categories,defaults)
            target=measure_target(prepared,public_target,epsilon=epsilon,radius=plan['radius'],
                                  quantization=plan['quantization'],se_weight=plan['se_weight'])
            return {'format':'S_Copula_bounded_Cox_target_release_v1','public_target':public_target,
                    'target':target,'accounting':{'additional_epsilon_exact':str(epsilon),
                    'previous_epsilon_exact':'14','composed_epsilon_exact':str(14+epsilon),'delta':0},
                    'selection_inputs':'Public fixed settings; exact integer DP output; no private diagnostics'}
        model,reused=reserve_and_measure(output,freeze,producer,ledger_dir=HERE/'release_ledger')
        if (output/'manifest.json').exists():
            manifest=read(output/'manifest.json')
            if sha(output/'C.csv')!=manifest['C_sha256'] or sha(output/'model.json')!=manifest['model_sha256']:
                raise ReleaseBlocked('Completed output changed')
            expected={**model['accounting'],'public_n':1049,'freeze_sha256':sha(output/'freeze.json'),
                      'base_C_sha256':freeze['base_C_sha256']}
            if any(manifest.get(k)!=v for k,v in expected.items()):
                raise ReleaseBlocked('Completed manifest identity or accounting mismatch')
            return {**model['accounting'],'reused_saved_model':True,'C_sha256':manifest['C_sha256']}
        before=pd.read_csv(base/'C.csv')
        candidate,trace=calibrate_target(before,model['target'],model['public_target'],schema['rare_detector'],
            seed=plan['postprocess_seed'],iterations=plan['iterations'],candidates=plan['candidates'],trials=plan['trials'])
        assert_preservation(before,candidate,schema['rare_detector'])
        if candidate.isna().any().any() or len(candidate)!=1049:
            raise ReleaseBlocked('Synthetic postprocessing validation failed; private budget remains spent')
        # Recreating a partial C is deterministic postprocessing, not a new measurement.
        candidate.to_csv(output/'C.csv',index=False)
        manifest={**model['accounting'],'public_n':1049,'freeze_sha256':sha(output/'freeze.json'),
            'model_sha256':sha(output/'model.json'),'C_sha256':sha(output/'C.csv'),
            'base_C_sha256':freeze['base_C_sha256'],'optimizer_trace':trace,
            'predictor_rows_preserved_exactly':True,'outcome_triples_preserved_per_rare_stratum':True,
            'private_measurement_count':1,'private_score_used_for_selection':False,
            'privacy_scope':'Saved target and C only; diagnostics and R outputs excluded'}
        _atomic_json(output/'manifest.json',manifest,exclusive=True)
        return {**model['accounting'],'reused_saved_model':reused,'C_sha256':manifest['C_sha256']}


if __name__=='__main__': print(json.dumps(run(),ensure_ascii=False))

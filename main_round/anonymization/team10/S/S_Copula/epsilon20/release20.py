"""One reservation, one standalone epsilon20 fit; no evaluator import."""
from pathlib import Path
import sys,json,hashlib
from fractions import Fraction
import pandas as pd
HERE=Path(__file__).resolve().parent;CORE=HERE.parent;TEAM=CORE.parents[1]
sys.path[:0]=[str(CORE),str(CORE/'expanded_budget')]
from release import _atomic_json,_exclusive_run,ReleaseBlocked
from expanded_release import reserve_and_measure
from pipeline20 import fit_once,construct
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES=['dp_copula.py','dp_cox.py','release.py','postprocessing_v2/histogram_refine.py',
         'postprocessing_v2/cox_optimizer.py','expanded_budget/dp_cox_target.py',
         'expanded_budget/target_calibration.py','expanded_budget/expanded_release.py',
         'epsilon20/influence_cox.py','epsilon20/pipeline20.py','epsilon20/release20.py']

def run():
    config=read(HERE/'config20.json');study=read(HERE/'public_final_check.json')
    if not study['complete'] or config['public_selection_sha256']!=sha(HERE/'public_final_check.json') or config['plan']!=study['plan']:
        raise ReleaseBlocked('Public selection mismatch')
    if (config['standalone_epsilon_exact'],config['additional_epsilon_exact'],config['previous_epsilon_exact'],config['composed_epsilon_exact'],config['delta'])!=('20','20','414','434',0):
        raise ReleaseBlocked('Unexpected accounting; cannot increase epsilon')
    plan=config['plan']
    if {n:sha(CORE/n) for n in SOURCES}!=config['source_sha256']:
        raise ReleaseBlocked('Frozen source changed; no private measurement allowed')
    if Fraction(plan['histogram_epsilon'])+sum(map(Fraction,plan['cox_round_epsilons']))!=20:
        raise ReleaseBlocked('Allocation does not sum to20')
    data=TEAM/'participant_data';output=HERE/'release';ledger=HERE/'release_ledger'
    for n,h in study['public_input_sha256'].items():
        if sha(data/n)!=h:raise ReleaseBlocked('Changed public input')
    for n,h in study['selection_evidence_sha256'].items():
        if sha(HERE/n)!=h:raise ReleaseBlocked('Changed public selection evidence')
    # Historical spent budget remains recorded even after old outputs were removed.
    events=[json.loads(x) for x in (CORE/'release_ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    old=sum(Fraction(r['accounting']['epsilon_exact']) for r in events if r['event']=='reserved')
    deleted=read(CORE/'expanded_budget/release_ledger/reservation.json')
    if old!=14 or Fraction(deleted['additional_epsilon_exact'])!=400 or Fraction(deleted['composed_epsilon_exact'])!=414:
        raise ReleaseBlocked('Historical ledger mismatch')
    freeze={**{k:config[k] for k in ('standalone_epsilon_exact','additional_epsilon_exact','previous_epsilon_exact','composed_epsilon_exact','delta','public_n')},
            'format':'S_Copula_standalone20_freeze_v1','config_sha256':sha(HERE/'config20.json'),
            'source_sha256':{n:sha(CORE/n) for n in SOURCES},
            'public_input_sha256':study['public_input_sha256'],
            'public_selection_sha256':config['public_selection_sha256'],
            'data_directory':str(data.resolve()),'output_directory':str(output.resolve()),
            'no_previous_DP_numeric_models':True,'one_candidate_no_private_selection':True}
    with _exclusive_run(ledger):
        public=pd.read_csv(data/'A_bg.csv');schema=read(data/'schema.json')
        def producer():
            # Only executed AFTER the durable reservation, never on resume.
            private=pd.read_csv(data/'B.csv')
            return fit_once(private,public,schema,plan)
        model,reused=reserve_and_measure(output,freeze,producer,ledger_dir=ledger)
        if (output/'manifest.json').exists():
            manifest=read(output/'manifest.json')
            if sha(output/'C.csv')!=manifest['C_sha256'] or sha(output/'model.json')!=manifest['model_sha256'] or manifest['freeze_sha256']!=sha(output/'freeze.json'):
                raise ReleaseBlocked('Saved output integrity mismatch')
        else:
            candidate,trace=construct(model)
            if len(candidate)!=1049 or candidate.isna().any().any():
                raise ReleaseBlocked('Invalid synthetic postprocessing; no new private query allowed')
            candidate.to_csv(output/'C.csv',index=False)
            manifest={**model['accounting'],'previous_epsilon_exact':'414','cumulative_spent_epsilon_exact':'434',
                'model_sha256':sha(output/'model.json'),'C_sha256':sha(output/'C.csv'),
                'freeze_sha256':sha(output/'freeze.json'),'optimizer_trace':trace,
                'no_previous_DP_numeric_models':True,'private_score_used_for_selection':False,
                'privacy_scope':'DP model and C only; non-DP diagnostics and internal operational metadata excluded'}
            _atomic_json(output/'manifest.json',manifest,exclusive=True)
        return {'standalone_epsilon':20,'cumulative_spent_epsilon':434,'reused_saved_model':reused,'C_sha256':manifest['C_sha256']}
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))

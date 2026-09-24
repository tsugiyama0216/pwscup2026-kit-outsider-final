"""At-most-once private fit under the user-authorized epsilon15 budget."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
CORE = HERE.parent
TEAM = CORE.parents[1]
sys.path[:0] = [str(HERE), str(CORE), str(CORE / 'expanded_budget')]
from release import _atomic_json, _exclusive_run, ReleaseBlocked
from expanded_release import reserve_and_measure
from pipeline15 import fit_once, construct


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight():
    config = read(HERE / 'config15.json')
    expected = {'standalone_epsilon_exact': '15', 'additional_epsilon_exact': '15',
                'previous_epsilon_exact': '434', 'composed_epsilon_exact': '449',
                'delta': 0, 'public_n': 1049}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ReleaseBlocked('Unexpected accounting')
    old_config = read(CORE / 'epsilon20/config20.json')
    plan = dict(old_config['plan'])
    plan['histogram_epsilon'] = '3'
    plan['cox_round_epsilons'] = ['12/5', '18/5', '6']
    if config['plan'] != plan or sha(CORE / 'epsilon20/config20.json') != config['parent_public_config_sha256']:
        raise ReleaseBlocked('Only the precommitted epsilon20 budget scaling is allowed')
    if Fraction(plan['histogram_epsilon']) + sum(map(Fraction, plan['cox_round_epsilons'])) != 15:
        raise ReleaseBlocked('Allocation must sum to15')
    if any(sha(CORE / name) != value for name, value in config['source_sha256'].items()):
        raise ReleaseBlocked('Frozen source changed')
    data = TEAM / 'participant_data'
    if any(sha(data / name) != value for name, value in config['public_input_sha256'].items()):
        raise ReleaseBlocked('Public input changed')
    events = [json.loads(x) for x in (CORE / 'release_ledger/events.jsonl').read_text(encoding='utf-8').splitlines()]
    initial = sum(Fraction(r['accounting']['epsilon_exact']) for r in events if r['event'] == 'reserved')
    old400 = read(CORE / 'expanded_budget/release_ledger/reservation.json')
    old20 = read(CORE / 'epsilon20/release_ledger/reservation.json')
    if initial != 14 or old400['additional_epsilon_exact'] != '400' or old20['additional_epsilon_exact'] != '20' or old20['composed_epsilon_exact'] != '434':
        raise ReleaseBlocked('Historical ledger mismatch')
    output = HERE / 'release'
    freeze = {**expected, 'format': 'S_Copula_standalone15_freeze_v1',
              'config_sha256': sha(HERE / 'config15.json'),
              'source_sha256': config['source_sha256'],
              'public_input_sha256': config['public_input_sha256'],
              'data_directory': str(data.resolve()), 'output_directory': str(output.resolve()),
              'no_previous_DP_numeric_models': True, 'one_candidate_no_private_selection': True}
    return config, freeze, data, output


def run():
    config, freeze, data, output = preflight()
    ledger = HERE / 'release_ledger'
    with _exclusive_run(ledger):
        public = pd.read_csv(data / 'A_bg.csv')
        schema = read(data / 'schema.json')
        def producer():
            # Read private rows only after durable reservation.
            private = pd.read_csv(data / 'B.csv')
            return fit_once(private, public, schema, config['plan'])
        model, reused = reserve_and_measure(output, freeze, producer, ledger_dir=ledger)
        if (output / 'manifest.json').exists():
            manifest = read(output / 'manifest.json')
            if sha(output / 'C.csv') != manifest['C_sha256'] or sha(output / 'model.json') != manifest['model_sha256'] or manifest['freeze_sha256'] != sha(output / 'freeze.json'):
                raise ReleaseBlocked('Saved output integrity mismatch')
        else:
            candidate, trace = construct(model)
            if len(candidate) != 1049 or candidate.isna().any().any():
                raise ReleaseBlocked('Invalid postprocessing; no new private fit allowed')
            candidate.to_csv(output / 'C.csv', index=False)
            manifest = {**model['accounting'], 'previous_epsilon_exact': '434',
                        'cumulative_spent_epsilon_exact': '449',
                        'model_sha256': sha(output / 'model.json'), 'C_sha256': sha(output / 'C.csv'),
                        'freeze_sha256': sha(output / 'freeze.json'), 'optimizer_trace': trace,
                        'no_previous_DP_numeric_models': True, 'private_score_used_for_selection': False,
                        'privacy_scope': 'DP model and C only; non-DP diagnostics and operational metadata excluded'}
            _atomic_json(output / 'manifest.json', manifest, exclusive=True)
        return {'standalone_epsilon': 15, 'cumulative_spent_epsilon': 449,
                'reused_saved_model': reused, 'C_sha256': manifest['C_sha256']}


if __name__ == '__main__':
    print(json.dumps(run()))

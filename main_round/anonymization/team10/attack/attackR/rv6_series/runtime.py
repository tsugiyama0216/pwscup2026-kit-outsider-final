"""Production I/O and dependency checks; the numerical model does not import this."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

HERE = Path(__file__).resolve().parent
SERIES = HERE.parent
ROOT = HERE.parents[5]
ATTACK = ROOT / 'main_round/attack'
sys.path.insert(0, str(ATTACK))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def sha(path):
    if Path(path).name.lower() == 'token.txt':
        raise ValueError('Do not hash tokens')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify():
    files = read(HERE / 'dependencies.json')['files']
    for name, expected in files.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError('Frozen dependency changed: ' + name)
    return len(files)


def actual():
    common = load('_rv6_common', ATTACK / 'Attack_R/v1/common.py')
    return common.load_actual()


def baseline_and_mask(actual_data):
    import pandas as pd
    w, _, _, _, _, _, b, _, cs, qs, _, _ = actual_data
    saved = ATTACK / 'Attack_R/v2/submission'
    oldm = pd.read_csv(saved / 'F_mia.csv', float_precision='round_trip').set_index('record_id').loc[b.record_id]
    olda = pd.read_csv(saved / 'F_aia.csv', float_precision='round_trip')
    mia = {k: oldm[str(k)].to_numpy() for k in cs}
    aia = {k: olda[olda.target_k == k].set_index('challenge_row_id').loc[qs[k].challenge_row_id].time_hat.to_numpy() for k in cs}
    maskmod = load('_rv6_mask', ATTACK / 'Attack_M/v1/actual.py')
    schema = read(w.ORIGINALS / 'schema.json')
    strings = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    mask, proof = maskmod.derive_private_mask(strings(w.ORIGINALS / 'B.csv'), strings(w.ORIGINALS / 'B_self.csv'),
        strings(w.TARGETS / 'aia_challenge_10.csv'), schema['aia_qi_columns'], schema['submitted_c_columns'])
    return mia, aia, mask, proof

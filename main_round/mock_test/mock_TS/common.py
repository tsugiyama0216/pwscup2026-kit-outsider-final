"""Anchored paths and serialization; no original inputs are written here."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MOCK_ROOT = HERE.parent
ROOT = HERE.parents[2]
ATTACK = ROOT / 'main_round/attack'
PARTICIPANT = ROOT / 'main_round/anonymization/team10/participant_data'
OUTPUT_ROOT = MOCK_ROOT / 'local-output/mock_TS'
KIT = ATTACK / 'official-kit/starter'
PACKAGE = KIT / 'src/pwscup2026'
FEATURES = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT',
            'smoking', 'FPG', 'time', 'onset', 'death']
QI = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
TARGET_ID = 0

# The pinned official source is imported without initializing legacy M/T code.
sys.path.insert(0, str(KIT / 'src'))
for name in ('pwscup2026', 'pwscup2026.scoring.attack_score',
             'pwscup2026.kit.codabench.scoring_io'):
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(PACKAGE.resolve()):
        raise ImportError(f'Expected pinned official package: {name}')


def cfg():
    spec = importlib.util.spec_from_file_location('_mock_ts_official_reference', KIT / 'reference/_common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_cfg()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def rare_mask(frame, schema):
    d = schema['rare_detector']
    diff = frame[d['columns']].to_numpy(float) - np.asarray(d['mu'])
    inv = np.linalg.pinv(np.asarray(d['sigma']))
    distance = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', diff, inv, diff), 0, None))
    return distance >= float(d['band']['lo'])


def ensure_new_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(OUTPUT_ROOT.resolve()) and path != OUTPUT_ROOT.resolve(),
            'Outputs must be a new directory under main_round/mock_test/local-output/mock_TS')
    require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
            'Refusing to overwrite a nonempty output directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def within(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), 'Manifest path must be relative')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Manifest path escapes root')
    return path


def file_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}


def verify_files(root, entries):
    require(isinstance(entries, dict) and bool(entries), 'Empty file manifest')
    for relative, digest in entries.items():
        path = within(root, relative)
        require(path.is_file() and sha256(path) == digest, f'Input changed: {relative}')


def code_hashes():
    paths = list(HERE.glob('*.py')) + [KIT / 'reference/_common.py',
        PACKAGE / 'kit/kit_config.yaml', PACKAGE / 'kit/codabench/scoring_io.py',
        PACKAGE / 'scoring/attack_score.py', PACKAGE / 'attack/aia.py', PACKAGE / 'attack/mia.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(paths)}


def verify_originals():
    """Check the existing official manifest without reading any credential."""
    manifest = read_json(ATTACK / 'source_manifest.json')
    for item in manifest['files']:
        path = (ATTACK / item['path']).resolve()
        require(path.is_relative_to((ROOT / 'main_round').resolve()), 'Original manifest escapes main_round')
        require(path.name.lower() != 'token.txt', 'Credentials do not belong in source verification')
        require(path.is_file() and sha256(path) == item['sha256'], f"Original hash mismatch: {item['path']}")
    return {'verified_files': len(manifest['files']),
            'manifest_sha256': sha256(ATTACK / 'source_manifest.json')}


def public_tables(directory):
    directory = Path(directory)
    tables = {'candidates': read_csv(directory / 'B.csv'),
              'background': read_csv(directory / 'A_bg.csv'),
              'challenge': read_csv(directory / 'aia_challenge_0.csv'),
              'schema': read_json(directory / 'schema.json'),
              'cfg': read_json(directory / 'cfg.json')}
    if (directory / 'C.csv').is_file():
        tables['c'] = read_csv(directory / 'C.csv')
    return tables

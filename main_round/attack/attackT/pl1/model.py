"""Preliminary algorithms adapted to main-round public tables, without target-ID routing."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ATTACK = HERE.parents[1]
for path in (ATTACK, ATTACK / 'Attak_T'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import attack_research  # Initializes the canonical shared compatibility runtime.
from prelim_transfer import methods

MODEL_NAME = 'attackT/pl1'
TARGETS = set(range(18)) - {10} | {22, 25, 26, 27, 28}


def predict_world(*, candidates, self_labels, background, self_challenge, targets, schema, cfg):
    if set(targets) != TARGETS or len(candidates) != 1049:
        raise ValueError('pl1 requires the main-round team10 world and all 22 targets')
    if candidates.record_id.duplicated().any():
        raise ValueError('Duplicate candidate IDs')
    releases = {k: targets[k]['release'].copy(deep=True) for k in sorted(targets)}
    challenges = {k: targets[k]['challenge'].copy(deep=True) for k in sorted(targets)}
    if any(len(q) != 326 or q.challenge_row_id.duplicated().any() for q in challenges.values()):
        raise ValueError('Invalid main-round challenges')
    modules = methods.load_legacy()
    mia, _, mia_details = methods.build_mia(modules, candidates.copy(deep=True),
        background.copy(deep=True), releases, self_challenge.copy(deep=True), schema)
    mia = mia.rename(columns={k: str(k) for k in targets})
    mia = mia[['record_id', *map(str, sorted(targets))]]
    print('pl1: preliminary v4 MIA completed', flush=True)
    # Fixed before evaluation. This is the portable v5 consensus family used by
    # preliminary routed attacks, not the v9 feedback-specific block lookup.
    variants, aia_details = methods.build_aia(modules, background, releases, challenges)
    aia = variants['structured'][['target_k', 'challenge_row_id', 'time_hat']].copy()
    if not np.isfinite(mia.iloc[:, 1:].to_numpy(float)).all() or not np.isfinite(aia.time_hat).all():
        raise ValueError('Nonfinite prediction')
    if ((mia.iloc[:, 1:] < 0) | (mia.iloc[:, 1:] > 1)).any().any() or (aia.time_hat < 0).any():
        raise ValueError('Invalid prediction range')
    return {'F_mia': mia, 'F_aia': aia, 'diagnostics': {
        'model': MODEL_NAME, 'mia': mia_details, 'aia': aia_details,
        'selected_aia': 'structured', 'historical_target_routing_used': False,
        'legacy_sources': methods.source_hashes()}}

"""Regenerate the frozen 25-case local proxy check for the published v88 method."""
from __future__ import annotations

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import attack_workspace as workspace
from attack_research.run import read_data
from attack_research.v88.predict import CombinedAttack
from local_eval.data import PRIMARY_STYLES, build_case, make_release
from local_eval.scoring import score_arrays, validate_predictions
from .check import verify_package


SEEDS = (971001, 971002, 971003, 971004, 971005)
EXPECTED_MIA = 0.7307692307692308
EXPECTED_AIA = 0.8473837794327597
EXPECTED_BOTH_ONE = 8


def main() -> int:
    verify_package()
    source, background, schema = read_data(pd)
    _, _, _, cfg, _ = workspace.load_runtime()
    attack = CombinedAttack()
    scores = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            case = build_case(source, background, schema, 'holdout', seed)
            for style in PRIMARY_STYLES:
                c = make_release(case.target, case.background, style, seed, schema)
                answers, inferred = attack.predict(c=c, background=case.background,
                    candidates=case.candidates, challenge=case.challenge, cfg=cfg)
                if inferred != style:
                    raise AssertionError(f'Public route differed for {seed} {style}: {inferred}')
                mia, aia = validate_predictions(answers['combined'], case)
                scored = score_arrays(mia, aia, case, cfg)
                scores.append((scored['mia'], scored['aia_no_phi']))
                print(f'{seed} {style}: MIA={scored["mia"]:.6f} AIA={scored["aia_no_phi"]:.6f}', flush=True)
    result = np.asarray(scores, dtype=float)
    mia_mean, aia_mean = result.mean(axis=0)
    both_one = int(np.count_nonzero(np.isclose(result[:, 0], 1.0) &
                                    np.isclose(result[:, 1], 1.0)))
    np.testing.assert_allclose([mia_mean, aia_mean], [EXPECTED_MIA, EXPECTED_AIA],
                               rtol=0, atol=1e-10)
    if both_one != EXPECTED_BOTH_ONE:
        raise AssertionError(f'Expected {EXPECTED_BOTH_ONE} joint-perfect cases, got {both_one}')
    print(f'Frozen local proxy reproduced: MIA={mia_mean:.6f}, '
          f'AIA={aia_mean:.6f}, both_1={both_one}/25')
    print('These are local proxy scores, not official contest scores.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""One trusted attack in a fresh process, with public inputs only.

Process state separation is not an OS security sandbox for hostile Python.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, rankdata

from .common import FEATURES, public_tables, require, write_json
from pwscup2026.attack.aia import aia_predict
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.common.schema import CONTINUOUS


def predict_builtin(name, tables):
    c, background = tables['c'], tables['background']
    candidates, challenge, config = tables['candidates'], tables['challenge'], tables['cfg']
    if name == 'official':
        mia = overlap_mia(c, background, candidates, config)
        _, aia = aia_predict(c, challenge, background, config)
    elif name == 'background_only':
        # Deliberately C-free density/rarity baseline. This is a diagnostic,
        # not a proof that all possible source classifiers fail.
        mean = background[CONTINUOUS].mean().to_numpy(float)
        sd = background[CONTINUOUS].std(ddof=0).to_numpy(float)
        sd = np.where(sd > 0, sd, 1)
        ref = (background[CONTINUOUS].to_numpy(float) - mean) / sd
        query = (candidates[CONTINUOUS].to_numpy(float) - mean) / sd
        try:
            log_density = gaussian_kde(ref.T).logpdf(query.T)
            mia = rankdata(-log_density, method='average') / (len(query) + 1)
        except np.linalg.LinAlgError:
            mia = np.full(len(query), .5)
        _, aia = aia_predict(background, challenge, background, config)
    elif name == 'constant':
        mia, aia = np.full(len(candidates), .5), np.full(len(challenge), 10.)
    else:
        raise ValueError(f'Unknown builtin: {name}')
    return {'mia': mia, 'aia': aia}


def predictions(tables, *, plugin=None, method='official'):
    if plugin:
        spec = importlib.util.spec_from_file_location('_mock_ts_candidate', Path(plugin).resolve())
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(spec.name, None)
            raise
        require(callable(getattr(module, 'predict', None)),
                'Plugin must implement predict(*, c, background, candidates, challenge, cfg)')
        args = {k: copy.deepcopy(tables[k]) for k in ('c', 'background', 'candidates', 'challenge', 'cfg')}
        answer = module.predict(**args)
    else:
        answer = predict_builtin(method, tables)
    require(isinstance(answer, dict), 'Plugin must return a dictionary')
    if 'F_mia' in answer and 'F_aia' in answer:
        require(isinstance(answer['F_mia'], pd.DataFrame) and isinstance(answer['F_aia'], pd.DataFrame),
                'F_mia and F_aia must be DataFrames')
        return answer['F_mia'], answer['F_aia']
    require('mia' in answer and 'aia' in answer, 'Expected mia and aia arrays (or F_mia/F_aia frames)')
    mia, aia = np.asarray(answer['mia'], dtype=float), np.asarray(answer['aia'], dtype=float)
    require(mia.shape == (len(tables['candidates']),) and aia.shape == (len(tables['challenge']),),
            'Prediction lengths differ from public inputs')
    return (pd.DataFrame({'record_id': tables['candidates'].record_id, '0': mia}),
            pd.DataFrame({'target_k': 0, 'challenge_row_id': tables['challenge'].challenge_row_id,
                          'time_hat': aia}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plugin')
    parser.add_argument('--method', default='official')
    args = parser.parse_args()
    tables = public_tables(Path.cwd())
    # No seed, model label, truth, input source index or prior predictions are passed.
    mia, aia = predictions(tables, plugin=args.plugin, method=args.method)
    output = Path('predictions')
    output.mkdir(exist_ok=False)
    mia.to_csv(output / 'F_mia.csv', index=False)
    aia.to_csv(output / 'F_aia.csv', index=False)
    write_json(output / 'worker.json', {'pid': os.getpid(), 'method': args.method,
               'public_arguments': ['c', 'background', 'candidates', 'challenge', 'cfg'],
               'fresh_process': True})


if __name__ == '__main__':
    main()

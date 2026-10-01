"""Try a Gaussian-copula time hit prior estimated from public release C."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

import attack_workspace as workspace


QI_NUMERIC = ('age', 'SBP', 'TG', 'HDL', 'ALT', 'sex')


def _cdf_rank(sorted_values, values):
    left = np.searchsorted(sorted_values, values, side='left')
    right = np.searchsorted(sorted_values, values, side='right')
    return np.clip((left + right + 1.)/(2.*(len(sorted_values)+1.)), 1e-5, 1.-1e-5)


def hit_prior(c, challenge, times, ridge):
    n = len(c)
    columns = QI_NUMERIC + ('time',)
    normal = np.column_stack([
        ndtri(np.clip((rankdata(c[col].to_numpy(float), method='average')-.5)/n,
                      1e-5, 1.-1e-5)) for col in columns])
    covariance = np.cov(normal, rowvar=False)
    xx = covariance[:-1, :-1]
    beta = np.linalg.solve(xx + ridge*np.eye(len(QI_NUMERIC)), covariance[:-1, -1])
    variance = max(covariance[-1, -1] - beta@covariance[:-1, -1], .05)
    query = np.column_stack([
        ndtri(_cdf_rank(np.sort(c[col].to_numpy(float)), challenge[col].to_numpy(float)))
        for col in QI_NUMERIC])
    mean = query@beta
    time_sorted = np.sort(c.time.to_numpy(float))
    lower = times - .5
    upper = times + .5
    lo_cdf = np.where(lower <= 0, 0.,
        np.searchsorted(time_sorted, lower, side='left')/n)
    hi_cdf = np.where(upper >= 10, 1.,
        np.searchsorted(time_sorted, upper, side='right')/n)
    lo_z = ndtri(np.clip(lo_cdf, 1e-8, 1.-1e-8))
    hi_z = ndtri(np.clip(hi_cdf, 1e-8, 1.-1e-8))
    prior = ndtr((hi_z-mean[:, None])/np.sqrt(variance)) - \
        ndtr((lo_z-mean[:, None])/np.sqrt(variance))
    return np.clip(prior, 0., 1.)


def main():
    workspace.verify_hashes()
    _, _, _, cfg, _ = workspace.load_runtime()
    from threadpoolctl import threadpool_limits
    from local_eval.data import build_case, make_release
    from local_eval.scoring import score_arrays, validate_predictions
    from local_eval.__main__ import save
    from ..run import read_data
    from ..v82.predict import CombinedAttack
    from ..v82.run import provenance, assets
    from ..v83.run import old_probabilities, decide

    root = Path(__file__).parent
    protocol = json.loads((root/'protocol.json').read_text())
    base = workspace.TS_OUTPUT/'attack-research'
    out = base/'round87-copula-time-covariance'
    if out.exists():
        raise ValueError('Preserve copula time-covariance comparison')
    selection = json.loads((base/'round82'/'selection.json').read_text())
    if selection['source_sha256'] != provenance() or selection['assets_sha256'] != assets():
        raise ValueError('Frozen selected attack changed')
    paths = {'membership': base/'round27'/'models.joblib',
             'grid': base/'round18-specialist'/'models.joblib'}
    membership = joblib.load(paths['membership'])['regularized']
    grid = joblib.load(paths['grid'])
    source, bg, schema = read_data(pd)
    attack = CombinedAttack()
    rows = []
    with threadpool_limits(limits=1):
        for seed in protocol['development_seeds']:
            case = build_case(source, bg, schema, protocol['split'], seed)
            c = make_release(case.target, case.background, protocol['style'], seed, schema)
            answers, inferred = attack.predict(c=c, background=case.background,
                candidates=case.candidates, challenge=case.challenge, cfg=cfg)
            if inferred != 'copula':
                raise ValueError(f'Unexpected public route {seed}: {inferred}')
            _, baseline = validate_predictions(answers['combined'], case)
            member, times, mh, ch = old_probabilities('copula', c,
                case.background, case.challenge, None, membership, grid)
            np.testing.assert_array_equal(decide('copula', member, times, mh, ch), baseline)
            predictions = {'previous_v82': baseline}
            for ridge in protocol['ridge']:
                prior = hit_prior(c, case.challenge, times, ridge)
                for mode in protocol['modes']:
                    for weight in protocol['weights']:
                        new_mh = (1-weight)*mh+weight*prior
                        new_ch = ch if mode == 'member_only' else (1-weight)*ch+weight*prior
                        predictions[f'ridge{ridge:g}_{mode}_w{weight:g}'] = decide(
                            'copula', member, times, new_mh, new_ch)
            for name, aia in predictions.items():
                scored = score_arrays(np.full(len(case.candidates), .5), aia, case, cfg)
                rows.append({'seed': seed, 'method': name,
                    'aia_no_phi': scored['aia_no_phi'],
                    'aia_all': scored['aia_all'], 'aia_rare': scored['aia_rare'],
                    'changed_answers': int((aia != baseline).sum())})
            print(f'Copula time covariance seed={seed}', flush=True)
    frame = pd.DataFrame(rows)
    out.mkdir()
    frame.to_csv(out/'results.csv', index=False)
    summary = frame.groupby('method').agg(aia=('aia_no_phi','mean'),
        minimum=('aia_no_phi','min'), all=('aia_all','mean'),
        rare=('aia_rare','mean'), changed=('changed_answers','sum'))
    summary.sort_values('aia', ascending=False).to_csv(out/'summary.csv')
    print(summary.sort_values('aia', ascending=False).to_string(), flush=True)
    save(out/'verification.json', {'scope': protocol['scope'],
        'frozen_selection_sha256': workspace.sha256(base/'round82'/'selection.json'),
        'model_sha256': {k:workspace.sha256(v) for k,v in paths.items()},
        'source_sha256': {str(p.relative_to(workspace.TS_ROOT)): workspace.sha256(p)
            for p in list(root.glob('*.py'))+[root/'protocol.json']},
        'input_verification': workspace.verify_hashes()})


if __name__ == '__main__':
    main()

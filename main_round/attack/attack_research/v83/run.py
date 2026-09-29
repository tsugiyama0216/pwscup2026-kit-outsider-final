"""Compare frozen QI time prior blends with hard-style AIA decisions."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

import attack_workspace as workspace


def old_probabilities(style, c, background, challenge, strong, copula_membership, copula_grid):
    from ..v5.conditioned import visible_conditioning
    from ..v12.features import extract
    from ..v27.features import extract as membership_extract

    x, times, hx, _ = extract(c, background, challenge)
    if style == 'strong_jitter':
        model = strong['grid']
        member = visible_conditioning(challenge,
            strong['membership'].predict_proba(x)[:, 1])
    elif style == 'copula':
        model = copula_grid['grid']
        mx = membership_extract(c, background, challenge, neighbor_x=x)
        p = np.clip(copula_membership.predict_proba(mx)[:, 1], 1e-6, 1-1e-6)
        member = visible_conditioning(challenge, expit(2*logit(p)))
    else:
        raise ValueError(f'Unexpected style {style}')
    hp = model.predict_proba(hx).reshape(len(x), times.shape[1], -1)
    index = {int(k): i for i, k in enumerate(model.classes_)}
    mh = hp[:, :, index[3]]/np.maximum(hp[:, :, index[2]]+hp[:, :, index[3]], 1e-8)
    ch = hp[:, :, index[1]]/np.maximum(hp[:, :, index[0]]+hp[:, :, index[1]], 1e-8)
    return member, times, mh, ch


def decide(style, member, times, mh, ch):
    from ..v13.decision import ratio_optimal

    if style == 'strong_jitter':
        return ratio_optimal(member[:, None]*mh,
            (1-member[:, None])*ch, times, len(member)//2)
    gain = member[:, None]*mh-.1*(1-member[:, None])*ch
    best = gain.argmax(axis=1)
    chosen = times[np.arange(len(member)), best].astype(float)
    return np.where(gain[np.arange(len(member)), best] > 0, chosen, 11.)


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
    from ..v80.run import sources as prior_sources

    root = Path(__file__).parent
    protocol = json.loads((root/'protocol.json').read_text())
    base = workspace.OUTPUT/'attack-research'
    out = base/'round83-qi-prior-transfer'
    if out.exists():
        raise ValueError('Preserve QI-prior transfer comparison')
    selection = json.loads((base/'round82'/'selection.json').read_text())
    if selection['source_sha256'] != provenance() or selection['assets_sha256'] != assets():
        raise ValueError('Frozen selected attack changed')
    prior_manifest = json.loads((base/'round80-public-qi-prior'/'model_manifest.json').read_text())
    prior_path = base/'round80-public-qi-prior'/'prior.joblib'
    if prior_manifest['source_sha256'] != prior_sources() or \
            prior_manifest['model_sha256'] != workspace.sha256(prior_path):
        raise ValueError('Frozen person-disjoint QI prior changed')
    prior = joblib.load(prior_path)
    model_paths = {'strong': base/'round12'/'models.joblib',
                   'copula_membership': base/'round27'/'models.joblib',
                   'copula_grid': base/'round18-specialist'/'models.joblib'}
    strong = joblib.load(model_paths['strong'])
    copula_membership = joblib.load(model_paths['copula_membership'])['regularized']
    copula_grid = joblib.load(model_paths['copula_grid'])
    source, bg, schema = read_data(pd)
    attack = CombinedAttack()
    out.mkdir()
    rows = []
    with threadpool_limits(limits=1):
        for seed in protocol['development_seeds']:
            case = build_case(source, bg, schema, protocol['split'], seed)
            for style in protocol['styles']:
                c = make_release(case.target, case.background, style, seed, schema)
                answers, inferred = attack.predict(c=c, background=case.background,
                    candidates=case.candidates, challenge=case.challenge, cfg=cfg)
                if inferred != style:
                    raise ValueError(f'Unexpected public style route: {seed} {style} -> {inferred}')
                _, baseline = validate_predictions(answers['combined'], case)
                member, times, mh, ch = old_probabilities(style, c, case.background,
                    case.challenge, strong, copula_membership, copula_grid)
                np.testing.assert_array_equal(decide(style, member, times, mh, ch), baseline)
                prior_hit = prior.hit_probability(case.challenge, times)
                predictions = {'previous_v82': baseline}
                for mode in protocol['modes']:
                    for weight in protocol['weights']:
                        new_mh = (1-weight)*mh+weight*prior_hit
                        new_ch = ch if mode == 'member_only' else (1-weight)*ch+weight*prior_hit
                        predictions[f'{mode}_w{weight:g}'] = decide(style, member, times, new_mh, new_ch)
                for name, aia in predictions.items():
                    scored = score_arrays(np.full(len(case.candidates), .5), aia, case, cfg)
                    rows.append({'seed': seed, 'style': style, 'method': name,
                        'aia_no_phi': scored['aia_no_phi'],
                        'aia_all': scored['aia_all'], 'aia_rare': scored['aia_rare'],
                        'changed_answers': int((aia != baseline).sum())})
                print(f'QI-prior transfer seed={seed} style={style}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'results.csv', index=False)
    for name, part in [('all_development', frame),
                       ('by_style', frame)]:
        group = ['method'] if name == 'all_development' else ['style','method']
        summary = part.groupby(group).agg(aia=('aia_no_phi','mean'),
            minimum=('aia_no_phi','min'), all=('aia_all','mean'),
            rare=('aia_rare','mean'), changed=('changed_answers','sum'))
        summary.sort_values('aia', ascending=False).to_csv(out/f'{name}_summary.csv')
        print(name, flush=True)
        print(summary.sort_values('aia', ascending=False).to_string(), flush=True)
    save(out/'verification.json', {'scope': protocol['scope'],
        'frozen_prior_sha256': workspace.sha256(prior_path),
        'frozen_selection_sha256': workspace.sha256(base/'round82'/'selection.json'),
        'frozen_model_sha256': {k: workspace.sha256(p) for k,p in model_paths.items()},
        'source_sha256': {str(p.relative_to(workspace.ROOT)): workspace.sha256(p)
            for p in list(root.glob('*.py'))+[root/'protocol.json']},
        'input_verification': workspace.verify_hashes()})


if __name__ == '__main__':
    main()

import argparse
from pathlib import Path
import runtime as rt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from model import predict_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=rt.read(rt.HERE / 'protocol.json')['series'])
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(rt.SERIES) or out.exists():
        raise ValueError('Choose a fresh output folder under team10/attack/attackR')
    verified = rt.verify()
    actual = rt.actual()
    w, _, _, validator, cfg, dist, b, bg, cs, qs, originals, versions = actual
    oldm, olda, mask, proof = rt.baseline_and_mask(actual)
    outputs, evidence = predict_all(b, bg, cs, qs, oldm, olda, mask, rt.read(rt.HERE / 'protocol.json'),
        progress=lambda k: print('Completed target', k, flush=True))
    selected = [args.model] if args.model else list(outputs)
    out.mkdir(parents=True)
    for name in selected:
        folder = out if args.model else out / name
        folder.mkdir(exist_ok=True)
        mia = pd.DataFrame({'record_id': b.record_id})
        aia, diagnostics = [], {}
        for k in sorted(cs):
            m, a = outputs[name]['mia'][k], outputs[name]['aia'][k]
            mia[str(k)] = m
            aia.append(pd.DataFrame({'target_k': k, 'challenge_row_id': qs[k].challenge_row_id, 'time_hat': a}))
            # Distinguish score rescaling from actual ranking/top-tail changes.
            use = ~mask
            oldtop = set(np.flatnonzero(use)[np.argsort(-oldm[k][use], kind='stable')[:30]])
            newtop = set(np.flatnonzero(use)[np.argsort(-m[use], kind='stable')[:30]])
            diagnostics[str(k)] = {
                'mia_values_changed': int(np.sum(m != oldm[k])),
                'mia_ranks_changed': int(np.sum(rankdata(m[use]) != rankdata(oldm[k][use]))),
                'mia_top30_replaced': len(oldtop - newtop),
                'aia_changed': int(np.sum(a != olda[k])),
                'aia_changed_over_half_year': int(np.sum(np.abs(a - olda[k]) > .5)),
                'known_negative_nonzero': int(np.sum(m[mask] != 0))}
        aia = pd.concat(aia, ignore_index=True)
        valid = w.validate_frames(np, pd, validator, mia, aia, cfg, dist)
        for n, f in [('F_mia.csv', mia), ('F_aia.csv', aia)]:
            f.to_csv(folder / n, index=False, lineterminator='\n', float_format='%.17g')
        read = lambda n: pd.read_csv(folder / n, float_precision='round_trip')
        w.validate_frames(np, pd, validator, read('F_mia.csv'), read('F_aia.csv'), cfg, dist)
        rt.write(folder / 'report.json', {'model': name, 'validation': valid, 'dependencies_verified': verified,
            'runtime_versions': versions, 'originals': originals, 'private_proof': proof,
            'targets': diagnostics, 'official_score_measured': False, 'mock_scores_used': False,
            'meaning': 'Public-input behavior diagnostics only; changes do not establish better scores.'})
    rt.write(out / 'attribute_evidence.json', evidence)
    rt.verify()


if __name__ == '__main__':
    main()

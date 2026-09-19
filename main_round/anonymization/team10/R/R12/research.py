"""A fixed six-candidate simplification study; select the fewest knots passing U."""
import io
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluation import HERE, DATA, digest, check_environment, read_training_data, evaluate_frame
from model import fit, generate, rare_mask, LABS

R11_SHA = '82f7a6295cc3f70a014491575a288bb623901c5bef5588fedd55cc8fd36a351d'
KNOT_GRID = [16, 32, 64, 128, 256, 512]


def baseline():
    path = HERE.parent / 'R11/C.csv'
    if digest(path) != R11_SHA:
        raise RuntimeError('Frozen R11 has changed')
    return pd.read_csv(path, float_precision='round_trip')


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


def round_trip(frame):
    return pd.read_csv(io.BytesIO(csv_bytes(frame)), float_precision='round_trip')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--without-boundary-protection', action='store_true')
    args = parser.parse_args()
    cfg, provenance = check_environment()
    b = read_training_data()
    teacher = baseline()
    detector = json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    mask = rare_mask(teacher, detector)
    results = []
    chosen = None
    with threadpool_limits(limits=1):
        old_score, _ = evaluate_frame(teacher, b, cfg)
        for count in KNOT_GRID:
            fitted = fit(teacher, detector, count, preserve_boundary=not args.without_boundary_protection)
            candidate = round_trip(generate(fitted))
            same_groups = bool(np.array_equal(mask, rare_mask(candidate, detector)))
            pd.testing.assert_frame_equal(teacher.loc[mask], candidate.loc[mask], check_exact=True)
            pd.testing.assert_frame_equal(teacher.drop(columns=LABS), candidate.drop(columns=LABS), check_exact=True)
            score, validation = evaluate_frame(candidate, b, cfg)
            eligible = bool(same_groups and score['utility'] >= old_score['utility'])
            row = dict(knots=count, score=score, validation=validation,
                       rare_membership_preserved=same_groups, eligible=eligible,
                       boundary_exception_rows=len(fitted.get('boundary_exceptions',{}).get('indices',[])),
                       changed_cells=int((candidate[LABS] != teacher[LABS]).to_numpy().sum()))
            results.append(row)
            print(json.dumps(dict(knots=count, U=score['utility'], facets=score['facets'],
                                  same_groups=same_groups, eligible=eligible)), flush=True)
            if eligible and chosen is None:
                chosen = count
    summary = dict(model='R12', fixed_candidate_grid=KNOT_GRID,
                   batch=1 if args.without_boundary_protection else 2,
                   previous_batch_sha256=None if args.without_boundary_protection else digest(HERE/'initial_summary.json'),
                   adaptation=None if args.without_boundary_protection else 'Restore R11 labs for majority rows crossing the rare threshold after interpolation',
                   objective='fewest majority marginal knots with U >= R11 and unchanged rare membership',
                   baseline_score=old_score, R11_C_sha256=R11_SHA, selected_knots=chosen,
                   candidates=results, independent_holdout=False, provenance=provenance)
    name = 'initial_summary_recheck.json' if args.without_boundary_protection else 'experiment_summary.json'
    (HERE/name).write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    if chosen is None:
        raise RuntimeError('No simplified candidate preserved R11 utility')


if __name__ == '__main__':
    main()

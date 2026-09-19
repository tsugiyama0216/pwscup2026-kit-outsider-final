"""Fit/reproduce R12, enforce its utility floor, and validate the submission ZIP."""
import argparse
import json
import time
import zipfile
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from evaluation import (HERE, DATA, ZIP_NAME, digest, check_environment,
                        read_training_data, evaluate_frame, write_package, evaluate_release)
from model import fit, generate, rare_mask, LABS
from research import baseline, csv_bytes, round_trip, R11_SHA

SOURCES = ['model.py', 'generate.py', 'evaluation.py', 'research.py', 'run.ps1']


def compare(teacher, candidate, detector):
    mask = rare_mask(teacher, detector)
    np.testing.assert_array_equal(mask, rare_mask(candidate, detector))
    pd.testing.assert_frame_equal(teacher.loc[mask], candidate.loc[mask], check_exact=True)
    pd.testing.assert_frame_equal(teacher.drop(columns=LABS), candidate.drop(columns=LABS), check_exact=True)
    return dict(rare_rows_unchanged=int(mask.sum()), fixed_columns_unchanged=True,
                rare_membership_unchanged=True,
                changed_cells=int((teacher[LABS] != candidate[LABS]).to_numpy().sum()),
                changed_rows=int((teacher[LABS] != candidate[LABS]).any(axis=1).sum()))


def gate(score, floor):
    if not np.isfinite(score['utility']) or score['utility'] < floor:
        raise RuntimeError('Utility fell below R11')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    cfg, provenance = check_environment()
    provenance.update(R11_C_sha256=R11_SHA, source_sha256={n:digest(HERE/n) for n in SOURCES},
                      study_sha256=digest(HERE/'experiment_summary.json'),
                      feedback_sha256=digest(HERE/'R11_user_reported_result.json'))
    teacher = baseline()
    detector = json.loads((DATA/'schema.json').read_text(encoding='utf-8'))['rare_detector']
    study = json.loads((HERE/'experiment_summary.json').read_text(encoding='utf-8'))
    eligible = [r for r in study['candidates'] if r['eligible']]
    knots = min(r['knots'] for r in eligible)
    if knots != study['selected_knots']:
        raise RuntimeError('Selected model is not the simplest eligible candidate')
    feedback = json.loads((HERE/'R11_user_reported_result.json').read_text(encoding='utf-8'))
    floor = max(study['baseline_score']['utility'], min(feedback['utility_facets'].values()))
    with threadpool_limits(limits=1):
        b = read_training_data()
        original_score, _ = evaluate_frame(teacher, b, cfg)
        if original_score != study['baseline_score']:
            raise RuntimeError('Baseline evaluation changed')
        fitted = fit(teacher, detector, knots)
        serialized = json.dumps(fitted, separators=(',', ':'), allow_nan=False)+'\n'
        # Persisting/reloading JSON must retain enough precision for exact reproduction.
        start = time.perf_counter()
        candidate = round_trip(generate(json.loads(serialized)))
        generation_seconds = time.perf_counter() - start
        audit = compare(teacher, candidate, detector)
        audit.update(boundary_exception_rows=len(fitted['boundary_exceptions']['indices']),
                     majority_marginal_values=6*knots,
                     rare_marginal_values=6*audit['rare_rows_unchanged'])
        payload = csv_bytes(candidate)
        score, _ = evaluate_frame(candidate, b, cfg)
        gate(score, floor)
        selected = next(r for r in study['candidates'] if r['knots']==knots)
        if score != selected['score']:
            raise RuntimeError('Candidate score changed')
        if args.verify_only:
            saved = json.loads((HERE/'report.json').read_text(encoding='utf-8'))
            if saved['provenance'] != provenance:
                raise RuntimeError('Provenance changed')
            if (HERE/'model.json').read_text(encoding='utf-8') != serialized:
                raise RuntimeError('Fitted model differs from its teacher')
            if (HERE/'C.csv').read_bytes() != payload:
                raise RuntimeError('CSV reproduction mismatch')
        else:
            targets = ['C.csv', 'model.json', 'report.json', 'submission/'+ZIP_NAME]
            if any((HERE/name).exists() for name in targets):
                raise RuntimeError('Refusing to overwrite release files')
            (HERE/'model.json').write_text(serialized, encoding='utf-8', newline='\n')
            (HERE/'C.csv').write_bytes(payload)
            write_package(HERE/'submission'/ZIP_NAME, payload, (DATA/'token.txt').read_bytes())
        zip_score, validation = evaluate_release(HERE, b, cfg)
        if zip_score != score:
            raise RuntimeError('ZIP score mismatch')
        with zipfile.ZipFile(HERE/'submission'/ZIP_NAME) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('ZIP CRC failure')
        gate(zip_score, floor)
        report = dict(model='R12', method='R11-distilled empirical copula with linear marginal interpolation',
                      majority_knots=knots, provenance=provenance, score=score,
                      baseline_score=original_score, utility_floor=floor, utility_delta=score['utility']-floor,
                      audit=audit, validation=validation, fitted_model_sha256=digest(HERE/'model.json'),
                      generation_seconds=generation_seconds, independent_holdout=False,
                      official_attack_evaluated=False, formal_differential_privacy=False,
                      official_submission_performed=False, exact_reproduction=True)
        if not args.verify_only:
            (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        elif any(saved[k]!=report[k] for k in report if k!='generation_seconds'):
            raise RuntimeError('Verified report differs from saved release')
        verification = dict(verified=True, exact_reproduction=True, ZIP_CRC_ok=True,
                            utility=score['utility'], facets=score['facets'], validation=validation,
                            audit=audit, majority_knots=knots)
        (HERE/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(verification),flush=True)


if __name__ == '__main__':
    main()

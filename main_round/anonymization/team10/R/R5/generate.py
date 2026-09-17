"""Generate or verify the frozen R5 sequential conditional synthesis release."""
import argparse
import json
from evaluation import (HERE, DATA, ZIP_NAME, digest, check_environment,
                        read_training_data, evaluate_release, write_package)
from sequential import synthesize

SOURCES = ('generate.py', 'sequential.py', 'survival.py', 'calibration.py', 'evaluation.py')

def payload(c):
    return c.to_csv(index=False, lineterminator='\n').encode('utf-8')

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify-only', action='store_true')
    group.add_argument('--reproduce-check', action='store_true')
    args = parser.parse_args()
    cfg, provenance = check_environment()
    from pwscup2026.scoring.result import json_safe
    b = read_training_data()
    selected = json.loads((HERE/'experiment_summary.json').read_text())['selected']
    provenance['source_sha256'] = {name: digest(HERE/name) for name in SOURCES}
    provenance['experiment_sha256'] = digest(HERE/'experiment_summary.json')
    if args.verify_only or args.reproduce_check:
        saved = json.loads((HERE/'report.json').read_text())
        if provenance != saved['provenance']:
            raise RuntimeError('Input, environment or source provenance changed')
        score, validation = evaluate_release(HERE, b, cfg)
        if validation != saved['validation'] or score['facets'] != saved['score']['facets']:
            raise RuntimeError('Release validation or scores changed')
        if args.reproduce_check:
            c, _ = synthesize(b, cfg, selected['leaf_size'], selected['strength'])
            if payload(c) != (HERE/'C.csv').read_bytes():
                raise RuntimeError('CSV byte reproduction failed')
        print(json.dumps({'verified': True, 'reproduced': args.reproduce_check, 'facets': score['facets']}))
        return
    if any((HERE/p).exists() for p in ('C.csv', 'report.json', 'submission/'+ZIP_NAME)):
        raise FileExistsError('Refusing to overwrite frozen release')
    c, audit = synthesize(b, cfg, selected['leaf_size'], selected['strength'])
    if any(audit['calibration']['clipped_counts'].values()):
        raise RuntimeError('Release rejects clipped calibration')
    data = payload(c)
    (HERE/'C.csv').write_bytes(data)
    write_package(HERE/'submission'/ZIP_NAME, data, (DATA/'token.txt').read_bytes())
    score, validation = evaluate_release(HERE, b, cfg)
    result = json_safe({'model': 'R5', 'provenance': provenance, 'settings': selected,
                       'audit': audit, 'validation': validation, 'score': score,
                       'training_inputs': ['participant_data/B.csv'],
                       'evaluation_only': ['B_self.csv', 'A_bg.csv', 'utility_ref.json'],
                       'formal_differential_privacy': False, 'official_attack_evaluated': False,
                       'official_submission_performed': False})
    (HERE/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'utility': score['utility'], 'facets': score['facets'], 'validation': validation}))

if __name__ == '__main__':
    main()

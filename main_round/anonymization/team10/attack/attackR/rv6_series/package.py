"""Create six local credential-bearing ZIPs only after two reproducible runs."""
import hmac
import io
import os
import subprocess
import sys
import zipfile
import runtime as rt
import numpy as np
import pandas as pd


def main():
    names = rt.read(rt.HERE / 'protocol.json')['series']
    if any((rt.SERIES / n / 'submission').exists() for n in names):
        raise ValueError('Never overwrite a previous submission')
    rt.verify()
    work = rt.HERE / '.work'
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    for i in (1, 2):
        subprocess.run([sys.executable, '-B', str(rt.HERE / 'infer.py'), '--output', str(work / f'pass{i}')], check=True, env=env)
    actual = rt.actual()
    w, _, _, validator, cfg, dist, *_ = actual
    for n, expected in rt.read(rt.HERE / 'preservation.json')['files'].items():
        if rt.sha(rt.ROOT / n) != expected:
            raise ValueError('Prior artifact changed: ' + n)
    results = {}
    for name in names:
        files = {n: (work / 'pass1' / name / n).read_bytes() for n in ['F_mia.csv', 'F_aia.csv']}
        if any(b != (work / 'pass2' / name / n).read_bytes() for n, b in files.items()):
            raise ValueError('Independent regeneration differs: ' + name)
        frames = {n: pd.read_csv(io.BytesIO(b), float_precision='round_trip') for n, b in files.items()}
        valid = w.validate_frames(np, pd, validator, frames['F_mia.csv'], frames['F_aia.csv'], cfg, dist)
        negative = {}
        for label in ['missing_mia', 'duplicate_mia', 'nan_mia', 'out_of_range', 'missing_aia', 'duplicate_aia', 'negative_time', 'bad_target']:
            m, a = frames['F_mia.csv'].copy(), frames['F_aia.csv'].copy()
            if label == 'missing_mia': m = m.iloc[:-1]
            elif label == 'duplicate_mia': m.loc[1, 'record_id'] = m.loc[0, 'record_id']
            elif label == 'nan_mia': m.iloc[0, 1] = np.nan
            elif label == 'out_of_range': m.iloc[0, 1] = 1.1
            elif label == 'missing_aia': a = a.iloc[:-1]
            elif label == 'duplicate_aia': a.loc[1, 'challenge_row_id'] = a.loc[0, 'challenge_row_id']
            elif label == 'negative_time': a.loc[0, 'time_hat'] = -1
            else: a.loc[0, 'target_k'] = 999
            try:
                w.validate_frames(np, pd, validator, m, a, cfg, dist)
            except (ValueError, AssertionError):
                negative[label] = True
            else:
                raise ValueError('Invalid submission accepted: ' + label)
        for n, b in files.items():
            if frames[n].to_csv(index=False, lineterminator='\n', float_format='%.17g').encode() != b:
                raise ValueError('CSV roundtrip differs')
        out = rt.SERIES / name / 'submission'
        out.mkdir()
        for n, b in files.items():
            (out / n).write_bytes(b)
        path = out / f'team10_{name}_attack.zip'
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as arc:
            for n, b in files.items(): arc.writestr(n, b)
            arc.writestr('token.txt', (w.ORIGINALS / 'token.txt').read_bytes())
        with zipfile.ZipFile(path) as arc:
            if arc.namelist() != ['F_mia.csv', 'F_aia.csv', 'token.txt'] or arc.testzip() is not None:
                raise ValueError('Invalid ZIP structure')
            if not all(arc.read(n) == b for n, b in files.items()):
                raise ValueError('ZIP prediction differs')
            if not hmac.compare_digest(arc.read('token.txt'), (w.ORIGINALS / 'token.txt').read_bytes()):
                raise ValueError('Credential differs')
        if not validator.validate_submission_zip(path, cfg, w.TARGETS, 'attack').ok:
            raise ValueError('Official ZIP validation failed')
        diagnostics = rt.read(work / 'pass1' / name / 'report.json')
        rt.write(rt.SERIES / name / 'production_diagnostics.json', diagnostics)
        report = {'model': name, 'validation': valid, 'two_independent_runs_byte_identical': True,
            'csv_roundtrip_identical': True, 'official_zip_valid': True, 'token_matches_original': True,
            'negative_checks': negative, 'prediction_sha256': {n: rt.sha(out / n) for n in files},
            'zip_sha256': rt.sha(path), 'dependency_manifest_sha256': rt.sha(rt.HERE / 'dependencies.json'),
            'official_score_measured': False, 'mock_scores_used': False, 'official_submission_performed': False}
        rt.write(rt.SERIES / name / 'verification.json', report)
        results[name] = {key: sum(v[key] for v in diagnostics['targets'].values()) for key in
            ['mia_values_changed', 'mia_ranks_changed', 'mia_top30_replaced', 'aia_changed', 'aia_changed_over_half_year']}
        results[name]['submission_zip_local'] = path.relative_to(rt.ROOT).as_posix()
        print('Verified local ZIP:', path, flush=True)
    rt.verify()
    rt.write(rt.HERE / 'production_summary.json', {'models': results, 'score_measured': False,
        'interpretation': 'Behavior differences versus Rv2, not score estimates.',
        'prior_files_preserved': len(rt.read(rt.HERE / 'preservation.json')['files'])})


if __name__ == '__main__':
    main()

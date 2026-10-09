"""Rebuild a local ZIP from the published, hash-verified frozen CSVs.

Useful on a fresh clone where submission CSVs already exist and package.py
therefore intentionally refuses to overwrite the submission directory.
"""
import argparse
import hmac
import zipfile
import runtime as rt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=rt.read(rt.HERE / 'protocol.json')['series'])
    args = parser.parse_args()
    rt.verify()
    w, np, pd, validator, cfg, dist, *_ = rt.actual()
    folder = rt.SERIES / args.model
    expected = rt.read(folder / 'verification.json')['prediction_sha256']
    out = folder / 'submission'
    data = {}
    for name in ['F_mia.csv', 'F_aia.csv']:
        path = out / name
        if rt.sha(path) != expected[name]:
            raise ValueError('Frozen submission CSV changed: ' + name)
        data[name] = path.read_bytes()
    read = lambda name: pd.read_csv(out / name, float_precision='round_trip')
    w.validate_frames(np, pd, validator, read('F_mia.csv'), read('F_aia.csv'), cfg, dist)
    token = (w.ORIGINALS / 'token.txt').read_bytes()
    target = out / f'team10_{args.model}_attack.zip'
    if target.exists():
        with zipfile.ZipFile(target) as arc:
            if arc.namelist() != ['F_mia.csv', 'F_aia.csv', 'token.txt'] or arc.testzip() is not None:
                raise ValueError('Existing ZIP differs; preserve it')
            if not all(arc.read(n) == b for n, b in data.items()) or not hmac.compare_digest(arc.read('token.txt'), token):
                raise ValueError('Existing ZIP differs; preserve it')
    else:
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as arc:
            for name, value in data.items():
                arc.writestr(name, value)
            arc.writestr('token.txt', token)
    if not validator.validate_submission_zip(target, cfg, w.TARGETS, 'attack').ok:
        raise ValueError('Official ZIP validation failed')
    print('Verified local ZIP:', target)


if __name__ == '__main__':
    main()

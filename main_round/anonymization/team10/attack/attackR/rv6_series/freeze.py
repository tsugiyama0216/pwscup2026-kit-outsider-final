"""Run once before production inference; never replace an existing freeze."""
from datetime import datetime, timezone
import runtime as rt


def main():
    destination = rt.HERE / 'dependencies.json'
    if destination.exists():
        raise ValueError('Existing freeze must be preserved')
    old = rt.read(rt.ATTACK / 'attackR/Rv5/dependencies.json')['files']
    files = {n: h for n, h in old.items() if not n.startswith('main_round/attack/attackR/Rv5/')}
    for n, expected in files.items():
        if rt.sha(rt.ROOT / n) != expected:
            raise ValueError('Previous dependency mismatch: ' + n)
    for directory in [rt.HERE] + [rt.SERIES / n for n in rt.read(rt.HERE / 'protocol.json')['series']]:
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix in ('.py', '.ps1', '.json'):
                files[p.relative_to(rt.ROOT).as_posix()] = rt.sha(p)
    now = datetime.now(timezone.utc).isoformat()
    rt.write(destination, {'created_utc': now, 'frozen_before_actual_predictions': True,
        'no_mock_score_selection': True, 'files': files})
    preserve = rt.read(rt.ATTACK / 'attackR/Rv5/preservation.json')['files']
    for base in [rt.ATTACK / 'attackR/Rv5', rt.SERIES / 'Rv4']:
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.py', '.ps1', '.json', '.csv', '.md', '.zip') and '.work' not in p.parts:
                preserve[p.relative_to(rt.ROOT).as_posix()] = rt.sha(p)
    rt.write(rt.HERE / 'preservation.json', {'captured_utc': now, 'files': preserve})
    print('Frozen dependencies:', len(files), 'Preserved prior artifacts:', len(preserve))


if __name__ == '__main__':
    main()

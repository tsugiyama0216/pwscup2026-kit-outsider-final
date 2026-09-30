"""Verify exactly the published v88 runtime sources and fitted assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import attack_workspace as workspace


MANIFEST = Path(__file__).with_name('manifest.json')


def verify_package() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    base = payload.get('path_base', 'main_round/attack')
    roots = {'main_round/attack': workspace.ROOT,
             'main_round': workspace.ROOT.parent,
             'repository': workspace.ROOT.parents[1]}
    if base not in roots:
        raise ValueError(f'Unsupported published manifest base: {base}')
    root = roots[base].resolve()
    for relative, expected in payload['files_sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f'Missing published v88 file: {relative}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Published v88 file changed: {relative}')
    originals = workspace.verify_hashes()
    return {'published_files': len(payload['files_sha256']),
            'verified_original_files': originals['verified_files'],
            'version': payload['version']}


def verify_runtime() -> dict:
    """Check canonical dependency origins and initialize the frozen attack."""
    import prelim_transfer.methods as adapter

    repository = workspace.ROOT.parents[1].resolve()
    expected_adapter = workspace.TS_ROOT / 'prelim_transfer' / 'methods.py'
    if Path(adapter.__file__).resolve() != expected_adapter.resolve():
        raise ImportError('v88 must use its bundled TS prelim_transfer adapter')
    modules = adapter.load_legacy()
    if not callable(adapter.build_aia) or not callable(
            modules['structured'].stratified_half_selection):
        raise ImportError('Missing historical AIA helpers required by v88')
    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    legacy_files = set()
    legacy_root = repository / 'prelim' / 'workspace'
    for name, module in tuple(sys.modules.items()):
        if name.startswith('attack_phase') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(legacy_root):
                raise ImportError(f'Historical dependency loaded outside this repository: {name}')
            relative = path.relative_to(repository).as_posix()
            if payload.get('path_base') != 'repository' or relative not in payload['files_sha256']:
                raise ImportError(f'Historical dependency absent from v88 manifest: {relative}')
            legacy_files.add(relative)
    from attack_research.v88.predict import CombinedAttack

    CombinedAttack()
    adapter.assert_official_origins()
    return {'runtime_initialized': True,
            'adapter': expected_adapter.relative_to(repository).as_posix(),
            'legacy_source_files': len(legacy_files)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify the frozen v88 package')
    parser.add_argument('--runtime', action='store_true',
                        help='also check dependency origins and initialize v88')
    args = parser.parse_args()
    result = verify_package()
    if args.runtime:
        result.update(verify_runtime())
    print(result)

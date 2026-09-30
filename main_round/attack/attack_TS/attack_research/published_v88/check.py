"""Verify exactly the published v88 runtime sources and fitted assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import attack_workspace as workspace


MANIFEST = Path(__file__).with_name('manifest.json')


def verify_package() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding='utf-8'))
    root = workspace.ROOT.resolve()
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


if __name__ == '__main__':
    print(verify_package())

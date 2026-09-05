"""Create and verify a flat v5 attack submission archive.

Packaging is deliberately separate from candidate generation because it reads
the team token.  The token is never printed or copied into the candidate
directory.  A temporary archive must pass the official two-layer validator
before it replaces the requested output path.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STARTER_SRC = REPO_ROOT / "starter" / "src"
for source_root in (REPO_ROOT, STARTER_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from pwscup2026.common import config as config_mod  # noqa: E402
from pwscup2026.kit import validate as official_validate  # noqa: E402


REQUIRED_CANDIDATE_FILES = ("F_mia.csv", "F_aia.csv")
ZIP_TIMESTAMP = (2026, 9, 4, 0, 0, 0)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _write_member(archive: zipfile.ZipFile, name: str, value: bytes) -> None:
    """Write deterministic flat archive members for reproducible SHA-256."""
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, value)


def package_candidate(
    candidate_dir: Path,
    token_path: Path,
    output_zip: Path,
    targets_dir: Path,
) -> dict[str, Any]:
    candidate_dir = candidate_dir.resolve()
    token_path = token_path.resolve()
    output_zip = output_zip.resolve()
    targets_dir = targets_dir.resolve()

    if not candidate_dir.is_dir():
        raise ValueError(f"candidate directory does not exist: {candidate_dir}")
    names = sorted(path.name for path in candidate_dir.iterdir())
    if names != sorted(REQUIRED_CANDIDATE_FILES):
        raise ValueError(
            "candidate directory must contain only F_mia.csv and F_aia.csv: "
            f"observed={names}"
        )
    if not token_path.is_file():
        raise ValueError(f"token file does not exist: {token_path}")

    source_bytes = {
        name: (candidate_dir / name).read_bytes()
        for name in REQUIRED_CANDIDATE_FILES
    }
    token_bytes = token_path.read_bytes()
    source_hashes = {name: _sha256_bytes(value) for name, value in source_bytes.items()}

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_zip.stem}-",
            suffix=".zip",
            dir=output_zip.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(
            temporary_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name in REQUIRED_CANDIDATE_FILES:
                _write_member(archive, name, source_bytes[name])
            _write_member(archive, "token.txt", token_bytes)

        with zipfile.ZipFile(temporary_path) as archive:
            archive_names = sorted(archive.namelist())
            if archive_names != sorted((*REQUIRED_CANDIDATE_FILES, "token.txt")):
                raise RuntimeError(f"archive file set is invalid: {archive_names}")
            archived_hashes = {
                name: _sha256_bytes(archive.read(name))
                for name in REQUIRED_CANDIDATE_FILES
            }
        if archived_hashes != source_hashes:
            raise RuntimeError("archive CSV hashes differ from candidate inputs")

        cfg = config_mod.load_config(official_validate.KIT_CONFIG_PATH)
        result = official_validate.validate_submission_zip(
            temporary_path, cfg, targets_dir, "attack"
        )
        if not result.ok:
            raise ValueError(f"official package validator failed: {result.errors}")

        archive_hash = _sha256_bytes(temporary_path.read_bytes())
        temporary_path.replace(output_zip)
        temporary_path = None
        return {
            "output_zip": str(output_zip),
            "archive_sha256": archive_hash,
            "candidate_sha256": source_hashes,
            "official_validator": "PASS",
        }
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    team_dir = REPO_ROOT / "submissions" / "prelim" / "team24"
    parser = argparse.ArgumentParser(
        description="v5 candidateをtoken付き提出ZIPへ梱包し、公式validatorで検査します。"
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--token",
        type=Path,
        default=team_dir / "submitted-v5" / "token.txt",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "team-shared" / "prelim" / "attack-targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = package_candidate(
        args.candidate_dir, args.token, args.out, args.targets
    )
    print("PASS: official package validator")
    print(f"archive SHA-256: {result['archive_sha256']}")
    for name, digest in result["candidate_sha256"].items():
        print(f"{name} SHA-256: {digest}")


if __name__ == "__main__":
    main()

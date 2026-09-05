"""Prepare an explicit, secret-scanned path list for the private prelim archive.

Only reads credential reference files into memory for exact-match detection;
never outputs their values/hashes or adds them to the archive. Existing Git
history is deliberately not rewritten. Run from this repository. The script
does not stage, commit, push, or alter candidate CSVs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath
from collections.abc import Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parent
ROOT = WORKSPACE_ROOT.parents[1]
WORKSPACE_PREFIX = WORKSPACE_ROOT.relative_to(ROOT).as_posix()
RESEARCH_EXTENSIONS = {".py", ".md", ".json", ".csv"}
MANIFEST = f"{WORKSPACE_PREFIX}/research_archive/prelim-2026-manifest.json"
PATHSPEC = f"{WORKSPACE_PREFIX}/local-output/prelim-archive-pathspec.nul"
KIT_SAMPLE_TOKEN_PATHS = {
    f"{WORKSPACE_PREFIX}/participant_data/token.txt",
    f"{WORKSPACE_PREFIX}/participant_data/sample_submissions/token.txt",
}
KIT_SAMPLE_ZIP_RELATIVE_PATHS = {
    "participant_data/sample_submissions/attack_sub.zip",
    "participant_data/sample_submissions/process_sub.zip",
}
KIT_SAMPLE_ZIP_PATHS = {
    f"{WORKSPACE_PREFIX}/{path}" for path in KIT_SAMPLE_ZIP_RELATIVE_PATHS
}
CACHE_PARTS = {
    "__pycache__", ".venv", ".tools", ".uv-cache", ".cache",
    ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints",
}
FORBIDDEN_RAW_HASHES = {
    "7A7AA39E16FA5C76EE9C8DC009928373AEC6CF47E7B1BD10DC4E6899B5D2AB3A",
    "297354A3C909FBA6BF4D33053CF91EFA9139892BF5269050A5607B86E0F33E35",
    "F4817FBA6C1434D6B74221C2B5ABE90E4A53E80D8FA17E12602D43ADAE0A3126",
}
SECRET_PATTERNS = (
    re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def git_paths(*arguments: str) -> set[str]:
    data = subprocess.check_output(["git", *arguments, "-z"], cwd=ROOT)
    return {p.decode("utf-8") for p in data.split(b"\0") if p}


def workspace_relative_path(path: str) -> str:
    """Accept Git-root paths and the historical workspace-relative API."""
    normalized = PurePosixPath(path.replace("\\", "/")).as_posix()
    prefix = WORKSPACE_PREFIX + "/"
    return normalized[len(prefix):] if normalized.startswith(prefix) else normalized


def is_research_path(path: str) -> bool:
    return workspace_relative_path(path).startswith("local-output/")


def excluded_reason(path: str) -> str | None:
    item = PurePosixPath(path.replace("\\", "/"))
    if item.name.lower() == "token.txt" or item.name.lower().startswith(".env"):
        return "credential file; not part of new archive"
    if (any(part.lower() in CACHE_PARTS or part.lower().endswith(".egg-info")
            for part in item.parts)
            or item.suffix.lower() in {".pyc", ".pyo"}):
        return "regenerable execution environment or cache"
    relative = workspace_relative_path(path)
    if item.suffix.lower() == ".zip" and relative not in KIT_SAMPLE_ZIP_RELATIVE_PATHS:
        return "submission archive may contain live token"
    if is_research_path(path):
        if item.suffix.lower() not in RESEARCH_EXTENSIONS:
            return "not an allowlisted research text artifact"
    return None


def assert_staged_scope(index_paths: Iterable[str], expected_paths: Iterable[str]) -> None:
    """Reject credentials/caches and unexamined additions anywhere in the index.

    The two original public kit/practice token files are retained deliberately.
    No other token path is exempt, including renamed anonymization submissions.
    This function only inspects names; it never reads or reports token contents.
    """
    indexed = set(index_paths)
    forbidden = sorted(path for path in indexed
                       if path not in KIT_SAMPLE_TOKEN_PATHS and excluded_reason(path))
    if forbidden:
        raise ValueError("excluded file is present in the staged tree: " + ", ".join(forbidden))
    unexpected = sorted(indexed - set(expected_paths) - KIT_SAMPLE_TOKEN_PATHS)
    if unexpected:
        raise ValueError("staged paths are absent from the inspected manifest: "
                         + ", ".join(unexpected))


def scan_payload(payload: bytes, secret_values: list[bytes]) -> list[str]:
    reasons = []
    if any(value in payload for value in secret_values):
        reasons.append("exact known credential value")
    if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
        reasons.append("credential/private-key pattern")
    if hashlib.sha256(payload).hexdigest().upper() in FORBIDDEN_RAW_HASHES:
        reasons.append("actual private raw distribution file")
    return reasons


def scan_existing_sample_zip(payload: bytes, secret_values: list[bytes]) -> list[str]:
    """Inspect existing kit sample members in memory without displaying values."""
    findings = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if member.file_size >= 100 * 1024 * 1024:
                raise ValueError("oversized existing sample archive member")
            reasons = scan_payload(archive.read(member), secret_values)
            if reasons:
                findings.append("credential/raw-data match inside existing sample archive")
    return findings


def verify_staged_manifest() -> None:
    """Compare every staged blob with the manifest; never print blob contents."""
    report = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    entries = report["files"] + [{"path": MANIFEST, "sha256": hashlib.sha256(
        (ROOT / MANIFEST).read_bytes()).hexdigest().upper()}]
    raw = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    index = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, path = item.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        if stage != b"0":
            raise ValueError("unmerged index is not an archive")
        index[path.decode("utf-8")] = oid.decode("ascii")
    assert_staged_scope(index, (entry["path"] for entry in entries))
    reader = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    hashes, problems = {}, []
    try:
        for entry in entries:
            relative = entry["path"]
            oid = index.get(relative)
            if oid is None:
                problems.append({"path": relative, "reason": "missing from staged tree"})
                continue
            if oid not in hashes:
                reader.stdin.write((oid + "\n").encode("ascii"))
                reader.stdin.flush()
                header = reader.stdout.readline().split()
                if len(header) != 3 or header[1] != b"blob":
                    raise ValueError("unexpected Git object in staged archive")
                payload = reader.stdout.read(int(header[2]))
                if reader.stdout.read(1) != b"\n":
                    raise ValueError("invalid Git batch response")
                hashes[oid] = hashlib.sha256(payload).hexdigest().upper()
            if hashes[oid] != entry["sha256"]:
                problems.append({"path": relative, "reason": "staged bytes differ from manifest"})
    finally:
        reader.stdin.close()
        reader.stdout.close()
        reader.wait()
    if problems:
        print(json.dumps({"staged_verification": "FAIL", "problems": problems}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"staged_verification": "PASS", "verified_paths": len(entries),
                      "live_token_in_current_index": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-staged", action="store_true")
    parser.add_argument("--credential-reference", type=Path, action="append",
                        help="local token file used only for in-memory exact-match scanning")
    args = parser.parse_args()
    if args.verify_staged:
        verify_staged_manifest()
        return
    if not args.credential_reference:
        parser.error("--credential-reference is required when preparing the archive")
    secrets = []
    for path in args.credential_reference:
        value = path.read_bytes().strip()
        if len(value) < 8:
            raise ValueError("credential reference is too short for a reliable exact-match audit")
        if value not in secrets:
            secrets.append(value)
    selected = git_paths("ls-files") | git_paths("ls-files", "--others", "--exclude-standard")
    research_root = WORKSPACE_ROOT / "local-output"
    selected.update(p.relative_to(ROOT).as_posix() for p in research_root.rglob("*") if p.is_file())
    selected.discard(MANIFEST)
    selected.discard(PATHSPEC)
    entries, excluded, findings = [], [], []
    for relative in sorted(selected):
        path = ROOT / relative
        reason = excluded_reason(relative)
        if reason:
            excluded.append({"path": relative, "reason": reason})
            continue
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"unsafe or missing archive path: {relative}")
        payload = path.read_bytes()
        if len(payload) >= 100 * 1024 * 1024:
            raise ValueError(f"file exceeds the private Git archive size gate: {relative}")
        matches = scan_payload(payload, secrets)
        if path.suffix.lower() == ".zip":
            if relative not in KIT_SAMPLE_ZIP_PATHS:
                raise ValueError(f"unexpected non-submission ZIP in archive selection: {relative}")
            matches.extend(scan_existing_sample_zip(payload, secrets))
        if matches:
            findings.append({"path": relative, "reasons": matches})
        entries.append({"path": relative, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest().upper(),
                        "kind": "research_output" if is_research_path(relative) else "workspace_artifact"})
    if findings:
        # Print paths/categories only, never any matched content or credential hash.
        print(json.dumps({"secret_scan": "FAIL", "findings": findings}, indent=2))
        raise SystemExit(1)
    report = {
        "archive": "PWSCUP2026 prelim complete work, private repository only",
        "official_final_score": .3407,
        "selection": "tracked/untracked project files plus explicitly allowlisted ignored research text",
        "secret_scan": "PASS: exact supplied credentials, common credential patterns, private raw file hashes",
        "history": "existing token history retained at the owner's explicit request; live token excluded from current new archive",
        "not_a_guarantee_all_possible_secrets_are_detected": True,
        "existing_kit_sample_zip_members_scanned_in_memory": True,
        "file_count": len(entries), "total_bytes": sum(e["bytes"] for e in entries),
        "files": entries, "excluded": excluded,
    }
    target = ROOT / MANIFEST
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage_paths = [e["path"] for e in entries] + [MANIFEST]
    (ROOT / PATHSPEC).write_bytes(b"\0".join(p.encode("utf-8") for p in stage_paths) + b"\0")
    print(json.dumps({"status": "PASS", "file_count": len(entries),
                      "total_bytes": report["total_bytes"], "excluded_count": len(excluded),
                      "manifest": MANIFEST, "pathspec": PATHSPEC}, indent=2))


if __name__ == "__main__":
    main()

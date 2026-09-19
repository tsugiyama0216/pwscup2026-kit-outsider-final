"""Package the frozen R10 CSV and verify with the pinned official kit."""
from pathlib import Path
import json
import zipfile
import hashlib
from threadpoolctl import threadpool_limits
import r10

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE / 'submission/team10_R10_defense.zip'
EXPECTED_CSV = '8e5797cbff0be3d312b7bd55584caef4ed91793fb7e5109e334849d6e4a658bb'

def main():
    cfg, _, b, base, detector = r10.inputs()
    stored = json.loads((HERE/'report.json').read_text(encoding='utf-8'))
    if r10.sources() != stored['provenance']['source_sha256']:
        raise RuntimeError('Original R10 generator or configuration changed')
    csv_bytes = (HERE/'C.csv').read_bytes()
    if hashlib.sha256(csv_bytes).hexdigest() != EXPECTED_CSV:
        raise RuntimeError('Original R10 CSV changed')
    token_bytes = (r10.DATA/'token.txt').read_bytes()
    if not ZIP_PATH.exists():
        ZIP_PATH.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in [('C.csv',csv_bytes),('token.txt',token_bytes)]:
                info = zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info,value)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.namelist() != ['C.csv','token.txt'] or archive.testzip() is not None:
            raise RuntimeError('Invalid archive contents or CRC')
        if archive.read('C.csv') != csv_bytes or archive.read('token.txt') != token_bytes:
            raise RuntimeError('Archive differs from original CSV or token')
    from pwscup2026.kit.validate import check_package, validate_submission_zip
    from pwscup2026.kit.selfscore import load_submitted_c
    checked = check_package(ZIP_PATH,cfg)
    validated = validate_submission_zip(ZIP_PATH,cfg,dist_dir=r10.DATA,kind_override='defense')
    if not checked.ok or not validated.ok or checked.kind != 'defense':
        raise RuntimeError('Official submission validation failed')
    with threadpool_limits(limits=1):
        c = load_submitted_c(ZIP_PATH)
        r10.fixed_checks(base,c,detector)
        score, details = r10.evaluate_frame(c,b,cfg)
    for name, value in stored['score']['facets'].items():
        if abs(score['facets'][name]-value) > 1e-12:
            raise RuntimeError('Packaged data utility changed')
    result = dict(package_ok=True, names=['C.csv','token.txt'],
                  csv_matches_frozen_R10=True, token_matches_original=True,
                  utility=score['utility'],facets=score['facets'],validation=details,
                  C_sha256=EXPECTED_CSV,ZIP_sha256=r10.digest(ZIP_PATH),
                  ZIP_bytes=ZIP_PATH.stat().st_size,
                  package_script_sha256=r10.digest(Path(__file__)),
                  official_R10_auc=None,codabench_submitted=False)
    (HERE/'package_verification.json').write_text(
        json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    main()

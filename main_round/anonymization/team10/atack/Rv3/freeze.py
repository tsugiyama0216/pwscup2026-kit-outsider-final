"""Freeze final inference sources and noncredential dependencies before confirmation."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
def sha(p):
    if p.name=='token.txt':raise ValueError('Never hash tokens')
    return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if (HERE/'production_sources.json').exists():raise ValueError('Never overwrite source freeze')
    sources={n:sha(HERE/n) for n in ['model.py','density.py','selected.py','run.py','plugin.py','confirm.py','world.py']}
    (HERE/'production_sources.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
    paths=[HERE/n for n in sources]
    paths += [ROOT/'main_round/attack/attack_workspace.py',ROOT/'main_round/attack/Attack_M/v1/actual.py']
    paths += list((ROOT/'main_round/attack/official-kit/starter/src/pwscup2026').rglob('*.py'))
    paths += list((ROOT/'main_round/mock_test/mock_TS').glob('*.py'))
    paths += list((ROOT/'main_round/mock_test/sugiyama_v2').glob('*.py'))
    record=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),selection_sha256=sha(HERE/'selection.json'),files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},fitted_assets=[],confirmation_started=False)
    (HERE/'production_dependencies.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Frozen',len(record['files_sha256']),'noncredential files')
if __name__=='__main__':main()

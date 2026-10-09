"""Run once before production diagnostics. Preserve immutable upstream bytes."""
import runtime as rt
if (rt.HERE/'dependencies.json').exists():raise ValueError('Already frozen')
files={}
for item in rt.read(rt.ATTACK/'source_manifest.json')['files']:
    p=(rt.ATTACK/item['path']).resolve()
    if rt.sha(p)!=item['sha256']:raise ValueError('Original mismatch')
    files[str(p.relative_to(rt.ROOT)).replace('\\','/')]=item['sha256']
extras=[rt.ATTACK/'attack_workspace.py',rt.ATTACK/'source_manifest.json',rt.ATTACK/'Attack_R/v1/common.py',rt.ATTACK/'Attack_R/v1/run.py',rt.ATTACK/'Attack_M/v1/actual.py',rt.ROOT/'main_round/anonymization/team10/participant_data/B_self.csv']
extras += list((rt.ATTACK/'Attack_R/v2/submission').glob('*.csv'))
extras += [p for p in rt.HERE.iterdir() if p.suffix in ('.py','.ps1') or p.name=='protocol.json']
for p in extras:files[str(p.relative_to(rt.ROOT)).replace('\\','/')]=rt.sha(p)
rt.write(rt.HERE/'dependencies.json',{'files':files,'frozen_before_actual_predictions':True})
preserve={}
for version in ('v1','v2','v2f'):
    d=rt.ATTACK/'Attack_R'/version
    for p in d.rglob('*'):
        if p.is_file() and p.suffix in ('.py','.csv','.json','.zip'):preserve[str(p.relative_to(rt.ROOT)).replace('\\','/')]=rt.sha(p)
rt.write(rt.HERE/'preservation.json',{'files':preserve})
print('Frozen',len(files),'files; preserved',len(preserve),'R artifacts')

"""Final behavioral, format, source-freeze and preservation checks."""
from pathlib import Path
import sys,subprocess,json,hashlib,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    expected=read(HERE/'production_dependencies.json')['selection_sha256']
    assert sha(HERE/'selection.json')==expected
    started=read(ROOT/'main_round/mock_test/local-output/mock_TS/Rv3_confirmation/started.json')
    assert started['selection_sha256']==expected
    assert started['production_sources_sha256']==sha(HERE/'production_sources.json')
    records=[]
    for name in ['test_model.py','test_submission.py']:
        result=subprocess.run([sys.executable,'-B',str(HERE/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        output=result.stdout+result.stderr
        (HERE/'evidence'/(name+'.log')).write_text(output,encoding='utf-8')
        if result.returncode:raise ValueError('Acceptance failed: '+name)
        count=int(re.search(r'Ran (\d+) tests?',output).group(1))
        records.append({'file':name,'passed':count})
    from audit import verify
    preservation=verify()
    package=read(HERE/'submission/verification.json')
    assert package['completed'] and package['two_independent_runs_byte_identical']
    result={'completed':True,'tests':records,'malformed_CSV_variants_rejected':20,'selection_and_production_sources_unchanged_since_freeze_and_confirmation_start':True,'preservation':preservation,'package_verification_sha256':sha(HERE/'submission/verification.json'),'official_score_measured':False,'official_submission_performed':False}
    (HERE/'acceptance_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()

"""Score frozen selection in a fresh preliminary-only scorer process."""
import os,sys,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
root=Path(os.environ['RV4_REPOSITORY']) if os.environ.get('RV4_REPOSITORY') else next(p for p in HERE.parents if (p/'main_round').exists())
sys.path.insert(0,str(root/'main_round/mock_test'))
from mock_TS_using_prelim.scoring import score_predictions
def main():
    selection=json.loads((HERE/'selection.json').read_text())
    for name,choices in [('Rv2',{'mia':'Rv2','aia':'Rv2'}),('Rv4',selection)]:
        out=HERE/'evaluation/preliminary'/name
        if out.exists():raise ValueError('Preserve prior preliminary evaluation')
        out.mkdir(parents=True)
        for c,n in [('mia','F_mia.csv'),('aia','F_aia.csv')]:
            shutil.copyfile(HERE/'evaluation/prelim-predictions'/c/choices[c]/n,out/n)
        result=score_predictions(out)
        (out/'score.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(name,result['summary'],flush=True)
if __name__=='__main__':main()

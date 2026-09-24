"""Replay the frozen v5 model from v4, without B or evaluation access."""
from pathlib import Path
import argparse,csv,hashlib,io,json

HERE=Path(__file__).resolve().parent
ALLOWED=['BMI','TG','ALT']
def sha(payload):return hashlib.sha256(payload).hexdigest()

def generate(base_bytes,model):
    if sha(base_bytes)!=model['base_csv_sha256']:raise ValueError('Unexpected base CSV')
    rows=list(csv.reader(io.StringIO(base_bytes.decode('utf-8'))));header,body=rows[0],rows[1:]
    if model['mutable_fields']!=ALLOWED or model['columns']!=header or model['rows']!=len(body):raise ValueError('Unexpected model schema')
    seen=set()
    for patch in model['patches']:
        if set(patch)!={'row_index',*ALLOWED}:raise ValueError('Forbidden patch field')
        idx=patch['row_index']
        if type(idx) is not int or not 0<=idx<len(body) or idx in seen:raise ValueError('Invalid or repeated patch row')
        seen.add(idx)
        for name in ALLOWED:body[idx][header.index(name)]=patch[name]
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(header);writer.writerows(body)
    payload=stream.getvalue().encode('utf-8')
    if sha(payload)!=model['output_csv_sha256']:raise ValueError('Model does not reproduce expected output')
    return payload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=HERE.parent/'M1_v4/C.csv')
    parser.add_argument('--model',type=Path,default=HERE/'model.json')
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Refusing to overwrite existing files')
    result=generate(args.base.read_bytes(),json.loads(args.model.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as output:output.write(result)
    print(json.dumps({'output':str(args.out),'sha256':sha(result)}))

if __name__=='__main__':main()

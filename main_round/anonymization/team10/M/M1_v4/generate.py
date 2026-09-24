"""Reproduce frozen M1_v4 from M1_v3 + saved patches, without private B access."""
from pathlib import Path
import argparse, csv, hashlib, io, json

HERE=Path(__file__).resolve().parent

def digest(payload): return hashlib.sha256(payload).hexdigest()

def generate(base_bytes: bytes, model: dict) -> bytes:
    if digest(base_bytes)!=model['base_csv_sha256']:
        raise ValueError('Base M1_v3 bytes differ from the frozen original')
    rows=list(csv.reader(io.StringIO(base_bytes.decode('utf-8'))))
    header,body=rows[0],rows[1:]
    if header!=model['columns'] or len(body)!=model['rows']:
        raise ValueError('Unexpected base schema')
    if model['mutable_fields']!=['time','onset']:
        raise ValueError('Only time and onset patches are allowed')
    seen=set()
    for patch in model['patches']:
        if set(patch)!={'row_index','time','onset'}:
            raise ValueError('Unexpected patch fields')
        i=patch['row_index']
        if type(i) is not int or i<0 or i>=len(body) or i in seen:
            raise ValueError('Invalid or repeated patch row')
        seen.add(i)
        for name in model['mutable_fields']:
            body[i][header.index(name)]=patch[name]
    stream=io.StringIO(newline=''); writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(header); writer.writerows(body)
    payload=stream.getvalue().encode('utf-8')
    if digest(payload)!=model['output_csv_sha256']:
        raise ValueError('Reproduced CSV hash does not match frozen candidate')
    return payload

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=HERE.parent/'M1_v3/C.csv')
    parser.add_argument('--model',type=Path,default=HERE/'model.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('Refusing to overwrite any existing file')
    payload=generate(args.base.read_bytes(),json.loads(args.model.read_text(encoding='utf-8')))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('xb') as file: file.write(payload)
    print(json.dumps({'output':str(args.out),'sha256':digest(payload)}))

if __name__=='__main__': main()

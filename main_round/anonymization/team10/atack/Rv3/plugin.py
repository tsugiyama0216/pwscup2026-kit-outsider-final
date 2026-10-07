"""Single-target mock_TS adapter; no original data or prior fitted assets."""
from pathlib import Path
import importlib.util,json,hashlib,sys
HERE=Path(__file__).resolve().parent
def predict(**public):
    s=json.loads((HERE/'selection.json').read_text(encoding='utf-8'))
    for n,h in s['source_sha256'].items():
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h:raise ValueError('Frozen source changed')
    for n,h in json.loads((HERE/'production_sources.json').read_text()).items():
        if hashlib.sha256((HERE/n).read_bytes()).hexdigest()!=h:raise ValueError('Frozen production source changed')
    sys.path.insert(0,str(HERE))
    spec=importlib.util.spec_from_file_location('_rv3_numeric',HERE/'selected.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.predict_selected(s,**public)

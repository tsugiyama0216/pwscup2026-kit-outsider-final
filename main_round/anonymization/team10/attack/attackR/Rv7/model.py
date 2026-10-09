"""共分散を考慮する距離MIA. Reusable API for a new evaluator."""
from pathlib import Path
import importlib.util
import json

SHARED = Path(__file__).resolve().parent.parent / 'rv6_series'

def predict(candidates, background, targets, challenges, baseline_mia, baseline_aia, known_negative):
    spec = importlib.util.spec_from_file_location('_rv6_series_public_model', SHARED / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = json.loads((SHARED / 'protocol.json').read_text(encoding='utf-8'))
    results, _ = module.predict_all(candidates, background, targets, challenges,
                                   baseline_mia, baseline_aia, known_negative, settings)
    return results['Rv7']

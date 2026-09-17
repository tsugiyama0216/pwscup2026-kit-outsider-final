"""Six settings fixed before scoring; no seed or row-order search."""
import io
import json
import pandas as pd
from evaluation import HERE, check_environment, read_training_data, evaluate_frame
from sequential import synthesize

GRID = [(leaf, strength) for leaf in (20, 40, 60) for strength in (0.0, 0.5)]

if __name__ == '__main__':
    cfg, _ = check_environment()
    b = read_training_data()
    results = []
    for leaf, strength in GRID:
        c, audit = synthesize(b, cfg, leaf, strength)
        c = pd.read_csv(io.StringIO(c.to_csv(index=False)), float_precision='round_trip')
        score, validation = evaluate_frame(c, b, cfg)
        result = {'leaf_size': leaf, 'strength': strength, 'utility': score['utility'],
                  'facets': score['facets'], 'clipped': audit['calibration']['clipped_counts'],
                  'rare': validation['rare_diagnostic']}
        results.append(result)
        print(json.dumps(result), flush=True)
    eligible = [r for r in results if not any(r['clipped'].values()) and not r['rare']['below_gate']]
    best = max(eligible, key=lambda r: r['utility'])
    (HERE/'experiment_summary.json').write_text(json.dumps({'grid': results, 'selected': best}, indent=2)+'\n', encoding='utf-8')

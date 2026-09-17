"""Sequential CART conditional synthesis; independent leaf draws per column."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from survival import synthesize_outcomes
from calibration import calibrate

SEED = 20260918
ORDER = ['sex', 'age', 'smoking', 'BMI', 'FPG', 'TG', 'HDL', 'SBP', 'ALT', 'prefecture']
CATEGORICAL = {'sex', 'smoking', 'prefecture'}

def synthesize(b, cfg, leaf_size=40, strength=0.5):
    if leaf_size < 10 or len(b) < 2 * leaf_size:
        raise ValueError('Insufficient conditional leaf support')
    rng = np.random.default_rng(SEED)
    c = pd.DataFrame(index=range(len(b)))
    audit = []
    previous = []
    for column in ORDER:
        values = b[column].to_numpy()
        labels = None
        if column == 'prefecture':
            labels, values = np.unique(values, return_inverse=True)
        transformed = np.log(values) if column in ('TG', 'ALT') else values.astype(float)
        if previous:
            cls = DecisionTreeClassifier if column in CATEGORICAL else DecisionTreeRegressor
            tree = cls(min_samples_leaf=leaf_size, max_depth=7, random_state=SEED)
            tree.fit(b[previous], transformed)
            source_leaves = tree.apply(b[previous])
            target_leaves = tree.apply(c[previous])
        else:
            source_leaves = np.zeros(len(b), dtype=int)
            target_leaves = np.zeros(len(b), dtype=int)
        output = np.empty(len(b))
        sizes = []
        for leaf in np.unique(target_leaves):
            pool = transformed[source_leaves == leaf]
            mask = target_leaves == leaf
            sizes.append(len(pool))
            # Empirical categorical draws; continuous inverse ECDF interpolation
            # produces new values within supported leaf ranges without row donors.
            if column in CATEGORICAL or column == 'age':
                output[mask] = rng.choice(pool, int(mask.sum()))
            else:
                output[mask] = np.quantile(pool, rng.random(int(mask.sum())), method='linear')
        if column in ('TG', 'ALT'):
            output = np.exp(output)
        c[column] = labels[output.astype(int)] if labels is not None else (output.astype(int) if column in CATEGORICAL or column == 'age' else output)
        audit.append({'column': column, 'conditions': list(previous),
                      'minimum_leaf_support': min(sizes), 'leaves_used': len(sizes)})
        previous.append(column)
    outcomes, survival_audit = synthesize_outcomes(b, c, float(cfg['onset']['horizon_years']), seed=SEED, calibration_steps=0)
    for name in ('time', 'onset', 'death'):
        c[name] = outcomes[name].to_numpy()
    c = c[['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'smoking', 'FPG', 'time', 'onset', 'death']]
    c, calibration_audit = calibrate(b, c, cfg, strength=strength)
    return c, {'sequence': audit, 'survival': survival_audit, 'calibration': calibration_audit}

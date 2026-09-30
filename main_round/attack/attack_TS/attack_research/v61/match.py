"""Full-record exact matching after the public rounded-release transform."""
import numpy as np
import pandas as pd

from local_eval.data import FEATURES


WIDTHS = {'age': 5., 'BMI': 1., 'SBP': 10., 'TG': 10., 'HDL': 5.,
          'ALT': 5., 'FPG': 10., 'time': .5}


def rounded_public_match(c, candidates, schema):
    transformed = candidates[FEATURES].copy().reset_index(drop=True)
    for column, width in WIDTHS.items():
        transformed[column] = np.round(transformed[column].to_numpy(float)/width)*width
    for column, bounds in schema['ranges'].items():
        transformed[column] = transformed[column].clip(*bounds)
    transformed['age'] = np.rint(transformed.age.clip(*schema['age_range'])).astype(int)
    transformed['time'] = transformed.time.clip(1/365.25, 10)
    query = pd.MultiIndex.from_frame(transformed[FEATURES])
    released = pd.MultiIndex.from_frame(c[FEATURES].reset_index(drop=True))
    return query.isin(released).astype(float)

"""Replace a bootstrap AIA estimate only when public QI uniquely fixes time."""
import numpy as np

from local_eval.data import QI


def correct(c,challenge,prediction):
    baseline=np.asarray(prediction,dtype=float)
    if baseline.shape!=(len(challenge),):raise ValueError('Prediction length differs from challenge')
    release=c[QI+['time']].groupby(QI,sort=False,dropna=False).time.agg(['nunique','first'])
    lookup=release.reindex(challenge[QI].set_axis(release.index.names,axis='columns').apply(tuple,axis=1))
    exact=lookup['nunique'].eq(1).to_numpy()
    observed=lookup['first'].to_numpy(float)
    corrected=np.where(exact,observed,baseline)
    if not np.isfinite(corrected).all():raise ArithmeticError('Nonfinite corrected time')
    return corrected,exact

"""Train a QI-only event/time prior on people disjoint from confirmation."""
import numpy as np
from scipy.special import ndtr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from local_eval.data import QI


NUMERIC = ('age', 'SBP', 'TG', 'HDL', 'ALT')
CATEGORICAL = ('sex', 'prefecture')


class PublicTimePrior:
    def __init__(self, event_model, event_numeric, event_sex, event_prefecture,
                 event_time, center, scale, non_event_time, bandwidth, time_sd):
        self.event_model = event_model
        self.event_numeric = event_numeric
        self.event_sex = event_sex
        self.event_prefecture = event_prefecture
        self.event_time = event_time
        self.center = center
        self.scale = scale
        self.non_event_time = float(non_event_time)
        self.bandwidth = float(bandwidth)
        self.time_sd = float(time_sd)

    @classmethod
    def fit(cls, development, protocol):
        x = development[QI]
        event = ((development.onset.to_numpy()+development.death.to_numpy()) > 0)
        preprocessor = ColumnTransformer([
            ('numeric', StandardScaler(), list(NUMERIC)),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), list(CATEGORICAL)),
        ])
        model = make_pipeline(preprocessor,
            LogisticRegression(C=float(protocol['event_logistic_C']), max_iter=1000,
                               random_state=20260927))
        model.fit(x, event.astype(int))
        center = x[list(NUMERIC)].to_numpy(float).mean(axis=0)
        scale = np.maximum(x[list(NUMERIC)].to_numpy(float).std(axis=0), 1e-6)
        event_rows = x.iloc[np.flatnonzero(event)]
        return cls(model, (event_rows[list(NUMERIC)].to_numpy(float)-center)/scale,
            event_rows.sex.to_numpy(), event_rows.prefecture.astype(str).to_numpy(),
            development.loc[event, 'time'].to_numpy(float), center, scale,
            float(np.median(development.loc[~event, 'time'])),
            protocol['kernel_bandwidth'], protocol['event_time_smoothing_sd'])

    def hit_probability(self, challenge, proposals):
        x = challenge[QI]
        event_probability = self.event_model.predict_proba(x)[:, 1]
        numeric = (x[list(NUMERIC)].to_numpy(float)-self.center)/self.scale
        delta = numeric[:, None, :]-self.event_numeric[None, :, :]
        distance_squared = np.einsum('ijk,ijk->ij', delta, delta)
        distance_squared += (x.sex.to_numpy()[:, None] != self.event_sex[None, :]).astype(float)
        distance_squared += .5*(x.prefecture.astype(str).to_numpy()[:, None]
                                != self.event_prefecture[None, :]).astype(float)
        log_weight = -.5*distance_squared/self.bandwidth**2
        log_weight -= log_weight.max(axis=1, keepdims=True)
        weight = np.exp(log_weight)
        weight /= weight.sum(axis=1, keepdims=True)
        difference = proposals[:, :, None]-self.event_time[None, None, :]
        per_event = ndtr((difference+.5)/self.time_sd)-ndtr((difference-.5)/self.time_sd)
        hit_event = np.einsum('ik,ijk->ij', weight, per_event, optimize=True)
        hit_non_event = (np.abs(proposals-self.non_event_time) <= .5).astype(float)
        result = event_probability[:, None]*hit_event+(1-event_probability[:, None])*hit_non_event
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ArithmeticError('Invalid public-QI time prior')
        return result

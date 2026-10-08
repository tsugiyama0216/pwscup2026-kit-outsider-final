"""Fixed-kernel Gen-LRA using only supplied continuous public columns.

The output is a strictly increasing transform of the local log likelihood
ratio, retaining tiny positive evidence without subtraction or density floors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

FEATURES = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
K_NEAREST = 10
EIGENVALUE_FLOOR = 1e-6
BATCH_SIZE = 256


def _numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    missing = set(FEATURES).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing continuous columns: {sorted(missing)}")
    values = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite continuous values are not supported")
    return values


def _log_softplus(values: np.ndarray) -> np.ndarray:
    """log(log(1+exp(z))) without losing negative-tail distinctions."""
    result = np.empty_like(values, dtype=float)
    negative = values < -30.0
    positive = values > 30.0
    middle = ~(negative | positive)
    # At z < -30 the relative error is <= exp(-30)/2.
    result[negative] = values[negative]
    result[positive] = np.log(np.logaddexp(0.0, values[positive]))
    result[middle] = np.log(np.logaddexp(0.0, values[middle]))
    return result


class FixedKernelGenLRA:
    """Reference-only preprocessing and covariance shared by R and R+x.

    This is an explicit fixed-kernel variant, not the public implementation's
    covariance/bandwidth re-fit for every augmented reference table.
    """

    def __init__(self, background: pd.DataFrame, *, k: int = K_NEAREST,
                 eigenvalue_floor: float = EIGENVALUE_FLOOR,
                 batch_size: int = BATCH_SIZE):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
            raise ValueError("eigenvalue_floor must be finite and positive")
        values = _numeric_frame(background)
        if len(values) < 2:
            raise ValueError("At least two background rows are required")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("Every reference column needs a finite observation")
        self.median = np.nanmedian(values, axis=0)
        values = np.where(np.isnan(values), self.median, values)
        self.center = np.mean(values, axis=0)
        self.scale = np.std(values, axis=0, ddof=1)
        self.scale = np.where(self.scale > 0.0, self.scale, 1.0)
        reference = (values - self.center) / self.scale
        self.n_reference, self.n_features = reference.shape
        covariance = np.cov(reference, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        self.covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.bandwidth = (
            self.n_reference * (self.n_features + 2.0) / 4.0
        ) ** (-1.0 / (self.n_features + 4.0))
        self.reference = reference
        self.reference_whitened = self._whiten(reference)
        self.k = int(k)
        self.batch_size = int(batch_size)

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = _numeric_frame(frame)
        values = np.where(np.isnan(values), self.median, values)
        return (values - self.center) / self.scale

    def _whiten(self, values: np.ndarray) -> np.ndarray:
        return solve_triangular(self.cholesky, values.T, lower=True).T / self.bandwidth

    def score(self, release: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
        """Return log positive evidence, monotone in fixed-kernel Gen-LRA.

        Let q(s,x)=K_H(s-x)/(m*p_R(s)).
        G(x)=sum(log1p(q))-k*log1p(1/m).
        This method returns log(sum(log1p(q))) instead, preserving the same
        ranking at fixed k while avoiding cancellation in low-density tails.
        Normalization of K_H cancels in q and is never exponentiated.
        """
        if len(release) == 0:
            raise ValueError("A nonempty release is required")
        synth = self._transform(release)
        # Canonical seven-feature order makes distance-boundary ties
        # independent of the input release row order. Equal feature rows
        # have identical KDE contributions, so their internal order is moot.
        canonical = np.lexsort(synth[:, ::-1].T)
        synth = synth[canonical]
        synth_w = self._whiten(synth)
        query = self._transform(candidates)
        query_w = self._whiten(query)
        k = min(self.k, len(synth))
        # log of sum of unnormalized reference kernels; equals log(m*p_R)
        # up to the Gaussian normalization shared with the candidate kernel.
        log_reference_sum = np.empty(len(synth))
        for start in range(0, len(synth), self.batch_size):
            stop = min(start + self.batch_size, len(synth))
            squared = cdist(synth_w[start:stop], self.reference_whitened, "sqeuclidean")
            log_reference_sum[start:stop] = logsumexp(-0.5 * squared, axis=1)
        result = np.empty(len(query))
        for start in range(0, len(query), self.batch_size):
            stop = min(start + self.batch_size, len(query))
            distances = cdist(query[start:stop], synth, "sqeuclidean")
            # Stable distance ties use the canonical seven-feature order above.
            neighbors = np.argsort(distances, axis=1, kind="stable")[:, :k]
            selected_w = synth_w[neighbors]
            diff = selected_w - query_w[start:stop, None, :]
            log_kernel = -0.5 * np.sum(diff * diff, axis=2)
            log_q = log_kernel - log_reference_sum[neighbors]
            result[start:stop] = logsumexp(_log_softplus(log_q), axis=1)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite Gen-LRA scores: input magnitude exceeds numeric range")
        return result

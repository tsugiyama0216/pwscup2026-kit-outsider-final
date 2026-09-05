"""Fixed-bandwidth Gaussian local likelihood influence attack.

Reference: Ward, Wang, Cheng (2025), https://arxiv.org/abs/2508.21146,
equation 3 and Algorithm 1. This implementation freezes the reference KDE
bandwidth when adding one query. The analytic mixture update is equivalent to
refitting that fixed-bandwidth KDE, without one estimator fit per query.

Only finite matrices are accepted. There is no file, token, network, or
submission functionality. A score is evidence, not a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from sklearn.neighbors import KernelDensity, NearestNeighbors


def _matrix(value: np.ndarray, name: str, dimensions: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (array.ndim != 2 or not array.shape[0] or not array.shape[1]
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    if dimensions is not None and array.shape[1] != dimensions:
        raise ValueError(f"{name} dimensions differ from reference")
    return array


@dataclass
class LocalLikelihoodInfluence:
    reference: np.ndarray
    bandwidth: float

    def __post_init__(self) -> None:
        self.reference = _matrix(self.reference, "reference").copy()
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite")
        self.bandwidth = float(self.bandwidth)
        self.model = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
        self.model.fit(self.reference)

    def score_many(
        self, queries: np.ndarray, released: np.ndarray, *,
        neighbours: Iterable[int] = (1, 3, 5),
    ) -> dict[str, np.ndarray]:
        """Score several local sizes and same-encoding comparison baselines.

        The released rows are sorted by coordinates for deterministic nearest
        neighbour ties. Repeated released rows remain repeated observations.
        The returned DOMIAS field is a fixed-bandwidth KDE baseline, not the
        repository's complete v4 ensemble.
        """
        dimensions = self.reference.shape[1]
        queries = _matrix(queries, "queries", dimensions)
        released = _matrix(released, "released", dimensions)
        sizes = tuple(neighbours)
        if (not sizes or len(set(sizes)) != len(sizes)
                or any(not isinstance(k, Integral) or isinstance(k, bool)
                       or not 1 <= k <= len(released) for k in sizes)):
            raise ValueError("neighbours must be unique integers within release size")
        order = np.lexsort(released.T[::-1])
        ordered = released[order]
        distance, indices = NearestNeighbors(n_neighbors=max(sizes)).fit(ordered).kneighbors(queries)
        if not np.isfinite(distance).all():
            raise ValueError("distance overflow")
        log_reference = self.model.score_samples(ordered)
        log_kernel = (-0.5 * dimensions * np.log(2 * np.pi)
                      - dimensions * np.log(self.bandwidth)
                      - 0.5 * (distance / self.bandwidth) ** 2)
        n = len(self.reference)
        # log[(n p_R(s) + K_h(s-x)) / ((n+1) p_R(s))].
        increment = (np.log(n / (n + 1.0))
                     + np.logaddexp(0.0, log_kernel - np.log(n) - log_reference[indices]))
        cumulative = np.cumsum(increment, axis=1)
        result = {f"gen_lra_k{k}": cumulative[:, k - 1] for k in sizes}
        result["dcr"] = -distance[:, 0]
        released_model = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(ordered)
        result["domias_kde"] = released_model.score_samples(queries) - self.model.score_samples(queries)
        if any(not np.isfinite(score).all() for score in result.values()):
            raise ValueError("non-finite density influence; rescale inputs")
        return result


def reference_bandwidth(reference: np.ndarray, *, multiplier: float = 1.0) -> float:
    """A reference-only bandwidth; never tuned on evaluation membership labels."""
    reference = _matrix(reference, "reference")
    if len(reference) < 2 or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("bandwidth requires two reference rows and positive multiplier")
    count = min(6, len(reference))
    distances = NearestNeighbors(n_neighbors=count).fit(reference).kneighbors(reference)[0][:, -1]
    positive = distances[distances > 1e-12]
    scale = float(np.median(positive)) if len(positive) else 1.0
    return max(1e-8, multiplier * scale)

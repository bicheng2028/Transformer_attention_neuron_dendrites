"""Resampling statistics for the delay-key experiments.

Every reported quantity in the revision is an average over independent seeds
with an interval attached. These helpers keep that uniform, and keep the
interval method explicit in one place so Methods can state it once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Estimate", "bootstrap_mean", "summarise"]


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap confidence interval.

    Attributes:
        mean: sample mean across seeds.
        lo: lower bound of the percentile bootstrap interval.
        hi: upper bound of the percentile bootstrap interval.
        n: number of independent seeds contributing.
        sd: sample standard deviation across seeds.
    """

    mean: float
    lo: float
    hi: float
    n: int
    sd: float

    def __str__(self) -> str:
        return f"{self.mean:.4g} [{self.lo:.4g}, {self.hi:.4g}] (n={self.n})"


def bootstrap_mean(
    values,
    rng: np.random.Generator,
    n_boot: int = 10_000,
    confidence: float = 0.95,
) -> Estimate:
    """Percentile bootstrap interval for the mean of ``values``.

    A percentile bootstrap over seeds is used rather than a normal
    approximation because several of the reported quantities are proportions
    that sit near zero, where a symmetric interval would cross into negative
    values.

    Args:
        values: one-dimensional sample, typically one entry per seed.
        rng: random generator for the resampling.
        n_boot: number of bootstrap resamples.
        confidence: coverage of the returned interval.

    Returns:
        An :class:`Estimate`. For a single observation the interval collapses to
        the observation itself.

    Raises:
        ValueError: if ``values`` is empty.
    """
    sample = np.asarray(values, dtype=np.float64).ravel()
    if sample.size == 0:
        raise ValueError("cannot summarise an empty sample")

    mean = float(sample.mean())
    sd = float(sample.std(ddof=1)) if sample.size > 1 else 0.0

    if sample.size == 1:
        return Estimate(mean=mean, lo=mean, hi=mean, n=1, sd=0.0)

    draws = rng.integers(0, sample.size, size=(n_boot, sample.size))
    boot_means = sample[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boot_means, [tail, 1.0 - tail])

    return Estimate(mean=mean, lo=float(lo), hi=float(hi), n=int(sample.size), sd=sd)


def summarise(
    values,
    rng: np.random.Generator,
    **kwargs,
) -> dict:
    """Return :func:`bootstrap_mean` as a plain dict, for tabular output."""
    est = bootstrap_mean(values, rng, **kwargs)
    return {"mean": est.mean, "lo": est.lo, "hi": est.hi, "n": est.n, "sd": est.sd}

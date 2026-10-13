"""Detector-class probabilities from the arrival statistics alone.

The dispersion sweep (``experiments/e1_dispersion.py``) classifies each neuron
from three yes-or-no answers: does it fire for event 1 alone (F1), for event 2
alone (F2), for the two together (F12)? Each answer is a first-passage
question about one branch's dendritic potential, and that potential depends on
nothing but when inputs arrive. This module computes the three probabilities
without simulating a neuron, and from them every class fraction. The
simulation then has to agree with it, which is the test that the sweep
measures geometry and not something else.

Arrivals
    Each of an event's ``n_per_event`` features connects to the branch with
    probability ``p``. A connected feature's spike leaves at ``onset + o``, ``o``
    uniform on ``{0..spread}``, and arrives ``d`` later, ``d`` uniform on
    ``{0..Delta}``. A feature therefore lands in bin ``b`` with probability
    ``p q(b)``, where ``q`` is the trapezoid that convolves the two uniforms,
    and it lands at most once. That last constraint matters. Treating bins as
    independent lets a feature arrive twice, which fattens the tail of the
    count and overstates firing by 15-30% at ``theta = 4.5``. So the
    calculation carries the number of features still to arrive. Given that
    ``r`` features are unresolved at bin ``t``, the number arriving is binomial
    in ``r``, with the conditional probability ``p q(t) / (1 - p Q(<t))``.

Passage
    The potential obeys ``x_t = a x_{t-1} + N_t`` with ``a = exp(-1/tau_den)``,
    exactly as in :func:`delaykey.core.simulate_population`. The neuron fires
    if ``x_t`` ever exceeds ``theta + eps``. The margin ``eps`` exists because a
    plateau of amplitude ``tanh(x - theta)`` must still drive the soma to
    threshold within the plateau. Its value follows from ``alpha``,
    ``tau_mem`` and the plateau length (:func:`soma_margin`): about 0.05 for
    the parameters used here. The joint distribution of potential and
    remaining features is propagated bin by bin, on a grid of ``x`` fine enough
    that an integer count shifts it exactly. Mass that crosses is removed, and
    what remains at the end is the probability of never firing. Nothing is
    sampled.

Classes
    F1 and F2 involve disjoint features, so they are independent. Before any
    crossing the potentials add, ``x12 = x1 + x2``, and ``x12`` staying below
    threshold forces both ``x1`` and ``x2`` to stay below it. So
    ``not F12`` implies ``not F1 and not F2``, and

        P(sequence) = P(not F1) P(not F2) - P(not F12)
        P(event1)   = P(F1) P(not F2)        P(dual) = P(F1) P(F2)

    match :func:`delaykey.detectors.classify_detectors` term for term. The
    two-event calculation carries both events' remaining features, because
    they overlap in time exactly where sequence detectors are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

__all__ = ["ClassProbabilities", "arrival_profile", "soma_margin",
           "never_fires", "class_probabilities", "crossover", "peak"]


@dataclass(frozen=True)
class ClassProbabilities:
    """Expected population fraction of each detector class."""

    sequence: float
    event1: float
    event2: float
    dual: float
    silent: float

    @property
    def event(self) -> float:
        """Event detectors of either event, the class the figure plots."""
        return self.event1 + self.event2


def arrival_profile(dispersion: int, spread: int) -> np.ndarray:
    """Probability that a connected spike arrives ``k`` steps after its onset.

    The convolution of uniform ``{0..spread}`` (offset) with uniform
    ``{0..dispersion}`` (delay): a trapezoid on ``{0..spread + dispersion}``.
    """
    offset = np.full(spread + 1, 1.0 / (spread + 1))
    delay = np.full(dispersion + 1, 1.0 / (dispersion + 1))
    return np.convolve(offset, delay)


def soma_margin(alpha: float, tau_mem: float, plateau_steps: int,
                v_threshold: float = 1.0, eta: float = 1.0) -> float:
    """How far past ``theta`` the potential must go for the soma to fire.

    A plateau of amplitude ``A = tanh(x - theta)`` drives the soma by
    ``alpha eta A`` per step for ``plateau_steps`` steps, against a leak of
    ``tau_mem``. The soma reaches threshold within the plateau only if ``A``
    exceeds ``v_threshold / (alpha eta S)``, where ``S`` sums the leak over the
    plateau.
    """
    s = np.sum(np.exp(-np.arange(plateau_steps) / tau_mem))
    a_min = v_threshold / (alpha * eta * s)
    return float(np.arctanh(min(a_min, 1.0 - 1e-12)))


def _arrival_rates(onset: int, profile: np.ndarray, p_connect: float,
                   n_time: int) -> np.ndarray:
    """Per-bin probability that a still-unresolved feature arrives now.

    A feature is unresolved at ``t`` if it has not arrived before ``t``; it may
    yet arrive, or never connect. Arrivals after the trial are never seen.
    """
    q = np.zeros(n_time)
    idx = onset + np.arange(profile.size)
    keep = idx < n_time
    q[idx[keep]] = profile[keep]
    resolved = p_connect * np.concatenate([[0.0], np.cumsum(q)[:-1]])
    return p_connect * q / (1.0 - resolved)


def _binomial_table(n: int, rate: float, arrived_max: int, k_max: int) -> np.ndarray:
    """``[a, k]``: probability that ``k`` of the ``n - a`` unresolved arrive."""
    table = np.zeros((arrived_max + 1, k_max + 1))
    for a in range(arrived_max + 1):
        r = n - a
        for k in range(min(k_max, r) + 1):
            table[a, k] = comb(r, k) * rate ** k * (1.0 - rate) ** (r - k)
    return table


def never_fires(rates1: np.ndarray, rates2: np.ndarray, n_per_event: int,
                threshold: float, tau_den: float, *, resolution: int = 1000,
                arrived_max: int = 16, k_max: int = 7) -> float:
    """Probability that the potential driven by up to two events never crosses.

    Args:
        rates1, rates2: per-bin arrival probability of an unresolved feature
            of each event (:func:`_arrival_rates`); all zeros for an absent
            event.
        n_per_event: features per event.
        threshold: firing level for the potential.
        tau_den: dendritic time constant, in steps.
        resolution: grid points per unit of potential. A count of ``k`` shifts
            the grid by exactly ``k * resolution`` points; the decay is rounded
            to the grid, which bounds the error in ``x`` by ``1.3 / resolution``.
        arrived_max, k_max: arrivals tracked per event in total and per bin.
            Mass beyond either is dropped; at ``p = 0.1`` with 40 features it
            is below ``1e-6``.
    """
    a = np.exp(-1.0 / tau_den)
    size = int(np.floor(threshold * resolution)) + 1
    dims = [arrived_max + 1 if np.any(r > 0) else 1 for r in (rates1, rates2)]
    mass = np.zeros((dims[0], dims[1], size))
    mass[0, 0, 0] = 1.0
    decayed = np.rint(np.arange(size) * a).astype(np.int64)
    flat = (np.arange(dims[0] * dims[1])[:, None] * size + decayed[None, :]).ravel()
    for r1, r2 in zip(rates1, rates2):
        mass = np.bincount(flat, weights=mass.ravel(),
                           minlength=mass.size).reshape(mass.shape)
        if r1 == 0.0 and r2 == 0.0:
            continue
        b1 = _binomial_table(n_per_event, r1, dims[0] - 1, k_max if r1 > 0 else 0)
        b2 = _binomial_table(n_per_event, r2, dims[1] - 1, k_max if r2 > 0 else 0)
        new = np.zeros_like(mass)
        for k1 in range(b1.shape[1]):
            for k2 in range(b2.shape[1]):
                shift = (k1 + k2) * resolution
                if shift >= size:
                    continue
                n1, n2 = dims[0] - k1, dims[1] - k2
                coef = b1[:n1, k1][:, None] * b2[:n2, k2][None, :]
                new[k1:, k2:, shift:] += coef[:, :, None] * mass[:n1, :n2, :size - shift]
        mass = new
    return float(mass.sum())


def class_probabilities(dispersion: int, interval: int, theta: float, *,
                        p_connect: float = 0.1, n_per_event: int = 40,
                        onset: int = 5, spread: int = 3, n_time: int = 60,
                        tau_den: float = 2.0, alpha: float = 2.5,
                        tau_mem: float = 10.0, plateau_steps: int = 15,
                        resolution: int = 1000) -> ClassProbabilities:
    """Expected class fractions for the dispersion sweep's stimulus and model.

    Defaults are the sweep's own (``experiments/e1_dispersion.py``,
    :class:`delaykey.core.NeuronParams`).
    """
    profile = arrival_profile(dispersion, spread)
    rates1 = _arrival_rates(onset, profile, p_connect, n_time)
    rates2 = _arrival_rates(onset + interval, profile, p_connect, n_time)
    none = np.zeros(n_time)
    level = theta + soma_margin(alpha, tau_mem, plateau_steps)

    def quiet(r1, r2):
        return never_fires(r1, r2, n_per_event, level, tau_den,
                           resolution=resolution)

    quiet1, quiet2, quiet12 = quiet(rates1, none), quiet(none, rates2), quiet(rates1, rates2)
    return ClassProbabilities(
        sequence=max(quiet1 * quiet2 - quiet12, 0.0),
        event1=(1.0 - quiet1) * quiet2,
        event2=(1.0 - quiet2) * quiet1,
        dual=(1.0 - quiet1) * (1.0 - quiet2),
        silent=quiet12,
    )


def crossover(dispersions, sequence, event) -> float:
    """First dispersion at which ``sequence`` reaches ``event``, interpolated."""
    d = np.asarray(dispersions, dtype=float)
    diff = np.asarray(sequence, dtype=float) - np.asarray(event, dtype=float)
    hits = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    if hits.size == 0:
        return float("nan")
    i = hits[0]
    return float(d[i] - diff[i] * (d[i + 1] - d[i]) / (diff[i + 1] - diff[i]))


def peak(dispersions, sequence) -> float:
    """Dispersion maximising ``sequence``, refined by a parabola through the top."""
    d = np.asarray(dispersions, dtype=float)
    s = np.asarray(sequence, dtype=float)
    i = int(np.argmax(s))
    if i == 0 or i == s.size - 1:
        return float("nan")
    denom = s[i - 1] - 2.0 * s[i] + s[i + 1]
    if denom == 0:
        return float(d[i])
    return float(d[i] - 0.5 * (d[i + 1] - d[i]) * (s[i + 1] - s[i - 1]) / denom)

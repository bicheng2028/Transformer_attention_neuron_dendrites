"""Tests for the analytic detector-class probabilities.

The load-bearing test is :func:`test_theory_matches_simulation`: the calculation
in :mod:`delaykey.theory` must reproduce what the simulator measures on fresh
seeds. The manuscript draws the theory through the sweep's points and quotes its
slopes, so that agreement is what licenses both.

Runs under pytest, or standalone with ``python tests/test_theory.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.core import NeuronParams, simulate_population
from delaykey.detectors import classify_detectors
from delaykey.stimuli import random_connectivity, two_event_stimulus
from delaykey.theory import arrival_profile, class_probabilities, soma_margin


def test_classes_partition_the_population():
    """The five class probabilities sum to one wherever they are evaluated."""
    for dispersion, interval, theta in ((0, 5, 1.5), (16, 15, 2.5),
                                        (30, 20, 4.5), (50, 10, 3.5)):
        c = class_probabilities(dispersion, interval, theta)
        total = c.sequence + c.event1 + c.event2 + c.dual + c.silent
        assert abs(total - 1.0) < 1e-12
        assert min(c.sequence, c.event1, c.event2, c.dual, c.silent) >= 0.0


def test_arrival_profile_is_a_distribution():
    """Offset convolved with delay: sums to one and spans spread + dispersion."""
    for dispersion in (0, 3, 20):
        q = arrival_profile(dispersion, spread=3)
        assert q.size == dispersion + 4
        assert abs(q.sum() - 1.0) < 1e-12


def test_no_sequence_detectors_without_overlap():
    """Volleys that cannot be brought together leave no sequence detectors.

    With delays on [0, 4] and a 20 ms interval, the arrival windows of the two
    events are 13 ms apart, and the dendritic potential forgets in 2 ms.
    """
    assert class_probabilities(4, 20, 4.5).sequence < 1e-8


def test_soma_margin_matches_the_plateau():
    """A plateau just past the margin brings the soma to threshold in time."""
    eps = soma_margin(alpha=2.5, tau_mem=10.0, plateau_steps=15)
    drive = 2.5 * np.tanh(eps * 1.001)
    v = 0.0
    for _ in range(15):
        v = v * np.exp(-1.0 / 10.0) + drive
    assert v >= 1.0


def test_theory_matches_simulation():
    """Class fractions from the calculation agree with the simulator.

    Fresh seeds, outside the 0-19 the manuscript reports, at the permissive
    gate where every class is populous enough to test tightly. The tolerance is
    four standard errors over seeds.
    """
    params = NeuronParams(theta=2.5, alpha=2.5, plateau_steps=15, refractory=2,
                          v_contrib=0.0)
    for dispersion in (8, 20):
        fractions = []
        for seed in range(100, 112):
            rng = np.random.default_rng(seed)
            stim = two_event_stimulus(rng, n_features=200, n_per_event=40,
                                      onset1=5, interval=15, spread=3, n_time=60)
            d, w = random_connectivity(rng, n_neurons=3000, n_features=200,
                                       dispersion=dispersion, n_branches=1,
                                       p_connect=0.1)
            counts = classify_detectors(
                simulate_population(stim.both, d, w, params),
                simulate_population(stim.event1_only, d, w, params),
                simulate_population(stim.event2_only, d, w, params))
            n = counts.n_neurons
            fractions.append([counts.sequence / n,
                              (counts.event1 + counts.event2) / n,
                              counts.dual / n])
        fractions = np.array(fractions)
        mean = fractions.mean(axis=0)
        se = fractions.std(axis=0, ddof=1) / np.sqrt(len(fractions))
        theory = class_probabilities(dispersion, 15, 2.5)
        expected = np.array([theory.sequence, theory.event, theory.dual])
        assert np.all(np.abs(mean - expected) <= 4 * se + 1e-4), (
            dispersion, mean, expected, se)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")

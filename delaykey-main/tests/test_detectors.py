"""Verification for the functional detector classification.

The classification is the operational definition behind E1's headline numbers,
so it is pinned down by construction here rather than only exercised through the
simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.core import NeuronParams, simulate_population
from delaykey.detectors import CLASSES, classify_detectors
from delaykey.stimuli import random_connectivity, two_event_stimulus


def test_classes_partition_the_population():
    """Every neuron lands in exactly one class, for arbitrary responses."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = 40
        both = rng.integers(0, 3, n)
        one = rng.integers(0, 3, n)
        two = rng.integers(0, 3, n)
        counts = classify_detectors(both, one, two)
        total = sum(getattr(counts, c) for c in CLASSES)
        assert total == n, f"classes sum to {total}, expected {n}"


def test_canonical_assignments():
    """Each class is produced by its defining response pattern."""
    #            sequence  event1  event2   dual   silent
    both  = np.array([3,       2,      2,      4,     0])
    only1 = np.array([0,       1,      0,      1,     0])
    only2 = np.array([0,       0,      1,      2,     0])

    counts = classify_detectors(both, only1, only2)
    assert counts.sequence == 1
    assert counts.event1 == 1
    assert counts.event2 == 1
    assert counts.dual == 1
    assert counts.silent == 1
    assert counts.other == 0


def test_dual_responsive_is_not_a_sequence_detector():
    """Responding to both events separately must not count as conjunctive.

    This is the distinction the manuscript checks informally; making it the
    definition is what lets E1 report sequence detectors without a separate
    control paragraph.
    """
    counts = classify_detectors(
        np.array([5]), np.array([2]), np.array([2])
    )
    assert counts.sequence == 0
    assert counts.dual == 1


def test_fraction_conversion():
    """``as_dict(fractions=True)`` divides by the population size."""
    counts = classify_detectors(
        np.array([1, 0, 0, 0]), np.zeros(4), np.zeros(4)
    )
    fractions = counts.as_dict(fractions=True)
    assert fractions["sequence"] == 0.25
    assert fractions["silent"] == 0.75
    assert abs(sum(fractions[c] for c in CLASSES) - 1.0) < 1e-12


def test_mismatched_shapes_rejected():
    try:
        classify_detectors(np.zeros(3), np.zeros(4), np.zeros(3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched response shapes")


def test_sequence_detectors_are_genuinely_conjunctive():
    """End-to-end: neurons E1 labels 'sequence' really are silent alone.

    Runs the simulator in a wide-dispersion regime, takes the neurons the
    classifier calls sequence detectors, and re-checks them directly against the
    two single-event stimuli.
    """
    rng = np.random.default_rng(4)
    params = NeuronParams(theta=2.5, alpha=2.5)
    stim = two_event_stimulus(rng, n_features=200, n_per_event=40,
                              interval=15, n_time=60)
    delays, weights = random_connectivity(
        rng, n_neurons=800, n_features=200, dispersion=25, p_connect=0.1
    )

    both = simulate_population(stim.both, delays, weights, params)
    one = simulate_population(stim.event1_only, delays, weights, params)
    two = simulate_population(stim.event2_only, delays, weights, params)

    counts = classify_detectors(both, one, two)
    assert counts.sequence > 0, "regime should produce some sequence detectors"

    is_sequence = (both > 0) & (one == 0) & (two == 0)
    assert is_sequence.sum() == counts.sequence
    assert one[is_sequence].max(initial=0) == 0
    assert two[is_sequence].max(initial=0) == 0
    assert both[is_sequence].min() > 0


def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

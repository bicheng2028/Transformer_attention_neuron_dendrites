"""Verification suite for the delay-key core.

The load-bearing test is :func:`test_matches_reference_implementation`: the fast
vectorised simulator and the transparent single-neuron reference must agree
exactly across randomised configurations. Everything downstream in the paper is
computed with the fast path, so that equivalence is what licenses its use.

The remaining tests pin down behaviour that has an answer independent of either
implementation -- delay compensation, refractory bounds, silence without input.

Runs under pytest, or standalone with ``python tests/test_core.py`` for
environments where pytest is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.core import NeuronParams, simulate_population, simulate_reference
from delaykey.stimuli import random_connectivity, two_event_stimulus


# --------------------------------------------------------------------------
# Equivalence of the fast and reference implementations
# --------------------------------------------------------------------------

def test_matches_reference_implementation():
    """Vectorised and reference simulators agree over randomised settings.

    Sweeps branch counts, delay dispersions, connection densities and the
    plateau/refractory parameters, since those interact through the plateau
    bookkeeping and are the most likely place for the two paths to diverge.
    """
    cases = [
        dict(n_branches=1, dispersion=0, p=0.10, plateau=15, refractory=2),
        dict(n_branches=1, dispersion=8, p=0.10, plateau=15, refractory=2),
        dict(n_branches=1, dispersion=20, p=0.20, plateau=1, refractory=0),
        dict(n_branches=3, dispersion=12, p=0.15, plateau=5, refractory=3),
        dict(n_branches=2, dispersion=30, p=0.05, plateau=25, refractory=1),
    ]

    for i, case in enumerate(cases):
        rng = np.random.default_rng(1000 + i)
        params = NeuronParams(
            theta=2.5,
            alpha=2.5,
            plateau_steps=case["plateau"],
            refractory=case["refractory"],
            v_contrib=0.05,
        )
        stim = two_event_stimulus(rng, n_features=60, n_per_event=12, n_time=40)
        delays, weights = random_connectivity(
            rng, n_neurons=25, n_features=60, dispersion=case["dispersion"],
            n_branches=case["n_branches"], p_connect=case["p"],
        )

        fast = simulate_population(stim.both, delays, weights, params)
        slow = simulate_reference(stim.both, delays, weights, params)

        assert np.array_equal(fast, slow), (
            f"case {i} {case}: fast and reference disagree; "
            f"max |diff| = {np.abs(fast - slow).max()}"
        )


def test_matches_reference_soft_gating():
    """The soft-gating path also agrees with the reference implementation.

    Soft gating is the control in the discriminability comparison, so its
    numbers carry the same weight as the hard-gating ones and need the same
    guarantee.
    """
    for i, dispersion in enumerate((0, 10, 25)):
        rng = np.random.default_rng(2000 + i)
        params = NeuronParams(theta=2.5, alpha=0.05, plateau_steps=8,
                              refractory=2, v_contrib=0.0)
        stim = two_event_stimulus(rng, n_features=60, n_per_event=12, n_time=40)
        delays, weights = random_connectivity(
            rng, n_neurons=20, n_features=60, dispersion=dispersion,
            p_connect=0.15,
        )
        fast = simulate_population(stim.both, delays, weights, params,
                                   gating="soft")
        slow = simulate_reference(stim.both, delays, weights, params,
                                  gating="soft")
        assert np.array_equal(fast, slow), f"dispersion {dispersion}: mismatch"


def test_matches_reference_with_proximal_content():
    """Equivalence also holds on the per-neuron proximal path.

    The S5 Text result rests entirely on ``proximal_drive`` and
    ``proximal_inhibition``, which vary neuron by neuron. Until the reference
    implementation accepted them, the equivalence guarantee stopped short of
    exactly the code path that experiment uses, so its numbers were produced by
    an unverified branch of the simulator.
    """
    for i, (n_branches, dispersion) in enumerate(((1, 12), (2, 20), (3, 6))):
        rng = np.random.default_rng(3000 + i)
        params = NeuronParams(theta=2.5, alpha=2.5, plateau_steps=10,
                              refractory=2, v_contrib=0.0)
        stim = two_event_stimulus(rng, n_features=60, n_per_event=12, n_time=40)
        delays, weights = random_connectivity(
            rng, n_neurons=18, n_features=60, dispersion=dispersion,
            n_branches=n_branches, p_connect=0.15,
        )
        n_neurons = delays.shape[0]
        drive = rng.uniform(0.0, 0.6, size=n_neurons)
        inhibition = rng.uniform(0.0, 1.2, size=n_neurons)

        fast = simulate_population(stim.both, delays, weights, params,
                                   proximal_drive=drive,
                                   proximal_inhibition=inhibition)
        slow = simulate_reference(stim.both, delays, weights, params,
                                  proximal_drive=drive,
                                  proximal_inhibition=inhibition)
        assert np.array_equal(fast, slow), (
            f"case {i}: proximal path disagrees; "
            f"max |diff| = {np.abs(fast - slow).max()}"
        )


def test_experiment1_worked_example_under_the_production_model():
    """Experiment 1's configuration, run through the simulator the paper uses.

    The Experiment 1 figure is drawn by a standalone script with its own
    ``lif()`` and ``integrator()`` helpers, which are an illustration rather
    than the model. This runs the same configuration -- two inputs, delays
    (5, 0) tuned to A->B, subthreshold proximal drive, the membrane parameters
    the caption quotes -- through ``simulate_population`` and asserts the three
    properties the text claims, so the illustration is pinned to the model
    rather than merely resembling it.
    """
    params = NeuronParams(theta=1.5, alpha=2.5, tau_mem=5.0, tau_den=2.0,
                          v_threshold=1.0, v_contrib=0.10, plateau_steps=10,
                          refractory=2)
    delays = np.array([[[5, 0]]], dtype=np.int64)
    weights = np.array([[[1.0, 1.0]]])

    forward = np.zeros((30, 2))
    forward[5, 0], forward[10, 1] = 1.0, 1.0     # A at t=5, then B at t=10
    reverse = np.zeros((30, 2))
    reverse[5, 1], reverse[10, 0] = 1.0, 1.0     # B first

    counts_f, membrane_f, spikes_f = simulate_population(
        forward, delays, weights, params, return_traces=True)
    counts_r, membrane_r, _ = simulate_population(
        reverse, delays, weights, params, return_traces=True)

    # (i) temporal selectivity: matched order fires, reversed order does not.
    assert counts_f[0] == 1, f"A->B should fire exactly once, got {counts_f[0]}"
    assert counts_r[0] == 0, "B->A should stay silent"

    # (ii) the spike lands at the coincidence, t = 10.
    assert int(np.flatnonzero(spikes_f[0])[0]) == 10

    # (iii) subthreshold V: without the distal vote the soma never approaches
    # threshold, so the proximal pathway alone cannot account for the spike.
    assert membrane_r[0].max() < params.v_threshold
    steady = params.v_contrib / (1.0 - np.exp(-1.0 / params.tau_mem))
    assert membrane_r[0].max() <= steady + 1e-9


def test_soft_gating_ignores_the_calcium_threshold():
    """Soft gating fires where hard gating is silenced by an unreachable theta.

    This is the property that makes it a meaningful control: the two paths
    differ in gating alone, not in delays or connectivity.
    """
    rng = np.random.default_rng(31)
    stim = two_event_stimulus(rng, n_features=60, n_per_event=20, n_time=40)
    delays, weights = random_connectivity(
        rng, n_neurons=40, n_features=60, dispersion=8, p_connect=0.4
    )
    params = NeuronParams(theta=1e6, alpha=0.5, v_contrib=0.0)

    hard = simulate_population(stim.both, delays, weights, params, gating="hard")
    soft = simulate_population(stim.both, delays, weights, params, gating="soft")

    assert hard.sum() == 0, "an unreachable threshold must silence hard gating"
    assert soft.sum() > 0, "soft gating should be unaffected by theta"


def test_peak_membrane_bounds_spiking():
    """A neuron spikes if and only if its peak potential reached threshold."""
    rng = np.random.default_rng(33)
    stim = two_event_stimulus(rng, n_features=80, n_per_event=20, n_time=45)
    delays, weights = random_connectivity(
        rng, n_neurons=60, n_features=80, dispersion=12, p_connect=0.25
    )
    params = NeuronParams(theta=2.5, alpha=2.5)

    counts, peak = simulate_population(stim.both, delays, weights, params,
                                       return_peak=True)
    assert np.all(peak[counts > 0] >= params.v_threshold)
    assert np.all(peak[counts == 0] < params.v_threshold)


def test_rejects_unknown_gating_mode():
    delays = np.zeros((1, 1, 3), dtype=np.int64)
    weights = np.ones((1, 1, 3))
    try:
        simulate_population(np.zeros((10, 3)), delays, weights, NeuronParams(),
                            gating="medium")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown gating mode")


def test_matches_reference_with_graded_input():
    """Equivalence also holds for non-binary input (event counts per bin)."""
    rng = np.random.default_rng(7)
    params = NeuronParams(theta=3.0, alpha=2.0, plateau_steps=6, refractory=2)

    x = rng.poisson(0.05, size=(35, 50)).astype(float)
    delays, weights = random_connectivity(
        rng, n_neurons=20, n_features=50, dispersion=10, p_connect=0.2
    )

    fast = simulate_population(x, delays, weights, params)
    slow = simulate_reference(x, delays, weights, params)
    assert np.array_equal(fast, slow)


# --------------------------------------------------------------------------
# Behaviour with an implementation-independent answer
# --------------------------------------------------------------------------

def test_delay_compensation_discriminates_order():
    """The Experiment 1 claim, as an assertion.

    Two inputs, A at t=5 and B at t=10, and a branch whose delays are tuned to
    A->B (d_A = 5, d_B = 0). The matched order must fire; the reversed order,
    with identical weights and identical total input, must stay silent.
    """
    params = NeuronParams(theta=1.5, alpha=2.5, v_threshold=1.0,
                          plateau_steps=10, refractory=2, v_contrib=0.0)

    delays = np.array([[[5, 0]]], dtype=np.int64)   # (1 neuron, 1 branch, 2 features)
    weights = np.array([[[1.0, 1.0]]])

    forward = np.zeros((30, 2))
    forward[5, 0] = 1.0     # A first
    forward[10, 1] = 1.0    # then B

    reverse = np.zeros((30, 2))
    reverse[5, 1] = 1.0     # B first
    reverse[10, 0] = 1.0    # then A

    assert simulate_population(forward, delays, weights, params)[0] > 0, \
        "matched order A->B should fire"
    assert simulate_population(reverse, delays, weights, params)[0] == 0, \
        "reversed order B->A should stay silent"


def test_silent_without_input():
    """No input and no proximal drive means no spikes."""
    rng = np.random.default_rng(3)
    params = NeuronParams(v_contrib=0.0)
    delays, weights = random_connectivity(
        rng, n_neurons=40, n_features=80, dispersion=15, p_connect=0.1
    )
    counts = simulate_population(np.zeros((50, 80)), delays, weights, params)
    assert counts.sum() == 0


def test_subthreshold_proximal_drive_alone_cannot_fire():
    """Proximal V below threshold never triggers a spike on its own.

    The steady state of a leaky membrane driven by a constant ``v_contrib`` is
    ``v_contrib / (1 - exp(-1/tau))``; the test picks a value comfortably under
    threshold and asserts silence over a long trial.
    """
    params = NeuronParams(tau_mem=10.0, v_threshold=1.0, v_contrib=0.05)
    steady = params.v_contrib / (1.0 - np.exp(-1.0 / params.tau_mem))
    assert steady < params.v_threshold, "test setup must stay subthreshold"

    delays = np.zeros((1, 1, 4), dtype=np.int64)
    weights = np.zeros((1, 1, 4))
    counts = simulate_population(np.zeros((400, 4)), delays, weights, params)
    assert counts[0] == 0


def test_refractory_period_bounds_spike_count():
    """Spike count can never exceed the refractory ceiling."""
    rng = np.random.default_rng(11)
    n_time, refractory = 60, 3
    params = NeuronParams(theta=0.5, alpha=50.0, refractory=refractory,
                          plateau_steps=30)

    x = rng.random((n_time, 40)) < 0.5
    delays, weights = random_connectivity(
        rng, n_neurons=30, n_features=40, dispersion=5, p_connect=0.5
    )
    counts = simulate_population(x.astype(float), delays, weights, params)

    ceiling = np.ceil(n_time / (refractory + 1))
    assert counts.max() <= ceiling, f"{counts.max()} exceeds ceiling {ceiling}"


def test_unreachable_threshold_gives_silence():
    """A calcium threshold above any attainable drive yields no plateaus."""
    rng = np.random.default_rng(5)
    stim = two_event_stimulus(rng, n_features=60, n_per_event=12, n_time=40)
    delays, weights = random_connectivity(
        rng, n_neurons=30, n_features=60, dispersion=10, p_connect=0.2
    )
    params = NeuronParams(theta=1e6, alpha=2.5, v_contrib=0.0)
    assert simulate_population(stim.both, delays, weights, params).sum() == 0


def test_delays_beyond_trial_contribute_nothing():
    """Inputs delayed past the end of the trial cannot drive the neuron."""
    params = NeuronParams(theta=0.5, alpha=5.0)
    n_time = 20

    x = np.zeros((n_time, 2))
    x[1, :] = 1.0

    late = np.array([[[n_time + 5, n_time + 5]]], dtype=np.int64)
    weights = np.array([[[1.0, 1.0]]])
    assert simulate_population(x, late, weights, params)[0] == 0


def test_population_is_independent_across_neurons():
    """A neuron's output does not depend on who else is in the population.

    Guards against the batch-coupled plateau reset in the notebook prototype,
    where any spiking element cleared every plateau in the batch.
    """
    rng = np.random.default_rng(17)
    params = NeuronParams(theta=2.0, alpha=2.5, plateau_steps=12, refractory=2)
    stim = two_event_stimulus(rng, n_features=70, n_per_event=15, n_time=45)
    delays, weights = random_connectivity(
        rng, n_neurons=24, n_features=70, dispersion=14, p_connect=0.15
    )

    together = simulate_population(stim.both, delays, weights, params)
    alone = np.array([
        simulate_population(stim.both, delays[i:i + 1], weights[i:i + 1], params)[0]
        for i in range(delays.shape[0])
    ])
    assert np.array_equal(together, alone)


def test_reproducible_from_seed():
    """Identical seeds reproduce identical output; different seeds do not.

    The connectivity here is deliberately dense enough to make the population
    fire. With a sparse projection the whole population falls silent, both runs
    return all-zeros, and the "different seeds differ" half of the test would
    pass vacuously -- so activity is asserted before the comparison.
    """
    params = NeuronParams(theta=2.5, alpha=2.5)

    def run(seed):
        rng = np.random.default_rng(seed)
        stim = two_event_stimulus(rng, n_features=60, n_per_event=20, n_time=40)
        delays, weights = random_connectivity(
            rng, n_neurons=20, n_features=60, dispersion=12, p_connect=0.5
        )
        return simulate_population(stim.both, delays, weights, params)

    baseline = run(42)
    assert baseline.sum() > 0, "test configuration must produce spikes to be meaningful"
    assert np.array_equal(baseline, run(42))
    assert not np.array_equal(baseline, run(43))


def _main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
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

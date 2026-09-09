"""Stimulus and connectivity generators for the delay-key experiments.

The two-event stimulus reproduces the design of manuscript Experiment 5 (see the
numbering table in ``delaykey/README.md``; the script that runs it is
``experiments/e1_dispersion.py``): two volleys of feature-neuron activity inside
a single gamma cycle, separated by an inter-event interval.

:class:`TwoEventStimulus` stores the *latent* description of a trial -- which
features belong to each event and each spike's offset within its volley -- and
renders the spike arrays on demand. That matters because several experiments
need to vary one property while holding everything else fixed. E1, E2 and E3
below are *script* names, not manuscript experiment numbers:

* E1 compares the two-event stimulus against each event alone, and the controls
  must carry the same spike times or the comparison is confounded.
* E2 sweeps the inter-event interval to map out aliasing, and the tuning curve
  is only interpretable if the identity and within-volley timing of every spike
  are unchanged as the interval moves.
* E3 needs the reversed-order distractor and jittered copies of both.

Re-drawing the stimulus for each condition would confound the manipulation with
resampling noise, so every variant is produced by re-rendering the same latent
description.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["TwoEventStimulus", "two_event_stimulus", "random_connectivity"]


@dataclass(frozen=True)
class TwoEventStimulus:
    """Two volleys of feature activity, with matched controls and variants.

    Attributes:
        features1: indices of the features recruited by event 1.
        features2: indices recruited by event 2, disjoint from ``features1``.
        offsets1: spike offset of each event-1 feature within its volley.
        offsets2: spike offset of each event-2 feature within its volley.
        onset1: onset of the first volley, in steps.
        interval: inter-event interval, in steps.
        n_features: size of the upstream feature population.
        n_time: trial length, in steps.
    """

    features1: np.ndarray
    features2: np.ndarray
    offsets1: np.ndarray
    offsets2: np.ndarray
    onset1: int
    interval: int
    n_features: int
    n_time: int

    # -- rendering ---------------------------------------------------------

    def _render(self, which: str) -> np.ndarray:
        """Render a spike array containing the requested events."""
        x = np.zeros((self.n_time, self.n_features))
        if which in ("both", "event1"):
            t1 = self.onset1 + self.offsets1
            keep = (t1 >= 0) & (t1 < self.n_time)
            x[t1[keep], self.features1[keep]] = 1.0
        if which in ("both", "event2"):
            t2 = self.onset1 + self.interval + self.offsets2
            keep = (t2 >= 0) & (t2 < self.n_time)
            x[t2[keep], self.features2[keep]] = 1.0
        return x

    @property
    def both(self) -> np.ndarray:
        """Both events present."""
        return self._render("both")

    @property
    def event1_only(self) -> np.ndarray:
        """Event 1 alone, with identical spike times."""
        return self._render("event1")

    @property
    def event2_only(self) -> np.ndarray:
        """Event 2 alone, with identical spike times."""
        return self._render("event2")

    @property
    def event_id(self) -> np.ndarray:
        """``(n_features,)`` array; 0 = silent, 1 = event 1, 2 = event 2."""
        ids = np.zeros(self.n_features, dtype=np.int8)
        ids[self.features1] = 1
        ids[self.features2] = 2
        return ids

    # -- variants ----------------------------------------------------------

    def with_interval(self, interval: int) -> "TwoEventStimulus":
        """Same features and within-volley timing, different interval.

        This is the manipulation E2 sweeps: everything about the trial is held
        fixed except how far apart the two volleys sit.
        """
        return replace(self, interval=int(interval))

    def reversed_order(self) -> "TwoEventStimulus":
        """Swap which volley comes first, preserving identities and offsets.

        Produces the distractor for the target-versus-distractor task in E3:
        the same features fire with the same within-volley timing, only the
        order is reversed, so nothing but temporal order distinguishes it from
        the target.
        """
        return replace(self, features1=self.features2, features2=self.features1,
                       offsets1=self.offsets2, offsets2=self.offsets1)

    def with_jitter(self, sigma: float, rng: np.random.Generator) -> "TwoEventStimulus":
        """Add Gaussian jitter, in steps, to every spike time independently."""
        if sigma <= 0:
            return self
        o1 = self.offsets1 + np.rint(rng.normal(0.0, sigma, self.offsets1.shape))
        o2 = self.offsets2 + np.rint(rng.normal(0.0, sigma, self.offsets2.shape))
        return replace(self, offsets1=o1.astype(np.int64),
                       offsets2=o2.astype(np.int64))


def two_event_stimulus(
    rng: np.random.Generator,
    n_features: int = 200,
    n_per_event: int = 40,
    onset1: int = 5,
    interval: int = 15,
    spread: int = 3,
    n_time: int = 60,
) -> TwoEventStimulus:
    """Draw two volleys of activity over a fixed pair of feature populations.

    Event 1 recruits features ``[0, n_per_event)`` and event 2 the disjoint
    block ``[n_per_event, 2 * n_per_event)``; the rest of the population stays
    silent. Each spike sits at an offset drawn uniformly from ``[0, spread]``
    after its volley's onset, and event 2's onset is ``interval`` steps after
    event 1's.

    Which features carry which event is fixed rather than redrawn per call,
    because it is a property of the circuit and not of the trial: the neurons
    coding an event do not exchange roles between runs, and a model that
    resamples them describes a different circuit on every trial. Fixing them
    costs nothing. :func:`random_connectivity` draws weights and delays i.i.d.
    across features, so feature indices are exchangeable and any assignment is
    a relabelling of any other -- which is also why the earlier random draw
    bought nothing. The randomness that matters is in the connectivity, still
    redrawn per seed.

    Args:
        rng: random generator; supplies the within-volley spike offsets.
        n_features: size of the upstream feature population.
        n_per_event: features recruited by each event.
        onset1: onset of event 1, in steps.
        interval: inter-event interval, in steps.
        spread: width of each volley, in steps.
        n_time: trial length, in steps.

    Returns:
        A :class:`TwoEventStimulus`.

    Raises:
        ValueError: if the population is too small to hold two disjoint events,
            or if the second volley would fall outside the trial.
    """
    if 2 * n_per_event > n_features:
        raise ValueError("events must recruit disjoint feature sets")
    if onset1 + interval + spread >= n_time:
        raise ValueError("event 2 falls outside the trial window")

    return TwoEventStimulus(
        features1=np.arange(n_per_event),
        features2=np.arange(n_per_event, 2 * n_per_event),
        offsets1=rng.integers(0, spread + 1, size=n_per_event),
        offsets2=rng.integers(0, spread + 1, size=n_per_event),
        onset1=int(onset1),
        interval=int(interval),
        n_features=int(n_features),
        n_time=int(n_time),
    )


def random_connectivity(
    rng: np.random.Generator,
    n_neurons: int,
    n_features: int,
    dispersion: int,
    n_branches: int = 1,
    p_connect: float = 0.1,
    weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random projections with delays uniform on ``[0, dispersion]``.

    Unconnected inputs get zero weight, so they contribute nothing regardless of
    their delay.

    Weights default to 1.0 for every realised connection, which makes the
    calcium threshold directly interpretable: with unit weights ``theta = 2.5``
    means "at least three inputs must arrive coincidentally", as the manuscript
    describes it. It also keeps branch drive on the same scale as the threshold
    however large the feature population is -- the notebook prototype drew
    weights from ``U[0.2, 0.5]`` across 2312 inputs, which put the drive two
    orders of magnitude above threshold and made every branch fire every step.

    Args:
        rng: random generator.
        n_neurons: population size.
        n_features: size of the upstream feature population.
        dispersion: maximum conduction delay; delays are drawn uniformly from
            the inclusive integer range ``[0, dispersion]``. ``0`` gives an
            entirely synchronous projection.
        n_branches: dendritic branches per neuron.
        p_connect: probability a given feature projects to a given branch.
        weight: synaptic weight of a realised connection.

    Returns:
        ``(delays, weights)``, both ``(neurons, branches, features)``; delays
        are integers.
    """
    if dispersion < 0:
        raise ValueError("dispersion must be non-negative")
    if not 0.0 <= p_connect <= 1.0:
        raise ValueError("p_connect must be a probability")

    shape = (n_neurons, n_branches, n_features)
    connected = rng.random(shape) < p_connect
    delays = rng.integers(0, dispersion + 1, size=shape).astype(np.int64)
    weights = np.where(connected, weight, 0.0)
    return delays, weights

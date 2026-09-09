"""Functional classification of integrator neurons into detector types.

The manuscript's Experiment 5 (this file's ``e1_dispersion.py``; see the
numbering table in ``delaykey/README.md``) classified a neuron by tracing which
event its branch inputs came from, then checked separately that the putative
sequence detectors stayed silent for either event alone. We invert that: the
silence check *is* the classification.

A neuron counts as a sequence detector only if it fires for the two-event
stimulus and for neither event on its own. That is the property a reviewer
actually cares about -- genuine conjunctivity rather than dual-event
responsiveness -- and it is measured rather than inferred from connectivity. It
also removes the need to pick a tracing rule for ``N1``/``N2``.

Because the three stimuli share their spike times (see
:func:`delaykey.stimuli.two_event_stimulus`), the comparison is within-neuron
and within-stimulus, with no resampling confound.

Conjunctivity is not order selectivity, and the distinction matters because the
manuscript claims the latter. The three stimuli above never present the two
events in the reverse order, so nothing in ``classify_detectors`` can speak to
order. :func:`sequence_mask` exposes the per-neuron mask precisely so that a
caller can intersect it with silence to the order-reversed stimulus and measure
order selectivity rather than assert it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DetectorCounts", "classify_detectors", "sequence_mask"]

#: Class labels, in the order used by :meth:`DetectorCounts.as_dict`.
CLASSES = ("sequence", "event1", "event2", "dual", "silent", "other")


@dataclass(frozen=True)
class DetectorCounts:
    """Population counts per detector class.

    Attributes:
        sequence: fired for both events together, for neither alone.
        event1: fired for event 1 alone but not event 2 alone.
        event2: fired for event 2 alone but not event 1 alone.
        dual: fired for each event alone -- responsive, but not conjunctive.
        silent: never fired under any of the three stimuli.
        other: fired for a single event but not for the two together. Possible
            in principle because a plateau triggered early can leave the neuron
            refractory later; tracked so the classes sum to the population.
        n_neurons: population size.
    """

    sequence: int
    event1: int
    event2: int
    dual: int
    silent: int
    other: int
    n_neurons: int

    def as_dict(self, fractions: bool = False) -> dict:
        """Return counts, or fractions of the population when ``fractions``."""
        scale = self.n_neurons if fractions and self.n_neurons else 1
        out = {c: getattr(self, c) / scale for c in CLASSES} if fractions else \
              {c: getattr(self, c) for c in CLASSES}
        out["n_neurons"] = self.n_neurons
        return out


def sequence_mask(
    response_both: np.ndarray,
    response_event1: np.ndarray,
    response_event2: np.ndarray,
) -> np.ndarray:
    """Boolean mask of the neurons classified as sequence detectors.

    The same rule :func:`classify_detectors` counts, exposed per neuron so that
    callers can intersect it with a further condition -- in particular silence
    to the order-reversed stimulus, which turns conjunctivity into measured
    order selectivity. Keeping the rule in one function is what stops the two
    definitions drifting apart.

    Args:
        response_both: spike counts under the two-event stimulus.
        response_event1: spike counts under event 1 alone.
        response_event2: spike counts under event 2 alone.

    Returns:
        Boolean array, ``True`` where the neuron fired for the two events
        together and for neither alone.
    """
    both = np.asarray(response_both) > 0
    only1 = np.asarray(response_event1) > 0
    only2 = np.asarray(response_event2) > 0
    if not (both.shape == only1.shape == only2.shape):
        raise ValueError("the three response arrays must have the same shape")
    return both & ~only1 & ~only2


def classify_detectors(
    response_both: np.ndarray,
    response_event1: np.ndarray,
    response_event2: np.ndarray,
) -> DetectorCounts:
    """Classify each neuron from its responses to the three matched stimuli.

    Args:
        response_both: spike counts under the two-event stimulus.
        response_event1: spike counts under event 1 alone.
        response_event2: spike counts under event 2 alone.

    Returns:
        A :class:`DetectorCounts` whose class counts sum to the population size.

    Raises:
        ValueError: if the three response arrays differ in length.
    """
    both = np.asarray(response_both) > 0
    only1 = np.asarray(response_event1) > 0
    only2 = np.asarray(response_event2) > 0

    sequence = sequence_mask(response_both, response_event1, response_event2)
    dual = only1 & only2
    event1 = only1 & ~only2
    event2 = only2 & ~only1
    silent = ~both & ~only1 & ~only2
    classified = sequence | dual | event1 | event2 | silent

    return DetectorCounts(
        sequence=int(sequence.sum()),
        event1=int(event1.sum()),
        event2=int(event2.sum()),
        dual=int(dual.sum()),
        silent=int(silent.sum()),
        other=int((~classified).sum()),
        n_neurons=int(both.size),
    )

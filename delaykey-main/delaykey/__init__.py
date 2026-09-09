"""Delay-key integrator neuron model and experiments.

Companion code for "Physical Delays as Keys: A Delay-Key Framework for Sequence
Coding in Neural Circuits".
"""

from .core import NeuronParams, simulate_population, simulate_reference
from .stats import Estimate, bootstrap_mean, summarise
from .stimuli import TwoEventStimulus, random_connectivity, two_event_stimulus

__version__ = "0.1.0"

__all__ = [
    "NeuronParams",
    "simulate_population",
    "simulate_reference",
    "Estimate",
    "bootstrap_mean",
    "summarise",
    "TwoEventStimulus",
    "two_event_stimulus",
    "random_connectivity",
]

"""Delay-key integrator neuron: vectorised population simulator + reference model.

The model is the one defined in Section 4 of the manuscript: a population of
integrator neurons, each carrying one or more distal dendritic branches. Every
branch applies a per-input conduction delay (the K matrix), sums the delayed
input, and triggers a calcium plateau when the local dendritic membrane crosses
theta. Plateau currents from all branches drive a leaky somatic membrane which
emits a spike and enters refractory.

Two implementations live here:

``simulate_population``
    Vectorised over neurons and branches. Applies delays as ``max_delay + 1``
    masked matrix products, then runs a single loop over time steps with all
    neurons advanced in lockstep. This is what the experiments call.

``simulate_reference``
    One neuron at a time, delays applied by explicit per-feature shifting, no
    vectorisation anywhere. Deliberately slow and written to be read. It exists
    so that ``tests/test_core.py`` can prove the fast path reproduces it, across
    both gating modes and both the constant and the per-neuron proximal paths.

Both follow the manuscript's update order exactly. One deviation from the
notebook prototype is deliberate and documented in ``simulate_population``:
plateau termination after a somatic spike is applied per neuron, not across the
whole population.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NeuronParams", "simulate_population", "simulate_reference"]


@dataclass(frozen=True)
class NeuronParams:
    """Parameters of the integrator neuron.

    Time constants and the refractory period are expressed in simulation steps.
    With the default 1 ms step used throughout the experiments they are
    numerically equal to milliseconds.
    """

    theta: float = 2.5          # calcium / coincidence threshold on the branch
    alpha: float = 2.5          # gain from summed plateau current to soma
    tau_den: float = 2.0        # dendritic membrane time constant
    tau_mem: float = 10.0       # somatic membrane time constant
    v_threshold: float = 1.0    # somatic spike threshold
    v_rest: float = 0.0         # reset potential
    refractory: int = 2         # somatic refractory period
    plateau_steps: int = 15     # plateau_duration_max
    g_slope: float = 1.0        # slope of the saturating plateau nonlinearity
    eta: float = 1.0            # branch -> soma transmission efficiency
    v_contrib: float = 0.0      # constant subthreshold proximal drive

    def __post_init__(self) -> None:
        if self.tau_den <= 0 or self.tau_mem <= 0:
            raise ValueError("time constants must be positive")
        if self.refractory < 0 or self.plateau_steps < 0:
            raise ValueError("refractory and plateau_steps must be non-negative")


def _delayed_drive(
    x: np.ndarray,
    delays: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the delayed, weighted input to every branch.

    ``drive[u, t] = sum_f weights[u, f] * x[t - delays[u, f], f]``, with inputs
    before ``t = 0`` treated as silent.

    Rather than materialising a ``(units, time, features)`` tensor -- which is
    what made the prototype run out of memory -- we group inputs by delay value.
    All inputs sharing delay ``k`` can be handled by one shift of ``x`` followed
    by a single matrix product, so the cost is ``max_delay + 1`` products of
    shape ``(units, features) @ (features, time)``.

    That cost grows with the square of the time resolution, since a finer step
    multiplies both the number of delay values and the length of the trial.
    When ``x`` is sparse -- the two-event stimuli carry one spike per active
    feature -- it is far cheaper to scatter each spike to its arrival times
    instead: ``drive[u, t + d[u, f]] += w[u, f] x[t, f]`` for every nonzero
    ``x[t, f]``, a cost proportional to the spikes times the units. Both paths
    compute the same sum; the scatter accumulates feature by feature, in the
    order the reference implementation does.

    Args:
        x: ``(time, features)`` input, spike counts or binary.
        delays: ``(units, features)`` non-negative integer delays.
        weights: ``(units, features)`` synaptic weights.

    Returns:
        ``(units, time)`` drive array.
    """
    n_time, n_features = x.shape
    n_units = delays.shape[0]

    times, feats = np.nonzero(x)
    if times.size <= 0.05 * x.size:
        return _scattered_drive(x, times, feats, delays, weights)

    drive = np.zeros((n_units, n_time), dtype=np.float64)

    # Delays at or beyond the window can never contribute within the trial.
    for k in range(int(delays.max()) + 1 if delays.size else 1):
        mask = delays == k
        if not mask.any() or k >= n_time:
            continue
        w_k = np.where(mask, weights, 0.0)
        shifted = np.zeros_like(x, dtype=np.float64)
        if k == 0:
            shifted[:] = x
        else:
            shifted[k:] = x[:-k]
        drive += w_k @ shifted.T

    return drive


def _scattered_drive(x, times, feats, delays, weights) -> np.ndarray:
    """The delayed drive of a sparse input, one spike at a time.

    See :func:`_delayed_drive`. Spikes are taken feature by feature, in
    ascending feature order, so each unit's drive accumulates in the same order
    as in :func:`simulate_reference`.
    """
    n_time = x.shape[0]
    n_units = delays.shape[0]
    index, value = [], []
    order = np.lexsort((times, feats))
    for t, f in zip(times[order], feats[order]):
        units = np.flatnonzero(weights[:, f])
        if units.size == 0:
            continue
        arrive = t + delays[units, f]
        keep = arrive < n_time
        index.append(units[keep] * n_time + arrive[keep])
        value.append(x[t, f] * weights[units[keep], f])
    if not index:
        return np.zeros((n_units, n_time), dtype=np.float64)
    flat = np.bincount(np.concatenate(index), weights=np.concatenate(value),
                       minlength=n_units * n_time)
    return flat.reshape(n_units, n_time)


def simulate_population(
    x: np.ndarray,
    delays: np.ndarray,
    weights: np.ndarray,
    params: NeuronParams,
    return_traces: bool = False,
    gating: str = "hard",
    return_peak: bool = False,
    proximal_drive: np.ndarray | float | None = None,
    proximal_inhibition: np.ndarray | float | None = None,
):
    """Simulate a population of integrator neurons on one input.

    Args:
        x: ``(time, features)`` input array.
        delays: ``(neurons, branches, features)`` integer conduction delays.
        weights: ``(neurons, branches, features)`` synaptic weights.
        params: model parameters.
        return_traces: also return per-step somatic potential and spike raster.
        proximal_drive: per-neuron proximal excitation, shape ``(neurons,)`` or
            a scalar. Overrides ``params.v_contrib`` when given. This is the
            :math:`\\sum_k w_k V'_k` term of the somatic equation. Passing a
            vector rather than the scalar ``v_contrib`` is what lets the
            proximal pathway carry actual content: with a constant it carries
            none, and somatic integration cannot be anything but a weighted sum.
        proximal_inhibition: per-neuron inhibitory drive, shape ``(neurons,)``
            or a scalar, subtracted from the somatic excitation. This is the
            :math:`I_\\mathrm{inh}` term. Making it depend on the agreement
            between the arriving content and the neuron's stored template is
            what turns somatic integration from a weighted sum into a
            conjunctive check: a strong distal match with mismatched content
            recruits enough inhibition to veto the output, an outcome no
            weighted sum of non-negative drives can produce.
        gating: ``"hard"`` applies the calcium threshold and plateau, as in the
            manuscript's model. ``"soft"`` bypasses both and passes the
            dendritic sum to the soma as a continuous activation -- the control
            used in the discriminability comparison. Both share the same delays,
            so the contrast isolates gating rather than delay tuning.
        return_peak: also return the peak somatic potential per neuron, a graded
            decision variable for ROC analysis (spike counts alone are too
            coarse to trace a curve).

    Returns:
        ``spike_counts`` of shape ``(neurons,)``. With ``return_peak``, a tuple
        ``(spike_counts, peak_membrane)``. With ``return_traces``, a tuple
        ``(spike_counts, membrane, spikes)`` with the latter two shaped
        ``(neurons, time)``.

    Raises:
        ValueError: on malformed inputs or an unknown ``gating`` mode.

    Note:
        The notebook prototype zeroed every plateau in the batch whenever *any*
        batch element spiked, because the reset was guarded by a scalar
        ``if spike.sum() > 0``. Neurons in a population are independent, so that
        coupling is a bug; here the reset is masked per neuron.
    """
    if gating not in ("hard", "soft"):
        raise ValueError(f"gating must be 'hard' or 'soft'; got {gating!r}")
    x = np.asarray(x, dtype=np.float64)
    delays = np.asarray(delays)
    weights = np.asarray(weights, dtype=np.float64)

    if delays.ndim != 3 or weights.ndim != 3:
        raise ValueError("delays and weights must be (neurons, branches, features)")
    if delays.shape != weights.shape:
        raise ValueError("delays and weights must have the same shape")
    if not np.issubdtype(delays.dtype, np.integer):
        raise ValueError("delays must be an integer array")
    if delays.min() < 0:
        raise ValueError("delays must be non-negative")

    n_neurons, n_branches, n_features = delays.shape
    n_time = x.shape[0]
    if x.shape[1] != n_features:
        raise ValueError("x and delays disagree on the number of features")

    def _per_neuron(value, default):
        if value is None:
            return np.full(n_neurons, float(default))
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n_neurons, float(arr))
        if arr.shape != (n_neurons,):
            raise ValueError(
                f"proximal terms must be scalar or shape ({n_neurons},); "
                f"got {arr.shape}")
        return arr

    v_drive = _per_neuron(proximal_drive, params.v_contrib)
    v_inhibition = _per_neuron(proximal_inhibition, 0.0)

    n_units = n_neurons * n_branches
    drive = _delayed_drive(
        x, delays.reshape(n_units, n_features), weights.reshape(n_units, n_features)
    )

    decay_den = np.exp(-1.0 / params.tau_den)
    decay_mem = np.exp(-1.0 / params.tau_mem)

    den = np.zeros(n_units)
    plateau_count = np.zeros(n_units)
    plateau_amp = np.zeros(n_units)

    membrane = np.full(n_neurons, params.v_rest, dtype=np.float64)
    refractory = np.zeros(n_neurons)
    spike_counts = np.zeros(n_neurons)

    membrane_trace = np.zeros((n_neurons, n_time)) if return_traces else None
    spike_trace = np.zeros((n_neurons, n_time)) if return_traces else None
    peak_membrane = np.full(n_neurons, -np.inf)

    for t in range(n_time):
        in_refractory = refractory > 0
        refractory = np.maximum(refractory - 1.0, 0.0)

        # --- distal dendrite -------------------------------------------------
        den = drive[:, t] if t == 0 else den * decay_den + drive[:, t]

        if gating == "hard":
            calcium = den - params.theta
            supra = calcium > 0
            amplitude_now = np.tanh(params.g_slope * calcium) * supra

            plateau_count = np.where(supra, params.plateau_steps,
                                     np.maximum(plateau_count - 1.0, 0.0))
            plateau_amp = np.where(supra, np.maximum(plateau_amp, amplitude_now),
                                   plateau_amp)
            active = plateau_count > 0
            branch_current = params.eta * plateau_amp * active
            plateau_amp = plateau_amp * active

            # A branch that has just fired its plateau resets its local membrane.
            den = den * ~supra
        else:
            # Soft gating: same saturating transfer function as the hard path,
            # but with no calcium threshold, no plateau hold and no dendritic
            # reset, so every input contributes continuously instead of being
            # filtered by a gate. Sharing the nonlinearity matters -- passing
            # the raw dendritic sum instead drives the soma several times
            # harder than the hard path, pins it permanently above threshold,
            # and degenerates its spike count into a measure of how long drive
            # was present rather than how well it matched.
            branch_current = params.eta * np.tanh(params.g_slope * den)

        # --- soma ------------------------------------------------------------
        distal = branch_current.reshape(n_neurons, n_branches).sum(axis=1)
        excitation = v_drive + params.alpha * distal - v_inhibition

        membrane = membrane * decay_mem + excitation * ~in_refractory
        peak_membrane = np.maximum(peak_membrane, membrane)
        spikes = (membrane >= params.v_threshold) & ~in_refractory

        membrane = np.where(spikes, params.v_rest, membrane)
        refractory = refractory + spikes * params.refractory
        spike_counts += spikes

        if gating == "hard" and spikes.any():
            silenced = np.repeat(spikes, n_branches)
            plateau_count = np.where(silenced, 0.0, plateau_count)
            plateau_amp = np.where(silenced, 0.0, plateau_amp)

        if return_traces:
            membrane_trace[:, t] = membrane
            spike_trace[:, t] = spikes

    if return_traces:
        return spike_counts, membrane_trace, spike_trace
    if return_peak:
        return spike_counts, peak_membrane
    return spike_counts


def simulate_reference(
    x: np.ndarray,
    delays: np.ndarray,
    weights: np.ndarray,
    params: NeuronParams,
    gating: str = "hard",
    proximal_drive: np.ndarray | float | None = None,
    proximal_inhibition: np.ndarray | float | None = None,
) -> np.ndarray:
    """Transparent single-neuron reference implementation.

    Deliberately unvectorised: delays are applied by shifting one feature at a
    time and every neuron is advanced in its own loop. Used only to verify
    ``simulate_population``; it is far too slow for the experiments.

    Args and return value match ``simulate_population`` without traces.
    ``proximal_drive`` and ``proximal_inhibition`` are accepted here for the
    same reason they exist there: the S5 Text experiment drives the soma with
    per-neuron content and content-dependent inhibition, and without them the
    equivalence test would not cover the code path that experiment's result
    rests on.
    """
    x = np.asarray(x, dtype=np.float64)
    delays = np.asarray(delays)
    weights = np.asarray(weights, dtype=np.float64)

    n_neurons, n_branches, n_features = delays.shape
    n_time = x.shape[0]

    def _per_neuron(value, default):
        if value is None:
            return np.full(n_neurons, float(default))
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n_neurons, float(arr))
        if arr.shape != (n_neurons,):
            raise ValueError(
                f"proximal terms must be scalar or shape ({n_neurons},); "
                f"got {arr.shape}")
        return arr

    v_drive = _per_neuron(proximal_drive, params.v_contrib)
    v_inhibition = _per_neuron(proximal_inhibition, 0.0)

    decay_den = np.exp(-1.0 / params.tau_den)
    decay_mem = np.exp(-1.0 / params.tau_mem)

    spike_counts = np.zeros(n_neurons)

    for n in range(n_neurons):
        # Apply this neuron's delays, one branch and one feature at a time.
        delayed = np.zeros((n_branches, n_time, n_features))
        for b in range(n_branches):
            for f in range(n_features):
                d = int(delays[n, b, f])
                if d == 0:
                    delayed[b, :, f] = x[:, f]
                elif d < n_time:
                    delayed[b, d:, f] = x[:-d, f]

        den = np.zeros(n_branches)
        plateau_count = np.zeros(n_branches)
        plateau_amp = np.zeros(n_branches)
        membrane = float(params.v_rest)
        refractory = 0.0

        for t in range(n_time):
            in_refractory = refractory > 0
            refractory = max(refractory - 1.0, 0.0)

            for b in range(n_branches):
                drive = float(np.dot(delayed[b, t, :], weights[n, b, :]))
                den[b] = drive if t == 0 else den[b] * decay_den + drive

            distal = 0.0
            for b in range(n_branches):
                if gating == "soft":
                    distal += params.eta * np.tanh(params.g_slope * den[b])
                    continue

                calcium = den[b] - params.theta
                supra = calcium > 0
                if supra:
                    plateau_count[b] = params.plateau_steps
                    plateau_amp[b] = max(
                        plateau_amp[b], np.tanh(params.g_slope * calcium)
                    )
                else:
                    plateau_count[b] = max(plateau_count[b] - 1.0, 0.0)

                if plateau_count[b] > 0:
                    distal += params.eta * plateau_amp[b]
                else:
                    plateau_amp[b] = 0.0

                if supra:
                    den[b] = 0.0

            excitation = v_drive[n] + params.alpha * distal - v_inhibition[n]
            membrane = membrane * decay_mem + (0.0 if in_refractory else excitation)

            if membrane >= params.v_threshold and not in_refractory:
                spike_counts[n] += 1
                membrane = params.v_rest
                refractory += params.refractory
                plateau_count[:] = 0.0
                plateau_amp[:] = 0.0

    return spike_counts

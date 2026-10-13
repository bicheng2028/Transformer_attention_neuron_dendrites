"""E1: does delay dispersion control the event/sequence detector split?

This script is manuscript **Experiment 5**; see the numbering table in
``delaykey/README.md``. It replaces the two-point narrow-vs-wide contrast of the
manuscript's earlier draft with a continuous sweep over dispersion, crossed with
the calcium threshold, the inter-event interval and the connection probability,
and repeated over independent seeds.

The claim under test is stronger than the original one. If the transition from
event detection to sequence detection is monotonic in dispersion, and if the
dispersion at which it happens tracks the inter-event interval, then dispersion
is not merely *associated* with detector class -- it sets it quantitatively, and
the relationship is falsifiable.

A fourth stimulus, the order-reversed sequence, is presented alongside the three
that define the classification. Firing for two events together and for neither
alone is *conjunctivity*; the manuscript's claim is *order selectivity*, and the
two come apart in principle -- a branch could compensate the separation without
caring which volley arrived first. The ``order_selective`` column counts the
sequence detectors that are additionally silent to the reversed order, so the
claim is measured rather than assumed.

Writes one tidy CSV row per (dispersion, theta, interval, p_connect, seed).
Figures are built separately from that CSV by ``figures/make_fig_e1.py``, so
replotting never requires recomputing.

Three flags exist to test the sweep against the model's simplifications rather
than to produce the reported grid (S2 Text); ``figures/report_e1_robustness.py``
prints the comparison:

*--steps-per-ms*     puts everything on a finer step, exactly as
                     ``e6_precision.py`` does: spike offsets and delays are
                     drawn on the finer grid, every duration is scaled, and
                     ``alpha`` is divided by the factor, so the soma integrates
                     the same charge per millisecond.
*--plateau-steps*    the plateau length, 15 ms in the reported model against a
                     biological 100-200 ms.
*--background-rate*  ongoing Poisson input on every feature for the whole
                     trial, in Hz, rendered as in ``e4_noise.py``: each
                     (step, feature) bin carries a spike independently, and a
                     spike landing on a stimulus spike leaves one spike there.

Durations on the command line are in milliseconds whatever the step, and are
written to the CSV that way. Left at their defaults, the three reproduce the
reported CSV byte for byte.

Usage::

    python experiments/e1_dispersion.py --quick          # ~1 min smoke test
    python experiments/e1_dispersion.py --seeds 20       # full grid

The exact invocation behind the committed CSV is recorded in ``run_all.sh``;
the defaults below do *not* reproduce it (the reported sweep runs to 50 ms).
"""

from __future__ import annotations

import os

# Each grid cell is small, so parallelism belongs at the process level. Left to
# itself every worker would also spin up a full complement of BLAS threads,
# oversubscribing the node by the worker count and making the run slower than
# serial. These must be set before numpy is imported anywhere.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.core import NeuronParams, simulate_population
from delaykey.detectors import CLASSES, classify_detectors, sequence_mask
from delaykey.stimuli import random_connectivity, two_event_stimulus

# Every row carries the full configuration that produced it, so a single row is
# reproducible in isolation and the figure caption can quote the setup without
# the parameters being restated by hand.
CONFIG_FIELDS = [
    "n_neurons", "n_features", "n_per_event", "n_branches", "onset1", "spread",
    "n_time", "alpha", "plateau_steps", "refractory",
]
# ``order_selective`` is a subset of ``sequence`` and so does not participate in
# the partition the CLASSES columns form; it is carried separately for that
# reason.
FIELDNAMES = [
    "dispersion", "theta", "interval", "p_connect", "seed",
    *CONFIG_FIELDS, *CLASSES, "order_selective",
]
# Diagnostics of the background run, counted per population:
# ``background_only`` fired for the background with no event at all;
# ``quiet_sequence`` are the sequence detectors of the same population without
# background; ``quiet_sequence_single`` are those of them that fire for one
# event alone once the background is added -- the false responses to a single
# event that the background rate is chosen to keep rare.
BACKGROUND_FIELDS = ["background_only", "quiet_sequence", "quiet_sequence_single"]


def fieldnames(steps_per_ms: int, background_rate: float) -> list[str]:
    """CSV columns for a run.

    The step and the background rate are written only when a run departs from
    the reported model, so that the reported invocation still writes the
    reported CSV byte for byte. Readers take a missing column as 1 step per ms
    and no background.
    """
    names = list(FIELDNAMES)
    if steps_per_ms != 1 or background_rate > 0:
        names += ["steps_per_ms", "background_rate"]
    if background_rate > 0:
        names += BACKGROUND_FIELDS
    return names


def run_cell(job: dict) -> dict:
    """Simulate one grid cell for one seed.

    Draws a stimulus and a connectivity matrix from the seed, runs the
    population against the two-event stimulus, both single-event controls and
    the order-reversed sequence, and classifies every neuron.

    Args:
        job: grid coordinates plus the fixed configuration.

    Returns:
        A row dict matching :func:`fieldnames`, with class counts and the count
        of sequence detectors that are also silent to the reversed order.
    """
    rng = np.random.default_rng(job["seed"])
    # Durations arrive in ms and are scaled to steps here, as in
    # e6_precision.py. At one step per ms every factor is 1 and the reported
    # model is untouched.
    r = job["steps_per_ms"]

    stim = two_event_stimulus(
        rng,
        n_features=job["n_features"],
        n_per_event=job["n_per_event"],
        onset1=job["onset1"] * r,
        interval=job["interval"] * r,
        spread=job["spread"] * r,
        n_time=job["n_time"] * r,
    )
    delays, weights = random_connectivity(
        rng,
        n_neurons=job["n_neurons"],
        n_features=job["n_features"],
        dispersion=job["dispersion"] * r,
        n_branches=job["n_branches"],
        p_connect=job["p_connect"],
    )
    base = NeuronParams()
    params = NeuronParams(
        theta=job["theta"],
        alpha=job["alpha"] / r,
        tau_den=base.tau_den * r,
        tau_mem=base.tau_mem * r,
        plateau_steps=job["plateau_steps"] * r,
        refractory=job["refractory"] * r,
        v_contrib=0.0,
    )

    # Drawn after the stimulus and the wiring, so that a seed's population is
    # the one the run without background classifies, and drawn once, so that
    # all four stimuli carry the same background. The classification compares
    # a neuron's responses across stimuli, and a fresh draw per stimulus would
    # let a lucky background on the two-event trial alone pass for a sequence
    # detector.
    background = None
    if job["background_rate"] > 0:
        per_step = job["background_rate"] / (1000.0 * r)
        background = (rng.random((stim.n_time, stim.n_features))
                      < per_step).astype(float)

    def respond(x, with_background=True):
        if background is not None and with_background:
            x = np.maximum(x, background)
        return simulate_population(x, delays, weights, params)

    response_both = respond(stim.both)
    response_event1 = respond(stim.event1_only)
    response_event2 = respond(stim.event2_only)

    # The order-reversed sequence: same features, same within-volley offsets,
    # only the order of the two volleys swapped, so nothing but temporal order
    # separates it from the target.
    response_reversed = respond(stim.reversed_order().both)

    counts = classify_detectors(response_both, response_event1, response_event2)
    is_sequence = sequence_mask(response_both, response_event1, response_event2)
    order_selective = int((is_sequence & ~(response_reversed > 0)).sum())

    row = {k: job[k] for k in
           ("dispersion", "theta", "interval", "p_connect", "seed")}
    row.update({k: job[k] for k in CONFIG_FIELDS})
    row.update(counts.as_dict())
    row["order_selective"] = order_selective
    row["steps_per_ms"] = r
    row["background_rate"] = job["background_rate"]

    if background is not None:
        # The same population without background, to count what the background
        # did to it: the detectors it turned into responders to a single event,
        # and the neurons it drives with no event at all.
        quiet = sequence_mask(respond(stim.both, False),
                              respond(stim.event1_only, False),
                              respond(stim.event2_only, False))
        single = (response_event1 > 0) | (response_event2 > 0)
        row["background_only"] = int((respond(np.zeros_like(stim.both)) > 0).sum())
        row["quiet_sequence"] = int(quiet.sum())
        row["quiet_sequence_single"] = int((quiet & single).sum())

    return {k: row[k] for k in fieldnames(r, job["background_rate"])}


def build_grid(args) -> list[dict]:
    """Expand the command-line arguments into one job per grid cell and seed."""
    fixed = dict(
        n_neurons=args.neurons,
        n_features=args.features,
        n_per_event=args.per_event,
        n_branches=args.branches,
        onset1=args.onset,
        spread=args.spread,
        n_time=args.time_steps,
        alpha=args.alpha,
        plateau_steps=args.plateau_steps,
        refractory=args.refractory,
        steps_per_ms=args.steps_per_ms,
        background_rate=args.background_rate,
    )
    grid = itertools.product(
        args.dispersions, args.thetas, args.intervals, args.p_connect,
        range(args.seed_base, args.seed_base + args.seeds),
    )
    return [
        dict(fixed, dispersion=d, theta=th, interval=iv, p_connect=p, seed=s)
        for d, th, iv, p, s in grid
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("results/e1_dispersion.csv"))
    parser.add_argument("--seeds", type=int, default=20,
                        help="independent seeds per grid cell")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes; 0 selects cpu_count() - 1")
    parser.add_argument("--quick", action="store_true",
                        help="small grid and few seeds, for a smoke test")

    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--per-event", type=int, default=40)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--onset", type=int, default=5)
    parser.add_argument("--spread", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=60)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--plateau-steps", type=int, default=15,
                        help="plateau length in ms")
    parser.add_argument("--refractory", type=int, default=2)
    parser.add_argument("--steps-per-ms", type=int, default=1,
                        help="simulation steps per millisecond (see the module "
                             "docstring); the reported sweep uses 1")
    parser.add_argument("--background-rate", type=float, default=0.0,
                        help="Poisson background on every feature, in Hz; the "
                             "reported sweep has none")

    parser.add_argument("--dispersions", type=int, nargs="+",
                        default=list(range(0, 31, 2)))
    parser.add_argument("--thetas", type=float, nargs="+",
                        default=[1.5, 2.5, 3.5, 4.5])
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=[5, 10, 15, 20])
    parser.add_argument("--p-connect", type=float, nargs="+", default=[0.1])

    args = parser.parse_args(argv)

    if args.quick:
        args.seeds = 3
        args.neurons = 300
        args.dispersions = [0, 5, 10, 15, 20, 25, 30]
        # Both axes stay plural even though a smoke test does not need the
        # resolution: make_fig_e1 highlights theta = 4.5 and fits crossover
        # against interval, so a grid with one theta or one interval makes the
        # figure script fail rather than run on reduced data.
        args.thetas = [2.5, 4.5]
        args.intervals = [5, 10, 15, 20]

    jobs = build_grid(args)
    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E1: {len(jobs)} cells "
          f"({len(args.dispersions)} dispersions x {len(args.thetas)} thetas x "
          f"{len(args.intervals)} intervals x {len(args.p_connect)} p x "
          f"{args.seeds} seeds), {args.neurons} neurons, {n_workers} workers, "
          f"{args.steps_per_ms} steps/ms, {args.plateau_steps} ms plateau, "
          f"{args.background_rate:g} Hz background")

    started = time.time()
    if n_workers == 1:
        rows = [run_cell(job) for job in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            rows = pool.map(run_cell, jobs, chunksize=4)
    elapsed = time.time() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames(args.steps_per_ms, args.background_rate))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {elapsed:.1f}s  ({elapsed / len(jobs) * 1000:.0f} ms/cell)")
    print(f"  wrote {args.out}")

    seq = np.array([r["sequence"] for r in rows]) / args.neurons
    print(f"  sequence-detector fraction: "
          f"min {seq.min():.4f}  max {seq.max():.4f}  mean {seq.mean():.4f}")

    # Order selectivity is pooled over seeds rather than averaged per seed:
    # at the operating threshold a single population contributes only a handful
    # of sequence detectors, and a ratio of small counts is better formed once.
    for theta in sorted({r["theta"] for r in rows}):
        n_seq = sum(r["sequence"] for r in rows if r["theta"] == theta)
        n_sel = sum(r["order_selective"] for r in rows if r["theta"] == theta)
        frac = f"{100.0 * n_sel / n_seq:.1f}%" if n_seq else "n/a"
        print(f"  theta {theta:>4}: order-selective {n_sel}/{n_seq} = {frac}")

    if args.background_rate > 0:
        for theta in sorted({r["theta"] for r in rows}):
            sub = [r for r in rows if r["theta"] == theta]
            n_quiet = sum(r["quiet_sequence"] for r in sub)
            n_false = sum(r["quiet_sequence_single"] for r in sub)
            frac = f"{100.0 * n_false / n_quiet:.1f}%" if n_quiet else "n/a"
            alone = sum(r["background_only"] for r in sub) / (len(sub) * args.neurons)
            print(f"  theta {theta:>4}: background alone drives "
                  f"{100.0 * alone:.2f}% of neurons; of the sequence detectors "
                  f"found without it, {n_false}/{n_quiet} = {frac} fire for "
                  f"one event alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

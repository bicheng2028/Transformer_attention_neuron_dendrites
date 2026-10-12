"""E2: aliasing, and the bound on unambiguously codable intervals.

This script is manuscript **Experiment 6**; see the numbering table in
``delaykey/README.md``.

A branch tuned to one inter-event interval cannot distinguish that interval from
others that produce the same pattern of coincident arrivals. The manuscript
raises this as "temporal folding" in Experiment 5 and presents it as an
explanatory mechanism; it is the same ambiguity a reviewer will raise as an
objection (Gap 9a). Measuring it turns the objection into a result, because the
width of the unambiguous range is exactly what bounds how long a sequence can be
before the code becomes degenerate -- the derivation the manuscript currently
asserts without one.

Procedure: build a population at a given dispersion, identify the neurons that
act as sequence detectors at a reference interval, then probe those same neurons
with the same stimulus retimed to every other interval. Because
:class:`~delaykey.stimuli.TwoEventStimulus` re-renders from a stored latent
description, feature identity and within-volley timing are held fixed as the
interval moves; only the separation changes.

Two statistics describe the sub-reference plateau and they are *not* the same
number, which is why both are recorded:

``response_fraction``
    the fraction of reference detectors responding at one probe interval. Its
    mean over the shorter probes is a *marginal* quantity -- the average height
    of the plateau.
``n_all_below`` / ``fraction_all_below``
    the fraction of reference detectors that respond at **every** probe interval
    strictly below the reference. This is the joint quantity, and it is strictly
    smaller than the mean of the marginals. It cannot be recovered from the
    per-probe rows, because those aggregate away detector identity, so it is
    accumulated inside ``run_cell`` while the per-detector masks still exist and
    then repeated on each row of its group.

Writes one row per (dispersion, theta, reference interval, probe interval, seed).

Usage::

    python experiments/e2_aliasing.py --quick
    python experiments/e2_aliasing.py --seeds 20

The exact invocation behind the committed CSV is recorded in ``run_all.sh``.
"""

from __future__ import annotations

import os

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

from delaykey.core import NeuronParams, simulate_population  # noqa: E402
from delaykey.stimuli import random_connectivity, two_event_stimulus  # noqa: E402

FIELDNAMES = [
    "dispersion", "theta", "reference_interval", "probe_interval", "seed",
    "n_neurons", "n_reference", "n_responding", "response_fraction",
    "n_all_below", "fraction_all_below",
]


def run_cell(job: dict) -> list[dict]:
    """Map the interval tuning curve of one population.

    Identifies sequence detectors at the reference interval, then probes them
    across every interval in the sweep, accumulating both the per-probe marginal
    response and the joint "responds at every shorter probe" mask.

    Returns:
        One row per probe interval. ``n_all_below`` and ``fraction_all_below``
        are properties of the whole group, so they take the same value on every
        row of it.
    """
    rng = np.random.default_rng(job["seed"])
    params = NeuronParams(theta=job["theta"], alpha=job["alpha"],
                          plateau_steps=job["plateau_steps"],
                          refractory=job["refractory"], v_contrib=0.0)

    stim = two_event_stimulus(
        rng, n_features=job["n_features"], n_per_event=job["n_per_event"],
        onset1=job["onset1"], interval=job["reference_interval"],
        spread=job["spread"], n_time=job["n_time"],
    )
    delays, weights = random_connectivity(
        rng, n_neurons=job["n_neurons"], n_features=job["n_features"],
        dispersion=job["dispersion"], n_branches=job["n_branches"],
        p_connect=job["p_connect"],
    )

    def responds(s):
        return simulate_population(s.both, delays, weights, params) > 0

    # Reference detectors: conjunctive at the reference interval.
    reference = (responds(stim)
                 & ~(simulate_population(stim.event1_only, delays, weights,
                                         params) > 0)
                 & ~(simulate_population(stim.event2_only, delays, weights,
                                         params) > 0))
    n_reference = int(reference.sum())

    # Running AND over the probes below the reference. Starts from the reference
    # mask itself so that a detector must satisfy every sub-reference probe to
    # survive; if there are no sub-reference probes the two statistics coincide,
    # which is the correct degenerate case.
    all_below = reference.copy()
    n_below_probes = 0

    rows = []
    for probe in job["probe_intervals"]:
        if job["onset1"] + probe + job["spread"] >= job["n_time"]:
            continue
        firing = responds(stim.with_interval(probe))
        n_responding = int((firing & reference).sum())
        if probe < job["reference_interval"]:
            all_below &= firing
            n_below_probes += 1
        rows.append({
            "dispersion": job["dispersion"],
            "theta": job["theta"],
            "reference_interval": job["reference_interval"],
            "probe_interval": int(probe),
            "seed": job["seed"],
            "n_neurons": job["n_neurons"],
            "n_reference": n_reference,
            "n_responding": n_responding,
            "response_fraction": (n_responding / n_reference) if n_reference else 0.0,
        })

    n_all_below = int(all_below.sum()) if n_below_probes else 0
    fraction_all_below = (n_all_below / n_reference) if n_reference else 0.0
    for row in rows:
        row["n_all_below"] = n_all_below
        row["fraction_all_below"] = fraction_all_below
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/e2_aliasing.csv"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--neurons", type=int, default=2000)
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--per-event", type=int, default=40)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--onset", type=int, default=5)
    parser.add_argument("--spread", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=100)
    parser.add_argument("--p-connect", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--plateau-steps", type=int, default=15)
    parser.add_argument("--refractory", type=int, default=2)

    parser.add_argument("--dispersions", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--thetas", type=float, nargs="+", default=[2.5])
    parser.add_argument("--reference-intervals", type=int, nargs="+",
                        default=[10, 15, 20])
    args = parser.parse_args(argv)

    if args.quick:
        # Two dispersions and both references the figure reads: panel b maps
        # 8 ms across dispersions and needs at least two rows to draw.
        args.seeds, args.neurons = 3, 500
        args.dispersions, args.reference_intervals = [10, 20], [8, 15]

    probe_intervals = list(range(0, args.time_steps - args.onset - args.spread))

    fixed = dict(
        n_neurons=args.neurons, n_features=args.features,
        n_per_event=args.per_event, n_branches=args.branches,
        onset1=args.onset, spread=args.spread, n_time=args.time_steps,
        p_connect=args.p_connect, alpha=args.alpha,
        plateau_steps=args.plateau_steps, refractory=args.refractory,
        probe_intervals=probe_intervals,
    )
    jobs = [dict(fixed, dispersion=d, theta=th, reference_interval=ri, seed=s)
            for d, th, ri, s in itertools.product(
                args.dispersions, args.thetas, args.reference_intervals,
                range(args.seed_base, args.seed_base + args.seeds))]

    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E2: {len(jobs)} populations x {len(probe_intervals)} probe intervals, "
          f"{args.neurons} neurons, {n_workers} workers")

    started = time.time()
    if n_workers == 1:
        nested = [run_cell(j) for j in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            nested = pool.map(run_cell, jobs, chunksize=1)
    rows = [r for group in nested for r in group]
    elapsed = time.time() - started

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {elapsed:.1f}s, wrote {args.out} ({len(rows)} rows)")

    ref = np.mean([r["n_reference"] for r in rows])
    print(f"  mean sequence detectors per population: {ref:.1f}")
    for d in args.dispersions:
        sub = [r for r in rows if r["dispersion"] == d
               and r["reference_interval"] == args.reference_intervals[0]]
        if not sub:
            continue
        by_probe = {}
        for r in sub:
            by_probe.setdefault(r["probe_interval"], []).append(r["response_fraction"])
        curve = {k: float(np.mean(v)) for k, v in sorted(by_probe.items())}
        best = max(curve, key=curve.get)
        side = {k: v for k, v in curve.items()
                if abs(k - args.reference_intervals[0]) > 3 and v > 0.2}
        print(f"  dispersion {d:2d}: peak at probe {best} ms "
              f"(fraction {curve[best]:.2f}); "
              f"{len(side)} interval(s) beyond the main lobe still above 0.2")

    # Marginal against joint, the distinction M9 turns on.
    print(f"\n  {'disp':>4} {'ref':>4} {'mean of marginals':>18} "
          f"{'responds at ALL below':>22}")
    for d in args.dispersions:
        for ref in args.reference_intervals:
            sub = [r for r in rows if r["dispersion"] == d
                   and r["reference_interval"] == ref]
            if not sub:
                continue
            below = [r["response_fraction"] for r in sub
                     if r["probe_interval"] < ref]
            joint = {(r["seed"], r["n_all_below"], r["n_reference"]) for r in sub}
            n_all = sum(a for _, a, _ in joint)
            n_ref = sum(b for _, _, b in joint)
            if not below or not n_ref:
                continue
            print(f"  {d:>4} {ref:>4} {100 * np.mean(below):>17.1f}% "
                  f"{100 * n_all / n_ref:>21.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

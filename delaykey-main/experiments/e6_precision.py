"""E6: how much conduction-velocity precision the delay-key scheme requires.

Sub-millisecond matching across a 10-30 ms delay spread implies a few per cent
precision on conduction velocity, which in vivo is temperature-, brain-state-
and activity-dependent. The manuscript contains no robustness analysis, and this
is the first objection a physiologist will raise (Gap 9b).

Two perturbations are applied to a population whose sequence detectors have
already been identified with exact delays:

*jitter*  each connection's delay is scaled by an independent draw of
          ``1 + N(0, s)``, modelling axon-to-axon variability in conduction
          velocity;
*scale*   every delay is scaled by a common factor ``k``, modelling a global
          shift such as a temperature change.

The reported quantity is the fraction of the original sequence detectors that
still respond conjunctively. Delays are re-rounded to whole steps after
perturbation, since the simulation is discrete-time.

Usage::

    python experiments/e6_precision.py --quick
    python experiments/e6_precision.py --seeds 20
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
    "perturbation", "magnitude", "dispersion", "theta", "interval", "seed",
    "n_neurons", "n_reference", "n_retained", "retention",
]


def run_cell(job: dict) -> list[dict]:
    """Identify sequence detectors, then perturb their delays and re-test."""
    rng = np.random.default_rng(job["seed"])
    params = NeuronParams(theta=job["theta"], alpha=job["alpha"],
                          plateau_steps=job["plateau_steps"],
                          refractory=job["refractory"], v_contrib=0.0)

    stim = two_event_stimulus(
        rng, n_features=job["n_features"], n_per_event=job["n_per_event"],
        onset1=job["onset1"], interval=job["interval"], spread=job["spread"],
        n_time=job["n_time"],
    )
    delays, weights = random_connectivity(
        rng, n_neurons=job["n_neurons"], n_features=job["n_features"],
        dispersion=job["dispersion"], n_branches=job["n_branches"],
        p_connect=job["p_connect"],
    )

    def conjunctive(d):
        return ((simulate_population(stim.both, d, weights, params) > 0)
                & ~(simulate_population(stim.event1_only, d, weights, params) > 0)
                & ~(simulate_population(stim.event2_only, d, weights, params) > 0))

    reference = conjunctive(delays)
    n_reference = int(reference.sum())
    if n_reference == 0:
        return []

    delays_ref = delays[reference]
    weights_ref = weights[reference]

    def retained(d):
        still = ((simulate_population(stim.both, d, weights_ref, params) > 0)
                 & ~(simulate_population(stim.event1_only, d, weights_ref,
                                         params) > 0)
                 & ~(simulate_population(stim.event2_only, d, weights_ref,
                                         params) > 0))
        return int(still.sum())

    rows = []
    for kind, magnitude in job["perturbations"]:
        if kind == "jitter":
            factor = 1.0 + rng.normal(0.0, magnitude, size=delays_ref.shape)
        else:  # global scale
            factor = np.full(delays_ref.shape, magnitude)
        perturbed = np.maximum(np.rint(delays_ref * factor), 0).astype(np.int64)

        n_retained = retained(perturbed)
        rows.append({
            "perturbation": kind,
            "magnitude": float(magnitude),
            "dispersion": job["dispersion"],
            "theta": job["theta"],
            "interval": job["interval"],
            "seed": job["seed"],
            "n_neurons": job["n_neurons"],
            "n_reference": n_reference,
            "n_retained": n_retained,
            "retention": n_retained / n_reference,
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/e6_precision.csv"))
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
    parser.add_argument("--time-steps", type=int, default=70)
    parser.add_argument("--p-connect", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--plateau-steps", type=int, default=15)
    parser.add_argument("--refractory", type=int, default=2)

    parser.add_argument("--dispersions", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--thetas", type=float, nargs="+", default=[2.5])
    parser.add_argument("--intervals", type=int, nargs="+", default=[15])
    parser.add_argument("--jitters", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.80, 0.90, 0.95, 1.05, 1.10, 1.20])
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.neurons = 3, 500
        args.dispersions = [20]

    perturbations = ([("jitter", m) for m in args.jitters]
                     + [("scale", k) for k in args.scales])

    fixed = dict(
        n_neurons=args.neurons, n_features=args.features,
        n_per_event=args.per_event, n_branches=args.branches,
        onset1=args.onset, spread=args.spread, n_time=args.time_steps,
        p_connect=args.p_connect, alpha=args.alpha,
        plateau_steps=args.plateau_steps, refractory=args.refractory,
        perturbations=perturbations,
    )
    jobs = [dict(fixed, dispersion=d, theta=th, interval=iv, seed=s)
            for d, th, iv, s in itertools.product(
                args.dispersions, args.thetas, args.intervals,
                range(args.seed_base, args.seed_base + args.seeds))]

    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E6: {len(jobs)} populations x {len(perturbations)} perturbations, "
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
    for kind, values in (("jitter", args.jitters), ("scale", args.scales)):
        print(f"\n  {kind}:")
        for m in values:
            sub = [r["retention"] for r in rows
                   if r["perturbation"] == kind and abs(r["magnitude"] - m) < 1e-9]
            if sub:
                label = f"{m*100:.0f}%" if kind == "jitter" else f"x{m:.2f}"
                print(f"    {label:>7}  retention {np.mean(sub):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

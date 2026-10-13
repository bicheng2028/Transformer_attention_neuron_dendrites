"""E6 alignment: why slowing costs less than speeding near the codable bound.

The precision sweep (``e6_precision.py``) finds that a common slowing of
conduction costs fewer sequence detectors than an equal speeding, but only where
the dispersion is close to the inter-event interval. This script measures the
cause. A detector is selected because some of its event-1 and event-2 inputs
arrive close enough together to cross threshold, and close enough is not the
same as exactly. Where delay differences long enough to compensate the interval
are scarce, the near-matches that pass tend to fall short of it, so event-1
inputs arrive early. Dilating every delay closes that gap, and compressing them
widens it.

For each sequence detector, the densest window of ``--window`` ms holding
arrivals from both events is found, and the mean event-1 arrival minus the mean
event-2 arrival inside it is recorded, in ms. A negative value means event 1
arrives early. One row per (dispersion, interval, seed) gives the number of
detectors, the mean offset, and the fractions early and late. The stimulus,
model and time step are those of the reported precision sweep.

Usage::

    python experiments/e6_alignment.py            # the reported grid
    python experiments/e6_alignment.py --quick
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
from delaykey.detectors import sequence_mask  # noqa: E402
from delaykey.stimuli import random_connectivity, two_event_stimulus  # noqa: E402

FIELDNAMES = ["dispersion", "theta", "interval", "seed", "n_neurons",
              "steps_per_ms", "window_ms", "n_detectors", "mean_offset_ms",
              "frac_early", "frac_late"]


def offset(a1: np.ndarray, a2: np.ndarray, width: int) -> float | None:
    """Mean event-1 minus event-2 arrival in the densest mixed window, in steps."""
    best = None
    for start in np.unique(np.concatenate([a1, a2])):
        in1 = a1[(a1 >= start) & (a1 < start + width)]
        in2 = a2[(a2 >= start) & (a2 < start + width)]
        if len(in1) and len(in2):
            count = len(in1) + len(in2)
            if best is None or count > best[0]:
                best = (count, float(in1.mean() - in2.mean()))
    return None if best is None else best[1]


def run_cell(job: dict) -> dict:
    rng = np.random.default_rng(job["seed"])
    r = job["steps_per_ms"]
    base = NeuronParams()
    params = NeuronParams(theta=job["theta"], alpha=job["alpha"] / r,
                          tau_den=base.tau_den * r, tau_mem=base.tau_mem * r,
                          plateau_steps=job["plateau_steps"] * r,
                          refractory=job["refractory"] * r, v_contrib=0.0)
    stim = two_event_stimulus(
        rng, n_features=job["n_features"], n_per_event=job["n_per_event"],
        onset1=job["onset1"] * r, interval=job["interval"] * r,
        spread=job["spread"] * r, n_time=job["n_time"] * r)
    delays, weights = random_connectivity(
        rng, n_neurons=job["n_neurons"], n_features=job["n_features"],
        dispersion=job["dispersion"] * r, n_branches=1,
        p_connect=job["p_connect"])
    detectors = sequence_mask(
        simulate_population(stim.both, delays, weights, params),
        simulate_population(stim.event1_only, delays, weights, params),
        simulate_population(stim.event2_only, delays, weights, params))

    t1 = stim.onset1 + stim.offsets1
    t2 = stim.onset1 + stim.interval + stim.offsets2
    offsets = []
    for n in np.flatnonzero(detectors):
        on1 = weights[n, 0, stim.features1] > 0
        on2 = weights[n, 0, stim.features2] > 0
        found = offset((t1 + delays[n, 0, stim.features1])[on1],
                       (t2 + delays[n, 0, stim.features2])[on2],
                       int(round(job["window_ms"] * r)))
        if found is not None:
            offsets.append(found / r)
    offsets = np.array(offsets)
    keep = ("dispersion", "theta", "interval", "seed", "n_neurons",
            "steps_per_ms", "window_ms")
    return dict({k: job[k] for k in keep},
                n_detectors=int(offsets.size),
                mean_offset_ms=float(offsets.mean()) if offsets.size else float("nan"),
                frac_early=float(np.mean(offsets < 0)) if offsets.size else float("nan"),
                frac_late=float(np.mean(offsets > 0)) if offsets.size else float("nan"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/e6_alignment.csv"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--neurons", type=int, default=2000)
    parser.add_argument("--steps-per-ms", type=int, default=10)
    parser.add_argument("--window-ms", type=float, default=2.0)
    parser.add_argument("--dispersions", type=int, nargs="+",
                        default=[10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--intervals", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--theta", type=float, default=2.5)
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds, args.neurons = 3, 500
        args.dispersions, args.intervals = [15, 30], [15]

    fixed = dict(n_neurons=args.neurons, n_features=200, n_per_event=40,
                 onset1=5, spread=3, n_time=70, p_connect=0.1, alpha=2.5,
                 plateau_steps=15, refractory=2, theta=args.theta,
                 steps_per_ms=args.steps_per_ms, window_ms=args.window_ms)
    jobs = [dict(fixed, dispersion=d, interval=iv, seed=s)
            for d, iv, s in itertools.product(args.dispersions, args.intervals,
                                              range(args.seeds))]
    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E6 alignment: {len(jobs)} populations, {args.neurons} neurons, "
          f"{args.steps_per_ms} steps/ms, {n_workers} workers")
    started = time.time()
    with mp.Pool(n_workers) as pool:
        rows = pool.map(run_cell, jobs, chunksize=1)
    print(f"  {time.time() - started:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {args.out}")
    for iv in args.intervals:
        for d in args.dispersions:
            cell = [r for r in rows if r["interval"] == iv and r["dispersion"] == d
                    and r["n_detectors"] > 0]
            n = sum(r["n_detectors"] for r in cell)
            if not n:
                continue
            early = sum(r["frac_early"] * r["n_detectors"] for r in cell) / n
            late = sum(r["frac_late"] * r["n_detectors"] for r in cell) / n
            mean = sum(r["mean_offset_ms"] * r["n_detectors"] for r in cell) / n
            print(f"  T={iv:2d} Delta={d:2d}: {n:5d} detectors, offset {mean:+.3f} ms, "
                  f"early {early:.0%}, late {late:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

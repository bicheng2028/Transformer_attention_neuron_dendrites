"""E1 theory: detector-class fractions computed from the arrival statistics alone.

The dispersion sweep (``e1_dispersion.py``) measures how often random wiring
yields each detector class. This script computes the same fractions without
simulating a neuron, by :func:`delaykey.theory.class_probabilities`, for the
same stimulus, model and grid. If the two agree, the sweep is measuring the
geometry of delayed arrivals and nothing else. The crossover and peak
dispersions then follow from a calculation rather than from estimates built on
the handful of sequence detectors one seed provides.

The interval grid is finer than the sweep's, so that the crossover and the peak
can be drawn as functions of the interval. The dispersion grid runs on every
integer, since delays are whole steps.

Writes one row per (theta, interval, dispersion). The figure reads it beside the
sweep's CSV.

Usage::

    python experiments/e1_theory.py            # the reported grid, a few minutes
    python experiments/e1_theory.py --quick    # smoke test
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.theory import class_probabilities, soma_margin  # noqa: E402

FIELDNAMES = [
    "dispersion", "theta", "interval", "p_connect", "n_per_event", "onset",
    "spread", "n_time", "tau_den", "alpha", "tau_mem", "plateau_steps",
    "soma_margin", "sequence", "event1", "event2", "dual", "silent",
]


def run_cell(job: dict) -> dict:
    probs = class_probabilities(
        job["dispersion"], job["interval"], job["theta"],
        p_connect=job["p_connect"], n_per_event=job["n_per_event"],
        onset=job["onset"], spread=job["spread"], n_time=job["n_time"],
        tau_den=job["tau_den"], alpha=job["alpha"], tau_mem=job["tau_mem"],
        plateau_steps=job["plateau_steps"])
    return dict(job, sequence=probs.sequence, event1=probs.event1,
                event2=probs.event2, dual=probs.dual, silent=probs.silent)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/e1_theory.csv"))
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes; 0 selects cpu_count() - 1")
    parser.add_argument("--quick", action="store_true")
    # The sweep's stimulus and model (e1_dispersion.py, NeuronParams).
    parser.add_argument("--p-connect", type=float, default=0.1)
    parser.add_argument("--per-event", type=int, default=40)
    parser.add_argument("--onset", type=int, default=5)
    parser.add_argument("--spread", type=int, default=3)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--tau-den", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--tau-mem", type=float, default=10.0)
    parser.add_argument("--plateau-steps", type=int, default=15)
    parser.add_argument("--dispersions", type=int, nargs="+",
                        default=list(range(0, 61)))
    parser.add_argument("--thetas", type=float, nargs="+",
                        default=[1.5, 2.5, 3.5, 4.5])
    parser.add_argument("--intervals", type=int, nargs="+",
                        default=list(range(3, 23)))
    args = parser.parse_args(argv)

    if args.quick:
        args.dispersions = list(range(0, 41, 2))
        args.thetas = [4.5]
        args.intervals = [5, 10, 15, 20]

    fixed = dict(p_connect=args.p_connect, n_per_event=args.per_event,
                 onset=args.onset, spread=args.spread, n_time=args.time_steps,
                 tau_den=args.tau_den, alpha=args.alpha, tau_mem=args.tau_mem,
                 plateau_steps=args.plateau_steps,
                 soma_margin=soma_margin(args.alpha, args.tau_mem,
                                         args.plateau_steps))
    jobs = [dict(fixed, dispersion=d, theta=th, interval=iv)
            for th, iv, d in itertools.product(args.thetas, args.intervals,
                                               args.dispersions)]
    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E1 theory: {len(jobs)} cells ({len(args.thetas)} thetas x "
          f"{len(args.intervals)} intervals x {len(args.dispersions)} dispersions), "
          f"{n_workers} workers")

    started = time.time()
    if n_workers == 1:
        rows = [run_cell(job) for job in jobs]
    else:
        with mp.Pool(n_workers) as pool:
            rows = pool.map(run_cell, jobs, chunksize=8)
    print(f"  {time.time() - started:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

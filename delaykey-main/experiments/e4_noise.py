"""E4: noise robustness of a single hard-gated K-line.

This script is manuscript **Experiment 2**; see the numbering table in
``delaykey/README.md``. It is a re-implementation. No generating code for the
originally reported version survived in the repository, so the figure behind it
was drawn by quoting the manuscript's own prose back at itself; nothing here is
quoted, and every number this writes is simulated against ``delaykey.core`` like
the rest of the suite.

The design is the manuscript's, deliberately: the *single K-line* of
Experiment 1 -- one dendritic branch carrying one delay set, no redundant
K-matrix ensemble -- driven with the A->B sequence under two kinds of noise, with
a soft-gating control that bypasses the calcium threshold. What is new is the
statistical footing: 20 independent seeds of 100 trials each, with percentile
bootstrap intervals over seeds, matching every other experiment in the suite.

Three noise conditions:

*jitter*    every spike time of the target displaced by an independent draw of
            ``N(0, sigma)``, rounded to whole steps.
*spurious*  every (time step, feature) bin carries an extra spike with
            probability ``density``, independently. The target is present.
*combined*  ``sigma`` = 2 jitter together with 20% spurious density, the cell
            the manuscript quotes its largest fold-difference from.

Two rates are measured per condition. *Accuracy* is the fraction of trials in
which the neuron fires with the target present. The *false-positive rate* is the
fraction of matched noise-only trials -- same noise process, target removed -- in
which it fires anyway; it is defined only where there is noise to fire on, so
the jitter arm leaves it empty.

Activation is reported as spikes emitted per trial, the same unit Experiment 3
uses, so the two experiments' activation figures are directly comparable. That
is a deliberate change from the original, whose "mean activation" was in
unstated units and could not be compared with anything.

A note on the threshold. With two inputs at unit weight a branch can carry at
most two coincident spikes, so ``theta`` must sit below 2 for the line to fire at
all; ``theta = 1.5`` is the two-input requirement, and is what
``tests/test_core.py::test_delay_compensation_discriminates_order`` already pins
the mechanism down with. The five-input operating point of Experiment 5 is not
reachable on a two-input branch, which is why this experiment cannot be run
there and says so rather than implying otherwise.

Usage::

    python experiments/e4_noise.py --quick
    python experiments/e4_noise.py --seeds 20
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

FIELDNAMES = [
    "noise", "level", "gating", "theta", "seed", "n_trials",
    "t_first", "t_second", "delay_a", "delay_b", "n_time",
    "accuracy", "false_positive_rate", "activation",
]


def render(rng, sigma: float, density: float, n_time: int,
           t_first: int, t_second: int, target: bool = True) -> np.ndarray:
    """One trial of the two-feature stimulus, with noise.

    Args:
        rng: random generator.
        sigma: s.d. of the Gaussian jitter applied to each target spike time.
        density: probability that any given (step, feature) bin carries a
            spurious spike.
        n_time: trial length in steps.
        t_first: nominal time of the A spike.
        t_second: nominal time of the B spike.
        target: include the A->B pair. ``False`` gives the noise-only trial the
            false-positive rate is measured on.

    Returns:
        ``(n_time, 2)`` binary spike array; feature 0 is A, feature 1 is B.
    """
    x = np.zeros((n_time, 2))
    if target:
        times = np.array([t_first, t_second], dtype=float)
        if sigma > 0:
            times = times + np.rint(rng.normal(0.0, sigma, size=2))
        for feature, t in enumerate(times):
            t = int(t)
            if 0 <= t < n_time:
                x[t, feature] = 1.0
    if density > 0:
        # Superimposed, not overwritten: a spurious spike landing on a target
        # spike leaves one spike there, which is what "isolated extra spikes"
        # means physically.
        x = np.maximum(x, (rng.random((n_time, 2)) < density).astype(float))
    return x


def run_cell(job: dict) -> list[dict]:
    """Run every noise condition for one seed, under both gating rules."""
    params = NeuronParams(
        theta=job["theta"], alpha=job["alpha"], tau_mem=job["tau_mem"],
        tau_den=job["tau_den"], v_threshold=1.0, v_contrib=job["v_contrib"],
        plateau_steps=job["plateau_steps"], refractory=job["refractory"],
    )
    # The single K-line: one neuron, one branch, delays tuned to A->B so that
    # the two arrivals coincide at t_second + delay_b.
    delays = np.array([[[job["delay_a"], job["delay_b"]]]], dtype=np.int64)
    weights = np.ones((1, 1, 2))

    rows = []
    for index, (kind, level) in enumerate(job["conditions"]):
        sigma, density = {
            "jitter": (level, 0.0),
            "spurious": (0.0, level),
            "combined": (job["combined_sigma"], level),
        }[kind]

        for gating in ("hard", "soft"):
            # A fresh generator per condition, re-seeded identically for the two
            # gating rules, so both see exactly the same noise realisations and
            # the comparison is paired trial by trial. Seeded from a SeedSequence
            # over (seed, condition index) rather than from hash(), which numpy
            # would accept but which Python salts per process.
            rng = np.random.default_rng([job["seed"], index])
            hits, spikes, false_hits = 0, [], 0
            for _ in range(job["n_trials"]):
                x = render(rng, sigma, density, job["n_time"],
                           job["t_first"], job["t_second"], target=True)
                count = simulate_population(x, delays, weights, params,
                                            gating=gating)[0]
                hits += count > 0
                spikes.append(float(count))
                if density > 0:
                    x0 = render(rng, sigma, density, job["n_time"],
                                job["t_first"], job["t_second"], target=False)
                    false_hits += simulate_population(
                        x0, delays, weights, params, gating=gating)[0] > 0

            rows.append({
                "noise": kind,
                "level": float(level),
                "gating": gating,
                "theta": job["theta"],
                "seed": job["seed"],
                "n_trials": job["n_trials"],
                "t_first": job["t_first"],
                "t_second": job["t_second"],
                "delay_a": job["delay_a"],
                "delay_b": job["delay_b"],
                "n_time": job["n_time"],
                "accuracy": hits / job["n_trials"],
                "false_positive_rate": (false_hits / job["n_trials"]
                                        if density > 0 else ""),
                "activation": float(np.mean(spikes)),
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("results/e4_noise.csv"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--time-steps", type=int, default=30)
    parser.add_argument("--t-first", type=int, default=5)
    parser.add_argument("--t-second", type=int, default=10)
    parser.add_argument("--delay-a", type=int, default=5)
    parser.add_argument("--delay-b", type=int, default=0)

    parser.add_argument("--thetas", type=float, nargs="+", default=[1.5])
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--tau-mem", type=float, default=5.0)
    parser.add_argument("--tau-den", type=float, default=2.0)
    parser.add_argument("--v-contrib", type=float, default=0.10)
    parser.add_argument("--plateau-steps", type=int, default=10)
    parser.add_argument("--refractory", type=int, default=2)

    parser.add_argument("--jitters", type=float, nargs="+",
                        default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Sampled finely at the low end: whatever advantage hard gating has lives
    # in the sparse regime, and a grid that starts at 0.1 steps straight over it.
    parser.add_argument("--densities", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--combined-sigma", type=float, default=2.0)
    parser.add_argument("--combined-densities", type=float, nargs="+",
                        default=[0.2])
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.n_trials = 3, 25

    conditions = ([("jitter", m) for m in args.jitters]
                  + [("spurious", d) for d in args.densities]
                  + [("combined", d) for d in args.combined_densities])

    fixed = dict(
        n_trials=args.n_trials, n_time=args.time_steps, t_first=args.t_first,
        t_second=args.t_second, delay_a=args.delay_a, delay_b=args.delay_b,
        alpha=args.alpha, tau_mem=args.tau_mem, tau_den=args.tau_den,
        v_contrib=args.v_contrib, plateau_steps=args.plateau_steps,
        refractory=args.refractory, combined_sigma=args.combined_sigma,
        conditions=conditions,
    )
    jobs = [dict(fixed, theta=th, seed=s)
            for th, s in itertools.product(
                args.thetas, range(args.seed_base, args.seed_base + args.seeds))]

    n_workers = args.jobs or max(1, min(mp.cpu_count() - 1, len(jobs)))
    print(f"E4: {len(jobs)} seeds x {len(conditions)} conditions x 2 gating "
          f"modes x {args.n_trials} trials, {n_workers} workers")

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

    def pick(kind, level, gating, key):
        vals = [r[key] for r in rows if r["noise"] == kind
                and abs(r["level"] - level) < 1e-9 and r["gating"] == gating
                and r[key] != ""]
        return float(np.mean(vals)) if vals else float("nan")

    for kind, levels in (("jitter", args.jitters),
                         ("spurious", args.densities),
                         ("combined", args.combined_densities)):
        print(f"\n  {kind}:  {'level':>6} {'hard acc':>9} {'soft acc':>9} "
              f"{'hard FP':>8} {'soft FP':>8} {'hard act':>9} {'soft act':>9} "
              f"{'fold':>6}")
        for m in levels:
            h, s = pick(kind, m, "hard", "activation"), pick(kind, m, "soft", "activation")
            fold = (s / h) if h > 0 else float("inf")
            print(f"          {m:>6.2f} {pick(kind,m,'hard','accuracy'):>9.3f} "
                  f"{pick(kind,m,'soft','accuracy'):>9.3f} "
                  f"{pick(kind,m,'hard','false_positive_rate'):>8.3f} "
                  f"{pick(kind,m,'soft','false_positive_rate'):>8.3f} "
                  f"{h:>9.3f} {s:>9.3f} {fold:>6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E3: discriminability of hard versus soft gating at matched false-positive rate.

This script is manuscript **Experiment 3**; see the numbering table in
``delaykey/README.md``.

Manuscript Experiment 2 reports that soft gating holds 100% detection at every
noise level while hard gating falls to 20% at sigma = 5. On the metric reported,
the baseline wins, and the 33x/63x headline measures activation rather than
performance (Gap 3). Two things are wrong with that comparison and both are
fixed here.

First, the soft-gating control in the manuscript has no delays at all, so its
failure or success confounds gating with delay tuning. Here both models receive
*identical* delays and connectivity and differ only in whether the calcium
threshold is applied, so the contrast isolates gating.

Second, detection rate at a single operating point is not discriminability. This
runs a target-versus-distractor task -- A->B against B->A, the same features and
the same within-volley timing, differing only in order -- and sweeps the readout
threshold to trace a full ROC. AUC is threshold-free and therefore comparable
across models at matched false-positive rate.

Activation is reported on its own axis as spikes emitted per trial, which is
comparable between the two models in the same units.

This experiment may not favour hard gating. A quantified
selectivity-versus-activation trade-off is a publishable result either way, and
is a more honest claim than the current framing.

Two choices in the design are open to challenge, and both are exposed as flags
rather than buried, so that the reported result can be checked against its
alternatives instead of being taken on trust:

``--readout``
    the decision variable. ``spikes`` (the default, and what the manuscript
    reports) totals somatic spikes over the trial. ``peak`` totals the peak
    somatic membrane potential instead. The signed AUC of the *soft* model is
    sensitive to this choice, because with no threshold and no dendritic reset
    a soft branch holds the soma at its refractory ceiling for as long as any
    drive is present -- so a spike count measures how *long* drive lasted, and a
    dispersed volley lasts longer than a coincident one.
``--tuned-by``
    which model selects the tuned subpopulation. ``hard`` (the default) selects
    once under hard gating and scores both models on those neurons, so the two
    read out the same cells; ``own`` lets each model select by its own criterion,
    which removes the cross-model selection but reintroduces a population
    difference between the arms.

Usage::

    python experiments/e3_discriminability.py --quick
    python experiments/e3_discriminability.py --seeds 20

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
    "gating", "readout", "tuned_by", "sigma", "dispersion", "theta", "seed",
    "n_tuned", "auc", "auc_unsigned", "d_prime", "activation_target",
    "activation_distractor",
]


def auc_from_scores(target: np.ndarray, distractor: np.ndarray) -> float:
    """Area under the ROC, computed as the Mann-Whitney statistic.

    Equal to the probability that a randomly drawn target trial scores above a
    randomly drawn distractor trial, with ties counted as half. Threshold-free,
    so it compares models whose scores live on different scales.
    """
    target = np.asarray(target, dtype=float)
    distractor = np.asarray(distractor, dtype=float)
    if target.size == 0 or distractor.size == 0:
        return float("nan")
    combined = np.concatenate([target, distractor])
    order = combined.argsort(kind="mergesort")
    ranks = np.empty(combined.size, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1, dtype=float)

    # Average ranks within ties, so tied scores contribute 0.5 rather than 1.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]

    rank_sum = ranks[:target.size].sum()
    u = rank_sum - target.size * (target.size + 1) / 2.0
    return float(u / (target.size * distractor.size))


def d_prime_from_scores(target: np.ndarray, distractor: np.ndarray) -> float:
    """Standard sensitivity index, pooling the two standard deviations."""
    target = np.asarray(target, dtype=float)
    distractor = np.asarray(distractor, dtype=float)
    pooled = np.sqrt(0.5 * (target.var(ddof=1) + distractor.var(ddof=1)))
    if pooled == 0:
        return float("nan")
    return float((target.mean() - distractor.mean()) / pooled)


def run_cell(job: dict) -> list[dict]:
    """Run one (dispersion, theta, sigma, seed) cell for both gating modes."""
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
    distractor_clean = stim.reversed_order()
    readout = job.get("readout", "spikes")
    tuned_by = job.get("tuned_by", "hard")

    def select(gating):
        """Neurons that fire for the clean target and not the clean distractor."""
        return ((simulate_population(stim.both, delays, weights, params,
                                     gating=gating) > 0)
                & ~(simulate_population(distractor_clean.both, delays, weights,
                                        params, gating=gating) > 0))

    # Tuned subpopulation: fires for the clean target, not the clean distractor.
    # Under the default it is selected once under hard gating and held fixed, so
    # both models read out the same neurons and the comparison is not confounded
    # by selection. That choice is itself contestable -- it scores the soft model
    # on cells chosen by its competitor's criterion -- so ``--tuned-by own`` lets
    # each model pick its own.
    tuned_shared = select("hard") if tuned_by == "hard" else None

    def score(x, gating):
        """The decision variable, summed over the tuned population."""
        if readout == "peak":
            _, peak = simulate_population(x, delays_g, weights_g, params,
                                          gating=gating, return_peak=True)
            return float(peak.sum())
        return float(simulate_population(x, delays_g, weights_g, params,
                                         gating=gating).sum())

    rows = []
    for gating in ("hard", "soft"):
        tuned = tuned_shared if tuned_shared is not None else select(gating)
        n_tuned = int(tuned.sum())
        if n_tuned == 0:
            continue
        delays_g = delays[tuned]
        weights_g = weights[tuned]

        target_scores, distractor_scores = [], []
        # Activation stays a spike count whatever the decision variable is: it is
        # a cost in emitted spikes, and a peak potential is not that.
        target_spikes = []
        for _ in range(job["n_trials"]):
            noisy_target = stim.with_jitter(job["sigma"], rng)
            noisy_distractor = distractor_clean.with_jitter(job["sigma"], rng)
            target_scores.append(score(noisy_target.both, gating))
            distractor_scores.append(score(noisy_distractor.both, gating))
            target_spikes.append(
                float(simulate_population(noisy_target.both, delays_g,
                                          weights_g, params,
                                          gating=gating).sum())
                if readout == "peak" else target_scores[-1])

        target_scores = np.array(target_scores, dtype=float)
        distractor_scores = np.array(distractor_scores, dtype=float)
        auc = auc_from_scores(target_scores, distractor_scores)
        rows.append({
            "gating": gating,
            "readout": readout,
            "tuned_by": tuned_by,
            "sigma": job["sigma"],
            "dispersion": job["dispersion"],
            "theta": job["theta"],
            "seed": job["seed"],
            "n_tuned": n_tuned,
            "auc": auc,
            # An AUC well below 0.5 does not mean the model is uninformative:
            # it means the readout separates the two classes with reversed
            # polarity, which a downstream decoder recovers by inverting its
            # decision rule. Reporting the signed value alone would understate
            # a model that is in fact discriminative, so the polarity-free
            # magnitude is carried alongside it.
            "auc_unsigned": max(auc, 1.0 - auc) if not np.isnan(auc) else auc,
            "d_prime": d_prime_from_scores(target_scores, distractor_scores),
            # Spikes emitted per trial per tuned neuron: the activation cost,
            # in units directly comparable between the two models. Always a
            # spike count, even when the decision variable is a peak potential.
            "activation_target": float(np.mean(target_spikes) / n_tuned),
            "activation_distractor": float(distractor_scores.mean() / n_tuned)
            if readout == "spikes" else float("nan"),
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("results/e3_discriminability.csv"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--features", type=int, default=200)
    parser.add_argument("--per-event", type=int, default=40)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--onset", type=int, default=10)
    parser.add_argument("--spread", type=int, default=3)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--time-steps", type=int, default=70)
    parser.add_argument("--p-connect", type=float, default=0.1)
    parser.add_argument("--n-trials", type=int, default=60)

    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--plateau-steps", type=int, default=15)
    parser.add_argument("--refractory", type=int, default=2)

    parser.add_argument("--dispersions", type=int, nargs="+", default=[20])
    parser.add_argument("--thetas", type=float, nargs="+", default=[2.5])
    parser.add_argument("--sigmas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Both default to what the manuscript reports; see the module docstring for
    # why each is contestable and what the alternative shows.
    parser.add_argument("--readout", choices=("spikes", "peak"), default="spikes",
                        help="decision variable: summed spike count, or summed "
                             "peak somatic membrane potential")
    parser.add_argument("--tuned-by", choices=("hard", "own"), default="hard",
                        help="which model selects the tuned subpopulation")
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.neurons, args.n_trials = 3, 400, 20
        args.sigmas = [0.0, 2.0, 5.0]

    fixed = dict(
        n_neurons=args.neurons, n_features=args.features,
        n_per_event=args.per_event, n_branches=args.branches,
        onset1=args.onset, spread=args.spread, interval=args.interval,
        n_time=args.time_steps, p_connect=args.p_connect,
        n_trials=args.n_trials, alpha=args.alpha,
        plateau_steps=args.plateau_steps, refractory=args.refractory,
        readout=args.readout, tuned_by=args.tuned_by,
    )
    jobs = [dict(fixed, dispersion=d, theta=th, sigma=sg, seed=s)
            for d, th, sg, s in itertools.product(
                args.dispersions, args.thetas, args.sigmas,
                range(args.seed_base, args.seed_base + args.seeds))]

    n_workers = args.jobs or max(1, mp.cpu_count() - 1)
    print(f"E3: {len(jobs)} cells x 2 gating modes x {args.n_trials} trials, "
          f"{args.neurons} neurons, readout={args.readout}, "
          f"tuned-by={args.tuned_by}, {n_workers} workers")

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
    print(f"\n  {'sigma':>6} {'hard AUC':>9} {'soft AUC':>9} "
          f"{'hard |AUC|':>11} {'soft |AUC|':>11} "
          f"{'hard spk':>9} {'soft spk':>9}")
    for sigma in args.sigmas:
        def pick(g, key):
            vals = [r[key] for r in rows
                    if r["gating"] == g and r["sigma"] == sigma
                    and not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")
        print(f"  {sigma:>6.1f} {pick('hard','auc'):>9.3f} "
              f"{pick('soft','auc'):>9.3f} "
              f"{pick('hard','auc_unsigned'):>11.3f} "
              f"{pick('soft','auc_unsigned'):>11.3f} "
              f"{pick('hard','activation_target'):>9.3f} "
              f"{pick('soft','activation_target'):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

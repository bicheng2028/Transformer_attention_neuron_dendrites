"""E3: discriminability of hard versus soft gating at matched false-positive rate.

This script is manuscript **Experiment 3**; see the numbering table in
``delaykey/README.md``.

-------------------------------------------------------------------------------
WHAT THIS SCRIPT MEASURES (2026-09 revision)
-------------------------------------------------------------------------------

The previous version reported the AUC of the *tuned* subpopulation as its
headline result, and that number cannot bear the weight put on it. The tuned
subpopulation is selected, by construction, as the neurons that fire for the
clean target and not for the clean distractor. Its AUC at sigma = 0 is therefore
1.0 by definition of the selection, not by measurement, and the shallow decline
of that AUC under jitter says only that jitter degrades the target and the
distractor responses symmetrically, leaving the rank order between them mostly
intact. A rank order that survives is not a population code that survives. Read
as robustness, the old result claimed a tolerance to perturbation that no
population of coincidence detectors could plausibly have.

The revision replaces the headline with four direct quantities, none of which
is fixed by the selection rule.

Definitions
-----------
The population is partitioned once, on the *clean* stimulus, into:

    tuned    := neurons that fire for the clean target and not for the clean
                distractor. Selected once per seed under hard gating
                (``--tuned-by hard``) or independently per gating mode
                (``--tuned-by own``), then held fixed across every jitter
                level, so the partition cannot drift with sigma.
    nontuned := every remaining neuron. Note that this pool is not "the silent
                neurons": it holds cells that answer both orders and cells that
                answer the distractor alone, as well as cells that were silent
                when clean. Its activity is therefore crosstalk in the general
                sense -- activity that carries no evidence for the target --
                and it is well above zero even at sigma = 0.

The four reported quantities, each a mean over trials:

1. ``tuned_activation``
       Spikes per tuned neuron per trial. The **signal channel**: how much
       drive the target still delivers to the cells tuned for it.

2. ``nontuned_activation``
       Spikes per non-tuned neuron per trial. The **crosstalk channel**: how
       much activity jitter pushes into cells that are not tuned for the
       target.

3. ``snr``
       tuned_activation / (nontuned_activation + eps). Ratio of the two
       channels; 1.0 when the population has stopped distinguishing them.

4. ``si``
       (tuned_activation - nontuned_activation)
       / (tuned_activation + nontuned_activation + eps).
       Population selectivity index, in [-1, 1]. 1 when all activity sits in
       the tuned pool, 0 when the two pools are equally active, negative when
       the non-tuned pool is the louder of the two.

Each is reported for the target and for the distractor stimulus separately, so
both the absolute level (how much activity survives jitter at all) and the
contrast between the two orders are visible.

Unlike the AUC, none of these is pinned by the selection rule. Selection fixes
which neurons are counted, not how hard they fire, so ``tuned_activation`` at
sigma = 0 is a measurement rather than a tautology, and ``nontuned_activation``
is free to rise with jitter without any compensating rise in the signal pool.

The AUC columns are retained as a secondary, threshold-free summary of the
signal-minus-crosstalk contrast, but they are no longer the result.

One cost note. Measuring the crosstalk channel means simulating the whole
population, not just the tuned pool, so this script is now roughly ``n_neurons
/ n_tuned`` times more expensive than the version that reported the AUC alone.
At the headline setting that is minutes; the ``theta = 4.5`` arm in
``run_all.sh`` runs 50,000 neurons and takes tens of minutes on a workstation.
The old metric was cheap precisely because it ignored the other 49,900 neurons.

-------------------------------------------------------------------------------
ORIGINAL NOTES (kept)
-------------------------------------------------------------------------------

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

This experiment may not favour hard gating. A quantified
selectivity-versus-activation trade-off is a publishable result either way, and
is a more honest claim than the current framing.

Two choices in the design are open to challenge, and both are exposed as flags
rather than buried, so that the reported result can be checked against its
alternatives instead of being taken on trust:

``--readout``
    the decision variable used for the secondary AUC. ``spikes`` (the default,
    and what the manuscript reports) uses spikes per neuron. ``peak`` uses the
    peak somatic membrane potential instead. The signed AUC of the *soft* model
    is sensitive to this choice, because with no threshold and no dendritic
    reset a soft branch holds the soma at its refractory ceiling for as long as
    any drive is present -- so a spike count measures how *long* drive lasted,
    and a dispersed volley lasts longer than a coincident one.
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

# Guards the ratio and the selectivity index when the crosstalk pool is silent.
EPS = 1e-9

FIELDNAMES = [
    "gating", "readout", "tuned_by", "sigma", "dispersion", "theta", "seed",
    "n_tuned", "n_nontuned",
    # The four primary quantities; see the module docstring for definitions.
    "tuned_activation_target", "tuned_activation_distractor",
    "nontuned_activation_target", "nontuned_activation_distractor",
    "snr_target", "snr_distractor",
    "si_target", "si_distractor",
    # Secondary threshold-free summary of the signal-minus-crosstalk contrast.
    "auc", "auc_unsigned", "d_prime",
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
    if target.size < 2 or distractor.size < 2:
        return float("nan")
    pooled = np.sqrt(0.5 * (target.var(ddof=1) + distractor.var(ddof=1)))
    if pooled == 0:
        return float("nan")
    return float((target.mean() - distractor.mean()) / pooled)


def _pool_mean(values: np.ndarray) -> float:
    """Mean over a pool, or NaN if the pool is empty."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else float("nan")


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
    # each model pick its own. Either way the partition is computed on the clean
    # stimulus alone and reused at every sigma, so it cannot track the jitter.
    tuned_shared = select("hard") if tuned_by == "hard" else None

    # The two gating arms must see the *same* jitter realisations, or the
    # contrast between them carries a resampling difference on top of the
    # gating difference. One generator shared across the loop does not give
    # that: the second arm consumes the draws that follow the first arm's. The
    # supplement's Methods already describe the paired design ("the generator
    # being re-seeded identically across the two gating rules"); snapshotting
    # the state here and restoring it per arm is what makes the code do it.
    # The first arm's draws are unchanged, so only the soft arm moves, and
    # only by the resampling it should not have had.
    trial_state = rng.bit_generator.state

    rows = []
    for gating in ("hard", "soft"):
        rng.bit_generator.state = trial_state
        tuned = tuned_shared if tuned_shared is not None else select(gating)
        n_tuned = int(tuned.sum())
        n_nontuned = int((~tuned).sum())
        if n_tuned == 0:
            continue

        def response(x):
            """Per-neuron spike count and peak potential for the whole population.

            ``simulate_population`` returns one value per neuron, and neurons in
            this model are independent -- no recurrence, no shared state -- so
            simulating the full population once and splitting it by the tuned
            mask gives exactly what simulating the two pools separately would,
            at half the cost. Verified against the split simulation for both
            gating modes.
            """
            spikes, peak = simulate_population(x, delays, weights, params,
                                               gating=gating, return_peak=True)
            return (np.asarray(spikes, dtype=float),
                    np.asarray(peak, dtype=float))

        tuned_target, tuned_distractor = [], []
        nontuned_target, nontuned_distractor = [], []
        readout_target, readout_distractor = [], []

        for _ in range(job["n_trials"]):
            noisy_target = stim.with_jitter(job["sigma"], rng)
            noisy_distractor = distractor_clean.with_jitter(job["sigma"], rng)

            sp_tgt, pk_tgt = response(noisy_target.both)
            sp_dis, pk_dis = response(noisy_distractor.both)

            # Activation is always a spike count, whatever the decision
            # variable is: it is a cost in emitted spikes, and a peak potential
            # is not that.
            tuned_target.append(_pool_mean(sp_tgt[tuned]))
            tuned_distractor.append(_pool_mean(sp_dis[tuned]))
            nontuned_target.append(_pool_mean(sp_tgt[~tuned]))
            nontuned_distractor.append(_pool_mean(sp_dis[~tuned]))

            # Secondary signal-minus-crosstalk readout for the AUC. Both pools
            # enter as per-neuron means rather than sums: the non-tuned pool
            # outnumbers the tuned one by more than an order of magnitude, and
            # a difference of sums would be a measure of the crosstalk channel
            # with the signal channel as rounding error.
            score_tgt, score_dis = (pk_tgt, pk_dis) if readout == "peak" else (sp_tgt, sp_dis)
            readout_target.append(
                _pool_mean(score_tgt[tuned]) - _pool_mean(score_tgt[~tuned]))
            readout_distractor.append(
                _pool_mean(score_dis[tuned]) - _pool_mean(score_dis[~tuned]))

        tuned_target = float(np.mean(tuned_target))
        tuned_distractor = float(np.mean(tuned_distractor))
        nontuned_target = float(np.mean(nontuned_target))
        nontuned_distractor = float(np.mean(nontuned_distractor))

        snr_target = tuned_target / (nontuned_target + EPS)
        snr_distractor = tuned_distractor / (nontuned_distractor + EPS)
        si_target = ((tuned_target - nontuned_target)
                     / (tuned_target + nontuned_target + EPS))
        si_distractor = ((tuned_distractor - nontuned_distractor)
                         / (tuned_distractor + nontuned_distractor + EPS))

        readout_target = np.array(readout_target, dtype=float)
        readout_distractor = np.array(readout_distractor, dtype=float)
        auc = auc_from_scores(readout_target, readout_distractor)

        rows.append({
            "gating": gating,
            "readout": readout,
            "tuned_by": tuned_by,
            "sigma": job["sigma"],
            "dispersion": job["dispersion"],
            "theta": job["theta"],
            "seed": job["seed"],
            "n_tuned": n_tuned,
            "n_nontuned": n_nontuned,
            # Spikes per trial per neuron in each pool. The target column of
            # the tuned pool is the activation cost the manuscript quotes, in
            # units directly comparable between the two models.
            "tuned_activation_target": tuned_target,
            "tuned_activation_distractor": tuned_distractor,
            "nontuned_activation_target": nontuned_target,
            "nontuned_activation_distractor": nontuned_distractor,
            "snr_target": snr_target,
            "snr_distractor": snr_distractor,
            "si_target": si_target,
            "si_distractor": si_distractor,
            "auc": auc,
            # An AUC well below 0.5 does not mean the model is uninformative:
            # it means the readout separates the two classes with reversed
            # polarity, which a downstream decoder recovers by inverting its
            # decision rule. Reporting the signed value alone would understate
            # a model that is in fact discriminative, so the polarity-free
            # magnitude is carried alongside it.
            "auc_unsigned": max(auc, 1.0 - auc) if not np.isnan(auc) else auc,
            "d_prime": d_prime_from_scores(readout_target, readout_distractor),
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
                        help="decision variable for the secondary AUC: spikes "
                             "per neuron, or peak somatic membrane potential")
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

    def pick(gating, sigma, key):
        vals = [r[key] for r in rows
                if r["gating"] == gating and r["sigma"] == sigma
                and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    # The four primary quantities, per gating mode, per sigma. Columns are
    # spikes per neuron per trial in the tuned and non-tuned pools on the
    # target, their ratio, and the selectivity index on each stimulus.
    for gating in ("hard", "soft"):
        print(f"\n  {gating.upper()} GATING "
              f"(tuned-by={args.tuned_by}, readout={args.readout})")
        print(f"  {'sigma':>6} {'tuned_tgt':>10} {'nontun_tgt':>11} "
              f"{'tuned_dis':>10} {'nontun_dis':>11} "
              f"{'SNR_tgt':>8} {'SI_tgt':>7} {'SI_dis':>7} {'|AUC|':>6}")
        for sigma in args.sigmas:
            print(f"  {sigma:>6.1f} "
                  f"{pick(gating, sigma, 'tuned_activation_target'):>10.3f} "
                  f"{pick(gating, sigma, 'nontuned_activation_target'):>11.3f} "
                  f"{pick(gating, sigma, 'tuned_activation_distractor'):>10.3f} "
                  f"{pick(gating, sigma, 'nontuned_activation_distractor'):>11.3f} "
                  f"{pick(gating, sigma, 'snr_target'):>8.2f} "
                  f"{pick(gating, sigma, 'si_target'):>7.3f} "
                  f"{pick(gating, sigma, 'si_distractor'):>7.3f} "
                  f"{pick(gating, sigma, 'auc_unsigned'):>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

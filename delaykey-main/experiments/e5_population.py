"""E5: population coding and the Q-sparsity dependence of false positives.

This script is manuscript **Experiment 4**; see the numbering table in
``delaykey/README.md``. Like ``e4_noise.py`` it is a re-implementation: no
generating code for the originally reported version survived, so its figure was
drawn from numbers quoted out of the manuscript and pixels digitised from a
legacy raster. Everything here is simulated against ``delaykey.core``.

The design is the manuscript's. A pool of integrator neurons is drawn with
random delays; the pool is screened for the neurons whose delays happen to match
the clean A->B target; then Poisson noise is superimposed on the Q signal at a
range of densities and two rates are followed:

*detection rate*      the fraction of matched neurons still firing for the
                      target under noise.
*false-positive rate* the fraction of unmatched neurons firing under noise with
                      no target present.

What is new is 20 seeds with bootstrap intervals instead of one draw of six
neurons, a sweep over the calcium threshold, and an explicit analytic baseline
for the match rate.

**The match-rate baseline.** The manuscript compares its observed rate against
"the naive estimate of 25% (a random delay from [0,20] ms matching a 5 ms spike
interval)". That figure is not reproducible under any reading, and the correct
one is much smaller. Delays are drawn independently and uniformly on the integer
range [0, D]. The two arrivals coincide exactly when

    t_first + d_A == t_second + d_B,   i.e.   d_A - d_B == t_second - t_first,

and for a target separation s in [0, D] the number of ordered pairs satisfying it
is D + 1 - s out of (D + 1)^2, giving :func:`exact_match_probability`. For
D = 20 and s = 5 that is 16/441 = 3.63%, which is *not* distinguishable from the
observed rate -- so the manuscript's explanation of the gap (a multi-input
threshold requirement) is explaining a gap that is not there. On a two-input
branch there is in any case no multi-input requirement beyond the two inputs.

The branch does tolerate near-misses, because the dendritic membrane has a time
constant: an arrival one step early still leaves ``exp(-1/tau_den)`` of its
depolarisation behind. The measured rate therefore sits at or above the exact
figure, and how far above is a property of ``theta`` and ``tau_den``, not of the
delay statistics. Both numbers are written to the CSV so the comparison is made
on the page rather than asserted.

Usage::

    python experiments/e5_population.py --quick
    python experiments/e5_population.py --seeds 20
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
    "density", "theta", "dispersion", "seed", "n_neurons", "n_trials",
    "t_first", "t_second", "n_time",
    "n_matched", "match_rate", "exact_match_probability", "tolerance_steps",
    "detection_rate", "false_positive_rate",
]


def exact_match_probability(dispersion: int, separation: int) -> float:
    """P(d_A - d_B == separation) for i.i.d. integer delays uniform on [0, D].

    This is the naive pairwise baseline for the match rate: the chance that a
    randomly drawn pair of delays compensates the target's spike-time difference
    exactly. Returns 0 when the separation exceeds the delay range, which is the
    statement that a delay pool cannot compensate a gap it cannot span.
    """
    if separation < 0 or separation > dispersion:
        return 0.0
    return (dispersion + 1 - separation) / float((dispersion + 1) ** 2)


def render(rng, density: float, n_time: int, t_first: int, t_second: int,
           target: bool = True) -> np.ndarray:
    """One trial: the A->B pair (optionally) plus Poisson noise on both features.

    ``density`` is in spikes per time step per feature neuron, the unit the
    manuscript states.
    """
    x = np.zeros((n_time, 2))
    if target:
        x[t_first, 0] = 1.0
        x[t_second, 1] = 1.0
    if density > 0:
        x = np.maximum(x, (rng.random((n_time, 2)) < density).astype(float))
    return x


def run_cell(job: dict) -> list[dict]:
    """Screen one pool, then sweep the noise density over it."""
    rng = np.random.default_rng([job["seed"], int(job["theta"] * 10)])
    params = NeuronParams(
        theta=job["theta"], alpha=job["alpha"], tau_mem=job["tau_mem"],
        tau_den=job["tau_den"], v_threshold=1.0, v_contrib=job["v_contrib"],
        plateau_steps=job["plateau_steps"], refractory=job["refractory"],
    )

    n = job["n_neurons"]
    shape = (n, 1, 2)
    delays = rng.integers(0, job["dispersion"] + 1, size=shape).astype(np.int64)
    weights = np.ones(shape)

    clean = render(rng, 0.0, job["n_time"], job["t_first"], job["t_second"])
    matched = simulate_population(clean, delays, weights, params) > 0
    n_matched = int(matched.sum())

    # How wide a mismatch the branch actually tolerates, measured rather than
    # assumed: the largest |d_A - d_B - separation| among the matched neurons.
    separation = job["t_second"] - job["t_first"]
    offsets = delays[:, 0, 0] - delays[:, 0, 1] - separation
    tolerance = int(np.abs(offsets[matched]).max()) if n_matched else 0

    exact = exact_match_probability(job["dispersion"], separation)

    rows = []
    for density in job["densities"]:
        detect, false_pos = [], []
        for _ in range(job["n_trials"]):
            if n_matched:
                x = render(rng, density, job["n_time"], job["t_first"],
                           job["t_second"], target=True)
                fired = simulate_population(x, delays[matched],
                                            weights[matched], params) > 0
                detect.append(float(fired.mean()))
            if n_matched < n:
                x0 = render(rng, density, job["n_time"], job["t_first"],
                            job["t_second"], target=False)
                fired0 = simulate_population(x0, delays[~matched],
                                             weights[~matched], params) > 0
                false_pos.append(float(fired0.mean()))
        rows.append({
            "density": float(density),
            "theta": job["theta"],
            "dispersion": job["dispersion"],
            "seed": job["seed"],
            "n_neurons": n,
            "n_trials": job["n_trials"],
            "t_first": job["t_first"],
            "t_second": job["t_second"],
            "n_time": job["n_time"],
            "n_matched": n_matched,
            "match_rate": n_matched / n,
            "exact_match_probability": exact,
            "tolerance_steps": tolerance,
            "detection_rate": float(np.mean(detect)) if detect else "",
            "false_positive_rate": float(np.mean(false_pos)) if false_pos else "",
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("results/e5_population.csv"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--neurons", type=int, default=200)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--time-steps", type=int, default=30)
    parser.add_argument("--t-first", type=int, default=5)
    parser.add_argument("--t-second", type=int, default=10)
    parser.add_argument("--dispersions", type=int, nargs="+", default=[20])

    parser.add_argument("--thetas", type=float, nargs="+",
                        default=[1.0, 1.5, 1.8])
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--tau-mem", type=float, default=5.0)
    parser.add_argument("--tau-den", type=float, default=2.0)
    parser.add_argument("--v-contrib", type=float, default=0.10)
    parser.add_argument("--plateau-steps", type=int, default=10)
    parser.add_argument("--refractory", type=int, default=2)

    parser.add_argument("--densities", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50])
    args = parser.parse_args(argv)

    if args.quick:
        # Seeds and trials only. The threshold axis is left at full width
        # because make_fig_e5 plots match rate against it and names all three
        # thresholds in its caption; reducing it makes the figure script fail
        # rather than run on reduced data.
        args.seeds, args.n_trials = 3, 10

    fixed = dict(
        n_neurons=args.neurons, n_trials=args.n_trials,
        n_time=args.time_steps, t_first=args.t_first, t_second=args.t_second,
        alpha=args.alpha, tau_mem=args.tau_mem, tau_den=args.tau_den,
        v_contrib=args.v_contrib, plateau_steps=args.plateau_steps,
        refractory=args.refractory, densities=args.densities,
    )
    jobs = [dict(fixed, dispersion=d, theta=th, seed=s)
            for d, th, s in itertools.product(
                args.dispersions, args.thetas,
                range(args.seed_base, args.seed_base + args.seeds))]

    n_workers = args.jobs or max(1, min(mp.cpu_count() - 1, len(jobs)))
    print(f"E5: {len(jobs)} pools x {len(args.densities)} densities x "
          f"{args.n_trials} trials, {args.neurons} neurons, {n_workers} workers")

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

    for disp in args.dispersions:
        for theta in args.thetas:
            sub = [r for r in rows if r["dispersion"] == disp
                   and r["theta"] == theta]
            if not sub:
                continue
            per_seed = {r["seed"]: r for r in sub}
            match = np.mean([r["match_rate"] for r in per_seed.values()])
            tol = max(r["tolerance_steps"] for r in per_seed.values())
            exact = sub[0]["exact_match_probability"]
            print(f"\n  dispersion {disp}, theta {theta}: match rate "
                  f"{100 * match:.2f}%  (exact-compensation baseline "
                  f"{100 * exact:.2f}%, tolerated mismatch up to {tol} step(s))")
            print(f"    {'density':>8} {'detection':>10} {'false pos':>10}")
            for d in args.densities:
                cells = [r for r in sub if abs(r["density"] - d) < 1e-12]
                det = [r["detection_rate"] for r in cells if r["detection_rate"] != ""]
                fp = [r["false_positive_rate"] for r in cells
                      if r["false_positive_rate"] != ""]
                print(f"    {d:>8.2f} "
                      f"{100 * np.mean(det) if det else float('nan'):>9.1f}% "
                      f"{100 * np.mean(fp) if fp else float('nan'):>9.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

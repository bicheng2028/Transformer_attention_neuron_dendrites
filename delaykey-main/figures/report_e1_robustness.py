"""Report how the dispersion sweep fares against three of the model's simplifications.

Reads the dispersion sweep of Fig 2 re-run on a reduced grid -- theta 2.5 and
4.5, a 15 ms interval, dispersions 0-50 ms, 20 seeds:

    results/e1_robustness_reported.csv         the reported model, 1 ms step
    results/e1_robustness_step_r{4,10,20}.csv  steps of 0.25, 0.1 and 0.05 ms
    results/e1_robustness_plateau.csv          a 100 ms plateau
    results/e1_robustness_background_*Hz.csv   Poisson background on every
                                               feature, eight rates, 100 seeds

and the theory of the models it covers:

    results/e1_theory.csv           the reported model (Fig 2's theory lines)
    results/e1_theory_step.csv      a 0.1 ms step
    results/e1_theory_plateau.csv   a 100 ms plateau

It prints, in order:

1. The background calibration. For each gate and rate, the share of the
   sequence detectors found without background that fire for one event alone
   once it is added, pooled over 100 seeds. Each gate's background condition is
   the highest rate that keeps that share below 5%, read on seeds 0-19 like
   every other condition.
2. Sequence detectors summed over the grid at every step, the evidence that
   0.1 ms is converged.
3. For each threshold, the class fractions at every dispersion: seed means, with
   the as-reported run's 95% bootstrap interval and a mark on any condition
   outside it, beside the theory.
4. The crossover dispersion Delta*, read off the seed-pooled curves with a seed
   bootstrap exactly as ``make_fig_e1.py`` reads it, beside the reported run
   and the theory.
5. The theory's slopes of Delta* and Delta_opt against the interval, fitted over
   the four simulated intervals as Fig 2c fits them.

S2 Text quotes this output.

Usage::

    python figures/report_e1_robustness.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from delaykey.stats import bootstrap_mean  # noqa: E402
from make_fig_e1 import (crossover_dispersion, load, load_theory,  # noqa: E402
                         peak_dispersion)

RESULTS = Path("results")
INTERVAL, THETAS, SEEDS = 15, (2.5, 4.5), 20
CLASSES = ("sequence", "event", "dual", "silent")
STEPS = (4, 10, 20)  # steps per ms besides the reported 1, as run_all.sh runs them
COMPARED_STEP = 10
RATES = ("0.25", "0.5", "0.75", "1", "2", "3", "4", "5")  # Hz, as in run_all.sh
FALSE_LIMIT = 0.05
SIM_INTERVALS = (5, 10, 15, 20)


def read(name: str) -> tuple[list[dict], list[dict]]:
    """Rows as ``make_fig_e1.load`` parses them, and the raw rows beside them."""
    path = RESULTS / name
    with path.open() as handle:
        return load(path), list(csv.DictReader(handle))


def seed_matrix(rows, theta, key, dispersions, seeds):
    cell = {(r["seed"], r["dispersion"]): r[key] for r in rows
            if r["theta"] == theta and r["interval"] == INTERVAL}
    return np.array([[cell[(s, d)] for d in dispersions] for s in seeds])


def crossover_ci(rows, theta, rng, n_boot=4000):
    """Delta* from the seed-pooled curves, with a seed bootstrap, as in Fig 2c."""
    dispersions = sorted({r["dispersion"] for r in rows if r["theta"] == theta})
    seeds = sorted({r["seed"] for r in rows if r["theta"] == theta})
    seq = seed_matrix(rows, theta, "sequence", dispersions, seeds)
    ev = seed_matrix(rows, theta, "event", dispersions, seeds)

    def at(idx):
        return crossover_dispersion(dispersions, seq[idx].mean(axis=0),
                                    ev[idx].mean(axis=0))

    point = at(np.arange(len(seeds)))
    boot = np.array([at(rng.integers(0, len(seeds), len(seeds)))
                     for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, lo, hi, int(np.isnan(boot).sum())


def calibrate() -> dict:
    """Each gate's background rate, and the calibration behind it."""
    print("Background: share of the sequence detectors found without it that "
          "fire for one event alone, 100 seeds")
    print(f"  {'rate':>6}" + "".join(f"{'theta ' + format(t, 'g'):>18}"
                                     for t in THETAS))
    chosen, table, passed = {}, {}, set(THETAS)
    for rate in RATES:
        _, raw = read(f"e1_robustness_background_{rate}Hz.csv")
        line = f"  {rate + ' Hz':>6}"
        for theta in THETAS:
            sub = [r for r in raw if float(r["theta"]) == theta]
            n_quiet = sum(int(r["quiet_sequence"]) for r in sub)
            n_false = sum(int(r["quiet_sequence_single"]) for r in sub)
            share = n_false / n_quiet
            table[(theta, rate)] = (n_false, n_quiet, share)
            line += f"{n_false:>7}/{n_quiet:<5} {100 * share:4.1f}%"
            # The highest rate below which every rate passes, so that noise in
            # one estimate cannot pick a rate above one that already failed.
            if theta in passed and share < FALSE_LIMIT:
                chosen[theta] = rate
            else:
                passed.discard(theta)
        print(line)
    for theta in THETAS:
        rate = chosen[theta]
        _, raw = read(f"e1_robustness_background_{rate}Hz.csv")
        sub = [r for r in raw if float(r["theta"]) == theta]
        alone = sum(int(r["background_only"]) for r in sub)
        total = sum(int(r["n_neurons"]) for r in sub)
        n_false, n_quiet, share = table[(theta, rate)]
        print(f"  theta {theta:g}: run at {rate} Hz ({100 * share:.1f}%); the "
              f"background alone drives {alone} of {total} neurons "
              f"({100 * alone / total:.4f}%)")
    return chosen


def main() -> int:
    rng = np.random.default_rng(0)
    chosen = calibrate()

    # The background condition: each gate at its own rate, seeds 0-19.
    background = []
    for theta, rate in chosen.items():
        rows, _ = read(f"e1_robustness_background_{rate}Hz.csv")
        background += [r for r in rows if r["theta"] == theta and r["seed"] < SEEDS]
    data = {"reported": read("e1_robustness_reported.csv")[0],
            "plateau": read("e1_robustness_plateau.csv")[0],
            "background": background}
    for r in STEPS:
        data[f"step_r{r}"] = read(f"e1_robustness_step_r{r}.csv")[0]
    shown = (("step_r10", "0.1 ms"), ("plateau", "100 ms pl."),
             ("background", "backgr."))
    theory = {"reported": load_theory(RESULTS / "e1_theory.csv"),
              "step": load_theory(RESULTS / "e1_theory_step.csv"),
              "plateau": load_theory(RESULTS / "e1_theory_plateau.csv")}

    # The as-reported condition is the reported model on a coarser grid, so
    # wherever the two grids meet its rows must be the reported run's own,
    # column for column.
    reported, reported_raw = read("e1_dispersion.csv")
    reported = [r for r in reported
                if r["theta"] in THETAS and r["interval"] == INTERVAL]

    def cell(raw):
        return raw["theta"], raw["interval"], raw["dispersion"], raw["seed"]

    ours = {cell(raw): raw for raw in read("e1_robustness_reported.csv")[1]}
    shared = [raw for raw in reported_raw if cell(raw) in ours]
    same = all(ours[cell(raw)] == raw for raw in shared)
    print(f"\nAs-reported rows against results/e1_dispersion.csv: {len(shared)} "
          f"shared, {'all identical' if same else 'DIFFERENT'}.")

    print("\nSequence detectors summed over the grid and the 20 seeds, by step:")
    for theta in THETAS:
        counts = [sum(r["n_sequence"] for r in data[key] if r["theta"] == theta)
                  for key in ["reported"] + [f"step_r{r}" for r in STEPS]]
        steps = ["1"] + [f"{1 / r:g}" for r in STEPS]
        print(f"  theta {theta:g}: " + ", ".join(
            f"{n} at {s} ms" for n, s in zip(counts, steps)))

    def at_interval(key, theta, cls, d):
        curve = theory[key][(theta, INTERVAL)]
        return float(np.interp(d, curve["dispersion"], curve[cls]))

    for theta in THETAS:
        print(f"\ntheta = {theta:g}, interval {INTERVAL} ms; background "
              f"{chosen[theta]} Hz")
        print("  Class fractions, % of the population, seed means; * lies outside "
              "the as-reported 95% bootstrap interval.")
        dispersions = sorted({r["dispersion"] for r in data["reported"]
                              if r["theta"] == theta})
        heads = "".join(f"{h:>11}" for _, h in shown)
        moved = {key: [] for key, _ in shown}
        for cls in CLASSES:
            print(f"\n  {cls:<8}  {'as reported [95% CI]':>27}{heads}   |"
                  f" theory: {'1 ms':>8}{'0.1 ms':>9}{'100 ms pl.':>11}")
            for d in dispersions:
                base = bootstrap_mean([r[cls] for r in data["reported"]
                                       if r["theta"] == theta
                                       and r["dispersion"] == d], rng)
                line = (f"  {d:>6}  {100 * base.mean:9.3f} "
                        f"[{100 * base.lo:7.3f}, {100 * base.hi:7.3f}]")
                for key, _ in shown:
                    mean = float(np.mean([r[cls] for r in data[key]
                                          if r["theta"] == theta
                                          and r["dispersion"] == d]))
                    out = not base.lo <= mean <= base.hi
                    if out:
                        moved[key].append((cls, d, mean - base.mean))
                    line += f"{100 * mean:10.3f}{'*' if out else ' '}"
                line += "   |        "
                line += "".join(f"{100 * at_interval(k, theta, cls, d):{w}.3f}"
                                for k, w in (("reported", 8), ("step", 9),
                                             ("plateau", 11)))
                print(line)

        print(f"\n  Outside the as-reported interval, of "
              f"{len(CLASSES) * len(dispersions)} class-dispersion cells:")
        for key, head in shown:
            worst = max(moved[key], key=lambda m: abs(m[2]), default=None)
            note = (f"; largest shift {100 * worst[2]:+.3f} points "
                    f"({worst[0]}, Delta = {worst[1]} ms)") if worst else ""
            print(f"    {head:<12} {len(moved[key]):>2}{note}")

        print("\n  Crossover Delta* (ms), seed-pooled curves, 95% bootstrap over "
              "seeds:")
        point, lo, hi, _ = crossover_ci(reported, theta, rng)
        print(f"    {'reported run, 2 ms grid':<29} {point:6.2f} "
              f"[{lo:5.2f}, {hi:5.2f}]")
        for key, head in (("reported", "as reported"), ("step_r4", "0.25 ms"),
                          ("step_r10", "0.1 ms"), ("step_r20", "0.05 ms"),
                          ("plateau", "100 ms plateau"),
                          ("background", "background")):
            point, lo, hi, n_nan = crossover_ci(data[key], theta, rng)
            gaps = f"  ({n_nan} resamples with no crossing)" if n_nan else ""
            print(f"    {head + ', reduced grid':<29} {point:6.2f} "
                  f"[{lo:5.2f}, {hi:5.2f}]{gaps}")
        for key, head in (("reported", "1 ms"), ("step", "0.1 ms"),
                          ("plateau", "100 ms plateau")):
            curve = theory[key][(theta, INTERVAL)]
            print(f"    {'theory, ' + head:<29} "
                  f"{crossover_dispersion(curve['dispersion'], curve['sequence'], curve['event']):6.2f}")

    print("\nTheory slopes against the interval, fitted over "
          f"{', '.join(map(str, SIM_INTERVALS))} ms as in Fig 2c:")
    for theta in THETAS:
        for key, head in (("reported", "1 ms"), ("step", "0.1 ms"),
                          ("plateau", "100 ms plateau")):
            cross, top = [], []
            for iv in SIM_INTERVALS:
                curve = theory[key][(theta, iv)]
                cross.append(crossover_dispersion(curve["dispersion"],
                                                  curve["sequence"], curve["event"]))
                top.append(peak_dispersion(curve["dispersion"], curve["sequence"]))
            print(f"  theta {theta:g}, {head:<15} Delta* "
                  f"{np.polyfit(SIM_INTERVALS, cross, 1)[0]:.3f}, Delta_opt "
                  f"{np.polyfit(SIM_INTERVALS, top, 1)[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

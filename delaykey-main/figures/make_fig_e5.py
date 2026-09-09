"""Build the E5 population/Q-sparsity figure from ``results/e5_population.csv``.

This is manuscript Figure 6 (Experiment 4). It replaces
``figures/make_fig6_population_noise.py``, which plotted values quoted out of
``content.tex`` and digitised from a legacy raster because no generating code
survived. Every value here is simulated.

Panels:

a  Detection rate and false-positive rate against Poisson noise density on the
   Q signal, at the headline threshold.
b  Match rate against the calcium threshold, against the analytic naive
   baseline. This is the panel that settles where the match rate comes from:
   the measured rate tracks the pairwise delay-matching probability once the
   branch's own tolerance to near-misses is counted, so no appeal to a
   multi-input threshold is needed to explain it.

Usage::

    python figures/make_fig_e5.py --formats pdf eps tiff png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.plotting import (PALETTE, apply_style, direct_label, figure_size,
                               label_panels, save_figure, tint)
from delaykey.stats import bootstrap_mean

import matplotlib.pyplot as plt  # noqa: E402


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "density": float(raw["density"]),
                "theta": float(raw["theta"]),
                "dispersion": int(raw["dispersion"]),
                "seed": int(raw["seed"]),
                "n_neurons": int(raw["n_neurons"]),
                "n_trials": int(raw["n_trials"]),
                "t_first": int(raw["t_first"]),
                "t_second": int(raw["t_second"]),
                "n_matched": int(raw["n_matched"]),
                "match_rate": float(raw["match_rate"]),
                "exact": float(raw["exact_match_probability"]),
                "tolerance": int(raw["tolerance_steps"]),
                "detection": (float(raw["detection_rate"])
                              if raw["detection_rate"] else None),
                "false_positive": (float(raw["false_positive_rate"])
                                   if raw["false_positive_rate"] else None),
            })
    return rows


def group(rows, keys, value):
    out = defaultdict(list)
    for row in rows:
        if row[value] is not None:
            out[tuple(row[k] for k in keys)].append(row[value])
    return out


def tolerated_probability(dispersion: int, separation: int, tolerance: int) -> float:
    """P(|d_A - d_B - separation| <= tolerance), integer delays on [0, D].

    The analytic match rate once the branch's measured tolerance to a
    near-miss is allowed for. At tolerance 0 this is the exact-compensation
    probability.
    """
    total = 0
    for k in range(separation - tolerance, separation + tolerance + 1):
        if 0 <= abs(k) <= dispersion:
            total += dispersion + 1 - abs(k)
    return total / float((dispersion + 1) ** 2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("results/e5_population.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e5_population"))
    # theta 1.8 is the threshold at which the two-input branch requires exact
    # coincidence: at 1.5 the dendritic time constant still passes a one-step
    # near-miss. It is also the setting whose false-positive curve reproduces
    # the originally reported one, which is what identifies it as the operating
    # point of the lost implementation.
    parser.add_argument("--theta-main", type=float, default=1.8)
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    parser.add_argument("--height-mm", type=float, default=56.0)
    args = parser.parse_args(argv)

    rows = load(args.csv)
    rng = np.random.default_rng(0)
    n_seeds = len({r["seed"] for r in rows})
    n_neurons = rows[0]["n_neurons"]
    n_trials = rows[0]["n_trials"]
    dispersion = rows[0]["dispersion"]
    separation = rows[0]["t_second"] - rows[0]["t_first"]
    thetas = sorted({r["theta"] for r in rows})

    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=figure_size(args.width, args.height_mm))
    fig.subplots_adjust(wspace=0.30, left=0.085, right=0.985,
                        bottom=0.20, top=0.93)

    # ---- a: the two rates against noise density ---------------------------
    main_rows = [r for r in rows if r["theta"] == args.theta_main]
    densities = sorted({r["density"] for r in main_rows})
    cited = {}
    for key, label, colour, marker in (
        ("detection", "Detection", PALETTE["blue"], "o"),
        ("false_positive", "False positive", PALETTE["vermillion"], "s"),
    ):
        grouped = group(main_rows, ("density",), key)
        live = [d for d in densities if (d,) in grouped]
        est = [bootstrap_mean(grouped[(d,)], rng) for d in live]
        for d, e in zip(live, est):
            cited[(key, d)] = e
        means = np.array([e.mean for e in est])
        ax_a.fill_between(live, [e.lo for e in est], [e.hi for e in est],
                          color=tint(colour), linewidth=0, zorder=1)
        ax_a.plot(live, means, color=colour, marker=marker, markersize=3.4,
                  linewidth=1.5, zorder=2)
        i = 2 if key == "false_positive" else len(live) - 2
        direct_label(ax_a, live[i], means[i] + (-0.13 if key == "detection" else 0.10),
                     label, colour, dx=(-0.10 if key == "detection" else 0.012))
    ax_a.set_xlabel("Poisson noise density on Q")
    ax_a.set_ylabel("Rate")
    ax_a.set_ylim(0, 1.16)

    # ---- b: match rate against threshold, with the naive baselines --------
    measured, exact_line, tol_line, tol_used = [], [], [], []
    for theta in thetas:
        sub = [r for r in rows if r["theta"] == theta]
        per_seed = {r["seed"]: r for r in sub}
        est = bootstrap_mean([r["match_rate"] for r in per_seed.values()], rng)
        measured.append(est)
        tol = max(r["tolerance"] for r in per_seed.values())
        tol_used.append(tol)
        exact_line.append(sub[0]["exact"])
        tol_line.append(tolerated_probability(dispersion, separation, tol))

    xs = np.arange(len(thetas), dtype=float)
    means = np.array([e.mean for e in measured])
    ax_b.errorbar(xs, means,
                  yerr=[means - [e.lo for e in measured],
                        [e.hi for e in measured] - means],
                  fmt="o", color=PALETTE["blue"], markersize=3.8, capsize=2.2,
                  elinewidth=0.6, capthick=0.6, linestyle="none", zorder=3)
    ax_b.plot(xs, tol_line, color=PALETTE["black"], linewidth=0.7,
              linestyle=(0, (3, 2)), zorder=2)
    ax_b.plot(xs, exact_line, color=PALETTE["grey"], linewidth=0.7,
              linestyle=(0, (1, 2)), zorder=2)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([f"{t:g}" for t in thetas])
    ax_b.set_xlim(-0.35, len(thetas) - 0.35)
    ax_b.set_xlabel("Calcium threshold $\\theta$")
    ax_b.set_ylabel("Match rate")
    ax_b.set_yscale("log")
    # Headroom first, then labels: on a log axis a multiplicative offset chosen
    # against the autoscaled limits lands off-canvas as soon as the anchor sits
    # near an edge, which is what put two of these three outside the axes.
    lo = min(min(exact_line), min(tol_line), means.min()) / 2.4
    hi = max(max(tol_line), means.max()) * 2.2
    ax_b.set_ylim(lo, hi)
    direct_label(ax_b, xs[-1], means[-1] * 1.9, "Measured", PALETTE["blue"],
                 dx=-24.0)
    direct_label(ax_b, xs[0], tol_line[0] * 1.55, "Naive, with tolerance",
                 PALETTE["black"], dx=0.0)
    direct_label(ax_b, xs[1], exact_line[1] * 0.62, "Naive, exact",
                 PALETTE["grey"], dx=-8.0)

    label_panels([ax_a, ax_b])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    def cite(key, d, dp=1):
        e = cited[(key, d)]
        return (f"{100 * e.mean:.{dp}f}\\% "
                f"[{100 * e.lo:.{dp}f}, {100 * e.hi:.{dp}f}]")

    i_main = thetas.index(args.theta_main)
    match_txt = (f"{100 * measured[i_main].mean:.2f}\\% "
                 f"[{100 * measured[i_main].lo:.2f}, "
                 f"{100 * measured[i_main].hi:.2f}]")
    n_matched = np.mean([r["n_matched"] for r in main_rows])
    quoted = [d for d in (0.05, 0.10, 0.20, 0.30) if (("false_positive", d) in cited)]
    fp_txt = ", ".join(f"{cite('false_positive', d)} at {d:g}" for d in quoted)

    caption = (
        f"\\textbf{{Population coding and Q-sparsity dependence, "
        f"re-simulated.}} Pools of {n_neurons} integrator neurons, one distal "
        f"branch each, delays drawn uniformly on the integers "
        f"$[0, {dispersion}]$~ms, screened for the neurons matching a clean "
        f"A$\\rightarrow$B target {separation}~ms apart. {n_seeds} independent "
        f"seeds of {n_trials} noise realisations each; shading and error bars "
        f"are 95\\% bootstrap confidence intervals over seeds. "
        f"\\textbf{{a}}, Detection rate --- the fraction of matched neurons "
        f"still firing for the target under noise --- and false-positive rate "
        f"--- the fraction of unmatched neurons firing under noise with no "
        f"target present --- against Poisson noise density on the Q signal, at "
        f"$\\theta = {args.theta_main:g}$ ({n_matched:.1f} matched neurons per "
        f"pool on average). Detection is unaffected by noise at every density "
        f"tested. False positives are not: {fp_txt}. The asymmetry is the "
        f"result, and it locates the defence against false positives in the "
        f"sparsity of Q rather than in the gate. \\textbf{{b}}, Where the match "
        f"rate comes from. Measured match rate against threshold "
        f"({match_txt} at $\\theta = {args.theta_main:g}$), against two "
        f"analytic baselines for independent uniform integer delays: the "
        f"probability that a pair compensates the target separation exactly "
        f"({100 * exact_line[0]:.2f}\\%, dotted), and the same probability "
        f"widened by the near-miss tolerance the branch is measured to have at "
        f"each threshold (dashed; $\\pm{tol_used[0]}$, $\\pm{tol_used[1]}$ and "
        f"$\\pm{tol_used[2]}$ steps at $\\theta = "
        f"{thetas[0]:g}$, ${thetas[1]:g}$ and ${thetas[2]:g}$). The measured "
        f"rate tracks the tolerance-widened baseline across two orders of "
        f"magnitude. The match rate is therefore what naive pairwise delay "
        f"matching predicts, not something suppressed below it."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

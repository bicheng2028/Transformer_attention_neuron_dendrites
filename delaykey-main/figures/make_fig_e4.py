"""Build the E4 noise-robustness figure from ``results/e4_noise.csv``.

This is manuscript Figure 5 (Experiment 2). It replaces
``figures/make_fig5_noise_robustness.py``, which plotted values quoted out of
``content.tex`` and digitised from a legacy raster because no generating code
survived. Every value here is simulated.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
7 pt maximum text, 8 pt bold lower-case panel letters, no panel titles, and
curves labelled in place rather than through legend boxes.

Panels:

a  Detection accuracy under temporal jitter, hard against soft gating.
b  False-positive rate under spurious spikes -- the panel that carries the
   result, because it is where the original claim fails.
c  Activation in spikes per trial, on a log axis, with the fold-difference
   between the two rules readable as a vertical distance.

Usage::

    python figures/make_fig_e4.py --formats pdf eps tiff png
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

GATING_STYLE = {
    "hard": ("Hard gating", PALETTE["blue"]),
    "soft": ("Soft gating", PALETTE["grey"]),
}


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "noise": raw["noise"],
                "level": float(raw["level"]),
                "gating": raw["gating"],
                "theta": float(raw["theta"]),
                "seed": int(raw["seed"]),
                "n_trials": int(raw["n_trials"]),
                "accuracy": float(raw["accuracy"]),
                "activation": float(raw["activation"]),
                "false_positive_rate": (float(raw["false_positive_rate"])
                                        if raw["false_positive_rate"] else None),
            })
    return rows


def group(rows, keys, value):
    out = defaultdict(list)
    for row in rows:
        if row[value] is not None:
            out[tuple(row[k] for k in keys)].append(row[value])
    return out


def band(ax, xs, estimates, colour, marker="o"):
    means = np.array([e.mean for e in estimates])
    lo = np.array([e.lo for e in estimates])
    hi = np.array([e.hi for e in estimates])
    ax.fill_between(xs, lo, hi, color=tint(colour), linewidth=0, zorder=1)
    ax.plot(xs, means, color=colour, marker=marker, markersize=3.4,
            linewidth=1.5, zorder=2)
    return means


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("results/e4_noise.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e4_noise"))
    # theta 1.5 is the two-input requirement on a two-input branch, the same
    # convention by which theta 2.5 requires three inputs and 4.5 requires five
    # elsewhere in the paper. theta 1.8, also in the CSV, additionally forbids
    # the one-step near-miss the dendritic time constant would otherwise pass.
    parser.add_argument("--theta-main", type=float, default=1.5)
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    parser.add_argument("--height-mm", type=float, default=52.0)
    args = parser.parse_args(argv)

    rows = [r for r in load(args.csv) if r["theta"] == args.theta_main]
    if not rows:
        raise SystemExit(f"no rows at theta = {args.theta_main} in {args.csv}")
    rng = np.random.default_rng(0)
    n_seeds = len({r["seed"] for r in rows})
    n_trials = rows[0]["n_trials"]

    apply_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=figure_size(args.width, args.height_mm))
    fig.subplots_adjust(wspace=0.42, left=0.075, right=0.985,
                        bottom=0.22, top=0.93)

    cited = {}

    # ---- a: accuracy under jitter -----------------------------------------
    jitters = sorted({r["level"] for r in rows if r["noise"] == "jitter"})
    for gating, (name, colour) in GATING_STYLE.items():
        sub = [r for r in rows if r["noise"] == "jitter" and r["gating"] == gating]
        grouped = group(sub, ("level",), "accuracy")
        est = [bootstrap_mean(grouped[(m,)], rng) for m in jitters]
        for m, e in zip(jitters, est):
            cited[("jitter", gating, m, "accuracy")] = e
        band(ax_a, jitters, est, colour)
        anchor = 0 if gating == "soft" else len(jitters) - 1
        direct_label(ax_a, jitters[anchor],
                     est[anchor].mean + (0.07 if gating == "soft" else 0.10),
                     name, colour, dx=(1.0 if gating == "soft" else -14.0))
    ax_a.set_xlabel("Temporal jitter $\\sigma$ (steps)")
    ax_a.set_ylabel("Detection accuracy")
    ax_a.set_ylim(0, 1.18)

    # ---- b: false positives under spurious spikes -------------------------
    densities = sorted({r["level"] for r in rows if r["noise"] == "spurious"})
    for gating, (name, colour) in GATING_STYLE.items():
        sub = [r for r in rows
               if r["noise"] == "spurious" and r["gating"] == gating]
        grouped = group(sub, ("level",), "false_positive_rate")
        live = [m for m in densities if (m,) in grouped]
        est = [bootstrap_mean(grouped[(m,)], rng) for m in live]
        for m, e in zip(live, est):
            cited[("spurious", gating, m, "fp")] = e
        band(ax_b, live, est, colour, marker="s")
        i = 1 if gating == "soft" else len(live) - 3
        direct_label(ax_b, live[i], est[i].mean + (0.09 if gating == "soft" else -0.16),
                     name, colour, dx=(0.4 if gating == "soft" else -6.0))
        # Accuracy on the same axis: it stays pinned at the ceiling, which is
        # the half of the result that does survive, and drawing it here rather
        # than in its own panel makes the contrast with the rising
        # false-positive curve immediate.
        if gating == "hard":
            acc = group(sub, ("level",), "accuracy")
            acc_est = [bootstrap_mean(acc[(m,)], rng) for m in densities]
            for m, e in zip(densities, acc_est):
                cited[("spurious", gating, m, "accuracy")] = e
            ax_b.plot(densities, [e.mean for e in acc_est], color=colour,
                      linewidth=0.7, linestyle=(0, (2, 1.5)), zorder=2)
            direct_label(ax_b, densities[-1], 1.06, "Hit rate, hard",
                         colour, dx=-16.0)
    ax_b.set_xlabel("Spurious spike density")
    ax_b.set_ylabel("Rate")
    ax_b.set_ylim(0, 1.18)

    # ---- c: activation cost ------------------------------------------------
    for gating, (name, colour) in GATING_STYLE.items():
        sub = [r for r in rows
               if r["noise"] == "spurious" and r["gating"] == gating]
        grouped = group(sub, ("level",), "activation")
        est = [bootstrap_mean(grouped[(m,)], rng) for m in densities]
        for m, e in zip(densities, est):
            cited[("spurious", gating, m, "activation")] = e
        band(ax_c, densities, est, colour)
        # Anchored two points in from the right edge: at the last point the
        # label runs off the axes, since direct_label draws left-aligned.
        i = len(densities) - 3
        direct_label(ax_c, densities[i], est[i].mean * (1.42 if gating == "soft" else 0.60),
                     name, colour, dx=-2.0)
    ax_c.set_xlabel("Spurious spike density")
    ax_c.set_ylabel("Spikes per trial")
    ax_c.set_yscale("log")

    label_panels([ax_a, ax_b, ax_c])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    def cite(kind, gating, level, key, dp=0):
        e = cited[(kind, gating, level, key)]
        return (f"{100 * e.mean:.{dp}f}\\% "
                f"[{100 * e.lo:.{dp}f}, {100 * e.hi:.{dp}f}]")

    def fold(level):
        h = cited[("spurious", "hard", level, "activation")].mean
        s = cited[("spurious", "soft", level, "activation")].mean
        return s / h

    peak_density = max(densities, key=fold)
    caption = (
        f"\\textbf{{Noise robustness of a single hard-gated K-line, "
        f"re-simulated.}} One dendritic branch carrying one delay set, driven "
        f"with the A$\\rightarrow$B sequence, with a soft-gating control that "
        f"bypasses the calcium threshold and is otherwise identical. "
        f"$\\theta = {args.theta_main:g}$, the two-input requirement on a "
        f"two-input branch. {n_seeds} independent seeds of {n_trials} trials "
        f"each; shading is the 95\\% bootstrap confidence interval over seeds. "
        f"\\textbf{{a}}, Detection accuracy under Gaussian jitter of the target "
        f"spike times. Hard gating falls from "
        f"{cite('jitter', 'hard', 0.0, 'accuracy')} when clean to "
        f"{cite('jitter', 'hard', max(jitters), 'accuracy')} at "
        f"$\\sigma = {max(jitters):g}$ steps as delayed spikes miss the "
        f"coincidence window; the soft control holds the ceiling throughout, "
        f"because it integrates every input whenever it arrives. "
        f"\\textbf{{b}}, Rate of firing to spurious spikes inserted "
        f"independently in each (step, input) bin. The hit rate for the target "
        f"is unaffected at any density (dashed). The false-positive rate is "
        f"not: hard gating fires to noise alone on "
        f"{cite('spurious', 'hard', 0.02, 'fp')} of trials at a density of "
        f"0.02 and {cite('spurious', 'hard', 0.1, 'fp')} at 0.1. Isolated "
        f"spikes are therefore \\emph{{not}} filtered by the threshold once the "
        f"input is dense enough to supply its own coincidences --- on a "
        f"two-input branch two spurious spikes at the right separation are "
        f"indistinguishable from the target. What the threshold buys is "
        f"a rate that stays low while the input stays sparse. "
        f"\\textbf{{c}}, Activation in spikes emitted per trial, the same unit "
        f"Experiment~3 reports, on a logarithmic axis. The advantage of hard "
        f"gating is largest in the sparse regime, peaking at "
        f"{fold(peak_density):.1f}-fold at a density of {peak_density:g}, and "
        f"narrows to {fold(max(densities)):.1f}-fold as the input saturates "
        f"both rules. It does not grow with noise."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

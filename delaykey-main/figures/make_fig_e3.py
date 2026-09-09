"""Build the E3 discriminability figure from ``results/e3_discriminability.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
165.1 mm PLOS width, 7 pt maximum text, 8 pt bold lower-case panel letters, no
panel titles, and curves labelled in place rather than through legend boxes.

Panels:

a  Discriminability (ROC AUC, threshold-free) against temporal jitter, hard
   versus soft gating, with 95% bootstrap confidence intervals.
b  Signed AUC for the same runs, showing that the soft readout carries the
   opposite polarity -- it responds more to the distractor than to the target.
c  Activation cost, spikes per trial per tuned neuron, on a log axis.

Usage::

    python figures/make_fig_e3.py --formats pdf eps tiff png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.plotting import (PALETTE, PT_ANNOT, apply_style, direct_label,
                               figure_size, label_panels, save_figure)
from delaykey.stats import bootstrap_mean

import matplotlib.pyplot as plt  # noqa: E402

GATING_STYLE = {
    "hard": ("Hard gate", PALETTE["blue"]),
    "soft": ("Soft gate", PALETTE["grey"]),
}


def load(path: Path) -> list[dict]:
    """Read the tidy CSV, coercing numerics."""
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "gating": raw["gating"],
                "sigma": float(raw["sigma"]),
                "dispersion": int(raw["dispersion"]),
                "theta": float(raw["theta"]),
                "seed": int(raw["seed"]),
                "n_tuned": int(raw["n_tuned"]),
                "auc": float(raw["auc"]),
                "auc_unsigned": float(raw["auc_unsigned"]),
                "activation_target": float(raw["activation_target"]),
                "activation_distractor": float(raw["activation_distractor"]),
            })
    return rows


def group(rows, keys, value):
    """Map each combination of ``keys`` to the list of ``value`` across seeds."""
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row[value])
    return out


def band(ax, xs, estimates, colour, **kwargs):
    """Plot the mean with its bootstrap interval as a shaded band."""
    means = np.array([e.mean for e in estimates])
    lo = np.array([e.lo for e in estimates])
    hi = np.array([e.hi for e in estimates])
    ax.fill_between(xs, lo, hi, color=colour, alpha=0.20, linewidth=0, zorder=1)
    ax.plot(xs, means, color=colour, zorder=2, **kwargs)
    return means


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("results/e3_discriminability.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e3_discriminability"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    parser.add_argument("--height-mm", type=float, default=64.0)
    # The restructured manuscript stacks this figure beneath the two-input
    # worked example inside one float, where these panels are d-f.
    parser.add_argument("--panel-letters", nargs=3, default=["a", "b", "c"])
    args = parser.parse_args(argv)

    rows = load(args.csv)
    rng = np.random.default_rng(0)

    sigmas = sorted({r["sigma"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    dispersion = rows[0]["dispersion"]
    theta = rows[0]["theta"]
    n_tuned = int(np.mean([r["n_tuned"] for r in rows]))

    apply_style()
    fig, axes = plt.subplots(
        1, 3, figsize=figure_size(args.width, args.height_mm))
    ax_a, ax_b, ax_c = axes
    # Margins in millimetres rather than axes fractions, matched to
    # figures/make_fig4_lif_vs_integrator.py (LEFT_MM = 15.0, TRIM_MM = 1.9).
    # The restructured manuscript stacks that figure directly above this one
    # inside a single float, and two halves whose axes start 2.6 mm apart read
    # as two figures glued together rather than as one. The left margin is the
    # binding constraint there -- it clears that figure's raster row names -- so
    # this figure yields to it.
    left_mm, trim_mm, width_mm = 17.5, 1.9, 165.1
    fig.subplots_adjust(wspace=0.46, left=left_mm / width_mm,
                        right=(width_mm - trim_mm) / width_mm,
                        bottom=0.22, top=0.895)  # headroom for 10 pt panel letters

    summary = {}
    for gating, (name, colour) in GATING_STYLE.items():
        subset = [r for r in rows if r["gating"] == gating]

        # a -- threshold-free discriminability, polarity removed.
        grouped = group(subset, ("sigma",), "auc_unsigned")
        est = [bootstrap_mean(grouped[(s,)], rng) for s in sigmas]
        band(ax_a, sigmas, est, colour)
        summary[(gating, "auc")] = est

        # b -- signed AUC, which is where the two models part company.
        grouped = group(subset, ("sigma",), "auc")
        est_signed = [bootstrap_mean(grouped[(s,)], rng) for s in sigmas]
        band(ax_b, sigmas, est_signed, colour)
        summary[(gating, "signed")] = est_signed

        # c -- activation cost in comparable units.
        grouped = group(subset, ("sigma",), "activation_target")
        est_act = [bootstrap_mean(grouped[(s,)], rng) for s in sigmas]
        band(ax_c, sigmas, est_act, colour)
        summary[(gating, "act")] = est_act

    hard_c, soft_c = PALETTE["blue"], PALETTE["grey"]
    direct_label(ax_a, sigmas[-1], summary[("hard", "auc")][-1].mean - 0.0055,
                 "Hard gate", hard_c, dx=-2.6)
    direct_label(ax_a, sigmas[-1], summary[("soft", "auc")][-1].mean + 0.0022,
                 "Soft gate", soft_c, dx=-2.6)
    direct_label(ax_b, sigmas[2], summary[("hard", "signed")][2].mean - 0.075,
                 "Hard gate", hard_c, dx=-0.6)
    direct_label(ax_b, sigmas[2], summary[("soft", "signed")][2].mean + 0.075,
                 "Soft gate", soft_c, dx=-0.6)
    # Anchored mid-panel: at sigmas[-1] the soft label ran off the figure's
    # right edge and the hard label fell below the axis floor, leaving this
    # panel with no legible key at all.
    direct_label(ax_c, sigmas[3], summary[("soft", "act")][3].mean * 0.70,
                 "Soft gate", soft_c, dx=2.0)
    direct_label(ax_c, sigmas[3], summary[("hard", "act")][3].mean * 1.45,
                 "Hard gate", hard_c, dx=2.0)

    ax_a.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_a.set_ylabel("ROC AUC (unsigned)")
    # Both models sit near the ceiling, so the axis is zoomed to the top 5% of
    # the range to make the hard-gate decline legible. Chance is 0.5, far below
    # the bottom of this axis; the caption says so.
    ax_a.set_ylim(0.972, 1.004)
    ax_a.set_yticks([0.98, 0.99, 1.00])

    ax_b.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_b.set_ylabel("ROC AUC (signed)")
    ax_b.set_ylim(-0.04, 1.04)
    # Both series are flat -- 1.000 and 0.000 at every jitter level -- so the
    # panel's content is which side of chance each one sits on, not its shape.
    # Shading the sub-chance half says that directly; without it the panel is
    # two horizontal lines separated by empty white.
    ax_b.axhspan(-0.04, 0.5, color=PALETTE["grey"], alpha=0.10, linewidth=0,
                 zorder=0)
    ax_b.axhline(0.5, color=PALETTE["black"], linewidth=0.9, linestyle=(0, (3, 2)),
                 zorder=1)
    ax_b.annotate("chance", xy=(sigmas[-1], 0.5), xytext=(sigmas[-1], 0.545),
                  color=PALETTE["black"], fontsize=PT_ANNOT, ha="right")
    # One label for the shaded half, set in the empty middle of the band so it
    # clears both the soft-gate trace at 0.0 and the chance rule at 0.5. The
    # unshaded half needs no counterpart: "not inverted" is the default reading.
    # Short enough to stay inside the panel at 8 pt; the caption carries
    # the full reading. The long form overran into the next panel.
    ax_b.annotate("inverted", xy=(sigmas[0] + 0.15, 0.24),
                  color=PALETTE["black"], fontsize=PT_ANNOT, alpha=0.75)

    ax_c.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_c.set_ylabel("Spikes per trial per neuron")
    ax_c.set_yscale("log")

    pa, pb, pc = args.panel_letters
    label_panels([ax_a, ax_b, ax_c], labels=[pa, pb, pc])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    a_h0 = summary[("hard", "auc")][0]
    a_hl = summary[("hard", "auc")][-1]
    ratio_lo = (summary[("soft", "act")][0].mean
                / summary[("hard", "act")][0].mean)
    ratio_hi = (summary[("soft", "act")][-1].mean
                / summary[("hard", "act")][-1].mean)
    caption = (
        f"\\textbf{{Hard and soft gating compared on the same task, the same "
        f"delays, and a threshold-free metric.}} A target-versus-distractor "
        f"discrimination, A$\\rightarrow$B against B$\\rightarrow$A, over "
        f"{n_tuned} tuned neurons drawn from a population whose delays are "
        f"uniform on $[0, {dispersion}]$~ms at $\\theta = {theta:g}$. Both "
        f"models receive identical delays, connectivity and stimuli and differ "
        f"only in whether the calcium threshold is applied, so the contrast "
        f"isolates gating rather than delay tuning. Shading is the 95\\% "
        f"bootstrap confidence interval over {n_seeds} independent seeds. "
        f"\\textbf{{{pa}}}, Discriminability as area under the ROC with polarity "
        f"removed; note the axis is zoomed to the top of the range, chance "
        f"being 0.5 and far below it. Both models separate the two orders "
        f"essentially perfectly at low jitter; hard gating declines only to "
        f"{a_hl.mean:.3f} [{a_hl.lo:.3f}, {a_hl.hi:.3f}] at "
        f"$\\sigma = {sigmas[-1]:g}$ from {a_h0.mean:.3f} when clean, so a "
        f"low detection rate at that jitter would be a property of the fixed "
        f"operating point at which it is measured and not a loss of "
        f"information. \\textbf{{{pb}}}, The same runs, showing the signed AUC. "
        f"Hard gating scores the target above the distractor (signed AUC "
        f"$\\approx$ 1.000). The soft-gated control, when read out by total "
        f"spike count, scores the distractor above the target (signed AUC "
        f"$\\approx$ 0.000), reflecting its lack of a refractory period rather "
        f"than a failure of selectivity. "
        f"\\textbf{{{pc}}}, Activation cost in comparable units, spikes emitted "
        f"per trial per tuned neuron, on a logarithmic axis. Hard gating spends "
        f"{ratio_lo:.0f}-fold less when clean and {ratio_hi:.0f}-fold less at "
        f"$\\sigma = {sigmas[-1]:g}$: the gap widens with jitter, because leaky "
        f"integration accumulates every displaced spike while the threshold "
        f"discards it."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

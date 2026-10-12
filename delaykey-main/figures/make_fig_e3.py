"""Build the E3 discriminability figure from ``results/e3_discriminability.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
165.1 mm PLOS width, 7 pt maximum text, 8 pt bold lower-case panel letters, no
panel titles, and curves labelled in place rather than through legend boxes.

2026-09 revision. The previous version led with the ROC AUC of the tuned
subpopulation, and that panel could not carry the claim made from it: the tuned
subpopulation is selected as the neurons that fire for the clean target and not
the clean distractor, so its AUC is 1.0 at sigma = 0 by construction, and its
shallow decline under jitter reports only that jitter degrades both stimuli
symmetrically enough to preserve the rank order between them. Read as
robustness, it asserted a tolerance to perturbation that a population of
coincidence detectors cannot have. The AUC now lives in the CSV as a secondary
summary and is off the figure; the panels plot the two activity channels
directly, which the selection rule does not pin.

Panels:

a  The three activity channels under hard gating: spikes per neuron per trial
   in the tuned pool on the target, in the tuned pool on the distractor, and in
   the non-tuned pool. This is the panel that answers how far jitter disturbs
   the population code -- the signal channel falls by more than half while the
   crosstalk floor stays flat, and the tuned pool's answer to the wrong order
   rises to meet that floor.
b  Population selectivity index, (tuned - non-tuned) / (tuned + non-tuned) on
   the target, hard versus soft gating. Hard gating starts far above soft and
   loses roughly half of its margin; soft gating never has one to lose.
c  Spikes per tuned neuron per trial, hard versus soft gating, on a log axis.

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

from delaykey.plotting import (PALETTE, apply_style, direct_label,
                               figure_size, label_panels, save_figure, tint)
from delaykey.stats import bootstrap_mean

import matplotlib.pyplot as plt  # noqa: E402

# Colour roles, held consistent with the neighbouring figures: blue is the
# headline condition, vermillion the contrasting one, grey the soft-gated
# control wherever it appears. Within panel a the blue family carries the
# target stimulus and saturation distinguishes the two pools, which keeps grey
# reserved for the soft model in panels b and c.
C_TUNED = PALETTE["blue"]
C_NONTUNED = tint(PALETTE["blue"], 0.45)
C_WRONG = PALETTE["vermillion"]
GATING_STYLE = {
    "hard": ("Hard gate", PALETTE["blue"]),
    "soft": ("Soft gate", PALETTE["grey"]),
}

CHANNELS = ("tuned_activation_target", "tuned_activation_distractor",
            "nontuned_activation_target", "nontuned_activation_distractor",
            "snr_target", "snr_distractor", "si_target", "si_distractor")


def load(path: Path) -> list[dict]:
    """Read the tidy CSV, coercing numerics."""
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row = {
                "gating": raw["gating"],
                "sigma": float(raw["sigma"]),
                "dispersion": int(raw["dispersion"]),
                "theta": float(raw["theta"]),
                "seed": int(raw["seed"]),
                "n_tuned": int(raw["n_tuned"]),
                "n_nontuned": int(raw["n_nontuned"]),
                "auc": float(raw["auc"]),
                "auc_unsigned": float(raw["auc_unsigned"]),
            }
            row.update({k: float(raw[k]) for k in CHANNELS})
            rows.append(row)
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
    n_pop = n_tuned + int(np.mean([r["n_nontuned"] for r in rows]))

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

    def curve(ax, subset, key, colour, **kwargs):
        grouped = group(subset, ("sigma",), key)
        est = [bootstrap_mean(grouped[(s,)], rng) for s in sigmas]
        band(ax, sigmas, est, colour, **kwargs)
        return est

    hard = [r for r in rows if r["gating"] == "hard"]
    soft = [r for r in rows if r["gating"] == "soft"]

    # a -- the three activity channels under hard gating, in spikes per neuron
    # per trial. Nothing here is fixed by the selection rule: selection chooses
    # which neurons are counted, not how hard they fire.
    a_tuned = curve(ax_a, hard, "tuned_activation_target", C_TUNED)
    a_wrong = curve(ax_a, hard, "tuned_activation_distractor", C_WRONG)
    a_cross = curve(ax_a, hard, "nontuned_activation_target", C_NONTUNED)

    # b -- population selectivity on the target, where the two models part
    # company: the quantity the old AUC panel was taken to measure, measured.
    b_hard = curve(ax_b, hard, "si_target", GATING_STYLE["hard"][1])
    b_soft = curve(ax_b, soft, "si_target", GATING_STYLE["soft"][1])

    # c -- spike counts in comparable units.
    c_hard = curve(ax_c, hard, "tuned_activation_target",
                   GATING_STYLE["hard"][1])
    c_soft = curve(ax_c, soft, "tuned_activation_target",
                   GATING_STYLE["soft"][1])

    direct_label(ax_a, sigmas[-1], a_tuned[-1].mean + 0.075, "tuned",
                 C_TUNED, dx=-1.05, ha="right")
    direct_label(ax_a, sigmas[-1], a_wrong[-1].mean + 0.068, "wrong order",
                 C_WRONG, dx=-1.05, ha="right")
    direct_label(ax_a, sigmas[-1], a_cross[-1].mean - 0.115, "non-tuned",
                 C_NONTUNED, dx=-1.05, ha="right")

    direct_label(ax_b, sigmas[-1], b_hard[-1].mean + 0.045, "Hard gate",
                 GATING_STYLE["hard"][1], dx=-1.05, ha="right")
    direct_label(ax_b, sigmas[-1], b_soft[-1].mean + 0.045, "Soft gate",
                 GATING_STYLE["soft"][1], dx=-1.05, ha="right")

    direct_label(ax_c, sigmas[-1], c_soft[-1].mean * 0.72, "Soft gate",
                 GATING_STYLE["soft"][1], dx=-1.05, ha="right")
    direct_label(ax_c, sigmas[-1], c_hard[-1].mean * 1.30, "Hard gate",
                 GATING_STYLE["hard"][1], dx=-1.05, ha="right")

    ax_a.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_a.set_ylabel("Spikes per neuron per trial")
    ax_a.set_ylim(0.0, max(e.hi for e in a_tuned) * 1.16)

    ax_b.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_b.set_ylabel("Selectivity index")
    # The axis floor is zero, which is exactly the value at which the tuned and
    # non-tuned pools are equally active -- so the soft-gate trace sitting on
    # the floor already reads as "no selectivity" without a rule to say so. A
    # dashed line at zero only collided with that trace.
    ax_b.set_ylim(0.0, max(e.hi for e in b_hard) * 1.20)

    ax_c.set_xlabel("Jitter $\\sigma$ (steps)")
    ax_c.set_ylabel("Spikes per neuron per trial")
    ax_c.set_yscale("log")

    pa, pb, pc = args.panel_letters
    label_panels([ax_a, ax_b, ax_c], labels=[pa, pb, pc])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    t0, t5 = a_tuned[0], a_tuned[-1]
    x0, x5 = a_cross[0], a_cross[-1]
    w5 = a_wrong[-1]
    s0, s5 = b_hard[0], b_hard[-1]
    f0, f5 = b_soft[0], b_soft[-1]
    drop = 100.0 * (1.0 - t5.mean / t0.mean)
    # The jitter level at which the tuned pool's answer to the wrong order first
    # reaches the crosstalk floor: the point at which the tuned pool stops
    # being silent for the reversed order, though it still prefers the target.
    # Read off the curves rather than asserted, so the sentence cannot drift
    # from the data.
    crossing = next((s for s, w, x in zip(sigmas, a_wrong, a_cross)
                     if w.mean >= x.mean), None)
    ratio_lo = c_soft[0].mean / c_hard[0].mean
    ratio_hi = c_soft[-1].mean / c_hard[-1].mean
    caption = (
        f"\\textbf{{Hard gating buys population selectivity with a fraction of "
        f"the soft rule's spikes, and loses most of its signal to the first "
        f"millisecond of jitter.}} A target-versus-distractor discrimination, "
        f"A$\\rightarrow$B against B$\\rightarrow$A, over a population of "
        f"{n_pop} neurons whose delays are uniform on $[0, {dispersion}]$~ms "
        f"at $\\theta = {theta:g}$. The population is partitioned once, on the "
        f"clean stimuli, into the {n_tuned} \\emph{{tuned}} neurons that fire "
        f"for the target and not the distractor and the non-tuned remainder; "
        f"the partition is then held fixed at every jitter level, so it cannot "
        f"track the perturbation. Both models receive identical delays, "
        f"connectivity and stimuli and differ only in whether the calcium "
        f"threshold is applied, so the contrast isolates gating rather than "
        f"delay tuning. Shading is the 95\\% bootstrap confidence interval "
        f"over {n_seeds} independent seeds. "
        f"\\textbf{{{pa}}}, The three activity channels under hard gating. The "
        f"signal channel -- the tuned pool's response to the target -- falls "
        f"{drop:.0f}\\% from {t0.mean:.2f} [{t0.lo:.2f}, {t0.hi:.2f}] spikes "
        f"per neuron per trial when clean to {t5.mean:.2f} "
        f"[{t5.lo:.2f}, {t5.hi:.2f}] at $\\sigma = {sigmas[-1]:g}$, while the "
        f"crosstalk floor is flat at {x0.mean:.2f} and {x5.mean:.2f}: the code "
        f"degrades by losing signal, not by recruiting noise. The tuned pool's "
        f"response to the reversed order rises from silence to {w5.mean:.2f}, "
        f"reaching the non-tuned floor by $\\sigma = {crossing:g}$: beyond "
        f"that jitter the tuned pool is no longer silent for the reversed "
        f"order, though it still prefers the target, {t5.mean:.2f} against "
        f"{w5.mean:.2f} at $\\sigma = {sigmas[-1]:g}$. "
        f"\\textbf{{{pb}}}, Population selectivity index, (tuned $-$ "
        f"non-tuned) / (tuned $+$ non-tuned) on the target; zero is the value "
        f"at which the two pools are equally active. Hard gating falls from "
        f"{s0.mean:.3f} [{s0.lo:.3f}, {s0.hi:.3f}] to {s5.mean:.3f} "
        f"[{s5.lo:.3f}, {s5.hi:.3f}], roughly half its clean margin. The "
        f"soft-gated control never rises above {max(f0.mean, f5.mean):.3f}: "
        f"sharing the delays is not enough, and without the threshold the "
        f"population has essentially no selectivity to lose. "
        f"\\textbf{{{pc}}}, Spikes per tuned neuron per trial, on a "
        f"logarithmic axis. The tuned pool emits {ratio_lo:.0f}-fold fewer "
        f"spikes under hard gating when clean and {ratio_hi:.0f}-fold fewer at "
        f"$\\sigma = {sigmas[-1]:g}$, the gap widening both because the "
        f"threshold discards the displaced spikes and because the hard-gated "
        f"response itself collapses."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

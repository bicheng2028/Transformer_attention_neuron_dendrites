"""Build the E7 verification figure from ``results/e7_verification.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
165.1 mm PLOS width, 7 pt maximum text, 8 pt bold lower-case panel letters, no
panel titles, and series labelled in place rather than through legend boxes.

Panels:

a  Firing probability across the content conditions at one representative
   inhibition strength, for both somatic rules.
b  The veto index against inhibition strength. The region above zero is
   unreachable by any weighted sum of non-negative drives, so the sign of this
   curve is the whole test.
c  What it costs to suppress mismatch: matched-content firing against
   mismatched-content firing as inhibition rises.

Usage::

    python figures/make_fig_e7.py --formats pdf eps tiff png
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

RULE_STYLE = {
    "conjunctive": ("Conjunctive", PALETTE["blue"]),
    "sum": ("Weighted sum", PALETTE["grey"]),
}
CONDITIONS = [
    ("fire_match_distal", "Matched"),
    ("fire_absent_distal", "Absent"),
    ("fire_mismatch_distal", "Mismatched"),
]


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row = {"rule": raw["rule"], "seed": int(raw["seed"]),
                   "inhibition": float(raw["inhibition"]),
                   "n_detectors": int(raw["n_detectors"]),
                   "content_dim": int(raw["content_dim"])}
            for key in raw:
                if key.startswith("fire_") or key in ("margin", "veto_index"):
                    row[key] = float(raw[key])
            rows.append(row)
    return rows


def group(rows, keys, value):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row[value])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("results/e7_verification.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e7_verification"))
    parser.add_argument("--inhibition-main", type=float, default=0.6)
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    parser.add_argument("--height-mm", type=float, default=57.0)
    args = parser.parse_args(argv)

    rows = load(args.csv)
    rng = np.random.default_rng(0)
    inhibitions = sorted({r["inhibition"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    n_det = int(np.mean([r["n_detectors"] for r in rows]))
    content_dim = rows[0]["content_dim"]

    apply_style()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=figure_size(args.width, args.height_mm))
    fig.subplots_adjust(wspace=0.40, left=0.075, right=0.985,
                        bottom=0.20, top=0.93)

    # a -- the 2x3 at one representative inhibition.
    width = 0.36
    cited = {}
    for k, (rule, (name, colour)) in enumerate(RULE_STYLE.items()):
        sel = [r for r in rows if r["rule"] == rule
               and r["inhibition"] == args.inhibition_main]
        means, los, his = [], [], []
        for field, _ in CONDITIONS:
            est = bootstrap_mean([r[field] for r in sel], rng)
            cited[(rule, field)] = est
            means.append(est.mean)
            los.append(est.mean - est.lo)
            his.append(est.hi - est.mean)
        xs = np.arange(len(CONDITIONS)) + (k - 0.5) * width
        ax_a.bar(xs, means, width=width, color=colour, linewidth=0, zorder=2)
        ax_a.errorbar(xs, means, yerr=[los, his], fmt="none",
                      ecolor=PALETTE["black"], elinewidth=0.9, capsize=2.2,
                      zorder=3)
    ax_a.set_xticks(np.arange(len(CONDITIONS)))
    ax_a.set_xticklabels([label for _, label in CONDITIONS])
    ax_a.set_ylabel("Fraction firing")
    ax_a.set_ylim(0, 1.16)
    ax_a.annotate("Conjunctive", xy=(0.02, 0.94), xycoords="axes fraction",
                  color=PALETTE["blue"], fontsize=PT_ANNOT)
    ax_a.annotate("Weighted sum", xy=(0.54, 0.94), xycoords="axes fraction",
                  color=PALETTE["grey"], fontsize=PT_ANNOT)

    # b -- the veto index, with the region a weighted sum cannot reach.
    ax_b.axhspan(0.0, 0.92, color=PALETTE["vermillion"], alpha=0.07,
                 linewidth=0, zorder=0)
    veto = {}
    for rule, (name, colour) in RULE_STYLE.items():
        sel = [r for r in rows if r["rule"] == rule]
        grouped = group(sel, ("inhibition",), "veto_index")
        est = [bootstrap_mean(grouped[(i,)], rng) for i in inhibitions]
        veto[rule] = est
        means = np.array([e.mean for e in est])
        lo = np.array([e.lo for e in est])
        hi = np.array([e.hi for e in est])
        ax_b.fill_between(inhibitions, lo, hi, color=colour, alpha=0.20,
                          linewidth=0, zorder=1)
        ax_b.plot(inhibitions, means, color=colour, marker="o", markersize=3.4,
                  linewidth=1.5, zorder=2)
    ax_b.axhline(0.0, color=PALETTE["black"], linewidth=0.9, zorder=1)
    direct_label(ax_b, inhibitions[-1], veto["conjunctive"][-1].mean - 0.10,
                 "Conjunctive", PALETTE["blue"], dx=-0.62)
    direct_label(ax_b, inhibitions[-1], 0.06, "Weighted sum", PALETTE["grey"],
                 dx=-0.62)
    ax_b.annotate("unreachable by any weighted sum", xy=(0.03, 0.83),
                  xycoords="axes fraction", color=PALETTE["vermillion"],
                  fontsize=PT_ANNOT)
    ax_b.set_xlabel("Inhibition strength")
    ax_b.set_ylabel("Veto index")
    ax_b.set_ylim(-0.14, 0.92)

    # c -- the cost of suppression: matched firing against mismatched firing.
    for rule, (name, colour) in RULE_STYLE.items():
        sel = [r for r in rows if r["rule"] == rule]
        gm = group(sel, ("inhibition",), "fire_mismatch_distal")
        gk = group(sel, ("inhibition",), "fire_match_distal")
        xs = [np.mean(gm[(i,)]) for i in inhibitions]
        ys = [np.mean(gk[(i,)]) for i in inhibitions]
        ax_c.plot(xs, ys, color=colour, marker="o", markersize=3.4,
                  linewidth=1.5, zorder=2)
    ax_c.plot([0, 1], [0, 1], color=PALETTE["black"], linewidth=0.9,
              linestyle=(0, (3, 2)), zorder=1)
    ax_c.annotate("Conjunctive", xy=(0.10, 0.92), xycoords="axes fraction",
                  color=PALETTE["blue"], fontsize=PT_ANNOT)
    ax_c.annotate("Weighted sum", xy=(0.40, 0.55), xycoords="axes fraction",
                  color=PALETTE["grey"], fontsize=PT_ANNOT)
    ax_c.set_xlabel("Fraction firing, mismatched")
    ax_c.set_ylabel("Fraction firing, matched")
    ax_c.set_xlim(0, 1.05)
    ax_c.set_ylim(0, 1.14)

    label_panels([ax_a, ax_b, ax_c])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    def pct(rule, field):
        e = cited[(rule, field)]
        return f"{100 * e.mean:.0f}\\% [{100 * e.lo:.0f}, {100 * e.hi:.0f}]"

    def silenced(rule, field):
        """The complement of a firing rate, with the interval reflected too.

        Quoting a firing rate under the word "silenced" is the error this
        exists to stop: the two are 1 - x of each other, and the bounds swap.
        """
        e = cited[(rule, field)]
        return (f"{100 * (1 - e.mean):.0f}\\% "
                f"[{100 * (1 - e.hi):.0f}, {100 * (1 - e.lo):.0f}]")

    sum_max = max(e.mean for e in veto["sum"])
    conj_max = max(e.mean for e in veto["conjunctive"])
    caption = (
        f"\\textbf{{Somatic integration is conjunctive, and the test that shows "
        f"it is the ordering against absent content.}} The distal sequence is "
        f"present throughout; only the content signal $V'$ varies. $V'$ is a "
        f"unit vector in $\\mathbb{{R}}^{{{content_dim}}}$ and agreement is its "
        f"rectified cosine with the neuron's stored template, measured over the "
        f"{n_det} neurons per population whose delays match the target "
        f"sequence. The two somatic rules are identical except in what sets "
        # No \ref to the somatic equation: this figure now lives in S5 Text of
        # the supplement, which is compiled as its own document, so a label
        # defined in content.tex resolves to nothing there.
        f"$I_\\mathrm{{inh}}$: a constant, giving exactly the weighted sum of "
        f"the main text's somatic rule as written, or the disagreement between "
        f"$V'$ and the template, gated by the presence of proximal input. "
        f"{n_seeds} independent seeds; error bars and shading are 95\\% "
        f"bootstrap confidence intervals. \\textbf{{a}}, Firing across the "
        f"three content conditions at an inhibition strength of "
        f"{args.inhibition_main:g}. The conjunctive soma fires to matched "
        f"content ({pct('conjunctive', 'fire_match_distal')}) and to absent "
        f"content ({pct('conjunctive', 'fire_absent_distal')}), and mismatched "
        f"content silences it on "
        f"{silenced('conjunctive', 'fire_mismatch_distal')} of trials --- it "
        f"still fires on the remaining "
        f"{pct('conjunctive', 'fire_mismatch_distal')}, so the effect is a "
        f"strong suppression and not silence. "
        f"The weighted sum fires slightly \\emph{{more}} to mismatched "
        f"({pct('sum', 'fire_mismatch_distal')}) than to absent "
        f"({pct('sum', 'fire_absent_distal')}) content. \\textbf{{b}}, The veto "
        f"index, $P(\\text{{fire}} \\mid \\text{{absent}}) - P(\\text{{fire}} "
        f"\\mid \\text{{mismatched}})$, against inhibition strength. Because a "
        f"rectified agreement term cannot be negative, any sum of non-negative "
        f"drives satisfies $P(\\text{{mismatch}}) \\geq P(\\text{{absent}})$ and "
        f"is confined to the unshaded region: the weighted sum never exceeds "
        f"{sum_max:.3f} at any setting, while the conjunctive rule reaches "
        f"{conj_max:.3f}. The sign of this index therefore decides the question "
        f"on its own. \\textbf{{c}}, The cost of suppression. Raising a constant "
        f"$I_\\mathrm{{inh}}$ moves the weighted sum down the diagonal, "
        f"silencing matched and mismatched content together; the conjunctive "
        f"rule moves left along the ceiling, discarding mismatched content "
        f"while retaining every matched one."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

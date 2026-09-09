"""Build the E1 dispersion figure from ``results/e1_dispersion.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
183 mm two-column width, 7 pt maximum text, 8 pt bold lower-case panel letters,
no panel titles, and curves labelled in place rather than through legend boxes.

Panels:

a  Sequence-detector fraction over dispersion x calcium threshold.
b  Detector-class fractions against dispersion, mean and 95% bootstrap CI.
c  Crossover and peak dispersion against inter-event interval, with fits.
d  Sequence-detector fraction against dispersion, one curve per interval.

The description that would otherwise sit above each panel is emitted as a
caption file next to the figure, ready to paste into the manuscript.

Usage::

    python figures/make_fig_e1.py --formats pdf eps tiff png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from delaykey.plotting import (COLUMN_MM, PALETTE, apply_style, direct_label,
                               figure_size, label_panels, save_figure, tint)
from delaykey.stats import bootstrap_mean

# delaykey.plotting selects the Agg backend on import, so these are safe here.
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CLASS_STYLE = {
    "sequence": ("Sequence", PALETTE["vermillion"]),
    "event": ("Event", PALETTE["blue"]),
    "dual": ("Dual", PALETTE["green"]),
    "silent": ("Silent", PALETTE["grey"]),
}


def load(path: Path) -> list[dict]:
    """Read the tidy CSV, coercing numerics and deriving the pooled event class."""
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            row = {
                "dispersion": int(raw["dispersion"]),
                "theta": float(raw["theta"]),
                "interval": int(raw["interval"]),
                "p_connect": float(raw["p_connect"]),
                "seed": int(raw["seed"]),
                "n": int(raw["n_neurons"]),
            }
            for key in ("n_features", "n_per_event", "n_branches"):
                if key in raw:
                    row[key] = int(raw[key])
            for key in ("sequence", "event1", "event2", "dual", "silent", "other"):
                row[key] = int(raw[key]) / row["n"]
            row["event"] = row["event1"] + row["event2"]
            # Raw counts, not fractions: order selectivity is a ratio within the
            # sequence class, so it has to be pooled as counts over seeds rather
            # than averaged as a per-seed ratio, which at the headline threshold
            # is a ratio of single-digit numbers.
            row["n_sequence"] = int(raw["sequence"])
            row["n_order_selective"] = int(raw.get("order_selective", 0))
            rows.append(row)
    return rows


def group(rows, keys, value):
    """Map each combination of ``keys`` to the list of ``value`` across seeds."""
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row[value])
    return out


def peak_dispersion(dispersions, sequence):
    """Dispersion maximising the sequence-detector fraction.

    Refines the grid argmax with a parabola through it and its neighbours, so
    the estimate is not quantised to the sweep step. Returns ``nan`` when the
    maximum lies on a boundary and is therefore outside the swept range.
    """
    sequence = np.asarray(sequence, dtype=float)
    dispersions = np.asarray(dispersions, dtype=float)
    i = int(np.argmax(sequence))
    if i == 0 or i == len(sequence) - 1:
        return np.nan
    y0, y1, y2 = sequence[i - 1], sequence[i], sequence[i + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(dispersions[i])
    step = dispersions[i + 1] - dispersions[i]
    return float(dispersions[i] - 0.5 * step * (y2 - y0) / denom)


def crossover_dispersion(dispersions, sequence, event):
    """Smallest dispersion at which sequence detectors overtake event detectors.

    Linearly interpolates between the bracketing grid points. Returns ``nan``
    when the curves never cross inside the swept range.
    """
    diff = np.asarray(sequence) - np.asarray(event)
    dispersions = np.asarray(dispersions, dtype=float)
    crossings = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0]
    if crossings.size == 0:
        return np.nan
    i = crossings[0]
    span = diff[i + 1] - diff[i]
    if span == 0:
        return float(dispersions[i])
    return float(dispersions[i] - diff[i] * (dispersions[i + 1] - dispersions[i]) / span)


def fit_line(x, y, rng, n_boot: int = 4000):
    """Least-squares line with a bootstrap CI on the slope."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    draws = rng.integers(0, x.size, size=(n_boot, x.size))
    fits = np.array([np.polyfit(x[d], y[d], 1) for d in draws
                     if np.unique(x[d]).size > 1])
    # Every draw is degenerate when x holds a single distinct value, which a
    # reduced grid can produce. The caller already treats None as "no fit".
    if fits.size == 0:
        return None
    ci = tuple(np.quantile(fits[:, 0], [0.025, 0.975]))
    return float(slope), float(intercept), ci


def at(xs, ys, target):
    """Interpolate ``ys`` at ``target`` for placing an in-figure label."""
    return float(np.interp(target, xs, ys))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("results/e1_dispersion.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e1_dispersion"))
    # theta 4.5 is the headline slice: with unit weights and a 15-step plateau it
    # requires five coincident inputs to fire, which is the effective requirement
    # of the prototype notebook (sub-unit weights, single-step plateau). See
    # sec:exp5_dispersion. theta 2.5 requires three and is kept as the
    # permissive comparison.
    parser.add_argument("--theta-main", type=float, default=4.5)
    parser.add_argument("--interval-main", type=int, default=15)
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos",
                        help="target column width: a key of COLUMN_MM "
                             "(plos, single, onehalf, double) or a width in mm")
    parser.add_argument("--height-mm", type=float, default=105.0)
    args = parser.parse_args(argv)

    rows = load(args.csv)
    rng = np.random.default_rng(0)

    dispersions = sorted({r["dispersion"] for r in rows})
    thetas = sorted({r["theta"] for r in rows})
    intervals = sorted({r["interval"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    n_neurons = rows[0]["n"]

    apply_style()
    width = args.width if args.width in COLUMN_MM else float(args.width)
    fig = plt.figure(figsize=figure_size(width, args.height_mm))
    # wspace has to clear panel a's colorbar label as well as panel b's y-label;
    # at the narrower PLOS width those two collide at the default spacing.
    grid = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.52,
                            left=0.095, right=0.980, top=0.925, bottom=0.090)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    # ---- a: phase diagram --------------------------------------------------
    at_interval = [r for r in rows if r["interval"] == args.interval_main]
    cells = group(at_interval, ("theta", "dispersion"), "sequence")
    heat = np.array([[np.mean(cells[(th, d)]) if (th, d) in cells else np.nan
                      for d in dispersions] for th in thetas])

    # White at zero, ramping to the same vermillion that marks sequence
    # detectors in b and c. A perceptual map such as magma puts near-black at
    # zero, turning the large "no sequence detectors" region into a heavy slab
    # that dominates the panel; white lets the informative region carry it.
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sequence", ["#FFFFFF", tint(PALETTE["vermillion"], 0.45),
                     PALETTE["vermillion"], "#7A2E00"])
    # Not rasterised: the mesh is only a few hundred cells, so vector output is
    # both smaller and sharper. Rasterising it at the 600 dpi save resolution
    # inflated the EPS from a few hundred kB to several MB, since EPS stores
    # raster data uncompressed.
    # Log norm: the sequence fraction spans three decades across theta, so on a
    # linear norm the single largest row saturates the map and every higher
    # threshold -- including the headline one -- reads as blank white.
    a_floor = 1.0 / (n_neurons * n_seeds)
    mesh = ax_a.pcolormesh(dispersions, thetas, np.maximum(heat, a_floor),
                           cmap=cmap, shading="nearest", rasterized=False,
                           norm=mcolors.LogNorm(vmin=a_floor,
                                                vmax=float(np.nanmax(heat))))
    bar = fig.colorbar(mesh, ax=ax_a, fraction=0.038, pad=0.02, aspect=18)
    bar.set_label("Sequence fraction", labelpad=1)
    bar.outline.set_linewidth(0.5)
    bar.ax.tick_params(width=0.5, size=1.8, pad=1.5)
    ax_a.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_a.set_ylabel("Threshold $\\theta$")
    ax_a.set_yticks(thetas)
    # Zero maps to white, so without a frame the low-dispersion region is
    # indistinguishable from the page and the panel's extent is invisible.
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    # ---- b: detector classes ----------------------------------------------
    main_rows = [r for r in at_interval if r["theta"] == args.theta_main]
    curves = {}
    for key in ("sequence", "event", "dual", "silent"):
        grouped = group(main_rows, ("dispersion",), key)
        stats = [bootstrap_mean(grouped[(d,)], rng) for d in dispersions]
        curves[key] = (np.array([s.mean for s in stats]),
                       np.array([s.lo for s in stats]),
                       np.array([s.hi for s in stats]))

    # One neuron in the whole pooled sample. Fractions below this are not
    # resolved by the sweep, so the axis stops there rather than running to zero
    # and making the log scale unusable.
    floor = 1.0 / (n_neurons * n_seeds)

    # A linear axis only works when the classes are of comparable size, which
    # they are at low theta. At the headline theta the population is >85%
    # silent and sequence detectors sit near 1e-3, so a linear axis collapses
    # every informative curve onto zero. Log keeps all four legible and leaves
    # the low-theta appearance essentially unchanged.
    for key in ("silent", "event", "dual", "sequence"):
        _, colour = CLASS_STYLE[key]
        mean, lo, hi = curves[key]
        ax_b.fill_between(dispersions, np.maximum(lo, floor),
                          np.maximum(hi, floor), color=tint(colour),
                          linewidth=0, zorder=1)
        ax_b.plot(dispersions, np.maximum(mean, floor), color=colour, zorder=2)

    # Curves are named where they are furthest apart, so no legend is needed.
    # Chosen from the data rather than hardcoded: the class ordering changes
    # completely between thetas, and fixed anchors put labels on top of each
    # other as soon as --theta-main moves.
    keys = ("silent", "event", "dual", "sequence")
    logs = {k: np.log10(np.maximum(curves[k][0], floor)) for k in keys}
    inside = [i for i, d in enumerate(dispersions) if d <= 0.78 * max(dispersions)]
    for key in keys:
        gaps = np.min([np.abs(logs[key] - logs[k]) for k in keys if k != key],
                      axis=0)
        # Never anchor on the resolution floor: a class is flat there, so the
        # gap is largest exactly where the curve carries no information.
        live = [i for i in inside if curves[key][0][i] > floor] or inside
        i = max(live, key=lambda j: gaps[j])
        label, colour = CLASS_STYLE[key]
        direct_label(ax_b, dispersions[i], 10 ** (logs[key][i] + 0.13),
                     label, colour)

    ax_b.axvline(args.interval_main, color="0.35", linestyle=(0, (2, 2)),
                 linewidth=0.9, zorder=0)
    ax_b.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_b.set_ylabel("Fraction of population")
    ax_b.set_yscale("log")
    ax_b.set_ylim(floor / 2.2, 2.6)

    # ---- c: how the transition point scales with the interval -------------
    seeds = sorted({r["seed"] for r in rows})

    def per_seed(interval, fn):
        values = []
        for seed in seeds:
            subset = [r for r in rows
                      if r["interval"] == interval and r["seed"] == seed
                      and r["theta"] == args.theta_main]
            if not subset:
                continue
            by_d = {r["dispersion"]: r for r in subset}
            ordered = [d for d in dispersions if d in by_d]
            value = fn(ordered, by_d)
            if not np.isnan(value):
                values.append(value)
        return values

    metrics = [
        ("Crossover $\\Delta^{*}$", PALETTE["vermillion"], "o",
         lambda o, b: crossover_dispersion(
             o, [b[d]["sequence"] for d in o], [b[d]["event"] for d in o])),
        ("Peak $\\Delta_\\mathrm{opt}$", PALETTE["blue"], "s",
         lambda o, b: peak_dispersion(o, [b[d]["sequence"] for d in o])),
    ]

    # Collected first and drawn second: the label for each fit has to sit inside
    # the axes, and the limits are not known until every series is measured.
    # Placing labels inside the loop clipped the steeper fit off the top.
    fit_notes, top, series = [], 0.0, []
    for label, colour, marker, fn in metrics:
        xs, means, errs, pool_x, pool_y = [], [], [], [], []
        for interval in intervals:
            values = per_seed(interval, fn)
            if len(values) < 2:
                continue
            est = bootstrap_mean(values, rng)
            xs.append(interval); means.append(est.mean)
            errs.append([est.mean - est.lo, est.hi - est.mean])
            pool_x.extend([interval] * len(values)); pool_y.extend(values)
        if not xs:
            continue
        top = max(top, max(means))
        ax_c.errorbar(xs, means, yerr=np.array(errs).T, fmt=marker, color=colour,
                      capsize=1.2, elinewidth=0.6, capthick=0.6, linestyle="none",
                      zorder=3)
        fit = fit_line(pool_x, pool_y, rng)
        if fit is not None:
            slope, intercept, ci = fit
            series.append((label, colour, slope, intercept))
            fit_notes.append((label, slope, ci))

    # Identical ranges on both axes so the identity line runs corner to corner
    # and "above identity" is visible rather than inferred. Forcing an equal
    # aspect as well would shrink the box and leave dead margins, so it is the
    # matched limits, not the aspect, that carry the comparison.
    lim = [0, max(max(intervals), top) * 1.12]
    # The two fits converge toward the top right, so their end labels are pushed
    # apart vertically -- at 8 pt they printed over each other. Which of them
    # goes up is read off the slopes rather than the order of ``metrics``: the
    # steeper fit runs above the shallower one where the labels sit, so a fixed
    # assignment can push both labels inward, into the gap between the lines --
    # and the identity line lies in that gap. Sorting by slope keeps each label
    # on the outside of the pair however the fits come out.
    # A vertical offset alone does not clear the steeper fit: its label runs
    # horizontally while the line climbs through it, which is how "Peak" came
    # to be struck through. The steeper fit is therefore set above and to the
    # left, into the empty wedge its own line leaves; the shallower one keeps
    # the below-right position, clear of both the other fit and the diagonal.
    by_slope = sorted(range(len(series)), key=lambda i: series[i][2], reverse=True)
    label_side = {}
    if len(by_slope) > 1:
        label_side[by_slope[0]] = (lim[1] * 0.055, "right", -1.5)
        label_side[by_slope[-1]] = (-lim[1] * 0.075, "left", 0.0)
    for i, (label, colour, slope, intercept) in enumerate(series):
        # Stop each fit where it would leave the box, so the line and its label
        # stay inside however steep the fit turns out to be.
        end = min(lim[1], (lim[1] * 0.90 - intercept) / slope) if slope > 0 \
            else lim[1]
        line_x = np.array([0.0, end])
        ax_c.plot(line_x, slope * line_x + intercept, color=colour,
                  linewidth=1.2, zorder=2)
        # Anchored inside the line rather than at its end: at 8 pt a label set
        # from the endpoint overflowed the panel into the gutter.
        dy, ha, dx = label_side.get(i, (0.0, "left", 0.0))
        x_lab = end * 0.62
        direct_label(ax_c, x_lab, slope * x_lab + intercept + dy, label,
                     colour, dx=dx, ha=ha)

    ax_c.plot(lim, lim, linestyle=(0, (1, 2)), color="0.55", linewidth=0.6, zorder=1)
    # Offset below the diagonal rather than sitting on it: when a fit comes out
    # near slope 1 it runs along the identity line, and the two labels would
    # otherwise print on top of each other.
    direct_label(ax_c, lim[1] * 0.34, lim[1] * 0.34 - lim[1] * 0.075,
                 "identity", "0.45", dx=1.5)
    ax_c.set_xlim(lim); ax_c.set_ylim(lim)
    ax_c.set_xlabel("Inter-event interval (ms)")
    ax_c.set_ylabel("Dispersion (ms)")

    # ---- d: unimodal tuning, one curve per interval -----------------------
    shades = [PALETTE["sky"], PALETTE["blue"], PALETTE["purple"],
              PALETTE["vermillion"]]
    curves = []
    for i, interval in enumerate(intervals):
        subset = [r for r in rows
                  if r["interval"] == interval and r["theta"] == args.theta_main]
        grouped = group(subset, ("dispersion",), "sequence")
        xs_d = [d for d in dispersions if (d,) in grouped]
        means = np.array([np.mean(grouped[(d,)]) for d in xs_d])
        colour = shades[i % len(shades)]
        ax_d.plot(xs_d, np.maximum(means, floor), color=colour, zorder=2)
        curves.append((interval, xs_d, means, colour))

    # Roughly what a "NN ms" label spans on this axis, as a fraction of the
    # sweep. Used only to sample clearance across the text rather than at its
    # anchor, so it does not have to be exact.
    label_span = 0.17 * (max(dispersions) - min(dispersions))

    def clearance(x, y, ha):
        """Smallest log-distance from the label to any curve, across its width.

        Two things this measures that the obvious version does not. It samples
        over the width the text occupies rather than at its anchor, because the
        label runs horizontally while the curves move underneath it -- a point
        measurement clears the anchor and still lets a curve cross the text.
        And it counts the label's *own* curve, which is the one most likely to
        cross it: the offset is taken from the curve's peak, but the label sits
        beside the peak, where the curve has already fallen away toward it.
        """
        probes = (np.linspace(x, x + label_span, 7) if ha == "left"
                  else np.linspace(x - label_span, x, 7))
        gaps = []
        for _, cx, cm, _ in curves:
            v = np.interp(probes, cx, np.maximum(cm, floor))
            gaps.append(float(np.min(np.abs(np.log10(max(y, floor))
                                            - np.log10(v)))))
        return min(gaps) if gaps else np.inf

    for i, (interval, xs_d, means, colour) in enumerate(curves):
        peak = peak_dispersion(xs_d, means)
        if np.isnan(peak):
            continue
        ax_d.plot([peak], [means.max()], marker="v", color=colour,
                  markersize=3.6, zorder=3)
        # Offset multiplicatively, not additively: on the log axis a fixed
        # additive offset is invisible at the small intervals and enormous at
        # the large ones. Neither side nor alignment is fixed -- the four
        # curves converge at wide dispersion, and at 15 and 20 ms the only
        # clear room is behind the peak rather than ahead of it. The placement
        # with the most room wins.
        above, below = means.max() * 1.7, means.max() / 2.4
        y, ha = max(((y_, ha_) for y_ in (above, below) for ha_ in ("left", "right")),
                    key=lambda o: clearance(peak, o[0], o[1]))
        direct_label(ax_d, peak, y, f"{interval} ms", colour, dx=-3.0, ha=ha)

    ax_d.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_d.set_ylabel("Sequence fraction")
    # Curves for the four intervals differ by two orders of magnitude at the
    # headline theta; on a linear axis only the shortest interval is visible.
    ax_d.set_yscale("log")
    # Headroom for the peak labels, which sit 1.7x above the highest curve.
    ax_d.set_ylim(floor / 2.2, ax_d.get_ylim()[1] * 3.0)

    label_panels([ax_a, ax_b, ax_c, ax_d])

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    # Whether a fit sits above the identity line depends on theta, so the claim
    # is read off the bootstrap CIs rather than asserted. At theta 2.5 both
    # slopes exclude 1; at theta 4.5 the crossover CI covers it, and saying
    # "both lie above identity" there would be false.
    # LaTeX thin-space thousands separator, built here rather than inline: a
    # .replace() on the caption f-string applies to the whole concatenated
    # literal and rewrites every comma in the caption.
    pooled = f"{n_neurons * n_seeds:,}".replace(",", "{,}")

    # Order selectivity, over the span where the sequence class exists at the
    # headline threshold and interval. Conjunctivity is what the classification
    # tests; this is the separate question of whether those neurons reject the
    # reversed order, and it is measured rather than assumed.
    live = [r for r in rows if r["theta"] == args.theta_main
            and r["interval"] == args.interval_main and r["n_sequence"] > 0]
    span = sorted({r["dispersion"] for r in live})
    n_seq_tot = sum(r["n_sequence"] for r in live)
    n_sel_tot = sum(r["n_order_selective"] for r in live)
    order_txt = (
        f"Across the $\\Delta = {min(span)}$--{max(span)}~ms span where the "
        f"class exists, {n_sel_tot} of {n_seq_tot} sequence detectors pooled "
        f"over all {n_seeds} seeds are also silent to the order-reversed "
        f"stimulus ({100 * n_sel_tot / n_seq_tot:.1f}\\%), so the class is "
        f"order-selective and not merely conjunctive."
    ) if n_seq_tot else ""
    above = [f for f in fit_notes if f[2][0] > 1.0]
    covers = [f for f in fit_notes if f[2][0] <= 1.0 <= f[2][1]]
    fit_text = "; ".join(f"{lab}  slope {s:.2f} [{c[0]:.2f}, {c[1]:.2f}]"
                         for lab, s, c in fit_notes)
    if len(above) == len(fit_notes):
        verdict = ("Both scale linearly with the interval but lie above the "
                   f"identity line ({fit_text}), so the transition is not "
                   "simply $\\Delta = $ interval.")
    elif above and covers:
        names = " and ".join(lab for lab, _, _ in covers)
        verdict = (f"Both scale linearly with the interval ({fit_text}). "
                   f"{above[0][0]} lies above the identity line, whereas "
                   f"{names} is statistically indistinguishable from it, so "
                   "sequence detectors overtake event detectors at roughly "
                   "the inter-event interval while the optimum falls beyond it.")
    else:
        verdict = (f"Both scale linearly with the interval ({fit_text}).")

    caption = (
        f"\\textbf{{Delay-key dispersion sets detector class.}} "
        f"Populations of {n_neurons} integrator neurons, one distal branch each, "
        f"receiving random projections (connection probability "
        f"{rows[0]['p_connect']:g}) from a {rows[0].get('n_features', 200)}-neuron "
        f"feature population, of which two disjoint sets of "
        f"{rows[0].get('n_per_event', 40)} neurons form two events inside a gamma "
        f"cycle. Conduction delays are drawn "
        f"uniformly from $[0, \\Delta]$. A neuron is classed a sequence detector "
        f"only if it fires for the two-event stimulus and for neither event "
        f"alone. A fourth stimulus, the order-reversed sequence, is presented "
        f"but takes no part in the classification, so order selectivity is "
        f"measured against it rather than assumed. {order_txt} "
        f"All panels show {n_seeds} independent seeds. "
        f"\\textbf{{a}}, Sequence-detector fraction across dispersion and calcium "
        f"threshold, at an inter-event interval of {args.interval_main} ms. "
        f"\\textbf{{b}}, Detector-class fractions against dispersion "
        f"($\\theta = {args.theta_main:g}$); shading is the 95\\% bootstrap "
        f"confidence interval and the dashed line marks the inter-event interval. "
        f"\\textbf{{c}}, Crossover dispersion $\\Delta^{{*}}$, at which sequence "
        f"detectors overtake event detectors, and peak dispersion "
        f"$\\Delta_\\mathrm{{opt}}$, against inter-event interval; lines are "
        f"least-squares fits and error bars are 95\\% bootstrap confidence "
        f"intervals. {verdict} \\textbf{{d}}, Sequence-detector fraction against "
        f"dispersion for each interval; triangles mark the fitted optimum. "
        f"Tuning is unimodal, and the interval sets where the optimum falls. "
        f"Panels \\textbf{{a}}, \\textbf{{b}} and \\textbf{{d}} use a logarithmic "
        f"scale, on which the floor at $1/{pooled}$ is a single neuron in the "
        f"pooled sample and marks the resolution of the sweep."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

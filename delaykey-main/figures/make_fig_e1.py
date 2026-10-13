"""Build the E1 dispersion figure from ``results/e1_dispersion.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
183 mm two-column width, 7 pt maximum text, 8 pt bold lower-case panel letters,
no panel titles, and curves labelled in place rather than through legend boxes.

Panels:

a  Sequence-detector fraction over dispersion x calcium threshold.
b  Detector-class fractions against dispersion: simulation as points with a
   95% bootstrap band, theory as lines.
c  Crossover and peak dispersion against inter-event interval: estimates from
   the seed-pooled curves with bootstrap intervals, theory as lines.
d  Sequence-detector fraction against dispersion, one interval per colour,
   simulation as points and theory as lines.

The theory is ``results/e1_theory.csv``, written by
``experiments/e1_theory.py`` from :mod:`delaykey.theory`. It has no free
parameter, so where its lines pass through the points the sweep is measuring
the geometry of delayed arrivals and nothing else.

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

from delaykey.plotting import (COLUMN_MM, PALETTE, PT_ANNOT, apply_style,
                               direct_label, figure_size, label_panels,
                               save_figure, tint)
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


def load_theory(path: Path) -> dict:
    """Theory curves keyed by (theta, interval), each an array over dispersion."""
    table = defaultdict(list)
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            table[(float(raw["theta"]), int(raw["interval"]))].append((
                int(raw["dispersion"]), float(raw["sequence"]),
                float(raw["event1"]) + float(raw["event2"]),
                float(raw["dual"]), float(raw["silent"])))
    curves = {}
    for key, entries in table.items():
        entries.sort()
        d, seq, ev, dual, silent = (np.array(c) for c in zip(*entries))
        curves[key] = {"dispersion": d, "sequence": seq, "event": ev,
                       "dual": dual, "silent": silent}
    return curves


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("results/e1_dispersion.csv"))
    parser.add_argument("--theory-csv", type=Path,
                        default=Path("results/e1_theory.csv"))
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
    if not args.theory_csv.exists():
        raise SystemExit(f"{args.theory_csv} not found: run experiments/e1_theory.py")
    theory = load_theory(args.theory_csv)
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
    th_b = theory[(args.theta_main, args.interval_main)]
    th_shown = th_b["dispersion"] <= max(dispersions)

    # One neuron in the whole pooled sample. Fractions below this are not
    # resolved by the sweep, so the axis stops there rather than running to zero
    # and making the log scale unusable.
    floor = 1.0 / (n_neurons * n_seeds)

    # A linear axis only works when the classes are of comparable size, which
    # they are at low theta. At the headline theta the population is >85%
    # silent and sequence detectors sit near 1e-3, so a linear axis collapses
    # every informative curve onto zero. Log keeps all four legible.
    #
    # Theory is the line and simulation the points on it. A theoretical zero
    # has no place on a log axis; a tiny stand-in sends the line out through
    # the bottom of the panel, which is how "none" should look. A simulated
    # zero is not drawn at all -- it is below the resolution floor, not a
    # measurement of zero.
    for key in ("silent", "event", "dual", "sequence"):
        _, colour = CLASS_STYLE[key]
        mean, lo, hi = curves[key]
        ax_b.fill_between(dispersions, np.maximum(lo, floor),
                          np.maximum(hi, floor), color=tint(colour),
                          linewidth=0, zorder=1)
        ax_b.plot(th_b["dispersion"][th_shown],
                  np.maximum(th_b[key][th_shown], 1e-12), color=colour,
                  zorder=2)
        seen = mean > 0
        ax_b.plot(np.asarray(dispersions)[seen], mean[seen], linestyle="none",
                  marker="o", markersize=2.3, color=colour, zorder=3)

    # Curves are named where they are furthest apart, so no legend is needed.
    # Chosen from the curves rather than hardcoded: the class ordering changes
    # completely between thetas, and fixed anchors put labels on top of each
    # other as soon as --theta-main moves.
    keys = ("silent", "event", "dual", "sequence")
    th_at = {k: np.interp(dispersions, th_b["dispersion"], th_b[k]) for k in keys}
    logs = {k: np.log10(np.maximum(th_at[k], floor)) for k in keys}
    inside = [i for i, d in enumerate(dispersions)
              if 0.06 * max(dispersions) <= d <= 0.78 * max(dispersions)]
    for key in keys:
        gaps = np.min([np.abs(logs[key] - logs[k]) for k in keys if k != key],
                      axis=0)
        # Never anchor on the resolution floor: a class is flat there, so the
        # gap is largest exactly where the curve carries no information.
        live = [i for i in inside if th_at[key][i] > floor] or inside
        i = max(live, key=lambda j: gaps[j])
        label, colour = CLASS_STYLE[key]
        lift = -0.30 if key == "silent" else 0.16
        direct_label(ax_b, dispersions[i], 10 ** (logs[key][i] + lift),
                     label, colour)

    # The key for the two marks, in the empty decade under the silent class.
    ax_b.text(0.97, 0.69, "\u2014 theory\n\u2022 simulation", transform=ax_b.transAxes,
              ha="right", va="top", fontsize=PT_ANNOT, color="0.25",
              linespacing=1.25)

    ax_b.axvline(args.interval_main, color="0.35", linestyle=(0, (2, 2)),
                 linewidth=0.9, zorder=0)
    ax_b.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_b.set_ylabel("Fraction of population")
    ax_b.set_yscale("log")
    ax_b.set_ylim(floor / 2.2, 2.6)

    # ---- c: how the transition point scales with the interval -------------
    # Estimated from the seed-pooled curves, not seed by seed. At wide
    # intervals a single population of 1000 holds no sequence detectors near
    # the crossing, so a per-seed crossover lands where the event class runs
    # out -- a zero-against-zero tie at the same dispersion whatever the
    # interval -- and averaging those flattened the slope. Pooling first and
    # bootstrapping the seeds keeps every population's information without
    # asking any one of them to resolve a fraction it cannot.
    seeds = sorted({r["seed"] for r in rows})

    def by_seed(interval, key):
        cell = {(r["seed"], r["dispersion"]): r[key] for r in rows
                if r["interval"] == interval and r["theta"] == args.theta_main}
        return np.array([[cell[(s_, d)] for d in dispersions] for s_ in seeds])

    pools = {iv: (by_seed(iv, "sequence"), by_seed(iv, "event")) for iv in intervals}

    def estimates(idx):
        cross, top_ = [], []
        for iv in intervals:
            seq, ev = pools[iv]
            seq_m, ev_m = seq[idx].mean(axis=0), ev[idx].mean(axis=0)
            cross.append(crossover_dispersion(dispersions, seq_m, ev_m))
            top_.append(peak_dispersion(dispersions, seq_m))
        return np.array([cross, top_])

    point = estimates(np.arange(len(seeds)))
    boot = np.array([estimates(rng.integers(0, len(seeds), len(seeds)))
                     for _ in range(4000)])

    th_ivs = sorted(iv for th, iv in theory if th == args.theta_main)
    th_curve = np.array([[crossover_dispersion(theory[(args.theta_main, iv)]["dispersion"],
                                               theory[(args.theta_main, iv)]["sequence"],
                                               theory[(args.theta_main, iv)]["event"]),
                          peak_dispersion(theory[(args.theta_main, iv)]["dispersion"],
                                          theory[(args.theta_main, iv)]["sequence"])]
                         for iv in th_ivs]).T

    metrics = [("Crossover $\\Delta^{*}$", PALETTE["vermillion"], "o"),
               ("Peak $\\Delta_\\mathrm{opt}$", PALETTE["blue"], "s")]
    fit_notes, top, series = [], 0.0, []
    xs = np.array(intervals, dtype=float)
    for m, (label, colour, marker) in enumerate(metrics):
        ys = point[m]
        lo = np.nanpercentile(boot[:, m, :], 2.5, axis=0)
        hi = np.nanpercentile(boot[:, m, :], 97.5, axis=0)
        ok = ~np.isnan(ys)
        ax_c.errorbar(xs[ok], ys[ok], yerr=np.array([ys - lo, hi - ys])[:, ok],
                      fmt=marker, color=colour, markersize=3.0, capsize=1.2,
                      elinewidth=0.6, capthick=0.6, linestyle="none", zorder=3)
        top = max(top, float(np.nanmax(ys)))
        slope = float(np.polyfit(xs[ok], ys[ok], 1)[0])
        draws = [np.polyfit(xs, b, 1)[0] for b in boot[:, m, :]
                 if not np.any(np.isnan(b))]
        ci = tuple(np.quantile(draws, [0.025, 0.975]))
        th_x = np.array(th_ivs, dtype=float)
        th_y = th_curve[m]
        keep = ~np.isnan(th_y)
        at_sim = np.isin(th_x, xs) & keep
        th_slope = float(np.polyfit(th_x[at_sim], th_y[at_sim], 1)[0])
        series.append((label, colour, th_x[keep], th_y[keep]))
        fit_notes.append((label, slope, ci, th_slope))

    # Identical ranges on both axes so the identity line runs corner to corner
    # and "above identity" is visible rather than inferred. Forcing an equal
    # aspect as well would shrink the box and leave dead margins, so it is the
    # matched limits, not the aspect, that carry the comparison. The widest
    # peak interval runs past the top: at a 20 ms interval the strict gate
    # leaves so few sequence detectors that the peak's interval is long, and
    # sizing the panel to it would squeeze every other point into a corner.
    lim = [0, max(max(intervals), top) * 1.12]
    # Theory as lines, labelled where the steeper one has pulled away from the
    # shallower: above and left of the steeper line, below and right of the
    # shallower, so neither label sits in the gap the identity line runs
    # through.
    steep = max(range(len(series)), key=lambda i: np.polyfit(series[i][2], series[i][3], 1)[0])
    for i, (label, colour, th_x, th_y) in enumerate(series):
        inside_ = (th_y <= lim[1] * 0.95) & (th_x <= lim[1])
        ax_c.plot(th_x[inside_], th_y[inside_], color=colour, linewidth=1.2, zorder=2)
        x_lab = 0.42 * lim[1]
        y_lab = float(np.interp(x_lab, th_x, th_y))
        if i == steep:
            direct_label(ax_c, x_lab, y_lab + lim[1] * 0.06, label, colour,
                         dx=-1.5, ha="right")
        else:
            direct_label(ax_c, x_lab + lim[1] * 0.08, y_lab - lim[1] * 0.07,
                         label, colour, dx=0.0, ha="left")

    ax_c.plot(lim, lim, linestyle=(0, (1, 2)), color="0.55", linewidth=0.6, zorder=1)
    # Offset below the diagonal rather than sitting on it: when a fit comes out
    # near slope 1 it runs along the identity line, and the two labels would
    # otherwise print on top of each other.
    direct_label(ax_c, lim[1] * 0.80, lim[1] * 0.80 - lim[1] * 0.065,
                 "identity", "0.45", dx=0.0, ha="left")
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
        # As in b: theory the line, simulation the points, zeros undrawn.
        th = theory[(args.theta_main, interval)]
        th_keep = th["dispersion"] <= max(dispersions)
        th_x, th_y = th["dispersion"][th_keep], th["sequence"][th_keep]
        ax_d.plot(th_x, np.maximum(th_y, 1e-12), color=colour, zorder=2)
        seen = means > 0
        ax_d.plot(np.asarray(xs_d)[seen], means[seen], linestyle="none",
                  marker="o", markersize=2.0, color=colour, zorder=3)
        curves.append((interval, th_x, th_y, colour,
                       np.asarray(xs_d)[seen], means[seen]))

    # A colour key rather than labels on the curves. The four converge at
    # wide dispersion and the two longest intervals peak beneath the shortest,
    # so no room beside the curves is free of a line or a point; the top right
    # of the panel is empty.
    ax_d.text(0.97, 0.95, "interval", transform=ax_d.transAxes, ha="right",
              va="top", fontsize=PT_ANNOT, color="0.35")
    for k, (interval, cx, cm, colour, _, _) in enumerate(curves):
        top_x = peak_dispersion(cx, cm)
        if not np.isnan(top_x):
            ax_d.plot([top_x], [float(np.interp(top_x, cx, cm))], marker="v",
                      color=colour, markersize=3.6, zorder=4)
        ax_d.text(0.97, 0.95 - 0.085 * (k + 1), f"{interval} ms",
                  transform=ax_d.transAxes, ha="right", va="top",
                  fontsize=PT_ANNOT, color=colour)

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
    (c_lab, c_sim, c_ci, c_th), (p_lab, p_sim, p_ci, p_th) = fit_notes

    # How closely the theory tracks the sweep, over every threshold, interval
    # and dispersion simulated. Restricted to fractions above 0.3% of the
    # population: below that a seed holds a handful of neurons, and a relative
    # difference measures the handful rather than the theory.
    sim_cells = defaultdict(list)
    for r in rows:
        for key in ("sequence", "event", "dual"):
            sim_cells[(r["theta"], r["interval"], r["dispersion"], key)].append(r[key])
    rel = []
    for (th, iv, d, key), vals in sim_cells.items():
        sim_mean = float(np.mean(vals))
        if sim_mean <= 0.003 or (th, iv) not in theory:
            continue
        curve = theory[(th, iv)]
        rel.append(abs(float(np.interp(d, curve["dispersion"], curve[key]))
                       - sim_mean) / sim_mean)
    agreement = (
        f"Over all four thresholds and intervals, theory and simulation differ "
        f"by a median of {100 * np.median(rel):.1f}\\% in the class fractions "
        f"above 0.3\\% of the population ({len(rel)} cells)."
    )
    print(f"  theory vs simulation: median |relative difference| "
          f"{100 * np.median(rel):.2f}% over {len(rel)} cells, "
          f"90th percentile {100 * np.quantile(rel, 0.9):.1f}%")
    slopes = (
        f"The theory gives slopes of {c_th:.2f} for {c_lab} and {p_th:.2f} for "
        f"{p_lab}; least-squares fits to the simulated points give "
        f"{c_sim:.2f} [{c_ci[0]:.2f}, {c_ci[1]:.2f}] and "
        f"{p_sim:.2f} [{p_ci[0]:.2f}, {p_ci[1]:.2f}]."
    )

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
        f"Points are simulation, {n_seeds} independent seeds; lines are the "
        f"analytic theory, computed from the arrival statistics with no free "
        f"parameter. {agreement} "
        f"\\textbf{{a}}, Sequence-detector fraction across dispersion and calcium "
        f"threshold, at an inter-event interval of {args.interval_main} ms. "
        f"\\textbf{{b}}, Detector-class fractions against dispersion "
        f"($\\theta = {args.theta_main:g}$); shading is the 95\\% bootstrap "
        f"confidence interval of the simulated mean, and the dashed line marks "
        f"the inter-event interval. "
        f"\\textbf{{c}}, Crossover dispersion $\\Delta^{{*}}$, at which sequence "
        f"detectors overtake event detectors, and peak dispersion "
        f"$\\Delta_\\mathrm{{opt}}$, against inter-event interval. Points are "
        f"read off the seed-pooled curves, with 95\\% bootstrap confidence "
        f"intervals over seeds. {slopes} "
        f"\\textbf{{d}}, Sequence-detector fraction against "
        f"dispersion for each interval; triangles mark the theoretical optimum. "
        f"Tuning is unimodal, and the interval sets where the optimum falls. "
        f"Panels \\textbf{{a}}, \\textbf{{b}} and \\textbf{{d}} use a logarithmic "
        f"scale, on which the floor at $1/{pooled}$ is a single neuron in the "
        f"pooled sample and marks the resolution of the sweep; simulated "
        f"fractions of zero are not drawn."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

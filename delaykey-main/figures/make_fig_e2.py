"""Build the E2 aliasing figure from ``results/e2_aliasing.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
165.1 mm PLOS width, 8-10 pt type, 10 pt bold lower-case panel letters, no
panel titles, and curves labelled in place rather than through legend boxes.

Panels:

a  Interval tuning of sequence detectors selected at one reference interval,
   probed across every other interval, as a fan over reference intervals at one
   dispersion. The asymmetry is the result: sharp rejection above the
   reference, a broad plateau of false acceptance below it.
b  The same measurement as a map, one row per dispersion at a fixed reference
   interval, so the plateau and the cutoff can be seen to move together with
   the dispersion rather than being asserted to.
c  The largest interval that elicits any response, against delay dispersion, at
   four gating thresholds. This is the bound on the unambiguously codable
   range, and running it across thresholds is what shows the bound to be set by
   the anatomy rather than by the gate.

The version this replaces drew three tuning curves at one dispersion and one
threshold, and three cutoff points at two thresholds. Three points cannot
distinguish a bound that tracks dispersion from two populations that happen to
differ, and a single threshold cannot show whether the bound is anatomical or
an artefact of the gate. Both sweeps are now dense enough to answer the
question they were drawn to answer.

Usage::

    python figures/make_fig_e2.py --formats pdf eps tiff png
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
                               figure_size, label_panels, save_figure, tint)
from delaykey.stats import bootstrap_mean

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RESPONSE_FLOOR = 0.01  # below this a probe interval counts as not encoded

#: Minimum reference detectors, summed over seeds, before a cell is drawn.
#:
#: A dispersion cannot encode an interval much longer than itself -- that is
#: this figure's own result -- so the corners of the (dispersion, reference)
#: grid hold cells with a handful of detectors or none at all, whose response
#: fraction is a ratio of single digits. Dropping them is not tidying: at
#: :math:`\Delta = 20`~ms and a 25~ms reference there are *zero* detectors, and
#: a curve drawn there would be a division by zero rendered as a flat line.
MIN_DETECTOR_TOTAL = 400

#: Threshold colours. Blue is the headline gate and vermillion the strict one
#: throughout the figure set, including the precision figure this one stacks
#: above; the two intermediate thresholds take tints of the same two hues so
#: the series reads in order rather than as four unrelated categories.
THETA_COLOUR = {
    1.5: tint(PALETTE["blue"], 0.42),
    2.5: PALETTE["blue"],
    3.5: tint(PALETTE["vermillion"], 0.48),
    4.5: PALETTE["vermillion"],
}


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "dispersion": int(raw["dispersion"]),
                "theta": float(raw["theta"]),
                "reference": int(raw["reference_interval"]),
                "probe": float(raw["probe_interval"]),
                "seed": int(raw["seed"]),
                "n_reference": int(raw["n_reference"]),
                "response": float(raw["response_fraction"]),
                # The joint statistic; constant within a
                # (dispersion, theta, reference, seed) group by construction.
                "n_all_below": int(raw["n_all_below"]),
            })
    return rows


def group(rows, keys, value):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row[value])
    return out


def cell(rows, dispersion, reference):
    """Rows for one (dispersion, reference) cell, or [] if it is under-powered.

    The reference count is carried once per seed and repeated on every probe
    row, so it is summed over seeds rather than over rows.
    """
    subset = [r for r in rows if r["dispersion"] == dispersion
              and r["reference"] == reference]
    if not subset:
        return []
    per_seed = {r["seed"]: r["n_reference"] for r in subset}
    if sum(per_seed.values()) < MIN_DETECTOR_TOTAL:
        return []
    return subset


def tuning(subset):
    """Mean response fraction against probe interval for one cell."""
    grouped = group(subset, ("probe",), "response")
    probes = sorted({p for (p,) in grouped})
    return probes, np.array([np.mean(grouped[(p,)]) for p in probes])


def reference_colours(references):
    """A sequential ramp over reference interval.

    Reference interval is ordered and there are now enough of them that a
    categorical palette would be read as unrelated conditions. Purple is the
    one Okabe-Ito hue this figure does not already spend on threshold (blue,
    vermillion) or on the response map.
    """
    ramp = LinearSegmentedColormap.from_list(
        "reference", [tint(PALETTE["purple"], 0.45), PALETTE["purple"],
                      (0.33, 0.10, 0.26)])
    lo, hi = min(references), max(references)
    span = (hi - lo) or 1
    return {r: ramp((r - lo) / span) for r in references}


def response_cmap():
    """White-to-blue sequential map for the response fraction in panel b."""
    return LinearSegmentedColormap.from_list(
        "response", ["#FFFFFF", tint(PALETTE["blue"], 0.35), PALETTE["blue"],
                     (0.0, 0.18, 0.30)])


def cutoffs(source, dispersions, rng, references=None):
    """Largest probe interval eliciting any response, per dispersion.

    Pooled over reference intervals and over seeds. A seed whose reference
    interval is not itself encoded contributes nothing: there is no bound to
    read off a population that never detected the sequence it was selected for.

    Args:
        references: the reference intervals to pool over. Panel c passes the
            set common to every threshold arm rather than each arm's own set,
            and the distinction is not cosmetic: detectors selected at a short
            reference are matched on smaller delay differences and carry a
            lower cutoff, so an arm swept over more short references would come
            out with a lower bound for a reason that has nothing to do with its
            threshold. Pooling over a common set is what makes the four series
            in that panel comparable.
    """
    if references is None:
        references = sorted({r["reference"] for r in source})
    out = []
    for disp in dispersions:
        cuts = []
        for seed in sorted({r["seed"] for r in source}):
            for ref in references:
                sub = [r for r in cell(source, disp, ref) if r["seed"] == seed]
                if not sub:
                    continue
                at_ref = [r["response"] for r in sub if r["probe"] == ref]
                if not at_ref or at_ref[0] < 0.5:
                    continue  # reference not encoded; nothing to bound
                live = [r["probe"] for r in sub
                        if r["response"] >= RESPONSE_FLOOR]
                if live:
                    cuts.append(max(live))
        out.append(bootstrap_mean(cuts, rng) if cuts else None)
    return out


def main(argv=None) -> int:
    global MIN_DETECTOR_TOTAL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("results/e2_aliasing.csv"))
    # Panel c plots one series per gating threshold. Each arm is a separate run
    # -- the populations have to be enlarged as the gate tightens, or the
    # detector count falls through the floor -- so each arrives as its own CSV
    # and any that are absent are simply not drawn.
    parser.add_argument("--csv-extra", type=Path, nargs="*", default=[
        Path("results/e2_aliasing_theta15.csv"),
        Path("results/e2_aliasing_theta35.csv"),
        Path("results/e2_aliasing_theta45.csv"),
    ])
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e2_aliasing"))
    parser.add_argument("--dispersion-main", type=int, default=20,
                        help="the dispersion whose interval tuning panel a draws")
    parser.add_argument("--map-reference", type=int, default=8,
                        help="the reference interval panel b maps across "
                             "dispersions. 8 ms is the longest one every "
                             "dispersion in the sweep can still encode, so the "
                             "map keeps all ten rows; at 10 ms the narrowest "
                             "population drops out and the row where the "
                             "cutoff is tightest is the one lost")
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    parser.add_argument("--height-mm", type=float, default=64.0)
    parser.add_argument("--panel-letters", nargs=3, default=["a", "b", "c"])
    parser.add_argument("--min-detectors", type=int, default=MIN_DETECTOR_TOTAL,
                        help="the detector floor below which a cell is not "
                             "drawn; run_all.sh --quick lowers it, because the "
                             "smoke-test grid never reaches the default")
    args = parser.parse_args(argv)
    MIN_DETECTOR_TOTAL = args.min_detectors

    rows = load(args.csv)
    extra = [(p, load(p)) for p in args.csv_extra if p.exists()]
    rng = np.random.default_rng(0)

    dispersions = sorted({r["dispersion"] for r in rows})
    references = sorted({r["reference"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})
    theta_main = sorted({r["theta"] for r in rows})[0]

    series = [(theta_main, rows)] + [
        (sorted({r["theta"] for r in body})[0], body) for _, body in extra]
    # The reference intervals every threshold arm has in common. Panel c pools
    # over these, and the caption's plateau and joint statistics are quoted over
    # them too. The second use matters as much as the first: both statistics
    # average over the probe intervals *below* the reference, so a 4 ms
    # reference averages four probes and a 20 ms reference twenty. Quoting a
    # range across references that differ that much in support would widen it
    # for a reason that has nothing to do with aliasing -- and would widen the
    # joint statistic most, which is the one the argument turns on.
    common_refs = sorted(set.intersection(
        *({r["reference"] for r in body} for _, body in series)))

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figure_size(args.width, args.height_mm),
                             gridspec_kw={"width_ratios": [1.12, 1.0, 0.98]})
    ax_a, ax_b, ax_c = axes
    # Margins in millimetres, matched to make_fig_e3.py and make_fig_e6.py so
    # the stacked halves of the manuscript's composite floats share one axis
    # origin rather than reading as two figures glued together.
    left_mm, trim_mm, width_mm = 17.5, 1.9, 165.1
    fig.subplots_adjust(wspace=0.50, left=left_mm / width_mm,
                        right=(width_mm - trim_mm) / width_mm,
                        bottom=0.21, top=0.895)  # headroom for 10 pt letters

    # -- a, the fan of tuning curves ---------------------------------------
    drawn_refs = [ref for ref in references
                  if cell(rows, args.dispersion_main, ref)]
    ref_colour = reference_colours(drawn_refs)
    plateaus, joints = {}, {}
    for ref in drawn_refs:
        subset = cell(rows, args.dispersion_main, ref)
        probes, means = tuning(subset)
        colour = ref_colour[ref]
        ax_a.plot(probes, means, color=colour, zorder=2, linewidth=1.2)
        ax_a.plot([ref], [means[probes.index(float(ref))]], marker="v",
                  color=colour, markersize=3.4, zorder=3)
        if ref not in common_refs:
            continue
        plateaus[ref] = float(np.mean([m for p, m in zip(probes, means) if p < ref]))
        # Pooled over seeds rather than averaged per seed: a single population
        # contributes few detectors at the longer references, and a ratio of
        # small counts is better formed once over the pool.
        per_seed = {r["seed"]: (r["n_all_below"], r["n_reference"]) for r in subset}
        n_all = sum(a for a, b in per_seed.values() if b)
        n_ref_tot = sum(b for _, b in per_seed.values())
        joints[ref] = (n_all / n_ref_tot) if n_ref_tot else float("nan")

    ax_a.set_xlabel("Probe inter-event interval (ms)")
    ax_a.set_ylabel("Fraction of detectors responding")
    ax_a.set_xlim(-1, max(drawn_refs) + 11)
    ax_a.set_ylim(-0.02, 1.10)
    # The panel carries one word of key and no more. Naming each curve is what
    # the three-curve version did; nine curves whose peaks stand 2 ms apart
    # would need nine labels 2 mm apart. Nor is there room to spell out what
    # the triangles mean: the fan fills the upper half, the plateau the middle,
    # and the descending tails sweep the lower left, which leaves only the
    # strip beyond the last cutoff -- wide enough for the dispersion and
    # nothing else. The triangles are read off the axis (each curve peaks at
    # its own reference) and named in the caption.
    ax_a.annotate(f"$\\Delta = {args.dispersion_main}$ ms",
                  xy=(0.98, 0.62), xycoords="axes fraction", fontsize=PT_ANNOT,
                  ha="right", va="center", color=PALETTE["black"])

    # -- b, the same measurement across dispersions ------------------------
    map_rows = [d for d in dispersions if cell(rows, d, args.map_reference)]
    probe_axis = sorted({r["probe"] for r in rows
                         if r["dispersion"] == map_rows[-1]
                         and r["reference"] == args.map_reference})
    x_max = min(max(map_rows) + 12, max(probe_axis))
    probe_axis = [p for p in probe_axis if p <= x_max]
    grid = np.full((len(map_rows), len(probe_axis)), np.nan)
    for i, disp in enumerate(map_rows):
        probes, means = tuning(cell(rows, disp, args.map_reference))
        lookup = dict(zip(probes, means))
        grid[i] = [lookup.get(p, np.nan) for p in probe_axis]

    # Row edges midway between the sampled dispersions, so a row's height is
    # the interval it stands for and the axis stays linear in dispersion even
    # where the sweep is not evenly spaced.
    y_edges = [map_rows[0] - (map_rows[1] - map_rows[0]) / 2.0]
    for lo, hi in zip(map_rows, map_rows[1:]):
        y_edges.append((lo + hi) / 2.0)
    y_edges.append(map_rows[-1] + (map_rows[-1] - map_rows[-2]) / 2.0)
    x_edges = [probe_axis[0] - 0.5] + [p + 0.5 for p in probe_axis]
    mesh = ax_b.pcolormesh(x_edges, y_edges, grid, cmap=response_cmap(),
                           vmin=0.0, vmax=1.0, rasterized=True, zorder=1)
    # The reference the row was selected at, and the identity line: everything
    # right of the reference is aliasing upward, everything left of it
    # downward, and the cutoff is where the row goes white.
    ax_b.axvline(args.map_reference, color=PALETTE["black"], linewidth=0.9,
                 linestyle=(0, (2.6, 1.8)), zorder=3)
    ax_b.plot([y_edges[0], y_edges[-1]], [y_edges[0], y_edges[-1]],
              color=PALETTE["black"], linewidth=0.9, linestyle=(0, (1.2, 1.4)),
              zorder=3)
    ax_b.annotate("$\\Delta$", xy=(y_edges[-1] - 1.0, y_edges[-1] - 3.0),
                  fontsize=PT_ANNOT, ha="right", va="top",
                  color=PALETTE["black"])
    ax_b.annotate("reference", xy=(args.map_reference + 1.2, y_edges[-1] - 1.4),
                  fontsize=PT_ANNOT, ha="left", va="top", color=PALETTE["black"])
    ax_b.set_xlabel("Probe inter-event interval (ms)")
    ax_b.set_ylabel("Delay dispersion $\\Delta$ (ms)")
    ax_b.set_yticks(map_rows[::2])

    # The colour bar sits inside the map, in the corner the data leaves empty:
    # long probes at short dispersions are exactly the combination nothing
    # responds to, so that corner is white by construction. In the gutter
    # instead -- the obvious place -- the bar and its rotated label ran into
    # panel c's y-axis label, and widening the gutter enough to separate them
    # took the width out of the three panels.
    cax = ax_b.inset_axes([0.50, 0.115, 0.45, 0.052])
    bar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                       ticks=[0.0, 0.5, 1.0])
    bar.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=PT_ANNOT, length=1.6, pad=1.4)
    cax.text(0.5, 1.9, "fraction responding", transform=cax.transAxes,
             fontsize=PT_ANNOT, ha="center", va="bottom",
             color=PALETTE["black"])

    # -- c, the bound, at every threshold measured --------------------------
    span = np.array([dispersions[0] - 4, dispersions[-1] + 4], dtype=float)
    # The identity line is the reference the claim is read against, so it is
    # drawn across the whole panel rather than only under the data.
    ax_c.plot(span, span, color=PALETTE["black"], linewidth=0.9,
              linestyle=(0, (3, 2)), zorder=1)
    direct_label(ax_c, span[0] + 2, span[0] + 2, "identity", PALETTE["black"],
                 dx=1.5)

    covered = {}
    for theta, body in sorted(series):
        disp_here = sorted({r["dispersion"] for r in body})
        est = cutoffs(body, disp_here, rng, references=common_refs)
        pts = [(x, e) for x, e in zip(disp_here, est) if e is not None]
        if not pts:
            continue
        # How far each arm reaches. The strict gate stops short, and for a
        # reason the reader has to be told rather than left to infer from a
        # curve that ends early: a threshold demanding five simultaneous
        # arrivals leaves too few detectors in a wide population to form a
        # response fraction at all, so those cells fall under the floor.
        covered[theta] = (pts[0][0], pts[-1][0])
        px = np.array([p[0] for p in pts])
        means = np.array([p[1].mean for p in pts])
        lo = np.array([p[1].lo for p in pts])
        hi = np.array([p[1].hi for p in pts])
        colour = THETA_COLOUR.get(theta, PALETTE["grey"])
        ax_c.errorbar(px, means, yerr=[means - lo, hi - means], color=colour,
                      marker="o", markersize=3.0, linewidth=1.2, capsize=1.8,
                      zorder=3)
        # Labelled off the right-hand end, staggered by threshold so four
        # labels on four converging series stay apart.
        dy = 3.2 * (2.5 - theta)
        direct_label(ax_c, px[-1], means[-1] + dy, f"$\\theta = {theta:g}$",
                     colour, dx=1.0)

    ax_c.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_c.set_ylabel("Largest encoded interval (ms)")
    ax_c.set_xlim(span[0], span[-1] + 9)
    ax_c.set_ylim(0, span[-1] + 6)

    label_panels(list(axes), labels=list(args.panel_letters))

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")

    pa, pb, pc = args.panel_letters
    plateau_lo, plateau_hi = min(plateaus.values()), max(plateaus.values())
    joint_lo, joint_hi = min(joints.values()), max(joints.values())
    theta_all = sorted(t for t, _ in series)
    # The bound, at the headline threshold, for the caption.
    main_cuts = cutoffs(rows, dispersions, rng, references=common_refs)
    cuts_txt = ", ".join(
        f"{d}~ms $\\rightarrow$ {e.mean:.0f}~ms [{e.lo:.0f}, {e.hi:.0f}]"
        for d, e in zip(dispersions, main_cuts) if e is not None)
    print(f"  bound at theta {theta_main:g}: {cuts_txt}")
    caption = (
        f"\\textbf{{A delay configuration selects an equivalence class of "
        f"intervals, and the dispersion bounds it.}} Sequence detectors were "
        f"identified at one reference inter-event interval and then probed with "
        f"the same stimulus retimed to every other interval; feature identity "
        f"and within-volley timing are held fixed, so only the separation "
        f"changes. {n_seeds} independent seeds. \\textbf{{{pa}}}, Interval "
        f"tuning at a dispersion of {args.dispersion_main}~ms, one curve per "
        f"reference interval from {drawn_refs[0]} to {drawn_refs[-1]}~ms "
        f"(triangles mark each curve's own reference). Selectivity is strongly "
        f"asymmetric. Above the reference the response collapses within about a "
        f"millisecond and reaches zero within a few. Below it the curves settle "
        f"onto a broad plateau: averaged over the probe intervals shorter than "
        f"the reference, {100 * plateau_lo:.0f}--{100 * plateau_hi:.0f}\\% of "
        f"detectors respond at any \\emph{{given}} shorter interval. That is a "
        f"mean of marginals and should not be read as a joint statement --- the "
        f"fraction responding at \\emph{{every}} shorter interval tested is far "
        f"smaller, {100 * joint_lo:.1f}--{100 * joint_hi:.1f}\\%. Both ranges "
        f"are quoted over the {', '.join(str(r) for r in common_refs)}~ms "
        f"references, whose sub-reference probe sets are comparable in size. "
        f"Aliasing is "
        f"therefore broad but not universal: a detector accepts many shorter "
        f"intervals, not all of them. This is the aliasing that Experiment~5 "
        f"reports as temporal folding. \\textbf{{{pb}}}, The same measurement "
        f"across dispersions, at a reference of {args.map_reference}~ms: each row "
        f"is one population's tuning curve rendered as colour, so the plateau "
        f"below the reference and the cutoff above it can be watched moving "
        f"with $\\Delta$. The cutoff edge tracks the dotted identity line, "
        f"which is the bound quantified in \\textbf{{{pc}}}. \\textbf{{{pc}}}, "
        f"The largest interval eliciting any response ($\\geq "
        f"{RESPONSE_FLOOR:g}$ of detectors), against delay dispersion, pooled "
        f"over the reference intervals every threshold arm has in common "
        f"({', '.join(str(r) for r in common_refs)}~ms), with 95\\% bootstrap "
        f"confidence intervals. "
        f"At $\\theta = {theta_main:g}$: {cuts_txt}. The bound tracks the "
        f"dispersion and sits just above the identity line: delays drawn from "
        f"$[0, \\Delta]$ cannot compensate a separation longer than roughly "
        f"$\\Delta$. Running the same sweep at "
        f"$\\theta = {', '.join(f'{t:g}' for t in theta_all)}$ moves it very "
        f"little, so the codable range is set by the anatomy rather than by the "
        f"gate: raising the threshold discards detectors, but the ones that "
        f"remain are bounded by the same delays. The strict arm is the shorter "
        f"one, reaching $\\Delta = {covered[theta_all[-1]][1]}$~ms against "
        f"{covered[theta_main][1]}~ms for the others; beyond that a gate "
        f"demanding five simultaneous arrivals leaves too few detectors in a "
        f"wide population to form a response fraction at all, which is a limit "
        f"on what can be measured there and not a bound that has moved. "
        f"For the 20--30~ms dispersion "
        f"of unmyelinated horizontal fibres the range coincides with one gamma "
        f"cycle."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

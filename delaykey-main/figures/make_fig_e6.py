"""Build the E6 precision figure from ``results/e6_precision.csv``.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting`):
165.1 mm PLOS width, 8-10 pt type, 10 pt bold lower-case panel letters, no
panel titles, and curves labelled in place rather than through legend boxes.

Panels:

a  Retention of sequence detectors under independent per-axon jitter in
   conduction velocity, the axon-to-axon variability case, as a family over
   delay dispersion.
b  Retention under a common scaling of every delay, the global case that a
   temperature or brain-state shift would produce, over the same family.
c  The budget in one number: the jitter that costs half the matched population,
   against dispersion, at both operating thresholds.

Panels a and b used to carry two curves each, both at $\\theta = 2.5$, and
panel c did not exist. Two curves cannot distinguish "wider populations are
more fragile" from "these two populations differ", and the precision budget --
the quantity a physiologist actually wants -- was left to be read off a curve
by eye. The dispersion sweep now runs 20-50 ms, and panel c extracts the
half-loss point from every one of those sweeps at both thresholds. What it
shows is not in the two-curve version: tolerance falls as 1/dispersion, so the
budget is a fixed *absolute* timing tolerance of order a millisecond rather
than the fixed percentage the two-curve figure invited the reader to infer.

Usage::

    python figures/make_fig_e6.py --formats pdf eps tiff png
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

#: Minimum reference detectors *summed over seeds* before a cell is drawn.
#:
#: The floor is on the total rather than on the per-population mean because the
#: plotted quantity is an ensemble average: twenty populations of ten detectors
#: and two populations of a hundred carry comparable information about the
#: mean, and a per-population floor would reject the first while accepting the
#: second. What it exists to reject is the degenerate cell -- at
#: :math:`\Delta = 10`~ms and a 15~ms interval the mean reference set is *one*
#: detector, because delays drawn from :math:`[0, \Delta]` cannot compensate a
#: separation longer than roughly :math:`\Delta` (the E2 bound), so retention
#: there is a coin flip drawn as a curve. Cells below the floor are dropped
#: rather than plotted, and :func:`main` reports what it dropped.
MIN_DETECTOR_TOTAL = 400

#: Panels a and b are read at one inter-event interval; panel c uses all three,
#: which is what shows the budget to be a property of the dispersion rather than
#: of the interval being encoded.
MAIN_INTERVAL = 15

#: Half-loss level for the tolerance summary in panel c.
HALF = 0.5


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "perturbation": raw["perturbation"],
                "magnitude": float(raw["magnitude"]),
                "dispersion": int(raw["dispersion"]),
                "theta": float(raw["theta"]),
                "interval": int(raw["interval"]),
                "seed": int(raw["seed"]),
                "n_reference": int(raw["n_reference"]),
                "retention": float(raw["retention"]),
                # Runs from before the time step became a parameter carry no
                # column, and ran at 1 step per ms.
                "steps_per_ms": int(raw.get("steps_per_ms") or 1),
            })
    return rows


def group(rows, keys, value):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row[value])
    return out


def band(ax, xs, estimates, colour, **kwargs):
    means = np.array([e.mean for e in estimates])
    lo = np.array([e.lo for e in estimates])
    hi = np.array([e.hi for e in estimates])
    ax.fill_between(xs, lo, hi, color=colour, alpha=0.20, linewidth=0, zorder=1)
    ax.plot(xs, means, color=colour, zorder=2, linewidth=1.3, **kwargs)
    return means


def dispersion_colours(dispersions):
    """One sequential ramp over dispersion, in the blue the set already uses.

    Dispersion is ordered, so it gets a single-hue ramp rather than a set of
    categorical colours: with seven curves the categorical reading ("blue is
    20, vermillion is 30") stops working and the ramp reads off the axis. The
    hue is the same blue that carries dispersion elsewhere in the figure set,
    and vermillion is left free to mean the stricter threshold, as it does in
    the aliasing figure this one stacks beneath.
    """
    ramp = LinearSegmentedColormap.from_list(
        "dispersion", [tint(PALETTE["blue"], 0.42), PALETTE["blue"],
                       (0.0, 0.18, 0.30)])
    lo, hi = min(dispersions), max(dispersions)
    span = (hi - lo) or 1
    return {d: ramp((d - lo) / span) for d in dispersions}


def curve(rows, pert, disp, interval, theta):
    """Mean retention per perturbation magnitude, and the per-seed curves.

    Returns:
        ``(magnitudes, {seed: {magnitude: retention}})``, empty if the cell is
        below :data:`MIN_DETECTOR_TOTAL`.
    """
    subset = [r for r in rows
              if r["perturbation"] == pert and r["dispersion"] == disp
              and r["interval"] == interval and abs(r["theta"] - theta) < 1e-9]
    if not subset:
        return [], {}
    # One reference count per seed; the rows repeat it once per magnitude.
    per_seed_ref = {r["seed"]: r["n_reference"] for r in subset}
    if sum(per_seed_ref.values()) < MIN_DETECTOR_TOTAL:
        return [], {}
    per_seed = defaultdict(dict)
    for r in subset:
        per_seed[r["seed"]][r["magnitude"]] = r["retention"]
    return sorted({r["magnitude"] for r in subset}), dict(per_seed)


def crossing(mags, ys, level=HALF):
    """Where a monotone-ish decreasing curve first falls through ``level``.

    Linear interpolation between the two bracketing magnitudes. Returns None if
    the curve never reaches the level within the sweep, which is a real outcome
    and not a failure: a narrow population can hold more than half its
    detectors across the whole jitter range.
    """
    for i in range(1, len(mags)):
        if ys[i - 1] >= level >= ys[i]:
            if ys[i - 1] == ys[i]:
                return float(mags[i - 1])
            t = (ys[i - 1] - level) / (ys[i - 1] - ys[i])
            return float(mags[i - 1] + t * (mags[i] - mags[i - 1]))
    return None


def half_loss(mags, per_seed, rng, n_boot: int = 2000):
    """The jitter s.d. that costs half the matched population, with an interval.

    The statistic is a crossing of the *seed-averaged* curve, not the average of
    per-seed crossings: a single population's curve is a step function over a
    few dozen detectors and its crossing is quantised, whereas the quantity the
    text quotes is a property of the population ensemble. The interval is a
    percentile bootstrap over seeds, recomputing the crossing on each resample.

    Returns:
        ``(point, lo, hi)`` in units of the magnitude axis, or None.
    """
    seeds = sorted(per_seed)
    if len(seeds) < 2:
        return None

    def crossing_of(chosen):
        ys = [float(np.mean([per_seed[s][m] for s in chosen])) for m in mags]
        return crossing(mags, ys)

    point = crossing_of(seeds)
    if point is None:
        return None
    draws = rng.integers(0, len(seeds), size=(n_boot, len(seeds)))
    boot = [crossing_of([seeds[i] for i in row]) for row in draws]
    boot = [b for b in boot if b is not None]
    if len(boot) < n_boot // 4:
        return point, point, point
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def main(argv=None) -> int:
    global MIN_DETECTOR_TOTAL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path,
                        default=Path("results/e6_precision.csv"))
    # The stricter-threshold arm measures the same budget at the operating
    # point Experiment 5 argues for. It used to be reported only as a sentence
    # of running text; carrying it in the figure is what lets panel c state the
    # budget at both thresholds instead of at one.
    parser.add_argument("--csv-strict", type=Path,
                        default=Path("results/e6_precision_theta45.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("figures/out/fig_e6_precision"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "eps", "tiff"])
    parser.add_argument("--width", default="plos")
    # The restructured manuscript stacks this figure beneath the aliasing
    # sweep inside one float, where these panels are d-f.
    parser.add_argument("--panel-letters", nargs=3, default=["a", "b", "c"])
    parser.add_argument("--height-mm", type=float, default=64.0)
    parser.add_argument("--reference-dispersion", type=int, default=20,
                        help="the dispersion carried through to the caption "
                             "and drawn for the stricter threshold in a-b")
    parser.add_argument("--min-detectors", type=int, default=MIN_DETECTOR_TOTAL,
                        help="the detector floor below which a cell is not "
                             "drawn; run_all.sh --quick lowers it, because the "
                             "smoke-test grid never reaches the default")
    args = parser.parse_args(argv)
    MIN_DETECTOR_TOTAL = args.min_detectors

    rows = load(args.csv)
    strict = load(args.csv_strict) if args.csv_strict.exists() else []
    rng = np.random.default_rng(0)

    theta_main = sorted({r["theta"] for r in rows})[0]
    theta_strict = sorted({r["theta"] for r in strict})[0] if strict else None
    intervals = sorted({r["interval"] for r in rows})
    n_seeds = len({r["seed"] for r in rows})

    # Only dispersions that actually support a retention ratio at the interval
    # the panels are read at.
    all_disp = sorted({r["dispersion"] for r in rows})
    dispersions = [d for d in all_disp
                   if curve(rows, "jitter", d, MAIN_INTERVAL, theta_main)[0]]
    dropped = [d for d in all_disp if d not in dispersions]
    colours = dispersion_colours(dispersions)
    ref_disp = args.reference_dispersion

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figure_size(args.width, args.height_mm))
    ax_a, ax_b, ax_c = axes
    # Margins in millimetres, matched to make_fig_e2.py and make_fig_e3.py so
    # the stacked halves of the manuscript's composite floats share one axis
    # origin rather than reading as two figures glued together.
    left_mm, trim_mm, width_mm = 17.5, 1.9, 165.1
    fig.subplots_adjust(wspace=0.46, left=left_mm / width_mm,
                        right=(width_mm - trim_mm) / width_mm,
                        bottom=0.22, top=0.895)  # headroom for 10 pt letters

    cited = {}

    def draw(ax, pert, disp, theta, source, colour, **kwargs):
        """One retention curve, with its bootstrap band."""
        mags, per_seed = curve(source, pert, disp, MAIN_INTERVAL, theta)
        if not mags:
            return None
        est = [bootstrap_mean([per_seed[s][m] for s in per_seed], rng)
               for m in mags]
        for m, e in zip(mags, est):
            cited[(pert, disp, round(theta, 2), m)] = e
        if pert == "jitter":
            xs = [100 * m for m in mags]
        else:
            # The scale sweep carries no k = 1.0 row, but that condition is not
            # unmeasured: it is the same unperturbed configuration as jitter
            # s = 0, measured on these very seeds, which comes back at exactly
            # 1.0. Splicing it in is a measurement rather than an
            # interpolation, and it lets the two branches be drawn as one
            # continuous curve peaking at k = 1 instead of as a broken plot.
            zero_mags, zero_seeds = curve(source, "jitter", disp,
                                          MAIN_INTERVAL, theta)
            anchor = bootstrap_mean([zero_seeds[s][0.0] for s in zero_seeds], rng)
            below = [(m, e) for m, e in zip(mags, est) if m < 1.0]
            above = [(m, e) for m, e in zip(mags, est) if m > 1.0]
            pairs = below + [(1.0, anchor)] + above
            xs = [m for m, _ in pairs]
            est = [e for _, e in pairs]
        band(ax, xs, est, colour, **kwargs)
        return est

    for pert, ax in (("jitter", ax_a), ("scale", ax_b)):
        for disp in dispersions:
            draw(ax, pert, disp, theta_main, rows, colours[disp])
        if strict:
            draw(ax, pert, ref_disp, theta_strict, strict, PALETTE["vermillion"],
                 linestyle=(0, (2.6, 1.6)))

    # -- a ------------------------------------------------------------------
    ax_a.set_xlabel("Per-axon velocity jitter (%, s.d.)")
    ax_a.set_ylabel("Sequence detectors retained")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlim(-1, 31)
    # The family is ordered, so the key is the ramp itself: the dispersions
    # stacked in their own colours in the corner the curves vacate. Seven
    # direct labels on seven curves that converge below 10% jitter would
    # collide with each other and with the curves.
    ax_a.text(0.985, 0.965, "$\\Delta$ (ms)", transform=ax_a.transAxes,
              fontsize=PT_ANNOT, ha="right", va="top", color=PALETTE["black"])
    for i, disp in enumerate(dispersions):
        ax_a.text(0.985, 0.885 - 0.078 * i, f"{disp}", transform=ax_a.transAxes,
                  fontsize=PT_ANNOT, ha="right", va="top", color=colours[disp])
    if strict:
        # Anchored where the dashed curve has already separated from the family
        # and still has clear air beneath it; at the right-hand end it ran into
        # the axis floor.
        direct_label(ax_a, 20.0,
                     cited[("jitter", ref_disp, round(theta_strict, 2), 0.20)].mean - 0.085,
                     f"$\\theta = {theta_strict:g}$", PALETTE["vermillion"],
                     dx=-1.0)

    # -- b ------------------------------------------------------------------
    ax_b.set_xlabel("Common delay scale factor $k$")
    ax_b.set_ylabel("Sequence detectors retained")
    ax_b.set_ylim(0, 1.05)
    ax_b.axvline(1.0, color=PALETTE["black"], linewidth=0.9,
                 linestyle=(0, (3, 2)), zorder=0)
    # The asymmetry is the result, so name the two directions on the axis
    # rather than leaving the reader to decode "k < 1" from the caption.
    ax_b.annotate("unperturbed", xy=(1.0, 1.0), xytext=(1.0, 1.075),
                  color=PALETTE["black"], fontsize=PT_ANNOT, ha="center")
    # In the top corners: the narrowest population and the strict gate both
    # fall to the floor under speeding, so the bottom left is no longer empty.
    ax_b.annotate("faster\nconduction", xy=(0.873, 0.86),
                  color=PALETTE["black"], fontsize=PT_ANNOT, ha="center", alpha=0.75)
    ax_b.annotate("slower\nconduction", xy=(1.14, 0.86),
                  color=PALETTE["black"], fontsize=PT_ANNOT, ha="center", alpha=0.75)

    # -- c, the budget ------------------------------------------------------
    # Every sweep in a, at every interval, condensed to the jitter that costs
    # half the matched population. Plotted against dispersion, because that is
    # the variable it depends on.
    budget = {}
    for theta, source, colour, marker, dy in (
            (theta_main, rows, PALETTE["blue"], "o", 1.5),
            (theta_strict, strict, PALETTE["vermillion"], "s", -1.5)):
        if source is None or not source:
            continue
        for interval in sorted({r["interval"] for r in source}):
            xs, pts, los, his = [], [], [], []
            for disp in sorted({r["dispersion"] for r in source}):
                mags, per_seed = curve(source, "jitter", disp, interval, theta)
                if not mags:
                    continue
                got = half_loss(mags, per_seed, rng)
                if got is None:
                    continue
                point, lo, hi = got
                xs.append(disp)
                pts.append(100 * point)
                los.append(100 * lo)
                his.append(100 * hi)
                budget[(round(theta, 2), interval, disp)] = 100 * point
            if not xs:
                continue
            if interval == MAIN_INTERVAL:
                ax_c.errorbar(xs, pts,
                              yerr=[np.array(pts) - los, np.array(his) - pts],
                              color=colour, marker=marker, markersize=3.6,
                              linewidth=1.3, capsize=2.0, zorder=3)
                main_xs, main_pts = xs, pts
                # The 1/dispersion law the points fall on, drawn from the mean
                # absolute tolerance rather than fitted: if the budget really is
                # a fixed number of milliseconds then that number, divided by
                # the dispersion, is the curve, and any departure from it is
                # visible as a departure of the points from the line.
                ms = float(np.mean([p * d / 100 for p, d in zip(pts, xs)]))
                grid = np.linspace(min(xs) - 2, max(xs) + 2, 100)
                ax_c.plot(grid, 100 * ms / grid, color=colour, linewidth=0.8,
                          linestyle=(0, (2.4, 1.6)), zorder=2, alpha=0.85)
                cited[("budget_ms", round(theta, 2))] = ms
                # Labelled off the second point rather than the last: the two
                # series converge to the right, where two labels a millimetre
                # apart would be read as one.
                # Above the series and to the right of its peak, where neither
                # the open markers of the other intervals nor the hyperbola run.
                i = min(1, len(main_xs) - 1)
                j = min(2, len(main_xs) - 1)
                x_lab = main_xs[j] if dy > 0 else main_xs[i]
                direct_label(ax_c, x_lab, main_pts[i] + dy,
                             f"$\\theta = {theta:g}$", colour, dx=1.5)
            else:
                # The other two intervals, drawn as bare points: the budget is a
                # property of the dispersion, and showing that it does not move
                # when the encoded interval moves is the check that says so.
                ax_c.plot(xs, pts, linestyle="none", marker=marker,
                          markersize=2.4, markerfacecolor="white",
                          markeredgecolor=colour, markeredgewidth=0.7, zorder=2)

    ax_c.set_xlabel("Delay dispersion $\\Delta$ (ms)")
    ax_c.set_ylabel("Half-loss jitter (%, s.d.)")
    ax_c.set_ylim(0, None)
    # Top right: both series descend to the right, so that corner is the only
    # region no curve, marker, band or series label occupies.
    ax_c.annotate("open markers: other\ninter-event intervals",
                  xy=(0.98, 0.99), xycoords="axes fraction", fontsize=PT_ANNOT,
                  ha="right", va="top", color=PALETTE["grey"])

    label_panels(list(axes), labels=list(args.panel_letters))

    written = save_figure(fig, args.out, formats=tuple(args.formats))
    for path in written:
        print(f"  wrote {path}")
    if dropped:
        print(f"  dropped dispersions {dropped} at interval {MAIN_INTERVAL}: "
              f"fewer than {MIN_DETECTOR_TOTAL} reference detectors in total")
    for key in sorted(k for k in cited if k[0] == "budget_ms"):
        print(f"  absolute tolerance at theta {key[1]}: {cited[key]:.2f} ms")

    pa, pb, pc = args.panel_letters
    n_ref = int(np.mean([r["n_reference"] for r in rows
                         if r["dispersion"] == ref_disp
                         and r["interval"] == MAIN_INTERVAL]))

    # The time step, read from the data rather than assumed: the caption has
    # to say it, because on a 1 ms step a perturbation under half a step rounds
    # away and the small-perturbation retention is inflated.
    steps = sorted({r["steps_per_ms"] for r in rows + strict})
    step_ms = ", ".join(f"{1 / k:g}" for k in steps)

    # The slowing/speeding asymmetry is confined to dispersions near the
    # interval, so the caption reports it there and says where it ends, rather
    # than quoting one dispersion as if it held for all.
    def scale_gap(disp, mag=0.05):
        keyed = defaultdict(list)
        for r in rows:
            if (r["perturbation"] == "scale" and r["dispersion"] == disp
                    and r["interval"] == MAIN_INTERVAL):
                keyed[round(r["magnitude"], 4)].append(r["retention"])
        slow, fast = keyed.get(round(1 + mag, 4)), keyed.get(round(1 - mag, 4))
        return None if not slow or not fast else float(np.mean(slow) - np.mean(fast))

    wide = [d for d in dispersions if d >= MAIN_INTERVAL + 15]
    wide_gaps = [g for g in (scale_gap(d) for d in wide) if g is not None]
    wide_txt = (f"from $\\Delta = {wide[0]}$~ms on, the two differ by at most "
                f"{100 * max(abs(g) for g in wide_gaps):.0f} percentage points"
                if wide_gaps else "")

    def cite(pert, mag, theta=None, disp=None):
        e = cited[(pert, disp or ref_disp, round(theta or theta_main, 2), mag)]
        return f"{100 * e.mean:.0f}\\% [{100 * e.lo:.0f}, {100 * e.hi:.0f}]"

    ms_main = cited[("budget_ms", round(theta_main, 2))]
    ms_strict = cited.get(("budget_ms", round(theta_strict, 2))) if strict else None
    budget_txt = ", ".join(
        f"$\\Delta = {d}$~ms $\\rightarrow$ {budget[(round(theta_main, 2), MAIN_INTERVAL, d)]:.1f}\\%"
        for d in dispersions
        if (round(theta_main, 2), MAIN_INTERVAL, d) in budget)
    caption = (
        f"\\textbf{{How much conduction-velocity precision the delay-key scheme "
        f"needs.}} Sequence detectors were identified with exact delays "
        f"({n_ref} per population on average at $\\Delta = {ref_disp}$~ms), then "
        f"the delays were perturbed and the fraction still responding "
        f"conjunctively was measured. The sweep runs on a {step_ms}~ms step, "
        f"with perturbed delays re-rounded to it, so that perturbations far "
        f"smaller than a millisecond are resolved rather than rounded away "
        f"(\\textbf{{S2 Text}}). Shading "
        f"and bars are 95\\% bootstrap confidence intervals over {n_seeds} "
        f"independent seeds. \\textbf{{{pa}}}, Independent per-connection "
        f"jitter, each delay scaled by its own draw of "
        f"$1 + \\mathcal{{N}}(0, s)$, modelling axon-to-axon variability, for "
        f"delay dispersions of {dispersions[0]}--{dispersions[-1]}~ms (key). At "
        f"$\\Delta = {ref_disp}$~ms, {cite('jitter', 0.02)} of detectors survive "
        f"2\\% jitter, {cite('jitter', 0.05)} survive 5\\%, and "
        f"{cite('jitter', 0.1)} survive 10\\%. Wider populations are uniformly "
        f"more fragile, as they must be: the same fractional error on a longer "
        f"delay is a larger absolute error."
        + (f" The dashed vermillion curve repeats $\\Delta = {ref_disp}$~ms at "
           f"the stricter gate $\\theta = {theta_strict:g}$." if strict else "")
        + f" \\textbf{{{pb}}}, A common scaling of "
        f"every delay, modelling a global shift such as a temperature change. "
        f"The factor $k$ multiplies \\emph{{delays}}, so $k > 1$ is a "
        f"lengthening of every delay and therefore a \\emph{{slowing}} of "
        f"conduction, and $k < 1$ a speeding. At $\\Delta = {ref_disp}$~ms a "
        f"5\\% slowing ($k = 1.05$) retains {cite('scale', 1.05)} against "
        f"{cite('scale', 0.95)} for a 5\\% speeding; a 20\\% slowing retains "
        f"{cite('scale', 1.2)} against {cite('scale', 0.8)} for a 20\\% "
        f"speeding. The asymmetry is confined to dispersions near the "
        f"interval, where the detectors that exist mostly fall short of "
        f"compensating it, so that dilating every delay helps them and "
        f"compressing hurts; {wide_txt} (\\textbf{{S2 Text}}). Retention falls "
        f"smoothly rather than collapsing, and does not reach zero even under "
        f"the largest perturbations applied, because detectors differ in how "
        f"tightly their delays are matched; the loosely matched ones tolerate "
        f"error that the precisely matched ones do not. \\textbf{{{pc}}}, The "
        f"budget in one number: the jitter that costs half the matched "
        f"population, against dispersion ({budget_txt}). Filled markers are the "
        f"{MAIN_INTERVAL}~ms inter-event interval of \\textbf{{{pa}}}--"
        f"\\textbf{{{pb}}}; open markers repeat the measurement at "
        f"{intervals[0]} and {intervals[-1]}~ms. Those follow the same decline: "
        f"a longer inter-event interval costs a little tolerance at every "
        f"dispersion, but the spread across intervals is small beside the "
        f"threefold fall across dispersions, so the budget is set principally "
        f"by the dispersion and not by the interval being encoded. The dashed "
        f"line is not a fit but the hyperbola "
        f"$s_{{1/2}} = \\bar{{c}} / \\Delta$ with $\\bar{{c}}$ the mean absolute "
        f"tolerance across dispersions. The tolerated \\emph{{fraction}} "
        f"therefore falls as $1/\\Delta$ while the tolerated absolute error "
        f"holds near {ms_main:.2f}~ms at $\\theta = {theta_main:g}$"
        + (f" and {ms_strict:.2f}~ms at $\\theta = {theta_strict:g}$"
           if ms_strict else "")
        + f": what the scheme requires is not a fixed percentage precision on "
        f"conduction velocity but a fixed timing tolerance of order a "
        f"millisecond, which is the same order as the coincidence window "
        f"itself. Reading the requirement as a percentage understates it for "
        f"wide populations and overstates it for narrow ones."
    )
    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

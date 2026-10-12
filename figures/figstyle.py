"""Shared figure style for the manuscript artwork.

Every figure in the paper is built through this module so that the whole set
reads as one visual system. The typographic and colour rules come from
``delaykey/delaykey/plotting.py``, which encodes the Nature *Guide to Preparing
Final Artwork*: 7 pt maximum text, 5 pt minimum, 8 pt bold lower-case panel
letters, Okabe-Ito colour, no panel titles, keys inside the figure.

What this module adds on top is the part the Nature spec implies but a plotting
library does not give you: **a drawing surface whose data units are
millimetres**, so a schematic is composed at the size it will be printed at
rather than drawn large and scaled down.

Why that matters here. LaTeX scales a graphic to the requested width, and that
scale divides every point size on the page. The figures this module replaces
were authored between 200 and 353 mm wide and placed at 133 mm, so nominal
10 pt text arrived on the page at 3.8-6.6 pt -- at or below the 5 pt floor.
Building at :data:`WIDTH` and placing at ``\\figwidth`` makes the scale exactly
1.00, and then the point sizes set here are the point sizes that print.

Usage for a plotted figure::

    from figstyle import WIDTH, apply_style, figure_size, save_figure
    apply_style()
    fig = plt.figure(figsize=figure_size(WIDTH, 70.0))
    ...
    save_figure(fig, "figures/fig_thing", formats=("pdf", "png"))

Usage for a schematic, where 1 data unit == 1 mm::

    from figstyle import canvas, soma, axon, spike_train, save_figure
    fig, ax = canvas(height_mm=52.0)
    soma(ax, 20, 30, label="P")
    axon(ax, (8, 45), (20, 33), curve=0.25)
    save_figure(fig, "figures/fig_thing", formats=("pdf", "png"))
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package sits beside figures/: as delaykey/ here and in the code archive,
# and as delaykey-main/ in the public code repository.
_ROOT = Path(__file__).resolve().parents[1]
_DELAYKEY = next((p for p in (_ROOT / "delaykey", _ROOT / "delaykey-main")
                  if (p / "delaykey" / "plotting.py").is_file()),
                 _ROOT / "delaykey")
if str(_DELAYKEY) not in sys.path:
    sys.path.insert(0, str(_DELAYKEY))

from delaykey.plotting import (COLUMN_MM, MAX_HEIGHT_MM,  # noqa: E402
                               PALETTE, apply_style, direct_label, figure_size,
                               label_panels, resolve_font, save_figure, tint)

# delaykey.plotting selects the Agg backend on import, so these are safe here.
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

__all__ = [
    # re-exported from delaykey.plotting
    "COLUMN_MM", "MAX_HEIGHT_MM", "PALETTE", "apply_style", "direct_label",
    "figure_size", "label_panels", "resolve_font", "save_figure", "tint",
    # added here
    "WIDTH", "WIDTH_MM", "C", "PT_LABEL", "PT_TICK", "PT_ANNOT", "PT_PANEL",
    "canvas", "axes_mm", "panel_letter", "soma", "axon", "spike_train",
    "time_axis", "bracket", "note", "trace_inset", "dashed", "curly",
    "clip_path_free", "audit_fonts",
]

#: The width every main-text figure is built at.
#:
#: ``\figwidth`` in this manuscript resolves to the PLOS Computational Biology
#: template text width (``\textwidth 5.25in`` = 133.35 mm). Building here and
#: placing there gives a scale factor of exactly 1.00.
WIDTH = "plos"
WIDTH_MM = COLUMN_MM[WIDTH]

# Point sizes, mirrored from delaykey.plotting so figure scripts can position
# text without reaching into private names.
#
# NOT PLOS-COMPLIANT, DELIBERATELY, FOR NOW. These are Nature's sizes. PLOS
# requires Arial, Times or Symbol at 8-12 pt in submitted artwork, so four of
# these five values sit below the floor and the typeface (Nimbus Sans / the
# Liberation Sans fallback) is a Helvetica clone rather than Arial. PLOS also
# accepts only TIFF or EPS for separately uploaded figures; the generators
# already emit both, so format is not the problem -- point size is.
#
# Retuning is a real job and it cannot be done by scaling. Because the toolchain
# composes at printed size and places at scale 1.00 (see the module docstring
# above), raising 5.5 pt to 8 pt makes every label physically larger on a canvas
# whose width is fixed by the column, so labels collide, direct-labelled curves
# need re-anchoring, and panel geometry has to be re-laid-out. Expect to touch
# every generator in figures/ and delaykey/figures/, and to re-check each figure
# by eye afterwards.
#
# It is deferred rather than forgotten because the manuscript is not at
# Editorial Accept, where PLOS enforces this, and because retuning now would
# have to be redone after any figure changes made in review. Do it once the
# science is settled. Changing the four values below and re-running
# `make figures` is the start of the job, not the whole of it.
# Mirrors delaykey.plotting's scale, which is set to PLOS's 8-12 pt
# artwork specification rather than to Nature's 7 pt ceiling. See the
# comment on _PT_LABEL there before changing these.
PT_LABEL = 9.0
PT_TICK = 8.0
PT_ANNOT = 8.0
PT_PANEL = 10.0

#: Semantic colour roles, so the same thing is the same colour in every figure.
#:
#: Bound to the Okabe-Ito palette and to the assignments already established by
#: the dispersion-sweep figure (sequence = vermillion, event = blue,
#: dual = green, silent = grey), which is the one figure in the paper that was
#: already built to specification.
C = {
    "a": PALETTE["vermillion"],   # first event / neuron A
    "b": PALETTE["blue"],         # second event / neuron B
    "c": PALETTE["green"],        # third event / neuron C
    "hit": PALETTE["orange"],     # matched, plateau-triggering
    "miss": PALETTE["grey"],      # unmatched, pruned, silent
    "distal": PALETTE["blue"],    # distal dendrite / Q-K pathway
    "proximal": PALETTE["green"], # proximal dendrite / V pathway
    "soma": PALETTE["orange"],    # soma / integration
    "rule": "0.35",               # thin reference rules, thresholds
    "faint": "0.72",              # de-emphasised structure
    "ink": "0.15",                # text and outlines
}

_MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Drawing surfaces
# ---------------------------------------------------------------------------

def canvas(width=WIDTH, height_mm: float = 60.0):
    """Return ``(fig, ax)`` whose data units are millimetres on the page.

    The axes span the whole canvas and carry an equal aspect with limits set to
    the physical size, so ``ax.plot([0, 10], ...)`` draws exactly 10 mm. That
    makes a schematic composable at print size: a 2 mm circle is 2 mm, and a
    label placed 3 mm from it stays 3 mm away in the PDF.

    Args:
        width: a key of :data:`COLUMN_MM` or an explicit width in millimetres.
        height_mm: canvas height, at most :data:`MAX_HEIGHT_MM`.

    Returns:
        The figure and a full-bleed, axis-free drawing axes.
    """
    apply_style()
    w_in, h_in = figure_size(width, height_mm)
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, w_in * _MM_PER_INCH)
    ax.set_ylim(0.0, h_in * _MM_PER_INCH)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def axes_mm(fig, x_mm: float, y_mm: float, w_mm: float, h_mm: float):
    """Place an ordinary plotting axes at millimetre coordinates on ``fig``.

    Coordinates are measured from the bottom-left of the canvas, matching
    :func:`canvas`, so a data panel can be dropped into a schematic without
    converting to figure fractions by hand.
    """
    fw_mm = fig.get_figwidth() * _MM_PER_INCH
    fh_mm = fig.get_figheight() * _MM_PER_INCH
    return fig.add_axes([x_mm / fw_mm, y_mm / fh_mm, w_mm / fw_mm, h_mm / fh_mm])


def panel_letter(ax, x_mm: float, y_mm: float, text: str) -> None:
    """Letter a schematic panel 8 pt bold lower case, per the Nature spec.

    :func:`label_panels` does this for gridded plotting axes; this is the
    millimetre-space equivalent for :func:`canvas` drawings.
    """
    ax.text(x_mm, y_mm, text, fontsize=PT_PANEL, fontweight="bold",
            va="top", ha="left", color="black", zorder=10)


# ---------------------------------------------------------------------------
# Schematic primitives. All positions and sizes are in millimetres.
# ---------------------------------------------------------------------------

def soma(ax, x: float, y: float, r: float = 1.9, color: str = C["miss"],
         label: str | None = None, label_dy: float = -1.2,
         filled: bool = True, lw: float = 0.5, zorder: float = 3,
         label_color: str | None = None, fontsize: float = PT_ANNOT):
    """Draw a cell body as a circle, optionally labelled beneath it.

    Args:
        x, y: centre, in millimetres.
        r: radius, in millimetres.
        color: fill colour when ``filled``, else the outline colour.
        label: text placed below the circle.
        label_dy: label offset from the circle edge; negative is below.
        filled: solid fill, or white fill with a coloured rim.

    Returns:
        The circle patch.
    """
    face = color if filled else "white"
    edge = "white" if filled else color
    circle = mpatches.Circle((x, y), r, facecolor=face, edgecolor=edge,
                             linewidth=lw, zorder=zorder)
    ax.add_patch(circle)
    if label is not None:
        below = label_dy < 0
        ax.text(x, y + (label_dy - r if below else label_dy + r), label,
                ha="center", va="top" if below else "bottom",
                fontsize=fontsize, color=label_color or C["ink"],
                zorder=zorder + 1)
    return circle


def axon(ax, start, end, curve: float = 0.2, color: str = C["miss"],
         lw: float = 0.6, ls: str = "-", arrow: bool = True,
         alpha: float = 1.0, zorder: float = 2, shrink: float = 2.2,
         head: float = 5.0):
    """Draw a curved axonal projection from ``start`` to ``end``.

    Args:
        start, end: ``(x, y)`` in millimetres.
        curve: arc bulge; positive bends left of the direction of travel.
        ls: ``"-"`` for a retained connection, ``"--"`` for a pruned one.
        shrink: gap in points left at each end so the line does not overlap
            the somata it connects.
        head: arrowhead size in points.

    Returns:
        The arrow patch.
    """
    style = "-|>" if arrow else "-"
    patch = mpatches.FancyArrowPatch(
        start, end, connectionstyle=f"arc3,rad={curve}", arrowstyle=style,
        mutation_scale=head, linewidth=lw, linestyle=ls, color=color,
        alpha=alpha, shrinkA=shrink, shrinkB=shrink, zorder=zorder,
        joinstyle="round", capstyle="round")
    ax.add_patch(patch)
    return patch


def spike_train(ax, x0: float, x1: float, y: float, times,
                color: str = C["ink"], height: float = 1.6,
                lw: float = 0.7, baseline: bool = True,
                baseline_color: str = C["faint"], label: str | None = None,
                label_gap: float = 1.5, zorder: float = 3):
    """Draw a spike raster row: a baseline with vertical ticks on it.

    Args:
        x0, x1: extent of the baseline, in millimetres.
        y: baseline height, in millimetres.
        times: tick positions in millimetres, already mapped to the axis.
        height: tick height above the baseline.
        label: row name, placed to the left of ``x0``.
    """
    if baseline:
        ax.plot([x0, x1], [y, y], color=baseline_color, lw=0.5,
                solid_capstyle="butt", zorder=zorder - 1)
    for t in times:
        ax.plot([t, t], [y, y + height], color=color, lw=lw,
                solid_capstyle="butt", zorder=zorder)
    if label is not None:
        ax.text(x0 - label_gap, y + height / 2.0, label, ha="right",
                va="center", fontsize=PT_ANNOT, color=C["ink"], zorder=zorder)


def time_axis(ax, x0: float, x1: float, y: float, ticks=(), labels=(),
              color: str = C["ink"], lw: float = 0.5, tick_len: float = 1.0,
              label: str | None = "$t$", tick_colors=None,
              fontsize: float = PT_ANNOT, zorder: float = 3):
    """Draw a horizontal time arrow with optional labelled ticks below it."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=4.0,
                                linewidth=lw, color=color, shrinkA=0,
                                shrinkB=0), zorder=zorder)
    colors = list(tick_colors) if tick_colors is not None else [color] * len(ticks)
    for i, t in enumerate(ticks):
        ax.plot([t, t], [y - tick_len, y], color=colors[i], lw=lw,
                solid_capstyle="butt", zorder=zorder)
    for t, text in zip(ticks, labels):
        ax.text(t, y - tick_len - 0.6, text, ha="center", va="top",
                fontsize=fontsize, color=C["ink"], zorder=zorder)
    if label:
        ax.text(x1 + 1.2, y, label, ha="left", va="center",
                fontsize=fontsize, color=color, zorder=zorder)


def bracket(ax, x0: float, x1: float, y: float, label: str | None = None,
            color: str = C["ink"], lw: float = 0.5, tick: float = 0.8,
            below: bool = False, fontsize: float = PT_ANNOT,
            zorder: float = 4, label_gap: float = 0.6):
    """Draw a dimension bracket spanning ``x0``-``x1``, labelled at its centre.

    Used for delays and distances, where the quantity being named is an extent
    rather than a point.
    """
    sign = -1.0 if below else 1.0
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
            zorder=zorder)
    for x in (x0, x1):
        ax.plot([x, x], [y, y + sign * tick], color=color, lw=lw,
                solid_capstyle="butt", zorder=zorder)
    if label:
        ax.text((x0 + x1) / 2.0, y + sign * (tick + label_gap), label,
                ha="center", va="bottom" if below is False else "top",
                fontsize=fontsize, color=color, zorder=zorder)
    return None


def note(ax, x: float, y: float, text: str, color: str = C["ink"],
         ha: str = "left", va: str = "center", fontsize: float = PT_ANNOT,
         weight: str = "normal", zorder: float = 6, **kwargs):
    """Place explanatory text in millimetre space.

    Kept deliberately plain: Nature figures carry no titles and no boxed
    callouts, so this is a bare text run with no frame behind it.
    """
    return ax.text(x, y, text, color=color, ha=ha, va=va, fontsize=fontsize,
                   fontweight=weight, zorder=zorder, **kwargs)


def trace_inset(fig, x_mm: float, y_mm: float, w_mm: float, h_mm: float,
                spines=("left", "bottom")):
    """A small waveform panel inside a schematic, with most furniture removed.

    Membrane traces in a schematic are illustrative, not quantitative, so the
    default keeps a left and bottom rule for orientation and drops the ticks.
    """
    ax = axes_mm(fig, x_mm, y_mm, w_mm, h_mm)
    ax.set_facecolor("none")
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_linewidth(0.5)
        spine.set_color(C["faint"])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def audit_fonts(pdf_path, expected=("LiberationSans",)):
    """Check that a figure PDF embeds only the intended faces.

    Nature forbids more than one typeface in a figure, and the way that rule
    gets broken is invisible: matplotlib substitutes a fallback face for a
    single missing glyph and embeds it silently. ``$\\rightarrow$`` on a host
    whose Helvetica lacks U+2192 pulls in STIXGeneral for one arrow, and
    nothing in the build says so -- it only shows up in ``pdffonts``.

    It also cannot catch outlined text, which embeds *no* font at all. Never
    apply ``patheffects.withStroke`` to a ``Text`` object to halo it over
    artwork: matplotlib converts the glyphs to paths, the PDF carries no font
    for them, and this check sees nothing to flag. Use ``illustrate.halo``,
    which draws offset copies and keeps the text live.

    Args:
        pdf_path: the figure PDF to inspect.
        expected: substrings of the acceptable PostScript names. Subset names
            carry a random ``ABCDEF+`` prefix, so matching is by substring.
            The default covers both weights: the regular Liberation face
            embeds as plain ``LiberationSans`` with no ``-Regular`` suffix,
            while the bold is ``LiberationSans-Bold``.

    Returns:
        A list of offending font names; empty means clean.

    Raises:
        RuntimeError: if ``pdffonts`` is unavailable.
    """
    import subprocess

    try:
        out = subprocess.run(["pdffonts", str(pdf_path)], capture_output=True,
                             text=True, check=True).stdout
    except FileNotFoundError as exc:                       # pragma: no cover
        raise RuntimeError("pdffonts (poppler) is required for audit_fonts") from exc

    names = [line.split()[0] for line in out.splitlines()[2:] if line.strip()]
    return [n for n in names if not any(e in n for e in expected)]


def clip_path_free(patch):
    """Detach a patch from its axes clip box so it may cross panel edges."""
    patch.set_clip_on(False)
    return patch


def dashed(pattern=(1.6, 1.4)):
    """A consistent dash pattern for pruned or inactive elements."""
    return (0, pattern)


def curly(ax, x: float, y0: float, y1: float, label: str | None = None,
          color: str = C["ink"], lw: float = 0.5, width: float = 0.9,
          fontsize: float = PT_ANNOT, zorder: float = 4):
    """A vertical brace spanning ``y0``-``y1``, for grouping rows."""
    ym = (y0 + y1) / 2.0
    verts = [(x, y0), (x + width, y0), (x + width, ym - 0.15),
             (x + 2 * width, ym), (x + width, ym + 0.15),
             (x + width, y1), (x, y1)]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor="none",
                                    edgecolor=color, linewidth=lw,
                                    zorder=zorder))
    if label:
        ax.text(x + 2 * width + 0.6, ym, label, ha="left", va="center",
                fontsize=fontsize, color=color, zorder=zorder)

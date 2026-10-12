"""Build Fig 3, the cortical-column diameter prediction.

Laid out to the Nature artwork specification through :mod:`figstyle`: the
manuscript's body text width (:data:`figstyle.WIDTH_MM`, 165.1 mm), 7 pt
maximum text, 8 pt bold lower-case panel letters, no panel titles, Okabe-Ito
colour, hairline rules, and every element positioned in millimetres so the
point sizes set here are the point sizes that print.

Panels:

a  The mechanism. Two LGN neurons with horizontally extending long axons
   (H1, H2) reach L4 through the white matter, shed their myelin, and run in
   opposite directions across columns A and B. Colour along each intracortical
   axon is the conduction delay accumulated from that axon's entry point, so
   ``d_A < d_B`` is a graded quantity rather than two labelled arrows. Below,
   the same H1 neuron fires at two latencies; because ``t_1 + d_B = t_2 + d_A``
   the earlier spike lands in register at the distal column and the later spike
   at the proximal one.
b  The prediction. Eq. (column_diameter) drawn as a relation --- column
   diameter against horizontal conduction velocity, banded over the coincidence
   window --- instead of stated as a sentence.

The version this replaces spent its whole area on anatomy (dozens of hand-drawn
dendritic arbors, thirty-odd myelin rectangles) and left the prediction, which
is the only falsifiable claim in this section, as running text under the
drawing. Here the anatomy is cut to what the argument needs and the diameter
relation gets a panel of its own.

The anatomy that survives is drawn through :mod:`illustrate` rather than out of
straight lines: :func:`~illustrate.axon_path` gives each afferent an organic
wander and, on its white-matter course, true internodes separated by nodes of
Ranvier, so the axon *losing* its myelin at the L4 boundary --- which is what
drops its conduction velocity into the range Eq. (column_diameter) uses --- is
visible anatomy instead of a caption sentence. The delay gradient rides on the
same curve through :func:`~illustrate.gradient_along`, and a keyline separates
the two afferents where they cross.

**On the numbers.** Every value plotted comes from the manuscript body:
``v_axon ~ 0.3 m/s`` and ``tau_window = 1-2 ms``, giving 300-600 um
(Eq.~\\ref{eq:column_diameter}, content.tex around line 830). The text asserts
agreement with "measured column dimensions in visual, somatosensory, and barrel
cortices" but attaches no numbers to those citations, so no measured diameters
are overlaid --- there is nothing to overlay that the manuscript itself states.
The velocity axis is swept only to give the relation an extent; the single
manuscript value on it is marked.

Usage::

    python3 figures/make_fig_column_diameter.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import (C, PALETTE, PT_ANNOT, WIDTH, WIDTH_MM,  # noqa: E402
                      audit_fonts, axes_mm, bracket, canvas, dashed, note,
                      panel_letter, resolve_font, save_figure, time_axis, tint)
from illustrate import (axon_path, bouton, branching_tree,  # noqa: E402
                        gradient_along, halo, ramp, shade, soma_shape)

# figstyle imports delaykey.plotting, which selects the Agg backend, so this is
# safe here.
import matplotlib.patches as mpatches  # noqa: E402

# ---------------------------------------------------------------------------
# The two quantities the figure exists to multiply, taken verbatim from the
# manuscript body (content.tex, "Estimating column diameter").
# ---------------------------------------------------------------------------
V_AXON = 0.3               # horizontal conduction velocity of LGN axons in L4, m/s
TAU_MIN, TAU_MAX = 1.0, 2.0   # effective coincidence window of L4 stellates, ms
V_MAX = 0.75               # extent of the swept velocity axis, m/s
                           # (the monkey mean, 0.62, must sit inside it)

# d in micrometres = v in m/s x tau in ms x 1000.
D_MIN = V_AXON * TAU_MIN * 1e3    # 300 um
D_MAX = V_AXON * TAU_MAX * 1e3    # 600 um

# ---------------------------------------------------------------------------
# The two areas in which horizontal intracortical conduction has been measured
# directly, in the layer the construction concerns, together with the column
# each area is conventionally credited with. These are literature values, not
# output of any experiment in this repository, so they are written here with
# their sources rather than read from a CSV.
#
#   Girard, Hupe & Bullier (2001) J Neurophysiol 85:1328. 156 stimulating-
#   recording pairs in V1 of three macaques; median orthodromic conduction
#   speed of intracortical horizontal axons 0.33 m/s. Ocular dominance column
#   width in macaque V1 is ~400 um.
#
#   Scheuer, Judge, Zhao & Jackson (2023) Cereb Cortex 33:9917. hVOS voltage
#   imaging of PV interneurons in mouse barrel cortex; L4 INTRALAMINAR
#   conduction 142 +/- 76 um/ms, i.e. 0.142 +/- 0.076 m/s. Mouse barrel
#   ~200-300 um across.
#
# The error bar plotted is on v only; the column figure is drawn as the range
# the area is quoted at, not as a measurement with its own interval.
# ---------------------------------------------------------------------------
# v is plotted as an interval, not a point, because each source reports a
# spread. Neither spread is the one the mechanism is about -- the spread of v
# across the afferents converging on one column -- so the bars show what each
# study measured and nothing more: Girard's median-to-mean skew across 156
# stimulating-recording pairs, and Scheuer's SD across four slices.
#
#   monkey V1     median 0.33, mean 0.62 m/s (Girard 2001).  The gap between
#                 the two is the skew: the distribution has a long fast tail,
#                 and Zhaoping (2025) shows it is distance-dependent.  Bar spans
#                 median to mean.
#   mouse barrel  142 +/- 76 um/ms (Scheuer 2023), mean +/- SD over four
#                 slices.  Bar spans one SD either side, clipped at zero.
# v_pt is the statistic each source actually reports and is where the marker
# sits; v_lo..v_hi is the spread around it.  Plotting the marker at the midpoint
# of the spread would put it at neither statistic.
MEASURED = (
    # label,          v_pt,  v_lo,  v_hi,  d_lo,  d_hi
    ("monkey V1",     0.33,  0.33,  0.62,  380.0, 420.0),   # median; bar to mean
    ("mouse barrel",  0.142, 0.066, 0.218, 200.0, 300.0),   # mean; bar is +/-1 SD
)

# ---------------------------------------------------------------------------
# Page geometry. Everything below is millimetres on the printed page.
#
# **Every horizontal position is derived from the canvas width**, never written
# down. That is not tidiness. This figure was first composed against 133.35 mm,
# the PLOS ``\textwidth`` -- which governs the *title page only*. The template
# issues ``\newgeometry{left=1in,right=1in}`` before the body, so ``\linewidth``
# (hence ``\figwidth``) is 6.5 in = 165.1 mm for every float. The hardcoded
# millimetres survived the correction to :data:`figstyle.WIDTH_MM` and left
# 35 mm of blank canvas down the right-hand side. Anchoring the sheet, the
# delay key, both entry points and the equation column to ``W`` means the next
# width change moves the drawing instead of stranding it.
# ---------------------------------------------------------------------------
W = WIDTH_MM                 # the canvas, and the origin of every x below
HEIGHT_MM = 92.0
EN = "–"                     # en dash; "--" is LaTeX input, not a glyph


def _lerp(a: float, b: float, t: float) -> float:
    """Point a fraction ``t`` of the way from ``a`` to ``b``.

    Waypoints along the two long axonal runs are placed as fractions of the run
    rather than as absolute millimetres, so an afferent keeps its wander when
    the run gets longer instead of straightening out over the new width.
    """
    return a + (b - a) * float(t)


# -- panel a, the cortical sheet -------------------------------------------
# The sheet is only as deep as the three things it has to hold -- H1, the
# terminal clusters and their stellate target, and H2. The axons need their
# full horizontal run for the delay gradient to be legible, so the box is
# hugged vertically; any more depth is empty box either side of the columns.
# Horizontally it takes the whole measure: the horizontal axon run *is* the
# argument of this panel, and every extra millimetre of it is another
# millimetre over which the delay ramp can be told apart from flat colour.
SHEET_X0, SHEET_X1 = 3.0, W - 2.2
L4_Y0, L4_Y1 = 66.8, 80.0         # the L4 sheet
WM_Y0, WM_Y1 = 50.5, 62.0         # white matter, below the intervening layers

Y_CORE = 72.6                     # the stellate cell at each column core
X_H1_IN = SHEET_X0 + 20.0         # H1's entry into L4, after its WM course
X_H1_END = SHEET_X1 - 2.0         # and its far tip, just inside the sheet
X_H2_IN = SHEET_X1 - 6.5          # H2 enters at the far end and runs back
X_H2_END = SHEET_X0 + 6.0
RUN = X_H1_END - X_H1_IN          # H1's intracortical run: the panel's ruler

COL_W = 18.0                      # every column is the same width: it is the
COL_A_C = X_H1_IN + 0.26 * RUN    # quantity panel b predicts. Their centres
COL_B_C = X_H1_IN + 0.70 * RUN    # are placed along H1's run, so the longer
                                  # run buys them separation rather than a
                                  # longer empty approach.

W_AXON = 0.26                     # drawn axon calibre, mm
# One internode length for both fibres, because internode length is what sets
# conduction velocity and these are two axons of the same population. Long
# internodes and a modest sheath on purpose: the myelin is here to be read as
# a fact about the axon, and a close-packed chain of fat sheaths would be the
# loudest thing in a panel whose subject is the gradient above it.
INTERNODE, NODE, SHEATH = 9.0, 1.3, 2.2

# Waypoints, proximal to distal. Each afferent is two courses: a myelinated one
# through the white matter that stops dead at the L4 boundary, and a bare one
# inside L4 that carries the delay gradient from that boundary onward. The
# approach and the climb are offsets from the entry point, because they are
# local shape; the long flat runs are fractions, because they are extent.
H1_WM = [(X_H1_IN - 19.0, 58.5), (X_H1_IN - 11.0, 58.1), (X_H1_IN - 4.5, 59.3),
         (X_H1_IN - 1.2, 62.6), (X_H1_IN, L4_Y0)]
_H1_RUN0 = X_H1_IN + 4.0          # where H1 levels off inside L4
H1_L4 = [(X_H1_IN, L4_Y0), (X_H1_IN + 0.6, 70.6), (X_H1_IN + 1.6, 74.4),
         (_H1_RUN0, 76.2),
         (_lerp(_H1_RUN0, X_H1_END, 0.14), 76.5),
         (_lerp(_H1_RUN0, X_H1_END, 0.40), 76.0),
         (_lerp(_H1_RUN0, X_H1_END, 0.71), 76.6),
         (X_H1_END, 76.2)]

_H2_WM0 = SHEET_X0 + 1.0
_H2_TURN = X_H2_IN - 9.2          # where H2 leaves the white matter for L4
H2_WM = [(_H2_WM0, 54.0),
         (_lerp(_H2_WM0, _H2_TURN, 0.27), 53.6),
         (_lerp(_H2_WM0, _H2_TURN, 0.58), 54.3),
         (_H2_TURN, 53.7),
         (X_H2_IN - 2.2, 57.2), (X_H2_IN - 0.2, 62.4), (X_H2_IN, L4_Y0)]
_H2_RUN0 = X_H2_IN - 4.2          # where H2 levels off inside L4
H2_L4 = [(X_H2_IN, L4_Y0), (X_H2_IN - 0.8, 68.2), (_H2_RUN0, 68.7),
         (_lerp(_H2_RUN0, X_H2_END, 0.19), 68.3),
         (_lerp(_H2_RUN0, X_H2_END, 0.47), 68.9),
         (_lerp(_H2_RUN0, X_H2_END, 0.75), 68.4),
         (X_H2_END, 68.8)]

# The delay key sits directly above the sheet and spans exactly H1's
# intracortical run, so its colour at any x is the colour of the axon below it.
Y_BAR0, Y_BAR1 = 80.5, 81.4
Y_BRACKET_A, Y_BRACKET_B = 82.8, 86.8

# -- panel a, the timing read-out ------------------------------------------
# The two arrows are the same two delays the brackets above measure, so they
# are drawn from those distances through one scale factor rather than placed
# by eye: the ratio of the arrows is then the ratio d_B : d_A, and the gap
# between the two spike times is exactly d_B - d_A, which is what the identity
# t_1 + d_B = t_2 + d_A asserts. The scale is whatever makes the row fill the
# axis, so the construction stretches with the canvas.
Y_AXIS = 41.0                     # H1's spike-time axis
X_T0, X_T1 = X_H1_IN, X_H1_END - 2.0
DELAY_A = COL_A_C - X_H1_IN       # H1's delay to column A, as drawn above
DELAY_B = COL_B_C - X_H1_IN       # and to column B
X_SPIKE_1 = X_T0 + 8.0            # t_1, the preferred stimulus
X_ARRIVE = X_T1 - 34.0            # where both, after their delays, land
_T_SCALE = (X_ARRIVE - X_SPIKE_1) / DELAY_B
X_SPIKE_2 = X_ARRIVE - _T_SCALE * DELAY_A   # t_2, the non-preferred one
TAU_MM = 2.8                      # drawn width of the coincidence window
Y_ARROW_A, Y_ARROW_B = 43.8, 47.2

# -- panel b ---------------------------------------------------------------
# The equation is three short lines, so the plot takes the width it leaves
# rather than an equal half of the row. Both are hung off the sheet's right
# edge, so the bottom row ends where panel a ends instead of stopping short.
EQ_W = 22.0                       # measure of the widest worked line
X_DERIV = SHEET_X1 - EQ_W - 8.5   # left edge of the worked equation
PB_X, PB_Y, PB_H = 15.0, 8.6, 23.0
PB_W = X_DERIV - PB_X - 13.0      # the plot fills what the equation leaves

# -- the orthogonal-crossing panel, when it is built ------------------------
# It sits between the plot and the worked equation, in width taken off the
# plot rather than off the canvas, so adding it does not change the figure's
# outline or move panel a. PB_W_GEOM is what the plot narrows to; the panel
# takes the difference.
# The geometry panel takes the LEFT of the bottom row and the plot shifts
# right, so the letters read a, b, c across the page. Putting it on the right
# instead lettered the row c then b, which a copy-editor would flag, and it
# also read out of order: the crossing geometry is what licenses the formula
# the plot then applies.
GEOM_CX = 30.0
GEOM_CY = PB_Y + PB_H / 2.0 + 0.9
GEOM_S = 5.0              # half-cell: the drawn v x tau is 2 * GEOM_S
GEOM_REACH = 9.6          # how far each afferent runs beyond the cell
GEOM_TICK = 5.6           # half-length of a drawn iso-arrival-time contour
PB_X_GEOM = 76.0                          # plot's left edge when b is present
PB_W_GEOM = X_DERIV - PB_X_GEOM - 13.0    # and the width that leaves it
PB_LETTER_X_GEOM = PB_X_GEOM - 16.0   # clear of the rotated y-label


def delay_cmap():
    """Sequential map for accumulated conduction delay.

    Purple is the one Okabe-Ito hue not already spoken for in this figure
    (column A is vermillion, column B blue), and delay is a continuous physical
    quantity, so it gets a ramp rather than a category colour. Built with
    :func:`illustrate.ramp` from a tint and a shade of that one entry, so the
    dark end is a darker purple rather than a hand-picked hex.
    """
    return ramp(tint(PALETTE["purple"], 0.34), PALETTE["purple"],
                shade(PALETTE["purple"], 0.45), name="delay")


CMAP = delay_cmap()
AXON_INK = shade(C["miss"], 0.30)   # the myelinated course, before L4


def arrow_head(ax, xy, xytext, color: str, lw: float = 1.0) -> None:
    """A bare arrowhead, for the delay arrows on the spike-time axis.

    Only for the quantities in the timing read-out. The axons themselves are
    anatomy and end in a bouton, not in an arrow.
    """
    ax.annotate("", xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=4.5,
                                linewidth=lw, color=color, shrinkA=0,
                                shrinkB=0), zorder=4)


def height_at(path, x: float) -> float:
    """Height of a drawn axon at horizontal position ``x``.

    The afferents wander, so a terminal cluster hung at a nominal height would
    float free of the axon it leaves. Both intracortical courses are monotonic
    in x, so a plain interpolation puts the cluster on the fibre.
    """
    P = np.asarray(path, dtype=float)
    if P[0, 0] > P[-1, 0]:
        P = P[::-1]
    return float(np.interp(x, P[:, 0], P[:, 1]))


# ---------------------------------------------------------------------------
# Panel a
# ---------------------------------------------------------------------------

def draw_sheet(ax) -> None:
    """L4 with the two columns tinted, and the white matter beneath it."""
    ax.add_patch(mpatches.Rectangle(
        (SHEET_X0, WM_Y0), SHEET_X1 - SHEET_X0, WM_Y1 - WM_Y0,
        facecolor=tint(C["ink"], 0.05), edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle(
        (SHEET_X0, L4_Y0), SHEET_X1 - SHEET_X0, L4_Y1 - L4_Y0,
        facecolor="white", edgecolor=C["faint"], linewidth=0.5, zorder=0.5))

    for centre, colour, name in ((COL_A_C, C["a"], "column A"),
                                 (COL_B_C, C["b"], "column B")):
        x0 = centre - COL_W / 2.0
        ax.add_patch(mpatches.Rectangle(
            (x0, L4_Y0), COL_W, L4_Y1 - L4_Y0, facecolor=tint(colour, 0.15),
            edgecolor="none", zorder=0.6))
        # The boundaries are the point: they run perpendicular to the axons.
        for x in (x0, x0 + COL_W):
            ax.plot([x, x], [L4_Y0, L4_Y1], color=C["rule"], lw=0.5,
                    linestyle=dashed((1.2, 1.2)), zorder=0.8)
        # Haloed, because the wandering axon passes within a millimetre of it.
        halo(ax, centre, L4_Y1 - 1.4, name, color=colour, rings=6)

    note(ax, SHEET_X0 + 1.5, L4_Y1 - 0.6, "L4", color=C["rule"], va="top")
    note(ax, SHEET_X1 - 1.0, WM_Y0 + 0.3, "white matter (myelinated)",
         color=C["rule"], ha="right", va="bottom")


def draw_afferent(ax, wm_points, l4_points, seed: int, zorder: float,
                  label: str, label_xy, label_ha: str):
    """One horizontally extending LGN axon, myelinated then bare.

    Two courses, and the boundary between them is the whole point. The
    white-matter course carries internodal sheaths pinched at nodes of
    Ranvier; the intracortical course has none, because the axon sheds its
    myelin on entering L4, and that is what drops it to the ~0.3 m/s that
    Eq. (column_diameter) multiplies. The delay gradient starts at exactly
    that boundary and darkens with distance travelled from it.

    Returns:
        The intracortical centreline, so terminals can be hung on it.
    """
    axon_path(ax, wm_points, width=W_AXON, colour=AXON_INK, wobble=0.22,
              wobble_scale=16.0, myelin=True, internode=INTERNODE, node=NODE,
              sheath=SHEATH, sheath_colour=tint(AXON_INK, 0.30),
              sheath_edge_mm=0.07, seed=seed, zorder=zorder, keyline_mm=0.10)

    # width 0 / colour "none" draws nothing: this call is here for the wandered
    # centreline, which gradient_along then fills with the delay ramp.
    course = axon_path(ax, l4_points, width=0.0, colour="none", wobble=0.20,
                       wobble_scale=26.0, seed=seed + 7, zorder=0)
    path = course["path"]
    gradient_along(ax, path, cmap=CMAP, w0=W_AXON, steps=120,
                   zorder=zorder + 0.2, keyline_mm=0.10)
    bouton(ax, path[-1], radius=0.42, colour=CMAP(1.0),
           angle=np.degrees(np.arctan2(*(path[-1] - path[-2])[::-1])),
           zorder=zorder + 0.3, keyline_mm=0.08)

    # Delay zero, at the boundary where the myelin stops.
    ax.add_patch(mpatches.Circle(tuple(path[0]), 0.5, facecolor=CMAP(0.0),
                                 edgecolor=CMAP(0.55), linewidth=0.4,
                                 zorder=zorder + 0.3))
    halo(ax, *label_xy, label, color=CMAP(0.80), ha=label_ha, rings=6)
    return path


def draw_afferents(ax):
    """Both afferents, running in opposite directions across the same sheet.

    H1 is drawn in front of H2, so the keyline on H1 makes the crossing at its
    entry read as one fibre passing over the other rather than as a junction.
    """
    h2 = draw_afferent(ax, H2_WM, H2_L4, seed=5, zorder=2.6, label="H$_2$",
                       label_xy=(X_H2_IN - 2.7, 70.9), label_ha="right")
    h1 = draw_afferent(ax, H1_WM, H1_L4, seed=11, zorder=3.0, label="H$_1$",
                       label_xy=(X_H1_IN + 4.5, 78.3), label_ha="left")
    note(ax, SHEET_X0 + 1.0, WM_Y1 - 0.6, "from LGN", color=C["rule"],
         va="top")
    return h1, h2


def _ramp_at(x: float, x_in: float, x_end: float) -> float:
    """Fraction of an axon's length travelled, for colouring its terminals."""
    return float(np.clip((x - x_in) / (x_end - x_in), 0.0, 1.0))


def draw_column_core(ax, centre: float, colour: str, h1, h2) -> None:
    """Local clustered terminals from both axons onto one L4 stellate cell.

    This is the "synchronous synaptic integration site" of the text: short
    branches, hence near-identical delays within the cluster, converging on a
    single postsynaptic target. Kept small and sparse on purpose --- the
    afferents and their gradient are what the panel is about, and a dense tuft
    here would out-shout them.

    Each tuft takes the delay colour its parent axon has reached at this
    column, so the two clusters in a column arrive at different delays and the
    pair of them differs from column to column. That difference is the
    iso-delay-difference line the text calls a column boundary.
    """
    for path, x_in, x_end, angle in ((h1, X_H1_IN, X_H1_END, 270.0),
                                     (h2, X_H2_IN, X_H2_END, 90.0)):
        shade_here = CMAP(_ramp_at(centre, x_in, x_end))
        tips, _ = branching_tree(
            ax, (centre, height_at(path, centre)), angle=angle, length=0.95,
            width=0.24, depth=3, length_decay=0.74, spread=68.0,
            curvature=0.24, min_width=0.06, colour=shade_here,
            seed=int(centre) + int(angle), zorder=3.4, keyline_mm=0.08)
        for tip in tips:
            bouton(ax, tip, radius=0.28, angle=angle, colour=shade_here,
                   zorder=3.5, keyline_mm=0.06)

    soma_shape(ax, centre, Y_CORE, width=2.1, height=2.4, colour=colour,
               pyramidal=False, zorder=4.2, keyline_mm=0.14)


def draw_delay_scale(ax) -> None:
    """The delay ramp as a key, and d_A and d_B as brackets off H1's entry."""
    y_bar = (Y_BAR0 + Y_BAR1) / 2.0
    gradient_along(ax, [[X_H1_IN, y_bar], [X_H1_END, y_bar]], cmap=CMAP,
                   w0=Y_BAR1 - Y_BAR0, steps=140, zorder=3)
    note(ax, X_H1_IN - 1.2, y_bar, "0", color=C["rule"], ha="right")
    note(ax, X_H1_END, Y_BAR1 + 0.6, "conduction delay", color=C["rule"],
         ha="right", va="bottom")

    # Both brackets run from H1's entry to a column *centre*, which is where
    # the clustered terminals integrate, and both get the same leader drawn on
    # top of the key -- otherwise d_A's, being the shorter drop, disappears
    # behind the key and the two are read on different rules.
    for centre, y_top, colour in ((COL_A_C, Y_BRACKET_A, C["a"]),
                                  (COL_B_C, Y_BRACKET_B, C["b"])):
        ax.plot([centre, centre], [L4_Y1, y_top], color=tint(colour, 0.75),
                lw=0.5, linestyle=dashed((0.8, 1.0)), zorder=5.5)

    bracket(ax, X_H1_IN, COL_A_C, Y_BRACKET_A, label="$d_A$", color=C["a"])
    bracket(ax, X_H1_IN, COL_B_C, Y_BRACKET_B, label="$d_B$", color=C["b"])

    # The quantity panel b predicts, hung directly under the column it is a
    # property of, where it cannot be confused with the two delay brackets.
    bracket(ax, COL_B_C - COL_W / 2.0, COL_B_C + COL_W / 2.0, L4_Y0 - 0.6,
            label="$d_\\mathrm{column}$", color=C["ink"], below=True)


def draw_timing(ax) -> None:
    """H1's two firing latencies, each carried to its column by its own delay.

    The construction is ``t_1 + d_B = t_2 + d_A``: the earlier spike buys the
    larger delay and lands at the distal column, the later spike the smaller
    one and lands at the proximal column, and both land inside a single
    coincidence window. Which is the whole argument for a column continuum.
    """
    ax.add_patch(mpatches.Rectangle(
        (X_ARRIVE - TAU_MM / 2.0, Y_AXIS), TAU_MM, Y_ARROW_B + 2.0 - Y_AXIS,
        facecolor=tint(C["ink"], 0.13), edgecolor="none", zorder=1))

    time_axis(ax, X_T0, X_T1, Y_AXIS,
              ticks=(X_SPIKE_1, X_SPIKE_2),
              labels=("$t_1$\npreferred", "$t_2$\nnon-preferred"),
              tick_colors=(C["b"], C["a"]), tick_len=1.6)
    note(ax, X_T0 - 1.5, Y_AXIS, "H$_1$ spike time", ha="right")

    for x_spike, y_arrow, colour, label in (
            (X_SPIKE_2, Y_ARROW_A, C["a"], "$+\\,d_A$ to column A"),
            (X_SPIKE_1, Y_ARROW_B, C["b"], "$+\\,d_B$ to column B")):
        ax.plot([x_spike, x_spike], [Y_AXIS, y_arrow], color=tint(colour, 0.5),
                lw=0.5, linestyle=dashed((0.8, 1.0)), zorder=2)
        arrow_head(ax, (X_ARRIVE, y_arrow), (x_spike, y_arrow), colour, lw=0.8)
        note(ax, (x_spike + X_ARRIVE) / 2.0, y_arrow + 0.4, label,
             color=colour, ha="center", va="bottom")

    note(ax, X_ARRIVE + TAU_MM / 2.0 + 1.4, (Y_ARROW_A + Y_ARROW_B) / 2.0,
         "arrival within one\ncoincidence window $\\tau$", color=C["rule"],
         va="center")


# ---------------------------------------------------------------------------
# Panel b
# ---------------------------------------------------------------------------

def draw_prediction(fig, canvas_ax) -> None:
    """Eq. (column_diameter) as a relation, plus the worked value beside it.

    Only manuscript numbers appear: the 1-2 ms window bounds the band, the
    0.3 m/s velocity is marked, and the 300-600 um it implies is the bar. The
    velocity axis is swept to give the product an extent, and the two areas in
    which horizontal intracortical conduction has been measured directly are
    overlaid on it (MEASURED). Two points are not a regression, and the caption
    says so; what they establish is that the relation is not being asked to
    hold at a single calibration point.
    """
    ax = axes_mm(fig, PB_X, PB_Y, PB_W, PB_H)
    colour = PALETTE["purple"]

    v = np.linspace(0.0, V_MAX, 101)
    lower, upper = v * TAU_MIN * 1e3, v * TAU_MAX * 1e3
    ax.fill_between(v, lower, upper, facecolor=tint(colour, 0.16),
                    linewidth=0, zorder=1)
    for edge in (upper, lower):
        ax.plot(v, edge, color=colour, lw=0.8, zorder=2)
    # Both bounds are labelled at the right, where the wedge is widest: a label
    # anywhere else is crossed by one of the two diverging lines.
    ax.text(V_MAX * 0.99, 980.0, f"$\\tau$ = {TAU_MAX:g} ms", color=colour,
            fontsize=PT_ANNOT, ha="right", va="top", zorder=5)
    ax.text(V_MAX * 0.99, 620.0, f"$\\tau$ = {TAU_MIN:g} ms", color=colour,
            fontsize=PT_ANNOT, ha="right", va="top", zorder=5)

    # The one velocity the manuscript supplies, and the diameter it implies.
    ax.plot([V_AXON, V_AXON], [0.0, 1285.0], color=C["rule"], lw=0.5,
            linestyle=dashed((1.4, 1.4)), zorder=2)
    for d in (D_MIN, D_MAX):
        ax.plot([0.0, V_AXON], [d, d], color=C["faint"], lw=0.5,
                linestyle=dashed((0.8, 1.0)), zorder=1)
    ax.add_patch(mpatches.Rectangle((V_AXON - 0.008, D_MIN), 0.016,
                                    D_MAX - D_MIN, facecolor=colour,
                                    edgecolor="none", zorder=4))
    ax.text(V_AXON - 0.018, 630.0, f"{D_MIN:.0f}{EN}{D_MAX:.0f} $\\mu$m",
            color=colour, fontsize=PT_ANNOT, ha="right", va="bottom", zorder=5)
    ax.text(V_AXON + 0.014, 1330.0, f"$v \\approx {V_AXON:g}$ m s$^{{-1}}$",
            color=C["rule"], fontsize=PT_ANNOT, ha="left", va="top", zorder=5)

    # The two areas with a directly measured horizontal conduction velocity.
    # Each is drawn as its column range against its measured v, so the reader
    # sees the datum as an interval and not as a point estimate it is not.
    for name, v_m, v_lo, v_hi, d_lo, d_hi in MEASURED:
        d_m = 0.5 * (d_lo + d_hi)
        # the column range, vertically
        ax.plot([v_m, v_m], [d_lo, d_hi], color=C["ink"], lw=1.1,
                solid_capstyle="butt", zorder=6)
        for d in (d_lo, d_hi):
            ax.plot([v_m - 0.007, v_m + 0.007], [d, d], color=C["ink"],
                    lw=0.8, zorder=6)
        # the measured velocity range, horizontally: a box, not a point
        ax.plot([v_lo, v_hi], [d_m, d_m], color=C["ink"], lw=0.6, zorder=5)
        for v in (v_lo, v_hi):
            ax.plot([v, v], [d_m - 18.0, d_m + 18.0], color=C["ink"],
                    lw=0.6, zorder=5)
        # Labels go on opposite sides: the two points sit close in x, and the
        # worked value's bar occupies the space between them.
        # both labels below their own marker: anywhere above collides with the
        # tau curve labels, which are anchored to the right edge
        note(ax, v_m, d_lo - 34.0, name, color=C["ink"], ha="center", va="top")

    ax.set_xlim(0.0, V_MAX * 1.03)
    ax.set_ylim(0.0, 1350.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.set_yticks([0, 300, 600, 900, 1200])
    ax.set_xlabel("Conduction velocity $v$ (m s$^{-1}$)")
    ax.set_ylabel("Column diameter ($\\mu$m)")

    # The equation itself, worked through, beside the relation it describes,
    # and nothing else: a sentence of interpretation is caption material, not
    # figure content. The falsifiability clause lives in caption().
    # Line spacing opened up for the 8-12 pt PLOS type scale (see
    # delaykey.plotting). At 5.5 pt these three lines cleared each other at
    # 4.8 mm apart and the summation rule sat in the gap; at 8 pt the rule fell
    # through the middle of the result line and read as a strikethrough.
    note(canvas_ax, X_DERIV, 28.0,
         "$d_\\mathrm{column} = v_\\mathrm{axon} \\times \\tau_\\mathrm{window}$",
         va="baseline")
    note(canvas_ax, X_DERIV + 3.0, 21.6,
         f"$=$ {V_AXON:g} m s$^{{-1}}$ $\\times$ ({TAU_MIN:g}{EN}{TAU_MAX:g} ms)",
         va="baseline")
    canvas_ax.plot([X_DERIV + 3.0, X_DERIV + 23.0], [18.4, 18.4],
                   color=C["rule"], lw=0.8, solid_capstyle="butt", zorder=3)
    note(canvas_ax, X_DERIV + 3.0, 14.6,
         f"$=$ {D_MIN:.0f}{EN}{D_MAX:.0f} $\\mu$m", va="baseline")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def draw_geometry(ax):
    """Plan view of L4: where two integration windows overlap is a column.

    A *tangential* section, looking down on the sheet that panel a cuts through
    from the side. Each afferent's iso-arrival-time contours run perpendicular
    to it and are spaced one window's conduction apart, so a target sitting
    between one pair of contours receives that afferent's input inside one
    window. A column is where one such strip from each of several afferents
    overlaps, and its width is therefore v x tau in every direction that has an
    afferent to set it.

    That is what makes the cross-section round. Two orthogonal afferents give a
    square cell; adding the two diagonals clips its corners to an octagon; and
    afferents arriving at many angles, which is the realistic case for L4,
    drive the cell to a disc. The circularity of a column in tangential section
    is thus a consequence of the same construction that fixes its diameter,
    rather than an extra assumption -- which is why the faint circle is drawn
    through the octagon here.

    Suggested by Cheng Bi.
    """
    cx, cy = GEOM_CX, GEOM_CY
    s_mm = GEOM_S            # half-width of the cell: v x tau is 2 * s_mm
    reach = GEOM_REACH

    # Four afferent orientations. 0 and 90 degrees are the orthogonal pair the
    # estimate adopts; the two diagonals are what round the cell off.
    angles = [0.0, 90.0, 45.0, 135.0]
    axon_colour = [C["b"], C["b"], tint(C["b"], 0.55), tint(C["b"], 0.55)]

    for ang, colour in zip(angles, axon_colour):
        t = np.radians(ang)
        ux, uy = np.cos(t), np.sin(t)
        # The afferent itself.
        ax.plot([cx - reach * ux, cx + reach * ux],
                [cy - reach * uy, cy + reach * uy],
                color=colour, lw=1.1, solid_capstyle="round", zorder=2)
        # Its two iso-arrival-time contours, perpendicular to it, one window's
        # conduction apart. Drawn only across the cell they bound, so four
        # orientations do not turn the panel into a lattice.
        for sign in (-1.0, 1.0):
            mx, my = cx + sign * s_mm * ux, cy + sign * s_mm * uy
            ax.plot([mx - GEOM_TICK * uy, mx + GEOM_TICK * uy],
                    [my + GEOM_TICK * ux, my - GEOM_TICK * ux],
                    color=C["rule"], lw=0.7, linestyle=dashed(), zorder=3)

    # The cell: the intersection of all four strips, which is a regular octagon.
    k = s_mm * (np.sqrt(2.0) - 1.0)
    octagon = [(s_mm, k), (k, s_mm), (-k, s_mm), (-s_mm, k),
               (-s_mm, -k), (-k, -s_mm), (k, -s_mm), (s_mm, -k)]
    ax.add_patch(mpatches.Polygon(
        [(cx + x, cy + y) for x, y in octagon], closed=True,
        facecolor=tint(C["a"], 0.38), edgecolor=C["a"], lw=1.0, zorder=4))

    # The limit the cell tends to. Every orientation imposes |p . n| < s_mm, so
    # adding orientations drives the cell to the disc of radius exactly s_mm --
    # the octagon's inradius, tangent to each of its eight edges. Drawn at that
    # radius rather than a decorative one, so the figure asserts the right
    # limit.
    ax.add_patch(mpatches.Circle((cx, cy), s_mm, facecolor="none",
                                 edgecolor=C["ink"], lw=0.6,
                                 linestyle=(0, (1.1, 1.5)), zorder=6))

    # v x tau, measured across the cell between one pair of contours.
    y_dim = cy - s_mm - 2.6
    ax.annotate("", xy=(cx + s_mm, y_dim), xytext=(cx - s_mm, y_dim),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.7,
                                shrinkA=0, shrinkB=0), zorder=6)
    note(ax, cx, y_dim - 1.2, "$v \\times \\tau$", color=C["ink"],
         ha="center", va="top")

    note(ax, cx + reach + 1.4, cy + s_mm + 1.0, "afferents",
         color=C["b"], ha="left", va="center")
    note(ax, cx + reach + 1.4, cy - s_mm - 1.0, "integration",
         color=C["rule"], ha="left", va="center")
    note(ax, cx + reach + 1.4, cy - s_mm - 4.2, "windows",
         color=C["rule"], ha="left", va="center")
    note(ax, cx - reach - 1.4, cy, "column", color=C["a"],
         ha="right", va="center")


def build(out_stem, formats, geometry: bool = False):
    """Compose the panels and write the figure.

    Args:
        geometry: include the orthogonal-crossing panel. Off by default so the
            figure the original manuscript includes is unchanged; the
            restructured manuscript builds it on.
    """
    fig, ax = canvas(WIDTH, HEIGHT_MM)
    ax.set_autoscale_on(False)   # the ramp patches must not move the limits

    draw_sheet(ax)
    h1, h2 = draw_afferents(ax)
    draw_column_core(ax, COL_A_C, C["a"], h1, h2)
    draw_column_core(ax, COL_B_C, C["b"], h1, h2)
    draw_delay_scale(ax)
    draw_timing(ax)
    draw_prediction(fig, ax)

    panel_letter(ax, 1.0, HEIGHT_MM - 0.7, "a")
    if geometry:
        draw_geometry(ax)
        panel_letter(ax, 1.0, 33.8, "b")
        panel_letter(ax, PB_LETTER_X_GEOM, 33.8, "c")
    else:
        panel_letter(ax, 1.0, 33.8, "b")

    return save_figure(fig, out_stem, formats=tuple(formats))


def caption(geometry: bool = False) -> str:
    """The replacement caption, with bold panel letters."""
    geom = (
        f"\\textbf{{b}}, Why the construction fixes a column's shape as well as "
        f"its size. A tangential view of L4, looking down on the sheet that "
        f"\\textbf{{a}} cuts through from the side. Each afferent's "
        f"iso-arrival-time contours (dashed) run perpendicular to it and are "
        f"spaced one window's conduction apart, so a target lying between one "
        f"pair receives that afferent's input inside a single window; the "
        f"column is where one such strip from each converging afferent "
        f"overlaps, and is therefore $v \\times \\tau$ across in every "
        f"direction an afferent constrains. Two orthogonal afferents -- the "
        f"pair the estimate adopts -- give a square cell; the two diagonals "
        f"drawn here clip its corners to an octagon; and afferents arriving at "
        f"many angles drive it to the inscribed disc (dotted), of the same "
        f"diameter. The approximately circular cross-section of a column in "
        f"tangential section is thus a consequence of the same construction "
        f"that fixes its diameter, not a separate fact about it. "
    ) if geometry else ""
    plot_letter = "c" if geometry else "b"
    return (
        f"\\textbf{{The delay gradient of LGN axons maps firing latency onto "
        f"columns and fixes their diameter.}} "
        f"\\textbf{{a}}, Two LGN neurons with horizontally extending long "
        f"axons (H$_1$, H$_2$) reach L4 through the white matter, where they "
        f"are myelinated (internodal sheaths separated by nodes of Ranvier), "
        f"shed that myelin at the L4 boundary, and "
        f"run in opposite directions across column~A and column~B. The "
        f"antiparallel arrangement is drawn because it makes the mapping from "
        f"firing latency to columnar position legible; it is not the geometry "
        + ("the diameter estimate assumes, which is the orthogonal crossing "
           "shown in tangential view in \\textbf{b}. "
           if geometry else
           "the diameter estimate assumes, which is orthogonal crossing (see "
           "text). Antiparallel afferents would halve the predicted diameter. ")
        + f"Colour "
        f"along each intracortical axon is the conduction delay accumulated "
        f"from that axon's own entry point (key above the sheet), which "
        f"increases continuously with distance; for H$_1$ the delay to "
        f"column~A is smaller than the delay to column~B ($d_A < d_B$, "
        f"brackets). The core of each column is a cluster of local terminals "
        f"with near-equal delays converging on an L4 stellate cell, which is "
        f"the synchronous integration site; column boundaries (dashed) run "
        f"perpendicular to the direction of axonal extension. Below, the same "
        f"H$_1$ neuron fires at two latencies under different stimuli---"
        f"earlier for a preferred stimulus ($t_1$), later for a non-preferred "
        f"one ($t_2$). Because $t_1 + d_B = t_2 + d_A$, the earlier spike "
        f"arrives in register with other LGN input at the more distal "
        f"column~B, and the later spike at the more proximal column~A, both "
        f"inside one coincidence window $\\tau$ (shaded). Neighbouring columns "
        f"therefore read out different firing latencies of the same LGN "
        f"neuron, and the arrangement of columns is continuous rather than "
        f"discrete. "
        f"{geom}"
        f"\\textbf{{{plot_letter}}}, The diameter that follows, Eq.~\\ref{{eq:column_diameter}}: "
        f"predicted column diameter against horizontal conduction velocity, "
        f"banded over the effective coincidence window that feedforward "
        f"inhibition can impose ($\\tau = {TAU_MIN:g}$--{TAU_MAX:g}~ms). At the "
        f"thalamocortical velocity reported for L4 ($v \\approx {V_AXON:g}$~m/s, "
        f"dashed line) the relation gives {D_MIN:.0f}--{D_MAX:.0f}~$\\mu$m "
        f"(bar), the range quoted in the text. The relation is falsifiable: an "
        f"effective window measured in the awake animal that is substantially "
        f"broader than {TAU_MAX:g}~ms, together with a failure of manipulations "
        f"of that window to alter columnar feature tuning, would refute the "
        f"mechanism."
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("figures/fig_column_diameter"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    # The restructured manuscript builds this on; the original does not, so its
    # figure and its two-panel caption stay exactly as they were.
    parser.add_argument("--geometry", action="store_true",
                        help="add the orthogonal-crossing panel between the "
                             "plot and the worked equation")
    args = parser.parse_args(argv)

    if args.geometry:
        # The panel takes its width off the plot, not off the canvas: the
        # figure's outline and panel a are untouched.
        globals()["PB_X"] = PB_X_GEOM
        globals()["PB_W"] = PB_W_GEOM

    written = build(args.out, args.formats, geometry=args.geometry)
    for path in written:
        print(f"  wrote {path}  ({path.stat().st_size / 1024:.0f} kB)")

    # A substituted glyph embeds a second face silently, and nothing else in
    # the build says so. Expect whatever figstyle resolved, so this keeps
    # tracking the shared module if its preference order changes again.
    for path in written:
        if path.suffix == ".pdf":
            stray = audit_fonts(path, expected=(resolve_font(warn=False)
                                                .replace(" ", ""),))
            print(f"  fonts: {'clean' if not stray else 'STRAY ' + str(stray)}")

    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption(geometry=args.geometry) + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

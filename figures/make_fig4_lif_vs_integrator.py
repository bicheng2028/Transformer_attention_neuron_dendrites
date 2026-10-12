"""Build Fig 4: a LIF control against the delay-key integrator neuron.

Laid out to the Nature artwork specification (see :mod:`delaykey.plotting` and
:mod:`figstyle`): authored at :data:`figstyle.WIDTH_MM`, the width the document
actually places the graphic at, so ``\\includegraphics[width=\\figwidth]``
scales it by exactly 1.00; all text between 5 and 7 pt; 8 pt bold lower-case
panel letters; no panel titles, no grid, and curves named in place rather than
through a legend box.

Panels:

a  Presynaptic spikes and their arrival times at the distal branch, for both
   orders, after the K-matrix delays are applied. This is the step the previous
   version of the figure never showed, and it is where the whole result comes
   from: under A->B the delays cancel the spike-time difference and both inputs
   land at t = 10; under B->A they land 10 steps apart.
b  Leaky integrate-and-fire control, both orders overlaid. The trajectories are
   the same array, so the panel shows one curve carrying two labels -- a more
   direct demonstration of order-blindness than two panels compared by memory.
c  Integrator neuron, both orders. A->B crosses threshold and fires at t = 10;
   B->A never leaves the subthreshold V level.

The simulation is the one the manuscript quotes and is deliberately untouched:
:func:`lif` and :func:`integrator` and every parameter below are carried over
verbatim from the generator this script replaces. Only the presentation changed.

Geometry is expressed in millimetres on the printed page, because the
annotations here (raster rows, delay arrows, in-axes notes) have to clear one
another at 5.5 pt and that is a physical constraint, not a proportional one.
Every horizontal position is *derived* from :data:`figstyle.WIDTH_MM` rather
than written out, so a correction to the target column width re-lays the figure
out instead of leaving a band of unused canvas down one side. The vertical
stack is written in millimetres directly, since it is set by type metrics that
do not move with the column width.

Usage::

    python figures/make_fig4_lif_vs_integrator.py --formats pdf png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import (C, PT_ANNOT, PT_TICK, WIDTH, WIDTH_MM,  # noqa: E402
                      apply_style, direct_label, figure_size, label_panels,
                      save_figure)

# figstyle imports delaykey.plotting, which selects the Agg backend, so this is
# safe here.
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Simulation. Every value here is quoted in the manuscript text or caption.
# ---------------------------------------------------------------------------

T = 30                  # steps simulated
TAU = 5.0               # membrane time constant tau_m
DECAY = np.exp(-1.0 / TAU)
THETA = 1.0             # firing threshold
W_LIF = 0.5             # LIF synaptic weight, equal for both inputs
V_CONTRIB = 0.10        # proximal V drive per step, held subthreshold
V0 = 0.10               # initial integrator potential
ALPHA = 2.5             # distal plateau drive delivered by a matched vote

T_FIRST, T_SECOND = 5, 10   # the two stimulus times
D_A, D_B = 5, 0             # K-matrix delays on the distal branch
DELAY = {"A": D_A, "B": D_B}
T_VOTE = 10                 # A->B coincidence, and the spike, land here


def lif(spikes, w=W_LIF):
    """Membrane trace of a leaky integrate-and-fire neuron with no delays.

    The control model: it sees a weighted sum of spikes and nothing about their
    order, so passing either stimulus set gives the same array.
    """
    v, out = 0.0, []
    for k in range(T):
        v *= DECAY
        if k in spikes:
            v += w
        out.append(v)
    return np.array(out)


def integrator(vote_time, v_drive=V_CONTRIB, alpha=ALPHA, v0=V0):
    """Membrane trace of the integrator neuron.

    Proximal V supplies ``v_drive`` every step, which alone saturates well below
    threshold. A matched distal coincidence at ``vote_time`` adds ``alpha``, the
    plateau drive that carries the soma over threshold. ``None`` means no
    coincidence occurred.
    """
    v, out = v0, []
    for k in range(T):
        v = v * DECAY + v_drive
        if vote_time is not None and k == vote_time:
            v += alpha
        out.append(v)
    return np.array(out)


def arrivals(order):
    """Arrival times at the distal branch for one stimulus order.

    Args:
        order: ``("A", "B")`` for A first, ``("B", "A")`` for B first.

    Returns:
        ``[(time, name), ...]`` sorted by arrival time, where ``time`` is the
        spike time plus that input's K-matrix delay.
    """
    fired = zip((T_FIRST, T_SECOND), order)
    return sorted((t + DELAY[name], name) for t, name in fired)


# ---------------------------------------------------------------------------
# Page layout.
# ---------------------------------------------------------------------------

MM_PER_IN = 25.4
HEIGHT_MM = 84.0

# Horizontal. Only the three margins are absolute; they are set by type that
# does not change size with the column width. Everything else follows from
# WIDTH_MM.
LEFT_MM = 17.5          # clears panel a's row names and b/c's axis furniture
GUTTER_MM = 17.0        # panel c's own y-axis label and ticks, plus its letter
TRIM_MM = 1.9           # bare canvas kept beyond the rightmost rule
RIGHT_MM = WIDTH_MM - TRIM_MM
ROW_W_MM = (RIGHT_MM - LEFT_MM - GUTTER_MM) / 2.0

# Vertical, bottom-up: axis furniture, the two trace panels, the raster.
#
# Re-spaced for the 8-12 pt PLOS type scale (see delaykey.plotting). At the old
# 5.5-7 pt sizes these panels sat 8.6 mm apart, which was enough; at 9 pt axis
# labels and 10 pt panel letters, panel a's "Time (steps)" and the b/c letters
# landed on the same line and overprinted. The gap is now 13 mm, and the trace
# panels are taller so the larger y-labels are not cramped against them.
BC_Y_MM, BC_H_MM = 10.5, 29.0
A_Y_MM, A_H_MM = 52.5, 27.0

LETTER_AB_X_MM = 5.0    # a and b share a letter column
LETTER_C_X_MM = LEFT_MM + ROW_W_MM + GUTTER_MM - 13.0

#: Time window for the trace panels: the traces are still decaying at t = 29.
TLIM = (-1.0, 29.8)
TTICKS = [0, 5, 10, 15, 20, 25]

#: Time window for the raster, which has no marks after the last arrival at
#: t = 15. Carrying it out to 29 to match b and c would spend a third of the
#: full-width panel on blank time; the two axes are separately labelled and
#: this panel is a raster rather than a trace, so the shorter window cannot be
#: mistaken for alignment with the curves below.
A_TLIM = (-1.0, 21.0)
A_TTICKS = [0, 5, 10, 15, 20]

# Panel a: raster row baselines. The gap between the two blocks is set wider
# than the gap between the rows inside a block, and the within-block gap is
# what has to hold a 5.5 pt delay label between two 5.5 pt event names.
ROW_GAP = 1.30
BLOCK_GAP = 1.90
ROW_AB_IN = 2.95
ROW_AB_ARR = ROW_AB_IN - ROW_GAP
ROW_BA_IN = ROW_AB_ARR - BLOCK_GAP
ROW_BA_ARR = ROW_BA_IN - ROW_GAP

TICK_HALF = 0.30        # raster tick half-height, in row units
HEAVY = 1.45            # how much taller a coincident arrival tick is drawn
# Bottom margin keeps the last row of ticks clear of the time axis; the top
# margin only has to clear the ticks, since the panel letter sits outside.
A_YLIM = (ROW_BA_ARR - TICK_HALF - 0.55, ROW_AB_IN + TICK_HALF + 0.20)

NAME_GAP_MM = 2.0       # event name to its tick
LABEL_GAP_MM = 1.2      # delay label to the arrow it names


def axes_at(fig, x_mm, y_mm, w_mm, h_mm):
    """Place an axes at millimetre coordinates measured from the bottom left."""
    fw = fig.get_figwidth() * MM_PER_IN
    fh = fig.get_figheight() * MM_PER_IN
    return fig.add_axes([x_mm / fw, y_mm / fh, w_mm / fw, h_mm / fh])


def letter_at(fig, ax, text, x_mm, gap_mm=0.6):
    """Letter a panel at an absolute millimetre position.

    :func:`label_panels` offsets by a fraction of each axes, which puts the
    letters of a wide panel and a narrow one in different columns. Converting a
    millimetre target back into that fraction keeps them on one alignment.
    """
    box = ax.get_position()
    fw = fig.get_figwidth() * MM_PER_IN
    fh = fig.get_figheight() * MM_PER_IN
    label_panels([ax], [text],
                 dx=(x_mm - box.x0 * fw) / (box.width * fw),
                 dy=gap_mm / (box.height * fh))


def mm_x(ax, mm):
    """A millimetre distance on the page, in this axes' x data units."""
    lo, hi = ax.get_xlim()
    return mm * (hi - lo) / (ax.get_position().width * WIDTH_MM)


def mm_y(ax, mm):
    """A millimetre distance on the page, in this axes' y data units."""
    lo, hi = ax.get_ylim()
    return mm * (hi - lo) / (ax.get_position().height * HEIGHT_MM)


def right_align(ax, texts, pad_mm=2.5):
    """Left-align a group of notes so the widest ends near the panel's edge.

    Anchoring a text block to the axes edge rather than to a data coordinate is
    what keeps it filling the panel when the target column width changes: a
    hardcoded data x that sat comfortably at one width leaves a band of white
    at another.

    Returns:
        The common left edge, in data units, so line samples or markers can be
        placed against it.
    """
    renderer = ax.figure.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer).width for t in texts)
    inverse = ax.transData.inverted()
    span = (inverse.transform((widest, 0.0))[0]
            - inverse.transform((0.0, 0.0))[0])
    x = ax.get_xlim()[1] - mm_x(ax, pad_mm) - span
    for text in texts:
        text.set_x(x)
    return x


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def raster_row(ax, y, events, colour):
    """Draw one raster row: a tick per event, each named just to its right.

    Names sit beside the ticks rather than above them because the rows are only
    a few millimetres apart; horizontal room is what this panel has spare.

    Args:
        events: ``(time, name, heavy)`` triples. ``heavy`` draws the tick both
            thicker and taller, which is how two coincident arrivals are told
            apart from a single one.
    """
    gap = mm_x(ax, NAME_GAP_MM)
    for time, name, heavy in events:
        half = TICK_HALF * (HEAVY if heavy else 1.0)
        ax.plot([time, time], [y - half, y + half], color=colour,
                lw=1.0 if heavy else 0.7, solid_capstyle="butt", zorder=3)
        ax.text(time + gap, y, name, color=colour, fontsize=PT_ANNOT,
                ha="left", va="center", zorder=3)


def delay_arrow(ax, t_in, y_in, t_out, y_out, half_out=TICK_HALF):
    """Carry one input from its spike time down to its arrival time.

    Drawn neutral grey in both blocks, because the delays are a property of the
    branch and are identical for the two orders; colouring them by sequence
    would suggest otherwise. ``half_out`` is the height of the tick being
    landed on, so the head stops on it rather than inside it.
    """
    ax.annotate("", xy=(t_out, y_out + half_out),
                xytext=(t_in, y_in - TICK_HALF),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=3.0,
                                linewidth=0.5, color=C["miss"],
                                shrinkA=0.0, shrinkB=0.0), zorder=2)


def threshold_rule(ax):
    """Draw and name the firing threshold as a hairline, not a legend entry."""
    ax.axhline(THETA, color=C["rule"], lw=0.5, linestyle=(0, (2.4, 1.8)),
               zorder=1)
    ax.text(ax.get_xlim()[1] - mm_x(ax, 0.7), THETA + mm_y(ax, 0.35),
            r"$\theta$", color=C["rule"], fontsize=PT_ANNOT, ha="right",
            va="bottom", zorder=2)


def coincidence_rule(ax, y_top):
    """Mark t = 10, the moment the two inputs meet, with a dotted guide."""
    ax.plot([T_VOTE, T_VOTE], [ax.get_ylim()[0], y_top], color=C["rule"],
            lw=0.4, linestyle=(0, (1.0, 1.6)), zorder=0)


def model_label(ax, text):
    """Name the model inside its axes, top left, in place of a panel title."""
    ax.text(ax.get_xlim()[0] + mm_x(ax, 1.2),
            ax.get_ylim()[1] - mm_y(ax, 0.6), text, color=C["ink"],
            fontsize=PT_ANNOT, ha="left", va="top")


def panel_a(ax):
    """Inputs and arrivals: where the order difference is actually created."""
    ax.set_xlim(*A_TLIM)
    ax.set_ylim(*A_YLIM)

    for order, y_in, y_arr, colour in (
            (("A", "B"), ROW_AB_IN, ROW_AB_ARR, C["a"]),
            (("B", "A"), ROW_BA_IN, ROW_BA_ARR, C["b"])):
        first, second = order
        raster_row(ax, y_in, [(T_FIRST, first, False),
                              (T_SECOND, second, False)], colour)

        landed = arrivals(order)
        coincident = landed[0][0] == landed[1][0]
        if coincident:
            names = [(landed[0][0], f"{first},{second}", True)]
        else:
            names = [(t, name, False) for t, name in landed]
        raster_row(ax, y_arr, names, colour)

        half_out = TICK_HALF * (HEAVY if coincident else 1.0)
        for spike, name in zip((T_FIRST, T_SECOND), order):
            delay_arrow(ax, spike, y_in, spike + DELAY[name], y_arr, half_out)

    # No coincidence rule here. At t = 10 it would lie exactly on top of the
    # zero-delay arrow and the B->A spike, and the three marks together read as
    # one long dotted connector. The converging arrowheads and the heavy
    # arrival tick already say where the two inputs meet.

    # Both delays are named in the A->B block, on the arrows that apply them,
    # and named once: the arrows are drawn identically in the B->A block, which
    # is the point -- one K matrix, two orders. d_A sits just above A's long
    # arrow at its midpoint. d_B is a stub, so its label goes beside it, in the
    # gap between the "B" and "A,B" names. That gap is what set this panel's
    # height: it has to clear 5.5 pt type above and below.
    start = (T_FIRST, ROW_AB_IN - TICK_HALF)
    end = (T_VOTE, ROW_AB_ARR + TICK_HALF * HEAVY)
    mid_x = 0.5 * (start[0] + end[0])
    mid_y = 0.5 * (start[1] + end[1])
    ax.text(mid_x, mid_y + mm_y(ax, 0.4), f"$d_A = {D_A}$", color=C["ink"],
            fontsize=PT_ANNOT, ha="center", va="bottom")
    ax.text(T_VOTE + mm_x(ax, LABEL_GAP_MM), 0.5 * (ROW_AB_IN + ROW_AB_ARR),
            f"$d_B = {D_B}$", color=C["ink"], fontsize=PT_ANNOT, ha="left",
            va="center")

    # One verdict per block, anchored to the right edge of the panel. They
    # state the consequence of the arrival pattern, which is the one thing the
    # ticks themselves cannot say.
    right_align(ax, [
        ax.text(0.0, 0.5 * (ROW_AB_IN + ROW_AB_ARR),
                f"coincidence at $t = {T_VOTE}$", color=C["a"],
                fontsize=PT_ANNOT, ha="left", va="center"),
        ax.text(0.0, 0.5 * (ROW_BA_IN + ROW_BA_ARR), "no coincidence",
                color=C["b"], fontsize=PT_ANNOT, ha="left", va="center"),
    ])

    ax.set_yticks([ROW_AB_IN, ROW_AB_ARR, ROW_BA_IN, ROW_BA_ARR])
    ax.set_yticklabels([r"A$\to$B", "arrival", r"B$\to$A", "arrival"])
    for label, colour in zip(ax.get_yticklabels(),
                             (C["a"], C["a"], C["b"], C["b"])):
        label.set_color(colour)
    # A raster's vertical axis is categorical: ticks and a spine on it would be
    # furniture with nothing to measure.
    ax.tick_params(axis="y", length=0, pad=2.0, labelsize=PT_TICK)
    ax.spines["left"].set_visible(False)
    ax.set_xticks(A_TTICKS)
    ax.set_xlabel("Time (steps)")


def panel_b(ax, lif_ab, lif_ba, t):
    """The LIF control, both orders on one axes, because they are one curve."""
    ax.set_xlim(*TLIM)
    ax.set_ylim(0.0, 1.30)
    threshold_rule(ax)

    ax.plot(t, lif_ab, color=C["a"], lw=1.0, zorder=3)
    ax.plot(t, lif_ba, color=C["b"], lw=1.0, dashes=(1.6, 1.4), zorder=4)

    model_label(ax, "LIF control")

    # A two-entry key drawn by hand: with the curves exactly superimposed there
    # is no separated point to label, and a framed legend is not an option. The
    # note is the widest line, so it sets where the block starts.
    note = ax.text(0.0, 0.54, "superimpose exactly", color=C["ink"],
                   fontsize=PT_ANNOT, ha="left", va="center")
    x_key = right_align(ax, [note])
    sample = mm_x(ax, 3.2)
    for y, colour, dashes in ((0.88, C["a"], None), (0.72, C["b"], (1.6, 1.4))):
        ax.plot([x_key, x_key + sample], [y, y], color=colour, lw=1.0,
                dashes=dashes or (1, 0), solid_capstyle="butt", zorder=3)
    direct_label(ax, x_key + sample, 0.88, r"A$\to$B", C["a"])
    direct_label(ax, x_key + sample, 0.72, r"B$\to$A", C["b"])

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(TTICKS)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Membrane potential")


def panel_c(ax, intg_ab, intg_ba, t):
    """The integrator: one order fires, the other stays on subthreshold V."""
    ax.set_xlim(*TLIM)
    ax.set_ylim(0.0, 3.55)
    threshold_rule(ax)
    coincidence_rule(ax, intg_ab.max())

    ax.plot(t, intg_ab, color=C["a"], lw=1.0, zorder=3)
    ax.plot(t, intg_ba, color=C["b"], lw=1.0, dashes=(1.6, 1.4), zorder=4)

    model_label(ax, "Integrator")

    # The spike is a tick over the peak, not a shout.
    peak = intg_ab[T_VOTE] + mm_y(ax, 1.0)
    ax.plot([T_VOTE], [peak], marker="v", markersize=2.2, color=C["a"],
            zorder=5)
    direct_label(ax, T_VOTE + mm_x(ax, 0.8), peak, f"spike at $t = {T_VOTE}$",
                 C["a"])

    direct_label(ax, 12.6, 2.32, r"A$\to$B", C["a"])
    # Below the flat trace: the two curves converge again towards t = 30, so
    # there is no room for a label above it that the A->B decay does not cross.
    # This is the only name the flat trace carries -- it is also the subthreshold
    # V level, but labelling one curve twice says less than the caption does.
    direct_label(ax, 14.4, 0.27, r"B$\to$A", C["b"])

    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks(TTICKS)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Membrane potential")


def caption() -> str:
    """The replacement caption, built from the constants so it cannot drift.

    Parameters keep the decimal places the superseded caption used, because the
    manuscript body quotes them in that form.
    """
    split = arrivals(("B", "A"))
    return (
        f"\\textbf{{Delay compensation, not synaptic weight, discriminates "
        f"temporal order.}} Two feature neurons, A and B, drive one model "
        f"neuron over {T} steps. The A$\\rightarrow$B pattern places A at "
        f"$t={T_FIRST}$ and B at $t={T_SECOND}$; B$\\rightarrow$A exchanges "
        f"them. \\textbf{{a}}, Presynaptic spikes and the times at which they "
        f"reach the distal branch, once the K-matrix delays $d_A={D_A}$ and "
        f"$d_B={D_B}$ have been applied. Under A$\\rightarrow$B the delays "
        f"cancel the spike-time difference and both inputs land together at "
        f"$t={T_VOTE}$ (heavy tick); under B$\\rightarrow$A they land at "
        f"$t={split[0][0]}$ and $t={split[1][0]}$, and no coincidence occurs. "
        f"\\textbf{{b}}, Leaky integrate-and-fire control with the same "
        f"membrane parameters ($\\tau_m={TAU:.1f}$, $\\theta={THETA:.1f}$), "
        f"equal synaptic weights $w={W_LIF:g}$ and no delays. The two orders give "
        f"identical trajectories, because integration depends only on the "
        f"weighted spike sum; the curves superimpose exactly and neither "
        f"approaches $\\theta$. \\textbf{{c}}, Integrator neuron, same "
        f"$\\tau_m$ and $\\theta$. Proximal V input is held subthreshold "
        f"($V_{{\\mathrm{{contrib}}}}={V_CONTRIB:.2f}$ per step from "
        f"$v_0={V0:.2f}$, threshold {THETA:.1f}), so V alone can never fire the "
        f"cell and distal coincidence detection is required. The matched "
        f"A$\\rightarrow$B coincidence delivers a plateau drive "
        f"$\\alpha={ALPHA:g}$, carrying the soma over $\\theta$ for a single "
        f"spike at $t={T_VOTE}$ (tick), whereas B$\\rightarrow$A never leaves "
        f"the subthreshold V level. All membrane parameters are identical "
        f"across the two models."
    )


def build():
    """Assemble the figure. Returns ``(fig, [ax_a, ax_b, ax_c])``."""
    t = np.arange(T)
    apply_style()
    fig = plt.figure(figsize=figure_size(WIDTH, HEIGHT_MM))
    ax_a = axes_at(fig, LEFT_MM, A_Y_MM, RIGHT_MM - LEFT_MM, A_H_MM)
    ax_b = axes_at(fig, LEFT_MM, BC_Y_MM, ROW_W_MM, BC_H_MM)
    ax_c = axes_at(fig, LEFT_MM + ROW_W_MM + GUTTER_MM, BC_Y_MM, ROW_W_MM,
                   BC_H_MM)

    panel_a(ax_a)
    panel_b(ax_b, lif({T_FIRST, T_SECOND}), lif({T_FIRST, T_SECOND}), t)
    panel_c(ax_c, integrator(T_VOTE), integrator(None), t)

    letter_at(fig, ax_a, "a", LETTER_AB_X_MM)
    letter_at(fig, ax_b, "b", LETTER_AB_X_MM)
    letter_at(fig, ax_c, "c", LETTER_C_X_MM)
    return fig, [ax_a, ax_b, ax_c]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path,
                        default=Path("figures/fig4_lif_vs_integrator"))
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = parser.parse_args(argv)

    lif_ab = lif({T_FIRST, T_SECOND})   # A at 5, B at 10
    lif_ba = lif({T_FIRST, T_SECOND})   # identical: the LIF is order-blind
    intg_ab = integrator(T_VOTE)        # coincidence at t = 10 -> vote -> spike
    intg_ba = integrator(None)          # no coincidence -> no spike

    fired = np.flatnonzero(intg_ab >= THETA)
    silent = np.flatnonzero(intg_ba >= THETA)
    print("dynamics")
    print(f"  arrivals A->B      {arrivals(('A', 'B'))}")
    print(f"  arrivals B->A      {arrivals(('B', 'A'))}")
    print(f"  LIF orders equal   {np.array_equal(lif_ab, lif_ba)}"
          f"  (peak {lif_ab.max():.4f} < theta {THETA:g})")
    print(f"  integrator A->B    first crossing t = {fired[0]}"
          f"  (peak {intg_ab.max():.4f} at t = {int(intg_ab.argmax())})")
    print(f"  integrator B->A    crossings {silent.tolist()}"
          f"  (peak {intg_ba.max():.4f} < theta {THETA:g})")
    print(f"  canvas             {WIDTH_MM:g} x {HEIGHT_MM:g} mm"
          f"  = {WIDTH_MM / MM_PER_IN * 72:.1f} x "
          f"{HEIGHT_MM / MM_PER_IN * 72:.1f} pt")

    fig, _ = build()
    for path in save_figure(fig, args.out, formats=tuple(args.formats)):
        print(f"  wrote {path}")

    caption_path = Path(args.out).with_suffix(".caption.tex")
    caption_path.write_text(caption() + "\n")
    print(f"  wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

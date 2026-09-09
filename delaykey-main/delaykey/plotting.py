"""Figure style built to the Nature artwork specification.

Sourced from Nature's *Guide to Preparing Final Artwork* and the Nature research
figure guide:

* Column widths are fixed: 89 mm (one column), 120 or 136 mm (1.5 columns),
  183 mm (two columns). Full page depth is 247 mm. Figures are supplied at the
  size they will be printed, so nothing is scaled afterwards and point sizes on
  the page are exactly the point sizes set here.
* All text is sans-serif, Helvetica or Arial, **7 pt maximum and 5 pt minimum**.
* Panels are lettered **8 pt bold, upright, lower case**: a, b, c.
* Text is never rasterised or converted to outlines, so fonts are embedded as
  TrueType (type 42).
* White space is kept to a minimum and panels are arranged on a tidy grid.
* Colour is supplied in RGB.

Two consequences worth stating, because they are what most self-made figures get
wrong. First, there are no panel titles: Nature figures carry a panel letter and
nothing else above the axes, and every word of description lives in the caption.
Second, keys belong inside the figure rather than as colour descriptions in the
caption, so curves are labelled directly wherever they can be.

The palette is Okabe-Ito, which stays distinguishable under all common forms of
colour vision deficiency and avoids the red/green pairing Nature warns against.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

__all__ = [
    "PALETTE", "COLUMN_MM", "MAX_HEIGHT_MM", "apply_style", "figure_size",
    "label_panels", "resolve_font", "save_figure", "tint", "direct_label",
    "PT_ANNOT", "PT_LABEL",
]

# Okabe-Ito. Colour-vision-deficiency safe; no red/green pairing.
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#8C8C8C",
}

#: Target widths in millimetres.
#:
#: The Nature entries are the only widths that journal accepts. Build to the
#: width the document will actually place the graphic at: a figure drawn at
#: 183 mm and dropped into a 133 mm column is scaled by 0.73, which silently
#: drags nominal 7 pt text down to about 5 pt on the printed page.
#:
#: ``plos`` is measured, not read off the template preamble, and the difference
#: is a trap worth stating. ``preamble-plos.tex`` sets ``\textwidth 5.25in``,
#: but that governs the **title page only**: the PLOS template then issues
#: ``\newgeometry{left=1in,right=1in}`` (``plos-body.tex``) before the body, so
#: ``\linewidth`` -- and therefore ``\figwidth`` -- is 6.5 in for every float in
#: the manuscript. Building to 5.25 in and placing at ``\figwidth`` silently
#: *enlarges* each figure by 1.24x, pushing nominal 7 pt text to about 8.7 pt.
#: Verify with ``\the\linewidth`` inside the body rather than trusting the
#: preamble; the two disagree.
COLUMN_MM = {
    "single": 89.0,          # Nature, one column
    "onehalf": 120.0,        # Nature, 1.5 columns
    "onehalf_wide": 136.0,   # Nature, 1.5 columns (wide variant)
    "double": 183.0,         # Nature, two columns
    "plos": 165.1,           # PLOS Comp Biol *body* text width (6.5 in)
    "plos_title": 133.35,    # PLOS title page only (5.25 in) -- not for figures
}
MAX_HEIGHT_MM = 247.0

_MM_PER_INCH = 25.4

# Type scale, set to the SUBMISSION target rather than to Nature's ceiling.
#
# This module used to sit at Nature's stated maximum for every size at once --
# 7 pt labels, 6 pt ticks, 5.5 pt annotations -- treating a ceiling as though it
# were a house style. Two things were wrong with that. Nature's 7 pt is an upper
# bound on a figure reproduced at final printed width, not an instruction to set
# everything as small as the rule allows; and this manuscript goes to PLOS
# Computational Biology, whose artwork specification asks for **8-12 pt Arial**.
# Four of the five sizes therefore sat *below the target journal's floor*, and
# the whole figure set read thin and washed out against surrounding body text.
#
# The scale below satisfies PLOS (nothing below 8 pt) and stays inside a range
# Nature would also accept for artwork reproduced at this width. Because the
# toolchain composes at printed size with scale 1.00, these are true points on
# the page: raising them makes labels physically larger on a canvas whose width
# is fixed by the column, so every generator's panel geometry was re-tuned to
# match. Do not change these without re-checking each figure at 100%.
_PT_LABEL = 9.0
_PT_TICK = 8.0
_PT_LEGEND = 8.0
_PT_ANNOTATION = 8.0
_PT_PANEL = 10.0

#: Public aliases, so generators can size in-axes annotations from the house
#: scale instead of hardcoding a number that then drifts when the scale moves.
PT_ANNOT = _PT_ANNOTATION
PT_LABEL = _PT_LABEL

#: Faces to try, in order. A fetched face comes first deliberately.
#:
#: Both fetched candidates are metric-compatible substitutes that satisfy
#: Nature's "Helvetica or Arial" requirement in substance, and both behave
#: identically on every machine because ``fonts/fetch.sh`` puts them there
#: rather than trusting whatever the host happens to have installed.
#:
#: Preferring the *system* face instead cost us twice, and both failures are
#: why a fetched face now leads:
#:
#: * macOS ships Helvetica as a ``.ttc`` collection that older matplotlib did
#:   not index and current matplotlib does. The same script therefore resolved
#:   to Nimbus Sans on one machine and Helvetica on another, so a figure's
#:   embedded fonts depended on the matplotlib version rather than on the code.
#: * macOS Helvetica has no U+2192. Any ``$\\rightarrow$`` in a label silently
#:   fell back to STIXGeneral for that one glyph, embedding a third face in the
#:   PDF -- two typefaces in one figure, which Nature forbids.
#:
#: Liberation Sans leads rather than Nimbus Sans because of how matplotlib
#: embeds fonts. ``pdf.fonttype: 42`` writes a ``/Subtype /CIDFontType2``
#: dictionary with a ``/FontFile2`` stream -- a *TrueType* declaration. Nimbus
#: Sans is CFF-flavoured OpenType, so that declaration wraps CFF bytes and the
#: result is malformed: poppler reports "Mismatch between font type and embedded
#: font file", and a strict preflight (Acrobat, PDF/X) may reject it. Liberation
#: Sans is genuine TrueType, so the declaration is true. It also draws a
#: conventional lowercase tau; Nimbus draws a flat-topped one that reads as a
#: Latin T at 5.5 pt.
#:
#: U+223C (``$\\sim$``) and U+207B (superscript minus) are absent from both, and
#: from most grotesques. Write "~" and compose superscripts through mathtext
#: (``$^{-1}$``) instead; :func:`~figstyle.audit_fonts` catches the fallout.
_PREFERRED_FONTS = ("Liberation Sans", "Nimbus Sans", "Helvetica", "Arial")

#: Optional directory of extra .ttf/.otf files registered before resolving.
FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"


def resolve_font(warn: bool = True) -> str:
    """Return the best available Nature-acceptable sans-serif face.

    Registers font files in ``fonts/`` alongside the package first, so a
    metric-compatible substitute can be vendored without root access.

    Args:
        warn: emit a warning when no accepted face is found.

    Returns:
        The resolved family name, falling back to ``"DejaVu Sans"``.
    """
    if FONT_DIR.is_dir():
        for path in sorted(FONT_DIR.glob("*.[to]tf")):
            font_manager.fontManager.addfont(str(path))

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _PREFERRED_FONTS:
        if name in available:
            return name

    if warn:
        warnings.warn(
            f"No Nature-accepted sans-serif face found "
            f"({', '.join(_PREFERRED_FONTS)}); falling back to DejaVu Sans. "
            f"Drop Helvetica, Arial or Liberation Sans TTFs into {FONT_DIR} "
            f"before final submission.",
            RuntimeWarning, stacklevel=2,
        )
    return "DejaVu Sans"


def apply_style() -> None:
    """Install the Nature-compliant rcParams. Call once per figure."""
    family = resolve_font()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],

        # Mathtext has its own font set, independent of font.family. Left at
        # the default, every $\Delta$ and $\theta$ renders in DejaVu while the
        # surrounding label renders in the chosen face -- two fonts in one
        # figure, which Nature explicitly forbids. Pointing mathtext at the
        # same family keeps symbols and prose consistent.
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.default": "regular",

        "font.size": _PT_LABEL,
        "axes.labelsize": _PT_LABEL,
        "axes.titlesize": _PT_LABEL,
        "xtick.labelsize": _PT_TICK,
        "ytick.labelsize": _PT_TICK,
        "legend.fontsize": _PT_LEGEND,

        # Weights raised with the type scale above. The previous hairlines were
        # chosen against 5.5-7 pt text; under 8-9 pt they read as faint and
        # unfinished, which is most of what made this figure set look washed
        # out. These are still light enough not to clutter at 165 mm.
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.0,
        "patch.linewidth": 0.8,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelpad": 3.0,
        "axes.titlepad": 3.0,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,

        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.2,
        "legend.columnspacing": 1.0,

        "figure.dpi": 600,
        # NOT "tight". A tight bounding box crops the canvas to whatever the
        # content happens to occupy, so the file is written at some width other
        # than the one requested and the journal then scales it -- shrinking
        # every point size away from the value set here. Column widths are
        # fixed, so the canvas is honoured exactly and margins are controlled
        # through the gridspec instead.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,

        # Nature: never outline text. Type 42 embeds TrueType and keeps it
        # selectable and editable in the vector output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure_size(width: str | float, height_mm: float) -> tuple[float, float]:
    """Return a figure size in inches from a Nature column width.

    Args:
        width: one of ``"single"``, ``"onehalf"``, ``"onehalf_wide"``,
            ``"double"``, or an explicit width in millimetres.
        height_mm: height in millimetres, at most :data:`MAX_HEIGHT_MM`.

    Returns:
        ``(width_in, height_in)``.

    Raises:
        ValueError: for an unknown keyword or an over-tall figure.
    """
    if isinstance(width, str):
        if width not in COLUMN_MM:
            raise ValueError(f"width must be one of {sorted(COLUMN_MM)} "
                             f"or a width in mm; got {width!r}")
        width_mm = COLUMN_MM[width]
    else:
        width_mm = float(width)
        if width_mm > COLUMN_MM["double"]:
            raise ValueError(f"width {width_mm} mm exceeds the two-column "
                             f"maximum {COLUMN_MM['double']} mm")

    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"height {height_mm} mm exceeds the page depth "
                         f"{MAX_HEIGHT_MM} mm")

    return width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH


def label_panels(axes, labels=None, dx: float = -0.055, dy: float = 0.035) -> None:
    """Letter panels 8 pt bold lower case, per the Nature specification.

    Letters are placed just outside the top-left corner of each axes so they
    never sit over data or shading.

    Args:
        axes: axes in reading order.
        labels: explicit labels; defaults to ``a``, ``b``, ``c``, ...
        dx: horizontal offset in axes fraction.
        dy: vertical offset in axes fraction.
    """
    axes = list(axes)
    if labels is None:
        labels = [chr(ord("a") + i) for i in range(len(axes))]
    for ax, text in zip(axes, labels):
        ax.text(dx, 1.0 + dy, text, transform=ax.transAxes,
                fontsize=_PT_PANEL, fontweight="bold", fontstyle="normal",
                va="bottom", ha="left")


def direct_label(ax, x, y, text, colour, dx: float = 1.2, ha: str = "left",
                 **kwargs) -> None:
    """Label a curve at a point instead of adding a legend entry.

    Nature asks for keys inside the figure rather than colour descriptions in
    the caption; labelling curves where they are separated is the most compact
    way to do that and removes the legend box entirely.

    ``ha`` selects which side of the anchor the text runs to. It matters for a
    steep curve: text runs horizontally while the curve climbs through it, so a
    vertical offset alone cannot clear a steep line and the label has to be set
    on whichever side the curve is leaving empty.
    """
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=colour, fontsize=_PT_ANNOTATION, va="center",
                ha=ha, **kwargs)


def tint(colour: str, strength: float = 0.20) -> tuple[float, float, float]:
    """Blend ``colour`` toward white and return an opaque RGB triple.

    Confidence bands drawn with an alpha channel render *opaque* through the
    PostScript backend, hiding the curve beneath them in the EPS that goes to
    the journal. Pre-blending against white looks identical in PDF, EPS and
    TIFF.

    Args:
        colour: any matplotlib colour specification.
        strength: 0 gives white, 1 gives the original colour.
    """
    r, g, b = matplotlib.colors.to_rgb(colour)
    return (1.0 - strength * (1.0 - r),
            1.0 - strength * (1.0 - g),
            1.0 - strength * (1.0 - b))


def save_figure(fig, stem, formats=("pdf", "eps", "tiff"), dpi: int = 600) -> list[Path]:
    """Write a figure in each format the submission pipeline needs.

    Args:
        fig: the figure.
        stem: output path without extension; parents are created.
        formats: any of ``pdf``, ``eps``, ``tiff``, ``png``.
        dpi: raster resolution; Nature requires at least 300, PLOS 300-600.

    Returns:
        The paths written.

    Note:
        TIFF is written flattened to RGB with LZW compression and a ``.tif``
        extension, because PLOS asks for exactly that: "Flattened, with no
        layers", "No alpha channels", "LZW compression is required", and file
        names of the form ``Fig1.tif``. matplotlib's TIFF writer produces RGBA
        by default, so an unflattened save carries an alpha channel that the
        journal rejects -- and nothing in the build would have told you.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)

    written = []
    for fmt in formats:
        if fmt in ("tiff", "tif"):
            # Render to RGBA in memory, composite onto white, then write RGB.
            # Compositing rather than dropping the alpha channel matters: a
            # bare .convert("RGB") maps transparent pixels to black.
            from PIL import Image  # noqa: PLC0415 -- optional until TIFF is asked for

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            buf.seek(0)
            rgba = Image.open(buf).convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[3])
            path = stem.with_suffix(".tif")
            flat.save(path, format="TIFF", compression="tiff_lzw",
                      dpi=(dpi, dpi))
        elif fmt == "png":
            path = stem.with_suffix(".png")
            fig.savefig(path, format="png", dpi=dpi)
        else:
            path = stem.with_suffix(f".{fmt}")
            fig.savefig(path, format=fmt)
        written.append(path)
    return written

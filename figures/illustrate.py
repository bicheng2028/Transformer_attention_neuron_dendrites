"""Illustration primitives: the drawing layer that sits on top of :mod:`figstyle`.

:mod:`figstyle` gives the manuscript a millimetre canvas and a fixed set of
colour roles, and its schematic primitives -- :func:`~figstyle.soma`,
:func:`~figstyle.axon` -- are *diagram* primitives: a circle, a constant-width
arc. They are correct, and they read as a diagram built out of circles and arcs.

The figures in a Nature paper do not read that way. Their schematics read as
drawings: a dendrite has a diameter that falls as it branches, a soma has a
shape rather than a radius, an axon curves the way an axon curves rather than
the way a circular arc does, and where two processes cross you can see which one
is in front. None of that is decoration. A constant-width line says every
process is equally important; a tapered one carries the cable geometry that the
paper's whole argument about conduction delay depends on.

This module supplies that layer, in the same millimetre space, from the same
palette, under the same typographic rules. Nothing here replaces or modifies
anything in :mod:`figstyle`; a figure imports from both.

Four groups of primitive
------------------------

**Real morphology.** :func:`load_swc` and :func:`draw_morphology` render an
actual reconstructed neuron as vector paths. The cached reconstruction is a
layer-5 thick-tufted pyramidal cell -- the manuscript's first stated biological
prototype -- and its provenance and the three citations its licence requires
live in ``data/morphology/provenance.json``; :func:`attribution` reads them.

**Procedural geometry.** :func:`dendrite`, :func:`branching_tree`,
:func:`axon_path`, :func:`soma_shape` and :func:`pyramidal_cell` draw organic
structure from parameters. They are the fallback if the cached reconstruction is
ever unavailable, and they are the right tool whenever a figure needs a *generic*
cell rather than a specific one -- a schematic of "an upstream neuron" should not
carry the accidental details of one rat.

**Depth.** :func:`keyline`, :func:`soft_shadow` and the depth ordering inside
:func:`draw_morphology` make overlapping processes read as layered instead of
merged. Used sparingly: a Nature figure separates two crossing lines, it does not
put drop shadows on things.

**Finish.** :func:`gradient_along` encodes a continuous quantity along a stroke,
:func:`tint_scale` derives related-but-distinct colours from one existing role
rather than inventing a hue, and :func:`halo` keeps 5.5 pt labels legible over
drawn structure without an opaque box behind them.

Rules this module keeps
-----------------------

*Vector only.* Every primitive emits filled paths or stroked paths. Nothing is
rasterised, so the output stays editable and stays sharp at any zoom.

*No alpha where it matters.* :func:`~delaykey.plotting.tint` exists because a
transparent fill renders opaque through the PostScript backend, hiding whatever
is beneath it in the EPS a journal receives. So the shadow and the tint helpers
here pre-blend against white rather than setting an alpha channel, and look the
same in PDF, EPS and TIFF.

*Text is never converted to outlines.* This is why :func:`halo` is built from
repeated offset copies rather than from ``matplotlib.patheffects``: a path effect
applied to a ``Text`` forces matplotlib down the draw-text-as-path route, and the
resulting PDF contains glyph outlines with *no font embedded at all* -- which
Nature forbids, and which :func:`~figstyle.audit_fonts` cannot catch, because a
figure with no fonts trivially passes a check for the wrong font. Path effects
are used freely on *patches*, where there is no such cost.

*No new hues.* Everything takes its colour from :data:`figstyle.C` or from
:func:`tint_scale` of one of those roles.

Usage::

    from figstyle import C, canvas, save_figure
    from illustrate import default_morphology, draw_morphology, halo

    fig, ax = canvas(height_mm=70.0)
    cell = draw_morphology(ax, default_morphology(), x=30.0, y=8.0,
                           height_mm=54.0)
    halo(ax, 30.0, 60.0, "apical tuft", color=C["distal"])
    save_figure(fig, "figures/fig_thing", formats=("pdf", "png"))
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figstyle import C, PT_ANNOT, tint  # noqa: E402

# figstyle imports delaykey.plotting, which selects the Agg backend, so these
# are safe here.
import matplotlib  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as mpe  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

__all__ = [
    # units
    "mm_to_pt", "pt_to_mm",
    # path geometry
    "smooth_path", "taper_widths", "taper_outline", "tapered_patch",
    # morphology
    "SWC_SOMA", "SWC_AXON", "SWC_BASAL", "SWC_APICAL", "KIND_NAME",
    "MORPHOLOGY_DIR", "DEFAULT_SWC", "Morphology", "Placement",
    "load_swc", "simplify", "default_morphology", "attribution",
    "draw_morphology", "COMPARTMENT_COLOUR",
    # procedural geometry
    "dendrite", "branching_tree", "axon_path", "bouton", "soma_shape",
    "pyramidal_cell",
    # depth and finish
    "keyline", "soft_shadow", "gradient_along", "tint_scale", "shade",
    "ramp", "halo",
]

_MM_PER_INCH = 25.4


# ---------------------------------------------------------------------------
# Units
#
# The canvas is in millimetres but matplotlib's line widths, marker sizes and
# path-effect widths are in points. Everything public in this module is
# specified in millimetres, and these two functions are where the conversion
# happens, once.
# ---------------------------------------------------------------------------

def mm_to_pt(mm: float) -> float:
    """Millimetres on the page to typographic points."""
    return mm * 72.0 / _MM_PER_INCH


def pt_to_mm(pt: float) -> float:
    """Typographic points to millimetres on the page."""
    return pt * _MM_PER_INCH / 72.0


# ---------------------------------------------------------------------------
# Path geometry
#
# The shared machinery under every organic shape below. A neurite is a
# *centreline plus a width profile*; these three functions turn that pair into
# a closed polygon that matplotlib can fill.
# ---------------------------------------------------------------------------

def smooth_path(points, samples: int = 12, closed: bool = False,
                alpha: float = 0.5) -> np.ndarray:
    """Resample a polyline as a centripetal Catmull-Rom spline.

    Catmull-Rom is the right curve here because it passes *through* its control
    points: a figure positions a process by naming the places it must go, and
    gets a smooth path that actually goes there, rather than a Bezier that is
    merely attracted towards them.

    The centripetal parameterisation (``alpha = 0.5``) is not a detail. The
    uniform variant (``alpha = 0``) forms cusps and self-intersecting loops
    wherever consecutive control points are unevenly spaced, which is exactly
    what happens when a hand-placed dendrite has one short segment among long
    ones. Centripetal is provably free of both.

    Args:
        points: ``(n, 2)`` control points, in millimetres.
        samples: output points per input span; higher is smoother and larger.
        closed: join the last point back to the first, for a blob outline.
        alpha: 0 uniform, 0.5 centripetal, 1 chordal.

    Returns:
        ``(m, 2)`` array of points along the curve.
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array of xy pairs")

    # Duplicate control points give a zero knot interval and a division by
    # zero; drop them rather than propagating NaNs into the fill.
    keep = np.ones(len(P), bool)
    keep[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9
    P = P[keep]
    if len(P) < 3:
        return P.copy()

    if closed:
        Q = np.vstack([P[-1], P, P[0], P[1]])
    else:
        # Reflect the end points outward so the curve is defined over the whole
        # span, including the first and last segments.
        Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])

    out = []
    for i in range(len(Q) - 3):
        p0, p1, p2, p3 = Q[i:i + 4]
        t = [0.0]
        for a, b in ((p0, p1), (p1, p2), (p2, p3)):
            d = float(np.linalg.norm(b - a))
            t.append(t[-1] + max(d, 1e-9) ** alpha)
        t0, t1, t2, t3 = t
        u = np.linspace(t1, t2, samples, endpoint=False)[:, None]

        a1 = (t1 - u) / (t1 - t0) * p0 + (u - t0) / (t1 - t0) * p1
        a2 = (t2 - u) / (t2 - t1) * p1 + (u - t1) / (t2 - t1) * p2
        a3 = (t3 - u) / (t3 - t2) * p2 + (u - t2) / (t3 - t2) * p3
        b1 = (t2 - u) / (t2 - t0) * a1 + (u - t0) / (t2 - t0) * a2
        b2 = (t3 - u) / (t3 - t1) * a2 + (u - t1) / (t3 - t1) * a3
        out.append((t2 - u) / (t2 - t1) * b1 + (u - t1) / (t2 - t1) * b2)

    curve = np.vstack(out)
    return np.vstack([curve, P[0]]) if closed else np.vstack([curve, P[-1]])


def taper_widths(n: int, w0: float, w1: float, profile: float = 1.0) -> np.ndarray:
    """A width profile from ``w0`` to ``w1`` over ``n`` points.

    Args:
        n: number of points along the path.
        w0, w1: full width at the start and at the end, in millimetres.
        profile: 1 is linear. Above 1 the taper is fast near the start and
            then flattens, which is how a real dendrite behaves -- most of the
            diameter is lost in the first stretch after a bifurcation.

    Returns:
        ``(n,)`` of full widths.
    """
    s = np.linspace(0.0, 1.0, max(int(n), 2))
    return w1 + (w0 - w1) * (1.0 - s) ** float(profile)


def taper_outline(points, widths) -> np.ndarray:
    """Offset a centreline by a varying half-width into a closed polygon.

    This is the single function that separates a drawing from a stick figure.
    A stroked line has one width for its whole length; a filled outline can
    start 0.4 mm across at the trunk and finish at a point, which is what a
    dendrite does.

    The offset direction at each vertex is the normal to the averaged tangent
    of its two adjacent segments, so the outline stays continuous through
    corners. No miter compensation is applied, so a sharp corner pinches
    slightly inward -- which reads as a rounded join and is invisible at the
    widths used here, whereas an uncompensated miter *spike* would not be.

    Args:
        points: ``(n, 2)`` centreline.
        widths: scalar or ``(n,)`` full widths, in millimetres.

    Returns:
        ``(2n, 2)`` closed polygon: one side out, the other back.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 2:
        raise ValueError("a tapered outline needs at least two points")
    h = np.broadcast_to(np.asarray(widths, dtype=float), (len(P),)) / 2.0

    d = np.diff(P, axis=0)
    length = np.linalg.norm(d, axis=1)
    unit = d / np.maximum(length, 1e-12)[:, None]

    tangent = np.empty_like(P)
    tangent[0] = unit[0]
    tangent[-1] = unit[-1]
    if len(P) > 2:
        tangent[1:-1] = unit[:-1] + unit[1:]
    norm = np.linalg.norm(tangent, axis=1)
    # A near-total reversal makes the averaged tangent vanish; fall back to the
    # incoming segment so the outline never collapses to a spike.
    bad = norm < 1e-9
    if bad.any():
        tangent[bad] = unit[np.clip(np.flatnonzero(bad) - 1, 0, len(unit) - 1)]
        norm = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(norm, 1e-12)[:, None]

    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    left = P + normal * h[:, None]
    right = P - normal * h[:, None]
    return np.vstack([left, right[::-1]])


def tapered_patch(points, widths, colour, zorder: float = 2,
                  edge: str | None = None, edge_mm: float = 0.0):
    """Build (but do not add) the filled patch for one tapered process."""
    poly = taper_outline(points, widths)
    return mpatches.Polygon(
        poly, closed=True, facecolor=colour,
        edgecolor=edge if edge is not None else colour,
        linewidth=mm_to_pt(edge_mm) if edge is not None else 0.0,
        joinstyle="round", zorder=zorder)


# ---------------------------------------------------------------------------
# Real morphology: SWC
#
# SWC is one row per sampled point: id, type, x, y, z, radius, parent. Type 1
# is soma, 2 axon, 3 basal dendrite, 4 apical dendrite, and keeping those
# separable is the whole point -- the manuscript's argument is about what each
# compartment of a pyramidal neuron does, so a figure has to be able to colour
# the apical tuft without colouring the basal skirt.
# ---------------------------------------------------------------------------

SWC_SOMA, SWC_AXON, SWC_BASAL, SWC_APICAL = 1, 2, 3, 4

#: Names used for compartments throughout this module's keyword arguments.
KIND_NAME = {SWC_SOMA: "soma", SWC_AXON: "axon",
             SWC_BASAL: "basal", SWC_APICAL: "apical"}
_NAME_KIND = {v: k for k, v in KIND_NAME.items()}

MORPHOLOGY_DIR = Path(__file__).resolve().parent / "data" / "morphology"

#: The cached reconstruction. Rat layer-5 thick-tufted pyramidal neuron,
#: NeuroMorpho.Org NMO_128418, archive Kole, from Cohen et al., *Cell* 2020.
#: See :func:`attribution` -- its licence requires three citations, not one.
DEFAULT_SWC = MORPHOLOGY_DIR / "11_06_13_-1_Cell_-3.CNG.swc"

#: Default compartment colours, bound to existing :data:`figstyle.C` roles so a
#: drawn cell already agrees with the schematics beside it: the apical tree is
#: the distal/query pathway, the basal tree is the proximal/content pathway,
#: and the axon is neutral because it is anatomy rather than a signal.
COMPARTMENT_COLOUR = {
    "apical": C["distal"],
    "basal": C["proximal"],
    "axon": C["miss"],
    "soma": C["soma"],
}


@dataclass(frozen=True)
class Morphology:
    """A parent-linked reconstruction, in micrometres.

    Attributes:
        name: the NeuroMorpho neuron name, or whatever the loader was given.
        xyz: ``(n, 3)`` sample positions.
        radius: ``(n,)`` sample radii.
        kind: ``(n,)`` SWC type codes.
        parent: ``(n,)`` index into these arrays; ``-1`` at a root.
        source: the file it came from, if any.
    """

    name: str
    xyz: np.ndarray
    radius: np.ndarray
    kind: np.ndarray
    parent: np.ndarray
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.xyz)

    @property
    def n_segments(self) -> int:
        """Segments, i.e. parent-child links -- what costs vector file size."""
        return int((self.parent >= 0).sum())

    def counts(self) -> dict[str, int]:
        """Sample count per named compartment."""
        return {name: int((self.kind == code).sum())
                for code, name in KIND_NAME.items()}

    def soma_centre(self) -> np.ndarray:
        """Centroid of the soma samples, or of everything if there are none."""
        m = self.kind == SWC_SOMA
        return self.xyz[m].mean(axis=0) if m.any() else self.xyz.mean(axis=0)

    def runs(self, kinds=None) -> list[np.ndarray]:
        """Split the tree into unbranched runs of one compartment each.

        A run is the natural unit of a *drawn* neuron: it is the stretch that
        gets one tapered outline, from one branch point to the next. Each run
        includes its parent sample, so consecutive runs abut and the arbor
        reads as continuous rather than as a scatter of separate strokes.

        Args:
            kinds: SWC type codes to include; ``None`` for all but the soma.

        Returns:
            A list of index arrays, each ordered from proximal to distal.
        """
        if kinds is None:
            kinds = (SWC_AXON, SWC_BASAL, SWC_APICAL)
        kinds = set(int(k) for k in kinds)

        children: dict[int, list[int]] = {}
        for i, p in enumerate(self.parent):
            if p >= 0:
                children.setdefault(int(p), []).append(i)

        out = []
        for i in range(len(self.xyz)):
            k = int(self.kind[i])
            if k not in kinds:
                continue
            p = int(self.parent[i])
            starts = (p < 0 or len(children.get(p, ())) > 1
                      or int(self.kind[p]) != k)
            if not starts:
                continue
            run = [p, i] if p >= 0 else [i]
            cur = i
            while True:
                kids = [c for c in children.get(cur, ()) if int(self.kind[c]) == k]
                if len(kids) != 1:
                    break
                cur = kids[0]
                run.append(cur)
            if len(run) >= 2:
                out.append(np.asarray(run, dtype=int))
        return out


def load_swc(path=DEFAULT_SWC, name: str | None = None) -> Morphology:
    """Read an SWC reconstruction into a :class:`Morphology`.

    Tolerant of the things real archive files do: comment banners, blank lines,
    non-contiguous or unsorted sample ids, and a parent that appears after its
    child. Sample ids are remapped to array indices so downstream code can
    index directly.

    Args:
        path: the ``.swc`` file.
        name: override the name; defaults to the file stem with ``.CNG``
            stripped, which recovers the NeuroMorpho neuron name.

    Returns:
        The reconstruction, coordinates and radii in micrometres.

    Raises:
        FileNotFoundError: with a pointer at :func:`attribution` if the cached
            file is missing, because re-fetching it means re-recording where it
            came from.
        ValueError: if the file contains no usable sample rows.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No SWC at {path}. The cached reconstruction and the citations "
            f"its licence requires are described in "
            f"{MORPHOLOGY_DIR / 'provenance.json'}; re-fetch from the "
            f"source_url recorded there, or draw a procedural cell with "
            f"illustrate.pyramidal_cell() instead.")

    ids, kinds, xyz, radii, parents = [], [], [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 7:
            continue
        ids.append(int(float(f[0])))
        kinds.append(int(float(f[1])))
        xyz.append((float(f[2]), float(f[3]), float(f[4])))
        radii.append(float(f[5]))
        parents.append(int(float(f[6])))

    if not ids:
        raise ValueError(f"{path} contains no SWC sample rows")

    index = {sid: i for i, sid in enumerate(ids)}
    parent = np.array([index.get(p, -1) if p >= 0 else -1 for p in parents],
                      dtype=int)

    return Morphology(
        name=name or path.name.replace(".CNG.swc", "").replace(".swc", ""),
        xyz=np.asarray(xyz, dtype=float),
        radius=np.asarray(radii, dtype=float),
        kind=np.asarray(kinds, dtype=int),
        parent=parent,
        source=path,
    )


def _rdp_keep(P: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: which points of a polyline to keep at ``eps``."""
    n = len(P)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = P[i], P[j] - P[i]
        seg = P[i + 1:j]
        denom = float(ab @ ab)
        if denom < 1e-18:
            dist = np.linalg.norm(seg - a, axis=1)
        else:
            t = np.clip((seg - a) @ ab / denom, 0.0, 1.0)
            dist = np.linalg.norm(seg - (a + t[:, None] * ab), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return keep


def _rebuild(morph: Morphology, keep: np.ndarray) -> Morphology:
    """Drop samples and re-parent survivors onto their nearest kept ancestor."""
    idx = np.flatnonzero(keep)
    remap = np.full(len(morph.xyz), -1, dtype=int)
    remap[idx] = np.arange(len(idx))

    parent = morph.parent
    new_parent = np.empty(len(idx), dtype=int)
    for out_i, i in enumerate(idx):
        p = int(parent[i])
        while p >= 0 and not keep[p]:
            p = int(parent[p])
        new_parent[out_i] = remap[p] if p >= 0 else -1

    return replace(morph, xyz=morph.xyz[idx], radius=morph.radius[idx],
                   kind=morph.kind[idx], parent=new_parent)


def _smooth_series(v: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge clamping, for a per-sample series along a run."""
    n = len(v)
    w = int(window) | 1
    if w < 3 or n < 3:
        return v
    w = min(w, n if n % 2 else n - 1)
    if w < 3:
        return v
    pad = w // 2
    padded = np.concatenate([np.full(pad, v[0]), v, np.full(pad, v[-1])])
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def simplify(morph: Morphology, max_segments: int = 1600,
             min_tip_length: float = 0.0, keep_kinds=None,
             radius_smooth: int = 9) -> Morphology:
    """Reduce a reconstruction to a print-sized number of segments.

    A published reconstruction is sampled for morphometry, not for drawing:
    the cached cell has about 13 600 samples, and a full arbor of 10^4 segments
    in a figure is several megabytes of vector data describing wiggles far below
    the resolution of the printed page. It also takes a PDF viewer noticeably
    long to open.

    The reduction is deliberately *topology preserving*. Every root, every
    bifurcation and every tip is kept, and only the redundant sampling along
    unbranched stretches is thinned, by Ramer-Douglas-Peucker with a tolerance
    bisected until the segment budget is met. So the cell keeps its exact
    branching pattern and simply stops describing a straight 40 um stretch with
    sixty points. Deleting branches instead -- the obvious alternative -- would
    quietly change the anatomy the figure is claiming to show.

    Pruning is separate and opt-in: ``min_tip_length`` removes terminal twigs
    shorter than that many micrometres, which is a real editorial choice (it
    declutters a dense tuft) and so is not done unless asked for.

    ``radius_smooth`` is not cosmetic tidying. Radii come from a human tracing
    diameters down a microscope, so consecutive samples disagree by tens of
    percent; drawn faithfully as a width, that noise becomes a visible lumpiness
    that reads as *texture on the dendrite* rather than as measurement error.
    Smoothing along each unbranched run, at full resolution and before any
    thinning, keeps the trend -- which is the real quantity -- and drops the
    jitter.

    Args:
        morph: the reconstruction.
        max_segments: target segment count. About 1200-2000 keeps a full
            pyramidal cell under 200 kB of PDF while staying visually complete.
        min_tip_length: prune terminal branches shorter than this, in
            micrometres. 0 disables pruning.
        keep_kinds: compartments to retain; ``None`` keeps everything.
        radius_smooth: moving-average window on radius along each run, in
            samples. 0 leaves the traced radii alone.

    Returns:
        A new :class:`Morphology`.
    """
    if radius_smooth and radius_smooth >= 3:
        radius = morph.radius.copy()
        for run in morph.runs():
            radius[run] = _smooth_series(morph.radius[run], radius_smooth)
        morph = replace(morph, radius=radius)

    keep = np.ones(len(morph.xyz), bool)

    if keep_kinds is not None:
        codes = {_NAME_KIND.get(k, k) for k in keep_kinds} | {SWC_SOMA}
        keep &= np.isin(morph.kind, sorted(codes))
        morph = _rebuild(morph, keep)
        keep = np.ones(len(morph.xyz), bool)

    if min_tip_length > 0.0:
        for _ in range(6):
            drop = _short_tips(morph, keep, float(min_tip_length))
            if not drop:
                break
            keep[list(drop)] = False
        morph = _rebuild(morph, keep)
        keep = np.ones(len(morph.xyz), bool)

    runs = morph.runs()
    if not runs:
        return morph

    span = float(np.linalg.norm(morph.xyz.max(axis=0) - morph.xyz.min(axis=0)))
    lo, hi = 0.0, span / 40.0
    best = None
    for _ in range(20):
        eps = 0.5 * (lo + hi)
        mask = np.zeros(len(morph.xyz), bool)
        # A run's own end points are branch points or tips of the whole tree,
        # so keeping them in every run keeps the topology intact.
        for run in runs:
            mask[run[_rdp_keep(morph.xyz[run], eps)]] = True
        mask[morph.kind == SWC_SOMA] = True
        mask[morph.parent < 0] = True
        n = int(mask.sum()) - int((morph.parent < 0).sum())
        if n <= max_segments:
            best = mask
            hi = eps
        else:
            lo = eps
        if best is not None and abs(n - max_segments) < max(8, max_segments // 50):
            break

    if best is None:                                    # budget unreachable
        best = np.ones(len(morph.xyz), bool)
    return _rebuild(morph, best)


def _short_tips(morph: Morphology, keep: np.ndarray, min_length: float) -> set:
    """Indices of terminal branches shorter than ``min_length`` micrometres."""
    children: dict[int, list[int]] = {}
    for i, p in enumerate(morph.parent):
        if p >= 0 and keep[i]:
            children.setdefault(int(p), []).append(i)

    drop: set[int] = set()
    for run in morph.runs():
        run = run[keep[run]]
        if len(run) < 2 or children.get(int(run[-1])):
            continue                                    # not a terminal run
        length = float(np.linalg.norm(np.diff(morph.xyz[run], axis=0),
                                      axis=1).sum())
        if length < min_length:
            drop.update(int(i) for i in run[1:])        # keep the branch point
    return drop


_CACHE: dict[Path, Morphology] = {}


def default_morphology(max_segments: int = 1600, min_tip_length: float = 0.0,
                       radius_smooth: int = 9) -> Morphology:
    """Load, simplify and cache the shipped layer-5 pyramidal reconstruction.

    Args:
        max_segments: passed to :func:`simplify`.
        min_tip_length: passed to :func:`simplify`.
        radius_smooth: passed to :func:`simplify`.

    Returns:
        The reconstruction, ready to draw.
    """
    key = (DEFAULT_SWC, max_segments, min_tip_length, radius_smooth)
    if key not in _CACHE:
        _CACHE[key] = simplify(load_swc(DEFAULT_SWC),
                               max_segments=max_segments,
                               min_tip_length=min_tip_length,
                               radius_smooth=radius_smooth)
    return _CACHE[key]


def attribution() -> dict:
    """The provenance record for the cached reconstruction.

    NeuroMorpho.Org is CC BY 4.0 and its terms require **three** citations for
    any reuse, not one: the paper that produced the reconstruction, the
    repository by its full name and RRID, and the repository's own reference
    paper. Shipping morphology without that trail is not an option, so the
    record is a file in the repository and this function reads it.

    Returns:
        The parsed ``data/morphology/provenance.json``. The three required
        citations are under ``["citation"]``.
    """
    return json.loads((MORPHOLOGY_DIR / "provenance.json").read_text())


@dataclass(frozen=True)
class Placement:
    """How a reconstruction was mapped onto the page.

    Returned by :func:`draw_morphology` so a figure can put something *on* the
    drawn cell -- a synapse on a named tuft branch, a bracket across the trunk
    -- without re-deriving the projection.
    """

    origin: np.ndarray          # micrometre point that landed on (x, y)
    offset: np.ndarray          # millimetre position of that point
    scale: float                # millimetres per micrometre
    azimuth: float
    tilt: float

    def to_mm(self, xyz) -> np.ndarray:
        """Project micrometre reconstruction coordinates to page millimetres."""
        p, _ = _project(np.atleast_2d(np.asarray(xyz, float)) - self.origin,
                        self.azimuth, self.tilt)
        out = p * self.scale + self.offset
        return out[0] if np.ndim(xyz) == 1 else out


def _project(xyz: np.ndarray, azimuth: float, tilt: float):
    """Project micrometre samples to a plane, and report their depth.

    ``azimuth`` spins the cell about its own vertical axis before projecting,
    which is the rotation that actually changes the picture -- it decides which
    branches come forward and which fall behind. ``tilt`` then rotates the
    resulting flat image, for a cell that should not stand perfectly upright.
    Depth is the out-of-plane coordinate, used only for draw order.
    """
    a = math.radians(azimuth)
    ca, sa = math.cos(a), math.sin(a)
    x = xyz[:, 0] * ca + xyz[:, 2] * sa
    y = xyz[:, 1]
    depth = -xyz[:, 0] * sa + xyz[:, 2] * ca

    if tilt:
        t = math.radians(tilt)
        ct, st = math.cos(t), math.sin(t)
        x, y = x * ct - y * st, x * st + y * ct
    return np.column_stack([x, y]), depth


def draw_morphology(ax, morph: Morphology, x: float, y: float,
                    scale: float | None = None, height_mm: float | None = None,
                    width_mm: float | None = None,
                    kinds=("axon", "basal", "apical"), colours=None,
                    w_min: float = 0.075, w_max: float = 0.46,
                    gamma: float = 0.80, tip_taper: float = 0.50,
                    azimuth: float = 0.0,
                    tilt: float = 0.0, anchor: str = "soma",
                    draw_soma: bool = True, soma_scale: float = 1.0,
                    keyline_mm: float = 0.0, keyline_colour: str = "white",
                    depth_sort: bool = True, zorder: float = 2.0,
                    zorder_step: float = 0.1):
    """Draw a reconstruction into millimetre space as tapered vector paths.

    Three choices make this look like a neuron rather than a wire model.

    *Tapered fills, not strokes.* Each unbranched run is drawn as a filled
    polygon whose width follows the reconstruction's own radii, so the apical
    trunk is visibly a trunk and the terminal twigs are hairlines.

    *A compressed width map.* Radii in a reconstruction span roughly 100:1 from
    soma to terminal. Drawn faithfully at print size, the thin end disappears
    below the resolution of the page while the thick end goes slab-like. The
    map here is ``w_min + (w_max - w_min) * (r / r_ref) ** gamma`` against the
    95th-percentile radius, with ``gamma < 1`` lifting the fine end. That is a
    deliberate, stated distortion of diameter -- lengths, angles and branching
    are untouched -- and it is the same one every published morphology plate
    makes, because the alternative is an unreadable figure.

    *Depth order.* Runs are drawn back to front by their out-of-plane
    coordinate, so a process in front of another looks like it.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        morph: the reconstruction; simplify it first for a sane file size.
        x, y: where ``anchor`` lands, in millimetres.
        scale: millimetres per micrometre. Mutually exclusive with
            ``height_mm`` and ``width_mm``.
        height_mm: fit the drawn compartments to this projected height.
        width_mm: fit them to this projected width. Given with ``height_mm``,
            the cell fits inside the box -- which is what you want in practice,
            because a pyramidal cell's horizontal axon collaterals are far
            wider than its arbor and will otherwise run into the panel beside
            it.
        kinds: compartment names to draw, in the order they are drawn.
        colours: name to colour; defaults to :data:`COMPARTMENT_COLOUR`.
            Pass ``tint(C["distal"], 0.35)`` to push a compartment back.
        w_min, w_max: full stroke width at the finest and thickest sample.
            Tuned for a cell 25-55 mm tall; scale them for a small inset.
        gamma: width-map exponent. Below 1 lifts the fine processes.
        tip_taper: width at the very end of a terminal branch, as a fraction
            of its traced width. Reconstructions stop at whatever diameter the
            tracer last measured, so terminals end in a blunt stub; bringing
            them to a point is the one place a drawing should overrule the
            data, because a dendrite does not actually end square. 1 disables.
        azimuth: degrees of rotation about the cell's vertical axis.
        tilt: degrees of in-plane rotation, applied after projection.
        anchor: ``"soma"`` puts the soma centre at ``(x, y)``; ``"base"`` puts
            the bottom-centre of the drawn extent there.
        draw_soma: draw the cell body as a smooth outline.
        soma_scale: enlarge or shrink the soma about its centre.
        keyline_mm: width of a page-coloured separation stroke behind each
            compartment. About 0.12 mm makes crossings read; 0 disables it.
        depth_sort: order runs back to front within each compartment.
        zorder: base draw order; each compartment adds ``zorder_step``.

    Returns:
        ``dict`` with one artist per drawn compartment name, ``"soma"`` if
        drawn, ``"placement"`` (a :class:`Placement`), and ``"tips"`` mapping
        each compartment to the millimetre positions of its terminal points.

    Raises:
        ValueError: if neither or both of ``scale`` and ``height_mm`` are given.
    """
    if (scale is None) == (height_mm is None and width_mm is None):
        raise ValueError("give either scale=, or height_mm= and/or width_mm=")

    codes = [_NAME_KIND.get(k, k) for k in kinds]
    colours = {**COMPARTMENT_COLOUR, **(colours or {})}

    origin = morph.soma_centre()
    flat, depth = _project(morph.xyz - origin, azimuth, tilt)

    drawn = np.isin(morph.kind, codes) | (morph.kind == SWC_SOMA)
    if not drawn.any():
        raise ValueError(f"no samples of kinds {tuple(kinds)} in {morph.name}")
    lo, hi = flat[drawn].min(axis=0), flat[drawn].max(axis=0)

    if scale is None:
        options = []
        if height_mm is not None:
            options.append(float(height_mm) / max(float(hi[1] - lo[1]), 1e-9))
        if width_mm is not None:
            options.append(float(width_mm) / max(float(hi[0] - lo[0]), 1e-9))
        scale = min(options)
    offset = np.array([x, y], dtype=float)
    if anchor == "base":
        offset = offset - np.array([0.5 * (lo[0] + hi[0]), lo[1]]) * scale
    elif anchor != "soma":
        raise ValueError("anchor must be 'soma' or 'base'")

    page = flat * scale + offset

    # Width map, referenced to the 95th percentile so one fat soma-adjacent
    # sample cannot set the scale for the whole cell.
    neurite = np.isin(morph.kind, (SWC_AXON, SWC_BASAL, SWC_APICAL))
    r_ref = float(np.percentile(morph.radius[neurite], 95)) if neurite.any() else 1.0
    r_ref = max(r_ref, 1e-6)
    unit = np.clip(morph.radius / r_ref, 0.0, 1.0) ** float(gamma)
    width = w_min + (w_max - w_min) * unit

    out: dict = {}
    tips: dict[str, np.ndarray] = {}
    effects = ([mpe.withStroke(linewidth=mm_to_pt(keyline_mm),
                               foreground=keyline_colour), mpe.Normal()]
               if keyline_mm > 0 else None)

    children = set(int(p) for p in morph.parent if p >= 0)

    for n, code in enumerate(codes):
        runs = [r for r in morph.runs((code,)) if len(r) >= 2]
        if not runs:
            continue
        if depth_sort:
            runs.sort(key=lambda r: float(depth[r].mean()))

        polys = []
        for r in runs:
            w = width[r]
            if tip_taper < 1.0 and len(r) >= 3 and int(r[-1]) not in children:
                k = max(2, len(r) // 3)
                w = w.copy()
                w[-k:] *= np.linspace(1.0, tip_taper, k)
            polys.append(taper_outline(page[r], w))

        coll = PolyCollection(
            polys, facecolors=colours[KIND_NAME[code]],
            edgecolors=colours[KIND_NAME[code]], linewidths=0.05,
            joinstyle="round", zorder=zorder + n * zorder_step,
            path_effects=effects)
        ax.add_collection(coll)
        out[KIND_NAME[code]] = coll

        ends = [int(r[-1]) for r in runs if int(r[-1]) not in children]
        tips[KIND_NAME[code]] = page[ends] if ends else np.empty((0, 2))

    if draw_soma:
        out["soma"] = _draw_swc_soma(ax, morph, page, scale, soma_scale,
                                     colours["soma"],
                                     zorder + len(codes) * zorder_step,
                                     effects)

    out["tips"] = tips
    out["placement"] = Placement(origin=origin, offset=offset, scale=scale,
                                 azimuth=azimuth, tilt=tilt)
    return out


def _draw_swc_soma(ax, morph, page, scale, soma_scale, colour, zorder, effects):
    """Draw the cell body as a smooth ovoid rather than a plotted circle.

    The CNG standardisation collapses a traced soma contour to three collinear
    samples: a centre and two points along the major axis. That is enough to
    recover an ellipse, which is then softened into a slightly irregular closed
    spline -- because a perfect ellipse is the one shape that immediately says
    "this was drawn by a plotting library".
    """
    m = morph.kind == SWC_SOMA
    if not m.any():
        return None

    pts = page[m]
    centre = pts.mean(axis=0)
    r_mm = float(morph.radius[m].mean()) * scale * soma_scale
    half = float(np.linalg.norm(pts - centre, axis=1).max()) * soma_scale
    # The contour samples sit at +/- half along the major axis and each carries
    # the soma radius, so the body reaches half + r, not max(half, r). Getting
    # this wrong halves the drawn soma and is why a reconstruction can come out
    # looking like it has no cell body at all.
    a = (half + r_mm) if half > 1e-9 else r_mm       # semi-major
    b = r_mm                                          # semi-minor

    if len(pts) > 1:
        v = pts[np.argmax(np.linalg.norm(pts - centre, axis=1))] - centre
        angle = math.atan2(v[1], v[0])
    else:
        angle = math.pi / 2.0

    th = np.linspace(0.0, 2.0 * math.pi, 11, endpoint=False)
    wobble = 1.0 + 0.05 * np.cos(3.0 * th + 0.7)
    local = np.column_stack([a * np.cos(th) * wobble, b * np.sin(th) * wobble])
    ca, sa = math.cos(angle), math.sin(angle)
    rot = np.array([[ca, -sa], [sa, ca]])
    outline = smooth_path(local @ rot.T + centre, samples=8, closed=True)

    patch = mpatches.Polygon(outline, closed=True, facecolor=colour,
                             edgecolor=colour, linewidth=0.0, zorder=zorder,
                             path_effects=effects)
    ax.add_patch(patch)
    return patch


# ---------------------------------------------------------------------------
# Procedural organic geometry
#
# For generic cells, for fallback if the cached reconstruction is unavailable,
# and for the many places a figure needs a *representative* process rather than
# a specific one.
# ---------------------------------------------------------------------------

def dendrite(ax, path_points, w0: float, w1: float, colour: str = C["distal"],
             profile: float = 1.35, smooth: bool = True, samples: int = 14,
             zorder: float = 2.0, keyline_mm: float = 0.0,
             keyline_colour: str = "white", **kwargs):
    """One tapered process, from ``w0`` at the base to ``w1`` at the tip.

    The default ``profile`` above 1 loses most of the diameter early, which is
    what a dendrite does after a bifurcation and what stops the shape reading as
    a mechanical wedge.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        path_points: ``(n, 2)`` millimetre points the process passes through.
        w0, w1: full width at base and tip, in millimetres. A ``w1`` of 0 ends
            the process in a point, which is right for a terminal twig.
        colour: fill colour; take it from :data:`figstyle.C`.
        profile: taper exponent; see :func:`taper_widths`.
        smooth: fit a spline through ``path_points`` first.
        keyline_mm: page-coloured separation stroke behind the fill.

    Returns:
        The added :class:`~matplotlib.patches.Polygon`.
    """
    pts = np.asarray(path_points, dtype=float)
    if smooth and len(pts) >= 3:
        pts = smooth_path(pts, samples=samples)
    elif len(pts) == 2:
        pts = np.linspace(pts[0], pts[1], max(samples, 6))

    patch = tapered_patch(pts, taper_widths(len(pts), w0, w1, profile),
                          colour, zorder=zorder, **kwargs)
    if keyline_mm > 0:
        keyline(patch, keyline_mm, keyline_colour)
    ax.add_patch(patch)
    return patch


def branching_tree(ax, origin, angle: float = 90.0, length: float = 9.0,
                   width: float = 0.55, depth: int = 4,
                   length_decay: float = 0.70, spread: float = 46.0,
                   rall: float = 1.5, curvature: float = 0.22,
                   jitter: float = 0.30, colour: str = C["distal"],
                   taper: float = 0.88, min_width: float = 0.07,
                   seed: int = 0, zorder: float = 2.0, keyline_mm: float = 0.0,
                   asymmetry: float = 0.26):
    """A recursive dendritic arbor with Rall-like diameter tapering.

    Three properties separate this from a fan of line segments.

    *Diameter obeys Rall.* At a bifurcation, ``d_parent ** 1.5`` equals the sum
    of the daughters' ``d ** 1.5``, so two equal daughters each come away at
    0.63 of the parent. That is why a real arbor thins so fast: after four
    bifurcations a process is a quarter of its trunk diameter, which is exactly
    the visual signature the eye reads as "dendrite".

    *Branches are not straight.* Each segment bows by ``curvature`` of its own
    length and each daughter angle carries noise, so no two branches are
    congruent.

    *Daughters are unequal.* ``asymmetry`` makes one daughter longer, thicker
    and *straighter* than its sister -- the dominant one continues the parent's
    direction while the minor one takes most of the turn. That is what keeps
    the arbor from looking like a snowflake, and it is the single strongest cue
    separating a drawn arbor from a recursive binary fan.

    The whole tree is seeded, so a figure regenerates identically on every
    build -- a figure that changes shape between runs cannot be proofread.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        origin: ``(x, y)`` base of the trunk, in millimetres.
        angle: direction of the trunk in degrees; 90 is straight up.
        length: trunk length in millimetres.
        width: trunk full width in millimetres.
        depth: number of bifurcations.
        length_decay: length multiplier per generation.
        spread: total angle between sister branches, in degrees.
        rall: branching exponent; 1.5 is Rall's 3/2 power law.
        curvature: lateral bow as a fraction of segment length.
        jitter: relative noise on angles and lengths.
        colour: fill colour.
        taper: within-segment tip width as a fraction of the base width.
        min_width: floor on drawn width, in millimetres.
        seed: RNG seed.
        keyline_mm: page-coloured separation stroke behind each segment.
        asymmetry: relative length/width imbalance between sisters.

    Returns:
        ``(tips, patches)`` -- terminal ``(x, y)`` positions, and the patches,
        so a figure can hang synapses on the tips or restyle one branch.
    """
    rng = np.random.default_rng(seed)
    tips: list[tuple[float, float]] = []
    patches: list = []
    ratio = 2.0 ** (-1.0 / float(rall))          # equal-daughter Rall factor

    def grow(base, ang, ln, w, level):
        a = math.radians(ang)
        direction = np.array([math.cos(a), math.sin(a)])
        normal = np.array([-direction[1], direction[0]])
        bow = curvature * ln * rng.uniform(-1.0, 1.0)
        end = base + direction * ln
        pts = np.vstack([
            base,
            base + direction * (0.35 * ln) + normal * (0.55 * bow),
            base + direction * (0.72 * ln) + normal * bow,
            end,
        ])
        w_base = max(w, min_width)
        w_tip = max(w * (taper if level < depth else 0.30), min_width * 0.6)
        if level == depth:
            w_tip = min_width * 0.55                 # terminal twigs end fine
        patches.append(dendrite(ax, pts, w_base, w_tip, colour=colour,
                                zorder=zorder, keyline_mm=keyline_mm,
                                samples=10))
        if level >= depth:
            tips.append((float(end[0]), float(end[1])))
            return

        half = spread / 2.0
        bias = rng.uniform(-1.0, 1.0) * asymmetry
        for sign, imbalance in ((+1.0, 1.0 + bias), (-1.0, 1.0 - bias)):
            # The angle is split *against* the imbalance: the dominant daughter
            # carries on close to the parent's direction and the minor one takes
            # most of the turn. Splitting it evenly instead is what makes a
            # recursive tree read as a binary fan rather than as an arbor --
            # in a real arbor one daughter is usually a continuation and the
            # other is a departure, and the eye reads that asymmetry directly.
            share = 1.0 + 2.0 * (1.0 - imbalance)
            child_angle = ang + sign * half * share * (1.0 + jitter * rng.normal() * 0.5)
            child_len = ln * length_decay * imbalance * (1.0 + jitter * rng.normal() * 0.4)
            child_w = w_tip * ratio * (imbalance ** 0.5)
            grow(end, child_angle, max(child_len, 0.6), child_w, level + 1)

    grow(np.asarray(origin, dtype=float), float(angle), float(length),
         float(width), 1)
    return tips, patches


def axon_path(ax, points, width: float = 0.30, colour: str = C["miss"],
              wobble: float = 0.0, wobble_scale: float = 3.0,
              myelin: bool = False, internode: float = 3.2,
              node: float = 0.55, sheath: float = 2.4,
              sheath_colour=None, sheath_edge_mm: float = 0.05,
              myelin_span: tuple[float, float] | None = None,
              terminal: float = 0.0, terminal_angle: float | None = None,
              seed: int = 0, samples: int = 16, zorder: float = 2.0,
              keyline_mm: float = 0.0, taper: float = 0.85):
    """A long-range projection with organic curvature, myelin and a terminal.

    An axon drawn as a circular arc reads as a connector in a flow chart: an arc
    has constant curvature, and nothing in an animal does. Here the centreline
    is a spline through the given waypoints plus, optionally, a low-frequency
    perpendicular wander whose wavelength is set by ``wobble_scale``, so the
    curvature varies along the path the way a real fibre's does.

    ``myelin`` draws the internodal sheaths as separate lozenges that pinch to
    the bare axon at each node of Ranvier. This is worth having in this
    manuscript specifically: internode length is what sets conduction velocity,
    so a drawn myelinated axon is showing the mechanism, not decorating it.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        points: ``(n, 2)`` waypoints in millimetres.
        width: full width of the bare axon, in millimetres.
        colour: fill colour.
        wobble: amplitude of the organic wander, in millimetres. 0 follows the
            waypoints exactly.
        wobble_scale: wavelength of the wander, in millimetres.
        myelin: draw internodal sheaths.
        internode: sheath length in millimetres.
        node: gap at each node of Ranvier, in millimetres.
        sheath: sheath width as a multiple of ``width``.
        sheath_colour: sheath fill; defaults to a pale tint of ``colour``.
        sheath_edge_mm: rim on the sheath, in the axon colour. Without it a
            pale lozenge over a dark core reads as a smudge; with it, it reads
            as something wrapped around the axon.
        myelin_span: ``(start, stop)`` as fractions of the axon's length, to
            sheathe only that stretch -- cortical axons are often myelinated
            in discontinuous profiles, some segments sheathed and others bare.
            The sheaths are centred in the span. ``None`` sheathes the whole
            axon from its first node, as before.
        terminal: bouton radius in millimetres; 0 for none.
        terminal_angle: bouton orientation in degrees; ``None`` continues the
            axon's own final direction.
        seed: RNG seed for the wander.
        keyline_mm: page-coloured separation stroke.
        taper: width at the far end as a fraction of ``width``.

    Returns:
        ``dict`` with ``"path"`` (the ``(m, 2)`` centreline, so a figure can
        put a delay label or a gradient on the same curve), ``"axon"``,
        ``"myelin"`` and ``"terminal"``.
    """
    pts = np.asarray(points, dtype=float)
    path = smooth_path(pts, samples=samples) if len(pts) >= 3 else \
        np.linspace(pts[0], pts[-1], max(samples * 2, 24))

    if wobble > 0.0:
        rng = np.random.default_rng(seed)
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(step)])
        # Two incommensurate components, so the wander never looks periodic,
        # tapered to zero at both ends so the endpoints stay where asked.
        disp = np.zeros_like(s)
        for k, amp in ((1.0, 1.0), (2.37, 0.45)):
            disp += amp * np.sin(2 * math.pi * k * s / max(wobble_scale, 1e-6)
                                 + rng.uniform(0, 2 * math.pi))
        envelope = np.sin(math.pi * np.clip(s / max(s[-1], 1e-9), 0, 1)) ** 0.7
        d = np.gradient(path, axis=0)
        n = np.column_stack([-d[:, 1], d[:, 0]])
        n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
        path = path + n * (wobble * disp * envelope / 1.45)[:, None]

    out: dict = {"path": path}
    widths = taper_widths(len(path), width, width * taper, profile=1.0)
    out["axon"] = dendrite(ax, path, width, width * taper, colour=colour,
                           smooth=False, zorder=zorder, keyline_mm=keyline_mm)

    if myelin:
        out["myelin"] = _draw_myelin(
            ax, path, widths, colour,
            tint(colour, 0.20) if sheath_colour is None else sheath_colour,
            sheath_edge_mm, internode, node, sheath, zorder + 0.05, keyline_mm,
            span=myelin_span)

    if terminal > 0.0:
        tangent = path[-1] - path[-2]
        ang = (math.degrees(math.atan2(tangent[1], tangent[0]))
               if terminal_angle is None else terminal_angle)
        out["terminal"] = bouton(ax, path[-1], terminal, angle=ang,
                                 colour=colour, zorder=zorder + 0.1,
                                 keyline_mm=keyline_mm)
    return out


def myelin_along(ax, path, width: float, colour: str, fill=None,
                 span: tuple[float, float] | None = None,
                 internode: float = 3.2, node: float = 0.55,
                 sheath: float = 2.4, edge_mm: float = 0.05,
                 zorder: float = 2.05, keyline_mm: float = 0.0):
    """Internodal sheaths on a centreline that something else has drawn.

    :func:`axon_path` sheathes the axons it draws itself. This is for a
    projection drawn another way -- a dashed hairline from
    :func:`figstyle.axon`, say -- whose centreline the figure can compute:
    the sheaths are laid on top of it, and the line shows through at each node
    of Ranvier.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        path: ``(n, 2)`` centreline in millimetres, densely sampled.
        width: width of the axon being sheathed, in millimetres; the sheath is
            ``sheath`` times this.
        colour: rim colour, normally the axon's own.
        fill: sheath fill; defaults to a pale tint of ``colour``.
        span: ``(start, stop)`` as fractions of the path length; ``None`` for
            the whole of it. See :func:`axon_path`.
        internode, node, sheath, edge_mm, keyline_mm: as for :func:`axon_path`.

    Returns:
        The sheath patches.
    """
    path = np.asarray(path, dtype=float)
    return _draw_myelin(ax, path, np.full(len(path), float(width)), colour,
                        tint(colour, 0.20) if fill is None else fill, edge_mm,
                        internode, node, sheath, zorder, keyline_mm, span=span)


def _draw_myelin(ax, path, widths, colour, fill, edge_mm, internode, node,
                 sheath, zorder, keyline_mm, span=None):
    """Internodal sheaths as rimmed lozenges, pinched to the axon at each node.

    Resampled to equal arc length inside each internode, so a sheath keeps its
    shape wherever the axon happens to be densely or sparsely sampled, and so
    short internodes do not collapse to two points.

    With ``span``, only that stretch of the axon is sheathed, and the sheaths
    are centred in it so the bare remainder splits evenly between its ends --
    a remainder left at one end reads as a sheath that went missing.
    """
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    total = float(s[-1])

    patches = []
    start = node
    period = internode + node
    if span is not None:
        lo, hi = span[0] * total, span[1] * total
        count = int((hi - lo + node + 1e-9) // period)
        start = lo + 0.5 * (hi - lo - (count * period - node))
        total = start + count * period - node
    while start + internode <= total + 1e-9:
        stop = start + internode
        t = np.linspace(start, stop, 44)
        seg = np.column_stack([np.interp(t, s, path[:, 0]),
                               np.interp(t, s, path[:, 1])])
        w0 = np.interp(t, s, widths)
        u = np.linspace(0.0, 1.0, len(t))
        # A flat-topped profile with a short, *rounded* pinch at each end: a
        # sheath is a cylinder that tapers over its last few percent, not a
        # rugby ball. A clipped sine gives the flat top but leaves a visible
        # kink where it saturates, so the shoulder is rounded by the exponent.
        shape = np.sin(math.pi * u) ** 0.32
        patch = tapered_patch(seg, w0 * (1.0 + (sheath - 1.0) * shape), fill,
                              zorder=zorder, edge=colour, edge_mm=edge_mm)
        if keyline_mm > 0:
            keyline(patch, keyline_mm, "white")
        ax.add_patch(patch)
        patches.append(patch)
        start += period
    return patches


def bouton(ax, position, radius: float = 0.7, angle: float = 0.0,
           colour: str = C["miss"], zorder: float = 3.0,
           keyline_mm: float = 0.0, elongation: float = 1.35):
    """A terminal swelling: a teardrop with its neck on the incoming axon.

    A synaptic terminal drawn as a circle is a dot, and a dot on the end of a
    line is a graph marker. The asymmetry -- narrow where the axon arrives,
    full at the far end -- is what makes it read as a bouton.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        position: ``(x, y)`` of the neck, in millimetres.
        radius: half-width of the swelling, in millimetres.
        angle: direction the terminal points, in degrees.
        elongation: length as a multiple of ``radius``.

    Returns:
        The added patch.
    """
    r = float(radius)
    L = r * float(elongation)
    local = np.array([
        [0.0, -0.42 * r], [0.55 * L, -0.95 * r], [1.15 * L, -0.55 * r],
        [1.30 * L, 0.0], [1.15 * L, 0.55 * r], [0.55 * L, 0.95 * r],
        [0.0, 0.42 * r],
    ])
    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    outline = smooth_path(local @ rot.T + np.asarray(position, float),
                          samples=9, closed=True)

    patch = mpatches.Polygon(outline, closed=True, facecolor=colour,
                             edgecolor=colour, linewidth=0.0, zorder=zorder)
    if keyline_mm > 0:
        keyline(patch, keyline_mm, "white")
    ax.add_patch(patch)
    return patch


def soma_shape(ax, x: float, y: float, width: float = 3.0,
               height: float = 4.2, colour: str = C["soma"],
               angle: float = 0.0, pyramidal: bool = True,
               edge: str | None = None, edge_mm: float = 0.0,
               zorder: float = 3.0, keyline_mm: float = 0.0):
    """A cell body with a shape, rather than a circle with a radius.

    ``pyramidal=True`` gives the profile of a pyramidal soma: a drawn-out apex
    where the apical trunk leaves, straight-ish flanks, and a broad rounded
    base where the basal dendrites emerge. That silhouette is diagnostic -- it
    is how a pyramidal cell is recognised in a plate -- and it costs one patch.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        x, y: centre, in millimetres.
        width, height: overall extent, in millimetres.
        angle: rotation in degrees; 0 points the apex up.
        pyramidal: the pyramidal profile, or a plain ovoid.
        edge: outline colour; ``None`` for a flat fill.

    Returns:
        The added patch.
    """
    w, h = float(width) / 2.0, float(height) / 2.0
    if pyramidal:
        # A rounded shoulder just below the apex, near-straight flanks, and a
        # broad base with distinct corners where the basal dendrites leave.
        # Without the shoulder point the spline runs straight from apex to
        # flank and the soma comes out as an ice-cream cone.
        local = np.array([
            [0.00, 1.00], [0.24, 0.78], [0.46, 0.32], [0.68, -0.24],
            [0.86, -0.68], [0.62, -0.96], [0.00, -1.02], [-0.62, -0.96],
            [-0.86, -0.68], [-0.68, -0.24], [-0.46, 0.32], [-0.24, 0.78],
        ])
    else:
        th = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        local = np.column_stack([np.cos(th), np.sin(th)])
    local = local * np.array([w, h])

    a = math.radians(angle)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    outline = smooth_path(local @ rot.T + np.array([x, y]), samples=9,
                          closed=True)

    patch = mpatches.Polygon(
        outline, closed=True, facecolor=colour,
        edgecolor=edge if edge else colour,
        linewidth=mm_to_pt(edge_mm) if edge else 0.0, zorder=zorder)
    if keyline_mm > 0:
        keyline(patch, keyline_mm, "white")
    ax.add_patch(patch)
    return patch


def pyramidal_cell(ax, x: float, y: float, height: float = 30.0,
                   apical_colour: str = C["distal"],
                   basal_colour: str = C["proximal"],
                   axon_colour: str = C["miss"],
                   soma_colour: str = C["soma"], seed: int = 3,
                   axon_length: float | None = None, myelin: bool = False,
                   keyline_mm: float = 0.10, zorder: float = 2.0):
    """A generic pyramidal neuron composed from the procedural primitives.

    This is the fallback if the cached reconstruction is ever unavailable, and
    it is the better choice whenever a figure needs *a* pyramidal cell rather
    than *this* rat's: a small inset, a population of nominally identical
    neurons, a cell that has to be a specific shape the argument depends on.
    A reconstruction carries accidental detail, which is a virtue when the
    figure is about anatomy and a distraction when it is not.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        x, y: soma centre, in millimetres.
        height: soma-to-tuft-top height, in millimetres.
        seed: RNG seed; the same seed always draws the same cell.
        axon_length: descending axon length; defaults to ``0.55 * height``.
        myelin: myelinate the axon.

    Returns:
        ``dict`` with ``"soma"``, ``"trunk"``, ``"tuft_tips"``,
        ``"basal_tips"`` and ``"axon"``.
    """
    h = float(height)
    soma_h = max(h * 0.090, 1.3)
    trunk_top = y + h * 0.60
    rng = np.random.default_rng(seed)

    trunk = np.array([
        [x, y + soma_h * 0.42],
        [x + rng.uniform(-0.25, 0.25) * h * 0.02, y + h * 0.22],
        [x + rng.uniform(-0.3, 0.3) * h * 0.02, y + h * 0.42],
        [x, trunk_top],
    ])
    trunk_patch = dendrite(ax, trunk, h * 0.030, h * 0.020,
                           colour=apical_colour, profile=1.1,
                           zorder=zorder, keyline_mm=keyline_mm)

    tuft_tips, _ = branching_tree(
        ax, (x, trunk_top), angle=90.0, length=h * 0.115, width=h * 0.020,
        depth=4, length_decay=0.76, spread=58.0, curvature=0.22,
        colour=apical_colour, seed=seed, zorder=zorder,
        keyline_mm=keyline_mm, min_width=h * 0.0032)

    basal_tips = []
    for k, ang in enumerate((196.0, 232.0, 268.0, 308.0, 344.0)):
        tips, _ = branching_tree(
            ax, (x, y - soma_h * 0.32), angle=ang, length=h * 0.105,
            width=h * 0.015, depth=3, length_decay=0.74, spread=50.0,
            curvature=0.26, colour=basal_colour, seed=seed * 17 + k,
            zorder=zorder, keyline_mm=keyline_mm, min_width=h * 0.0032)
        basal_tips.extend(tips)

    axon_len = 0.55 * h if axon_length is None else float(axon_length)
    axon = axon_path(
        ax, [[x, y - soma_h * 0.5],
             [x - h * 0.03, y - soma_h * 0.5 - axon_len * 0.45],
             [x + h * 0.02, y - soma_h * 0.5 - axon_len]],
        width=h * 0.013, colour=axon_colour, wobble=h * 0.012,
        wobble_scale=axon_len * 0.55, myelin=myelin,
        internode=axon_len * 0.24, node=axon_len * 0.045,
        terminal=h * 0.022 if not myelin else 0.0, seed=seed,
        zorder=zorder - 0.05, keyline_mm=keyline_mm)

    soma = soma_shape(ax, x, y, width=soma_h * 0.78, height=soma_h,
                      colour=soma_colour, zorder=zorder + 0.4,
                      keyline_mm=keyline_mm)

    return {"soma": soma, "trunk": trunk_patch, "tuft_tips": tuft_tips,
            "basal_tips": basal_tips, "axon": axon}


# ---------------------------------------------------------------------------
# Depth and finish
# ---------------------------------------------------------------------------

def keyline(artist, width_mm: float = 0.12, colour: str = "white"):
    """Put a page-coloured separation stroke behind a patch or collection.

    This is the occlusion primitive, and it is why a plate of overlapping
    neurites reads as layered instead of as a mat. Where two processes cross,
    the upper one carries a thin band of page colour around it, so the lower one
    visibly passes behind. Illustrators have done this by hand forever; here it
    is one path effect, drawn beneath the fill and half-covered by it, so the
    visible band is only ``width_mm / 2`` wide.

    Do not use it on :class:`~matplotlib.text.Text` -- a path effect on text
    converts the glyphs to outlines and embeds no font. Use :func:`halo`.

    Args:
        artist: any patch, line or collection.
        width_mm: full stroke width; about 0.10-0.16 mm is right at this scale.
        colour: normally ``"white"``, or whatever the panel background is.

    Returns:
        The artist, so this composes in an expression.
    """
    artist.set_path_effects([
        mpe.withStroke(linewidth=mm_to_pt(width_mm), foreground=colour),
        mpe.Normal()])
    return artist


def soft_shadow(patch, dx_mm: float = 0.14, dy_mm: float = -0.14,
                strength: float = 0.13, colour: str = "black"):
    """Offset a faint copy of a patch behind it, to lift it off the page.

    Deliberately weak, and deliberately rare. Nature figures use depth to
    disambiguate structure, never to ornament it; if the shadow is the first
    thing you notice, ``strength`` is too high. Anything above about 0.2 is
    a slide, not a research figure.

    The shadow colour is pre-blended against white rather than set with an
    alpha channel, for the same reason :func:`figstyle.tint` exists: alpha is
    flattened to opaque by the PostScript backend, so an alpha shadow would
    become a solid grey slab in the EPS a journal receives.

    Args:
        patch: a patch or collection. Not text.
        dx_mm, dy_mm: offset, in millimetres. Keep it under about 0.25 mm.
        strength: 0 is invisible, 1 is the full colour.
        colour: what to blend from; a neutral is almost always right.

    Returns:
        The patch.
    """
    patch.set_path_effects([
        mpe.SimplePatchShadow(offset=(mm_to_pt(dx_mm), mm_to_pt(dy_mm)),
                              shadow_rgbFace=tint(colour, strength),
                              alpha=1.0, rho=1.0),
        mpe.Normal()])
    return patch


def shade(colour, strength: float = 0.20):
    """Blend ``colour`` toward near-black and return an opaque RGB triple.

    The counterpart of :func:`figstyle.tint`, which blends toward white. The
    target is a very dark neutral rather than pure black, because blending a
    saturated hue into pure black desaturates it into mud before it gets dark.

    Args:
        colour: any matplotlib colour.
        strength: 0 leaves it alone, 1 gives the dark neutral.
    """
    r, g, b = mcolors.to_rgb(colour)
    t = 0.10
    k = float(strength)
    return (r + k * (t - r), g + k * (t - g), b + k * (t - b))


def tint_scale(colour, n: int, lo: float = 0.28, hi: float = 1.30):
    """A ramp of tints and shades of one colour role.

    A figure that needs five related-but-distinct elements has two options:
    five hues, or five values of one hue. The first breaks the manuscript's
    colour semantics -- every hue in :data:`figstyle.C` already means something
    -- and risks the red/green pairing Nature warns against. The second says
    "these are five of the same kind of thing", which is usually the truth.

    Values at or below 1 are tints toward white; above 1 they are shades toward
    a dark neutral, so a single call can span light to dark through the pure
    colour.

    Args:
        colour: the base colour, normally a :data:`figstyle.C` role.
        n: how many steps.
        lo: lightest step; 0 is white, 1 is the pure colour.
        hi: darkest step; above 1 shades past the pure colour.

    Returns:
        A list of ``n`` opaque RGB triples, light to dark.
    """
    out = []
    for t in np.linspace(float(lo), float(hi), max(int(n), 1)):
        out.append(tint(colour, t) if t <= 1.0 else shade(colour, t - 1.0))
    return out


def ramp(*colours, name: str = "illustrate_ramp"):
    """Build a colormap through the given colours, for :func:`gradient_along`.

    Kept separate so a figure declares its ramp from :data:`figstyle.C` roles
    explicitly, rather than reaching for a matplotlib colormap and importing a
    hue the manuscript does not use anywhere else.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name, [mcolors.to_rgb(c) for c in colours])


def gradient_along(ax, path, cmap=None, w0: float = 0.5,
                   w1: float | None = None, steps: int = 90,
                   smooth: bool = False, samples: int = 14,
                   zorder: float = 2.0, keyline_mm: float = 0.0,
                   values=None):
    """Fill a tapered stroke with a colour gradient along its length.

    For encoding a continuous quantity *on the structure that carries it* --
    conduction delay accumulating down an axon, calcium decaying along a branch
    -- instead of in a separate plot beside it. That is a strong device and an
    easy one to overuse: it earns its place when the quantity varies along a
    path the figure is drawing anyway.

    Implemented as a strip of short filled quadrilaterals, one per step, each a
    flat colour. That keeps the output pure vector, where a true PDF shading
    would not survive every downstream tool. Adjacent quads are stroked in their
    own fill colour so the seams between them close.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        path: ``(n, 2)`` centreline in millimetres -- e.g. the ``"path"``
            returned by :func:`axon_path`, so the gradient lands on that curve.
        cmap: a colormap, or ``None`` for a light-to-dark ramp of
            ``C["distal"]`` built with :func:`tint_scale`.
        w0: full width at the start, in millimetres.
        w1: full width at the end; ``None`` keeps it constant.
        steps: number of quads. 60-120 is smooth at print size.
        values: optional ``(steps,)`` in 0-1 giving the quantity at each step;
            defaults to arc length, i.e. a linear ramp along the path.

    Returns:
        The added :class:`~matplotlib.collections.PolyCollection`.
    """
    P = np.asarray(path, dtype=float)
    if smooth and len(P) >= 3:
        P = smooth_path(P, samples=samples)

    # Resample to equal arc length, so the gradient is uniform in distance
    # rather than in however densely the input happened to be sampled.
    step = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    target = np.linspace(0.0, s[-1], int(steps) + 1)
    P = np.column_stack([np.interp(target, s, P[:, 0]),
                         np.interp(target, s, P[:, 1])])

    widths = taper_widths(len(P), w0, w0 if w1 is None else w1, profile=1.0)
    outline = taper_outline(P, widths)
    left, right = outline[:len(P)], outline[len(P):][::-1]

    quads = [np.array([left[i], left[i + 1], right[i + 1], right[i]])
             for i in range(len(P) - 1)]

    if cmap is None:
        cmap = ramp(*tint_scale(C["distal"], 6))
    elif isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    elif not callable(cmap):
        cmap = ramp(*cmap)

    u = (np.linspace(0.0, 1.0, len(quads)) if values is None
         else np.clip(np.asarray(values, float), 0.0, 1.0))
    faces = cmap(u)

    coll = PolyCollection(quads, facecolors=faces, edgecolors=faces,
                          linewidths=0.28, zorder=zorder)
    if keyline_mm > 0:
        keyline(coll, keyline_mm, "white")
    ax.add_collection(coll)
    return coll


def halo(ax, x: float, y: float, text: str, color: str = C["ink"],
         halo_mm: float = 0.17, halo_colour: str = "white", rings: int = 8,
         fontsize: float = PT_ANNOT, zorder: float = 6.0, **kwargs):
    """Small text with a page-coloured outline, so it survives over drawing.

    A 5.5 pt label over a dendritic arbor is unreadable, and the usual fixes are
    both bad: an opaque box behind it punches a hole in the anatomy, and moving
    the label away from what it names costs the reader the association. A thin
    outline in the page colour solves it -- the label sits *in* the drawing and
    is still legible.

    The outline is built from copies of the string offset around the position,
    not from a stroke path effect. Path effects on text send matplotlib down its
    draw-text-as-path route: the PDF then contains glyph outlines and embeds no
    font at all, which Nature forbids ("text is never converted to outlines")
    and which :func:`figstyle.audit_fonts` cannot detect, since a figure with
    no fonts passes a check for the wrong font trivially. Offset copies keep
    every glyph as live, embedded, selectable text.

    The cost is ``rings + 1`` text objects instead of one; at 5.5 pt that is a
    few hundred bytes. Use it only where a label really is over structure.

    Args:
        ax: a :func:`figstyle.canvas` axes.
        x, y: position, in millimetres.
        text: the label.
        color: ink colour; take it from :data:`figstyle.C`.
        halo_mm: outline thickness, in millimetres. 0.15-0.20 is right at
            5.5 pt; more starts to look like a sticker.
        rings: copies around the position. 8 is smooth; 4 is enough for short
            horizontal labels and halves the object count.
        fontsize: at most 7 pt and at least 5 pt, per the Nature spec.

    Returns:
        The foreground :class:`~matplotlib.text.Text`.
    """
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    for k in range(int(rings)):
        a = 2.0 * math.pi * k / float(rings)
        ax.text(x + halo_mm * math.cos(a), y + halo_mm * math.sin(a), text,
                color=halo_colour, fontsize=fontsize, zorder=zorder - 0.01,
                **kwargs)
    return ax.text(x, y, text, color=color, fontsize=fontsize, zorder=zorder,
                   **kwargs)

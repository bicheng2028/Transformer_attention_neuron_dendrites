"""Report how the precision sweep's numbers change with the simulation step.

Reads ``results/e6_convergence_r{1,4,10,20}.csv``: the precision sweep at 1, 4,
10 and 20 steps per millisecond, on a reduced grid (theta 2.5, a 15 ms interval,
dispersions 15, 20, 30 and 50 ms). It prints the retention the text quotes, and
the half-loss jitter and tolerated absolute error that Fig 3f plots, computed
exactly as ``make_fig_e6.py`` computes them. S2 Text quotes this table. It is
the evidence that 10 steps per ms, the reported run's step, is converged, and
that 1 step per ms is not.

Usage::

    python figures/report_e6_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_fig_e6 import curve, half_loss, load  # noqa: E402

RESOLUTIONS = (1, 4, 10, 20)
INTERVAL, THETA = 15, 2.5


def main() -> int:
    rng = np.random.default_rng(0)
    print(f"theta {THETA}, interval {INTERVAL} ms; retention is the seed mean")
    for r in RESOLUTIONS:
        rows = load(Path(f"results/e6_convergence_r{r}.csv"))
        print(f"\n{r} step(s) per ms")
        tolerances = []
        for disp in sorted({row["dispersion"] for row in rows}):
            cells = {}
            for pert, mag in (("jitter", 0.02), ("jitter", 0.05),
                              ("jitter", 0.10), ("scale", 0.95),
                              ("scale", 1.05)):
                vals = [row["retention"] for row in rows
                        if row["perturbation"] == pert and row["dispersion"] == disp
                        and abs(row["magnitude"] - mag) < 1e-9]
                cells[(pert, mag)] = float(np.mean(vals)) if vals else float("nan")
            mags, per_seed = curve(rows, "jitter", disp, INTERVAL, THETA)
            got = half_loss(mags, per_seed, rng) if mags else None
            s_half = got[0] if got else float("nan")
            tolerances.append(s_half * disp)
            print(f"  Delta {disp:2d} ms: jitter 2% {cells[('jitter', 0.02)]:.2f}"
                  f"  5% {cells[('jitter', 0.05)]:.2f}"
                  f"  10% {cells[('jitter', 0.10)]:.2f}"
                  f" | x0.95 {cells[('scale', 0.95)]:.2f}"
                  f"  x1.05 {cells[('scale', 1.05)]:.2f}"
                  f" | half-loss jitter {100 * s_half:.1f}%"
                  f" = {s_half * disp:.2f} ms")
        print(f"  mean tolerated absolute error {np.nanmean(tolerances):.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
#
# Build the shareable archive of the experiment code: dist/delaykey-code-<date>.zip
#
#     bash delaykey/package.sh          # or: make code-archive
#
# What goes in, and why it is not simply `zip -r delaykey/`:
#
#   * The figure generator for manuscript Experiment 1 lives outside the
#     package, in <repo>/figures/, and reaches back into it through
#     figstyle.py. Both files come along, at the same relative depth, so that
#     path still resolves inside the archive.
#   * The reported CSVs in results/ are included even though they are
#     gitignored. Without them a collaborator cannot redraw a figure without
#     first spending 13 minutes re-running the simulations.
#   * The font binaries are left out, as they are in the repository: Nimbus
#     Sans is AGPLv3, and the fetch step covers both faces rather than only the
#     one that could be vendored. fonts/fetch.sh comes along instead. They are
#     staged temporarily so the artwork in this archive is typeset in the right
#     face, then removed before the zip is written.
#   * Smoke-test output (results/quick/, figures/out/quick/) is left out. It is
#     three-seed data and would be indistinguishable from the real thing once
#     it is sitting in someone else's checkout.
#
# The script regenerates the figures from the CSVs it is about to ship and runs
# the test suite inside the staged copy, so a zip that builds is a zip that has
# been shown to work standalone.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
STAMP="$(date +%Y-%m-%d)"
NAME="delaykey-code-$STAMP"
DIST="dist"
STAGE="$DIST/$NAME"

echo "=== staging $STAGE ==="
rm -rf "$STAGE"
mkdir -p "$STAGE"

# --- the package ----------------------------------------------------------
rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.gitignore' \
    --exclude 'results/quick/' \
    --exclude 'results/*_quick.csv' \
    --exclude 'figures/out/' \
    --exclude 'fonts/*.otf' \
    --exclude 'fonts/*.ttf' \
    --exclude 'package.sh' \
    delaykey/ "$STAGE/delaykey/"

# --- the Experiment 1 figure generator, at its original depth --------------
mkdir -p "$STAGE/figures"
cp figures/figstyle.py figures/make_fig4_lif_vs_integrator.py "$STAGE/figures/"

# --- provenance -----------------------------------------------------------
{
    echo "delaykey experiment code -- archive provenance"
    echo
    echo "packaged        $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "source repo     $(git remote get-url origin 2>/dev/null || echo '(no remote)')"
    echo "commit          $(git rev-parse HEAD 2>/dev/null || echo '(not a git checkout)')"
    echo "branch          $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
    if [ -n "$(git status --porcelain delaykey figures/figstyle.py \
                figures/make_fig4_lif_vs_integrator.py 2>/dev/null)" ]; then
        echo "working tree    MODIFIED relative to that commit. These paths differ,"
        echo "                so checking the commit out does not reproduce this"
        echo "                archive:"
        git status --porcelain delaykey figures/figstyle.py \
            figures/make_fig4_lif_vs_integrator.py | sed 's/^/                  /'
    else
        echo "working tree    clean"
    fi
    echo
    echo "verified with"
    echo "  python        $("$PYTHON" -V 2>&1)"
    echo "  numpy         $("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
    echo "  matplotlib    $("$PYTHON" -c 'import matplotlib; print(matplotlib.__version__)')"
    echo "  platform      $(uname -srm)"
    echo
    echo "The CSVs in delaykey/results/ are the ones behind the reported numbers."
    echo "The artwork in delaykey/figures/out/ and figures/ was regenerated from"
    echo "those CSVs at packaging time, with the fonts fetch.sh installs."
} > "$STAGE/PROVENANCE.txt"

# --- cover README ---------------------------------------------------------
cat > "$STAGE/README.md" <<'COVER'
# delaykey — experiment code

Companion code for *Physical Delays as Keys: A Delay-Key Framework for
Sequence Coding in Neural Circuits*. Everything needed to reproduce every
reported number and every generated figure, plus the data behind them.

## Start here

```bash
cd delaykey
pip install -r requirements.txt
bash fonts/fetch.sh                 # figures need this; see "Fonts" below

python3 tests/test_core.py          # 16 checks on the simulator
python3 tests/test_detectors.py     # 6 checks on the classification
bash run_all.sh --quick             # the whole pipeline on a reduced grid, ~10 s
```

If those four commands pass, the environment is good and you can read
`delaykey/README.md`, which is the real documentation: what each module does,
which script corresponds to which manuscript experiment (the numbering does
*not* line up, and the table there is the authority), and how the two
independent implementations of the simulator check each other.

## Layout

```
delaykey/            the package, self-contained
  README.md          full documentation — read this one
  requirements.txt
  run_all.sh         the exact command lines behind every CSV in results/
  delaykey/          model, stimuli, detector classification, stats, plot style
  experiments/       one script per experiment, each writes a tidy CSV
  figures/           one script per figure, each reads a CSV, writes PDF/EPS/TIFF
    out/             the artwork as submitted, with its captions
  results/           the CSVs behind the reported numbers
  tests/             verification suite
  fonts/             fetch.sh and licences (binaries are not shipped)
figures/             the Experiment 1 figure generator and the shared house
                     style, kept at this depth because figstyle.py resolves
                     ../delaykey from here
PROVENANCE.txt       commit, packaging date, and the versions used
```

## Three things worth knowing before you run anything

**The results are included.** `delaykey/results/` holds the CSVs behind the
reported numbers, so any figure can be redrawn in seconds without re-running
the simulations. A full re-run is `bash run_all.sh` and takes about 13 minutes
on 16 cores.

**`--quick` cannot damage them.** Smoke-test runs write to `results/quick/` and
`figures/out/quick/`. The reported CSVs and artwork are never overwritten by a
`--quick` run, so it is safe as a first command in a fresh copy.

**Fonts are fetched, not shipped.** The figures are typeset in Liberation Sans,
with Nimbus Sans as a fallback; `fonts/fetch.sh` downloads both and
`fonts/LICENSE.*` explains the choice. The binaries are not redistributed here
because Nimbus Sans is AGPLv3. Run `bash fonts/fetch.sh` before regenerating
artwork: without it matplotlib falls back to DejaVu Sans, which is not a face
Nature or PLOS accepts, and `delaykey.plotting` warns rather than failing — so
the figures will look fine and be unpublishable. The artwork already in
`figures/out/` was generated with the correct fonts.

## Licence

The source repository carries no LICENSE file, so this archive does not either.
Check with the authors before redistributing it.
COVER

# --- regenerate the artwork this archive ships ----------------------------
# The fonts have to be present for this and absent from the zip, so they are
# staged, used, and removed. Without them the output silently falls back to
# DejaVu Sans.
echo
echo "=== regenerating artwork from the staged CSVs ==="
if ls delaykey/fonts/*.otf >/dev/null 2>&1; then
    cp delaykey/fonts/*.otf delaykey/fonts/*.ttf "$STAGE/delaykey/fonts/"
else
    echo "WARNING: no fonts in delaykey/fonts/ -- run fonts/fetch.sh first."
    echo "         The artwork in this archive would fall back to DejaVu Sans."
    exit 1
fi

for stem in e1_dispersion e2_aliasing e3_discriminability e4_noise \
            e5_population e6_precision e7_verification; do
    ( cd "$STAGE/delaykey" && "$PYTHON" "figures/make_fig_${stem%%_*}.py" \
        --csv "results/$stem.csv" \
        --out "figures/out/fig_$stem" \
        --formats pdf png >/dev/null )
    echo "  fig_$stem"
done

# Experiment 1's figure, from the repo-level generator.
( cd "$STAGE" && "$PYTHON" figures/make_fig4_lif_vs_integrator.py \
    --formats pdf png >/dev/null )
echo "  fig4_lif_vs_integrator"

rm -f "$STAGE/delaykey/fonts"/*.otf "$STAGE/delaykey/fonts"/*.ttf

# --- verify the staged copy stands on its own -----------------------------
echo
echo "=== verifying the staged copy ==="
( cd "$STAGE/delaykey" && "$PYTHON" tests/test_core.py | tail -1 )
( cd "$STAGE/delaykey" && "$PYTHON" tests/test_detectors.py | tail -1 )

# --- zip ------------------------------------------------------------------
echo
echo "=== writing the archive ==="
# Regenerating the artwork and running the tests both import from the staged
# copy, which leaves bytecode caches behind. They are excluded from the rsync
# above but created after it, so they have to be swept here as well.
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$STAGE" -name '.DS_Store' -delete
rm -f "$DIST/$NAME.zip"
( cd "$DIST" && zip -rqX "$NAME.zip" "$NAME" )

echo
echo "$DIST/$NAME.zip  ($(du -h "$DIST/$NAME.zip" | cut -f1), $(unzip -l "$DIST/$NAME.zip" | tail -1 | awk '{print $2}') files)"

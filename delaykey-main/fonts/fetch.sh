#!/usr/bin/env bash
# Fetch the sans-serif faces used for every figure.
#
# Nature requires Helvetica or Arial and forbids mixing faces within a figure.
# Neither ships with matplotlib or with most cluster images, so figures silently
# fall back to DejaVu Sans, which is not an accepted face.
#
# Two metric-compatible substitutes are fetched. Both are accepted in substance;
# they differ in the font technology underneath, and that difference decides
# which one delaykey.plotting prefers:
#
#   Liberation Sans  Arial-metric, TrueType (glyf outlines), SIL OFL 1.1.
#                    PREFERRED. matplotlib's `pdf.fonttype: 42` writes a
#                    /Subtype /CIDFontType2 dictionary with a /FontFile2 stream,
#                    which is a *TrueType* declaration. Feeding it a TrueType
#                    font makes that declaration true.
#
#   Nimbus Sans      Helvetica-metric, CFF-flavoured OpenType, AGPLv3 + font
#                    exception. Kept as a fallback and for the closer Helvetica
#                    match, but its CFF outlines wrapped in a CIDFontType2
#                    declaration are malformed: poppler reports "Mismatch
#                    between font type and embedded font file", and a strict
#                    preflight (Acrobat, PDF/X) is liable to reject it. Nimbus
#                    also draws a flat-topped lowercase tau that reads as a
#                    Latin T at 5.5 pt.
#
# Licences. Liberation Sans is SIL OFL 1.1, which explicitly permits bundling
# and redistribution -- it could be vendored into the repository outright if you
# would rather not depend on a fetch step. Nimbus Sans is AGPLv3 with a font
# exception permitting embedding in PostScript or PDF output "regardless of the
# conditions or license applying to the document itself", which is exactly how
# the figures use it; its binaries are gitignored rather than vendored so the
# repository does not redistribute AGPL binaries.
#
# Usage:  bash fonts/fetch.sh

set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Liberation Sans (preferred: TrueType, OFL) ------------------------------
lib_ver="2.1.5"
lib_url="https://github.com/liberationfonts/liberation-fonts/files/7261482/liberation-fonts-ttf-${lib_ver}.tar.gz"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

printf '  fetching Liberation Sans %s\n' "${lib_ver}"
curl -fsSL -o "${tmp}/liberation.tar.gz" "${lib_url}"
tar xzf "${tmp}/liberation.tar.gz" -C "${tmp}"

for face in Regular Bold Italic BoldItalic; do
    src="${tmp}/liberation-fonts-ttf-${lib_ver}/LiberationSans-${face}.ttf"
    printf '    LiberationSans-%s.ttf\n' "${face}"
    cp "${src}" "${dir}/LiberationSans-${face}.ttf"
done
cp "${tmp}/liberation-fonts-ttf-${lib_ver}/LICENSE" "${dir}/LICENSE.liberation"

# -- Nimbus Sans (fallback: CFF, AGPL + font exception) ----------------------
base="https://github.com/ArtifexSoftware/urw-base35-fonts/raw/master"

for face in Regular Bold Italic BoldItalic; do
    name="NimbusSans-${face}.otf"
    printf '  fetching %s\n' "$name"
    curl -fsSL -o "${dir}/${name}" "${base}/fonts/${name}"
done

printf '  fetching LICENSE.urw\n'
curl -fsSL -o "${dir}/LICENSE.urw" "${base}/LICENSE"

printf '\nDone. delaykey.plotting.resolve_font() will now pick up Liberation Sans.\n'

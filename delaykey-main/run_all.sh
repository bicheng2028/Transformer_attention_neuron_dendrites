#!/usr/bin/env bash
#
# The exact invocations behind every committed CSV in results/.
#
# This file exists because the scripts' argparse defaults do NOT reproduce the
# reported grids. e1_dispersion defaults to dispersions 0-30, but the manuscript
# sweeps 0-50; every experiment defaults to fewer seeds than were run. Without a
# record of the command lines, "one script per experiment" is not enough to
# reproduce a number, and the Data Availability Statement would be promising
# something the repository could not deliver.
#
# Run from inside delaykey/, because each script writes to a path relative to
# the working directory ("results/...") and the figure scripts read from there:
#
#     cd delaykey && bash run_all.sh
#
# Pass --quick to run every experiment on its smoke-test grid instead, which
# takes about a minute in total and is enough to confirm the environment works.
#
# Wall-clock on a 16-core laptop, full grids: on the order of an hour and a
# half, of which the aliasing sweep and its three threshold arms are most.
# Nothing here needs a GPU or a cluster. Memory is a few hundred MB per worker
# except in the two arms that enlarge the population to keep the detector count
# up at a strict gate -- Experiment 7 at theta = 4.5 runs 120k neurons on four
# workers, around 2 GB in total.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

# --quick sends every output to a subdirectory rather than alongside the
# reported output. The experiment scripts' --quick flag reduces the grid but
# leaves --out at its default, so without this redirection a smoke test would
# write three-seed data over the CSVs behind the manuscript, and the figure
# step would then redraw the reported artwork from it. Both are silent: the
# files are the expected ones and the numbers are wrong.
QUICK=""
RESULTS="results"
FIGURES="figures/out"
FORMATS=(pdf eps tiff)
if [ "${1:-}" = "--quick" ]; then
    QUICK="--quick"
    RESULTS="results/quick"
    FIGURES="figures/out/quick"
    FORMATS=(pdf png)
    echo "running every experiment on its smoke-test grid"
    echo "writing to $RESULTS/ and $FIGURES/; reported output is left untouched"
fi
mkdir -p "$RESULTS" "$FIGURES"

run() {
    echo
    echo "=== $1 ==="
    shift
    # shellcheck disable=SC2086
    "$PYTHON" "$@" $QUICK
}

# --------------------------------------------------------------------------
# Manuscript Experiment 5 -- dispersion sets detector class.
# The reported sweep runs to 50 ms; the script's default stops at 30.
# --------------------------------------------------------------------------
run "Experiment 5: dispersion sweep" \
    experiments/e1_dispersion.py \
    --seeds 20 \
    --dispersions 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 \
    --thetas 1.5 2.5 3.5 4.5 \
    --intervals 5 10 15 20 \
    --p-connect 0.1 \
    --out "$RESULTS/e1_dispersion.csv"

# The same classes computed from the arrival statistics, for Fig 2's theory
# lines. Deterministic, so it takes no seeds; its defaults are the reported grid.
run "Fig 2 theory: detector classes without simulation" \
    experiments/e1_theory.py \
    --out "$RESULTS/e1_theory.csv"

# The same sweep against three of the model's simplifications, on a reduced
# grid: as reported, on finer steps, with a 100 ms plateau, and under Poisson
# background input (S2 Text). figures/report_e1_robustness.py prints the
# comparison.
run "Fig 2 robustness: as reported" \
    experiments/e1_dispersion.py --seeds 20 \
    --dispersions 0 5 10 15 20 25 30 40 50 --thetas 2.5 4.5 --intervals 15 \
    --out "$RESULTS/e1_robustness_reported.csv"
# 0.1 ms is the step compared cell by cell; 0.25 and 0.05 ms show it converged.
for r in 4 10 20; do
    run "Fig 2 robustness: $r steps per ms" \
        experiments/e1_dispersion.py --seeds 20 --steps-per-ms $r \
        --dispersions 0 5 10 15 20 25 30 40 50 --thetas 2.5 4.5 --intervals 15 \
        --out "$RESULTS/e1_robustness_step_r$r.csv"
done
run "Fig 2 robustness: 100 ms plateau" \
    experiments/e1_dispersion.py --seeds 20 --plateau-steps 100 \
    --dispersions 0 5 10 15 20 25 30 40 50 --thetas 2.5 4.5 --intervals 15 \
    --out "$RESULTS/e1_robustness_plateau.csv"
# Background at eight rates, over 100 seeds. The report runs each gate at the
# highest rate that keeps the sequence detectors found without background from
# answering one event alone in more than 5% of cases, and estimating that at the
# strict gate takes more than the hundred-odd detectors 20 seeds hold. It
# compares the other conditions with seeds 0-19 of that rate, which are the rows
# a 20-seed run would write.
for rate in 0.25 0.5 0.75 1 2 3 4 5; do
    run "Fig 2 robustness: background at $rate Hz" \
        experiments/e1_dispersion.py --seeds 100 --background-rate $rate \
        --dispersions 0 5 10 15 20 25 30 40 50 --thetas 2.5 4.5 --intervals 15 \
        --out "$RESULTS/e1_robustness_background_${rate}Hz.csv"
done
# Theory for the two departures it covers, on the four simulated intervals so
# that Fig 2c's slopes can be compared. The finer step takes a quarter of an hour
# on 16 cores, which a smoke test cannot spare.
run "Fig 2 robustness: theory with a 100 ms plateau" \
    experiments/e1_theory.py --plateau-steps 100 \
    --thetas 2.5 4.5 --intervals 5 10 15 20 \
    --out "$RESULTS/e1_theory_plateau.csv"
if [ -z "$QUICK" ]; then
    run "Fig 2 robustness: theory on a 0.1 ms step" \
        experiments/e1_theory.py --steps-per-ms 10 \
        --thetas 2.5 4.5 --intervals 5 10 15 20 \
        --out "$RESULTS/e1_theory_step.csv"
fi

# --------------------------------------------------------------------------
# Manuscript Experiment 6 -- aliasing and the bound on codable intervals.
#
# The dispersion sweep runs 5-50 ms because the figure's third panel plots the
# bound *against* dispersion: three dispersions cannot distinguish a bound that
# tracks the anatomy from two populations that happen to differ. The reference
# intervals run 4-30 ms because its first panel is a fan of tuning curves, one
# per reference. The corners of that grid hold cells with no detectors at all
# -- a 20 ms dispersion cannot encode a 25 ms interval, which is the result --
# and the figure drops them on a detector-count floor rather than plotting a
# ratio with a zero denominator.
# --------------------------------------------------------------------------
run "Experiment 6: aliasing" \
    experiments/e2_aliasing.py \
    --seeds 20 \
    --dispersions 5 10 15 20 25 30 35 40 45 50 \
    --thetas 2.5 \
    --reference-intervals 4 6 8 10 12 14 15 16 18 20 25 30 \
    --out "$RESULTS/e2_aliasing.csv"

# --------------------------------------------------------------------------
# Manuscript Experiment 3 -- discriminability at matched false-positive rate.
# --readout and --tuned-by are left at their defaults, which are what the
# manuscript reports; see the module docstring for the alternatives and why
# they matter.
# --------------------------------------------------------------------------
run "Experiment 3: discriminability" \
    experiments/e3_discriminability.py --seeds 20 \
    --out "$RESULTS/e3_discriminability.csv"

# --------------------------------------------------------------------------
# Manuscript Experiment 2 -- noise robustness of a single K-line.
# Both thresholds are run: 1.5 is the two-input requirement and is what the
# text reports; 1.8 additionally forbids the one-step near-miss that the
# dendritic time constant would otherwise pass, and is the operating point
# Experiment 4 uses.
# --------------------------------------------------------------------------
run "Experiment 2: noise robustness" \
    experiments/e4_noise.py --seeds 20 --thetas 1.5 1.8 \
    --out "$RESULTS/e4_noise.csv"

# --------------------------------------------------------------------------
# Manuscript Experiment 4 -- population coding and Q-sparsity dependence.
# --------------------------------------------------------------------------
run "Experiment 4: population coding" \
    experiments/e5_population.py --seeds 20 \
    --out "$RESULTS/e5_population.csv"

# --------------------------------------------------------------------------
# Manuscript Experiment 7 -- the precision budget.
#
# The jitter list is APPEND-ONLY, and must stay that way. run_cell() draws each
# jitter perturbation from one per-population RNG in loop order, so inserting a
# magnitude ahead of an existing one re-rolls every draw after it and silently
# moves numbers already quoted in the manuscript. Appending leaves the earlier
# draws bit-identical; the figure sorts magnitudes before plotting. Scale
# perturbations take no draw and can be listed in any order.
#
# Dispersion runs 10-50 ms and the interval sweep 10-20 ms because the figure's
# third panel extracts the half-loss jitter from every one of these sweeps and
# plots it against dispersion: that is what shows the budget to be a fixed
# absolute timing tolerance rather than the fixed percentage two curves at one
# interval invited the reader to infer.
# --------------------------------------------------------------------------
# On 10 steps per ms, not 1: on a 1 ms step a perturbation under half a
# millisecond rounds back to the delay it started from, which inflated the
# small-perturbation retention the text once quoted. The convergence runs below
# are the evidence that 10 is enough.
run "Experiment 7: precision budget" \
    experiments/e6_precision.py --seeds 20 --steps-per-ms 10 \
    --dispersions 10 15 20 25 30 35 40 45 50 \
    --thetas 2.5 --intervals 10 15 20 \
    --jitters 0.0 0.02 0.05 0.10 0.20 0.30 0.01 0.03 0.075 0.15 0.25 \
    --scales 0.80 0.85 0.90 0.925 0.95 0.975 1.025 1.05 1.075 1.10 1.15 1.20 \
    --out "$RESULTS/e6_precision.csv"

# The same sweep on a reduced grid at four time steps, for the S2 Text
# convergence check (figures/report_e6_convergence.py prints the table).
for r in 1 4 10 20; do
    run "Experiment 7 convergence: $r steps per ms" \
        experiments/e6_precision.py --seeds 20 --steps-per-ms $r \
        --dispersions 15 20 30 50 --thetas 2.5 --intervals 15 \
        --jitters 0.0 0.02 0.05 0.10 0.20 0.30 0.01 0.03 0.075 0.15 0.25 \
        --scales 0.80 0.85 0.90 0.925 0.95 0.975 1.025 1.05 1.075 1.10 1.15 1.20 \
        --out "$RESULTS/e6_convergence_r$r.csv"
done

# Why slowing costs less than speeding near the codable bound: the signed
# alignment of the selected detectors' two volleys (S2 Text).
run "Experiment 7 alignment" \
    experiments/e6_alignment.py --seeds 20 \
    --out "$RESULTS/e6_alignment.csv"

# --------------------------------------------------------------------------
# Manuscript S5 Text (was Experiment 8) -- conjunctive versus additive
# somatic integration.
# --------------------------------------------------------------------------
run "S5 Text: somatic verification" \
    experiments/e7_verification.py --seeds 20 \
    --out "$RESULTS/e7_verification.csv"

# --------------------------------------------------------------------------
# Threshold-consistency arms.
#
# Experiments 3, 6, 7 and 8 are reported at theta = 2.5 while Experiment 5
# argues that theta = 4.5 is the correct operating point, so each is repeated
# there. The populations are enlarged because the detector count falls by
# roughly thirtyfold at the stricter gate and the smaller populations would
# otherwise yield too few detectors to measure anything.
#
# These write to *_theta45.csv and do not overwrite the headline results.
# --------------------------------------------------------------------------
if [ -z "$QUICK" ]; then
    run "Experiment 3 at theta = 4.5" \
        experiments/e3_discriminability.py \
        --seeds 20 --thetas 4.5 --neurons 50000 --jobs 6 \
        --out "$RESULTS/e3_discriminability_theta45.csv"

    # 120k neurons, not the 40k the other strict arms use: the precision figure
    # now carries this arm across the whole dispersion sweep, and the detector
    # count falls with dispersion as well as with threshold. At 40k the widest
    # populations came back with eight detectors each, which the figure's floor
    # rejects; at 120k the thinnest cell (50 ms) clears it.
    run "Experiment 7 at theta = 4.5" \
        experiments/e6_precision.py --steps-per-ms 10 \
        --seeds 20 --thetas 4.5 --neurons 120000 --jobs 4 \
        --dispersions 20 25 30 35 40 45 50 --intervals 15 \
        --jitters 0.0 0.02 0.05 0.10 0.20 0.30 0.01 0.03 0.075 0.15 0.25 \
        --scales 0.80 0.85 0.90 0.925 0.95 0.975 1.025 1.05 1.075 1.10 1.15 1.20 \
        --out "$RESULTS/e6_precision_theta45.csv"

    run "S5 Text at theta = 4.5" \
        experiments/e7_verification.py \
        --seeds 20 --theta 4.5 --n-neurons 40000 --jobs 6 \
        --out "$RESULTS/e7_verification_theta45.csv"

    # Experiment 6's own threshold arms. The aliasing figure's third panel
    # plots the codable bound against dispersion at every threshold measured,
    # which is what distinguishes a bound set by the anatomy from one set by
    # the gate; a single strict arm could not make that comparison. The three
    # arms are pooled over the reference intervals they share with the headline
    # run (10, 15, 20 ms), so the reference grid is deliberately the narrow one
    # here even though the headline run sweeps 4-30 ms.
    #
    # The population size tracks the threshold, for the same reason as above.
    run "Experiment 6 at theta = 1.5" \
        experiments/e2_aliasing.py \
        --seeds 20 --thetas 1.5 --neurons 2000 \
        --dispersions 5 10 15 20 25 30 40 50 \
        --reference-intervals 10 15 20 \
        --out "$RESULTS/e2_aliasing_theta15.csv"

    run "Experiment 6 at theta = 3.5" \
        experiments/e2_aliasing.py \
        --seeds 20 --thetas 3.5 --neurons 8000 --jobs 12 \
        --dispersions 5 10 15 20 25 30 40 50 \
        --reference-intervals 10 15 20 \
        --out "$RESULTS/e2_aliasing_theta35.csv"

    # The long one: about half an hour.
    run "Experiment 6 at theta = 4.5" \
        experiments/e2_aliasing.py \
        --seeds 20 --thetas 4.5 --neurons 20000 --jobs 8 \
        --dispersions 5 10 15 20 25 30 40 50 \
        --reference-intervals 10 15 20 \
        --out "$RESULTS/e2_aliasing_theta45.csv"
fi

# --------------------------------------------------------------------------
# Figures. Each reads a CSV written above and writes to $FIGURES/.
#
# The figure script for a CSV named e<n>_<thing>.csv is make_fig_e<n>.py and
# its output stem is fig_e<n>_<thing>, so one stem drives all three names.
# --------------------------------------------------------------------------
for stem in e1_dispersion e2_aliasing e3_discriminability e4_noise \
            e5_population e6_precision e7_verification; do
    echo
    echo "=== figure ${stem%%_*} ==="
    extra=()
    [ "$stem" = e1_dispersion ] && extra=(--theory-csv "$RESULTS/e1_theory.csv")
    # The smoke-test grid finds tens of detectors where the reported runs find
    # thousands, so --quick lowers the two figures' detector floor rather than
    # leave them nothing to draw.
    if [ -n "$QUICK" ] && { [ "$stem" = e2_aliasing ] || [ "$stem" = e6_precision ]; }; then
        extra+=(--min-detectors 1)
    fi
    "$PYTHON" "figures/make_fig_${stem%%_*}.py" \
        --csv "$RESULTS/$stem.csv" \
        --out "$FIGURES/fig_$stem" \
        --formats "${FORMATS[@]}" ${extra[@]+"${extra[@]}"}
done

echo
echo "Done. CSVs in $RESULTS/, artwork and captions in $FIGURES/."

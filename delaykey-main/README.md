# delaykey

Companion code for *From anatomy to computation: axonal delay dispersion shifts
neurons from event to sequence detection* (Bi and Sun).

This repository is the target of the manuscript's Data Availability Statement.
It contains the integrator-neuron model, the experiment scripts that produce
every reported number, and the figure scripts that turn those numbers into the
submitted figures.

## Requirements

`numpy` for the model and experiments; `matplotlib` additionally for figures.
No GPU, no PyTorch, no compiled extensions.

```bash
pip install -r requirements.txt
bash fonts/fetch.sh
```

`fonts/fetch.sh` downloads the two metric-compatible sans faces the figures are
typeset in: Liberation Sans, which `delaykey.plotting` prefers, and Nimbus Sans
behind it. Without them matplotlib falls back to DejaVu Sans, which is not a
face Nature or PLOS accept; `delaykey.plotting` warns rather than failing so
this is visible rather than silent. The binaries are gitignored, not vendored,
because Nimbus Sans is AGPLv3 — the licence's font exception covers embedding
it in the PDF and EPS output, which is all the figures do. `fonts/fetch.sh` and
`fonts/LICENSE.*` set out why the preference falls the way it does.

## Layout

```
delaykey/          model and shared utilities
  core.py          vectorised simulator + transparent reference implementation
  stimuli.py       two-event stimuli and random delay-key connectivity
  detectors.py     functional classification into detector types
  stats.py         bootstrap confidence intervals
  plotting.py      PLOS-compliant figure style and export
experiments/       one script per experiment; each writes a tidy CSV
figures/           one script per figure; each reads a CSV, writes PDF/EPS/TIFF
tests/             verification suite
results/           generated CSVs (gitignored; shipped in the code archive)
```

Experiments write CSVs and figures read them. Replotting therefore never
re-runs a simulation, and a figure can be regenerated from the committed
results alone.

## Which script is which experiment

The script numbering (`e1`–`e7`) is the order in which the scripts were
written, and it has never matched the manuscript's order. Nothing in the code
can make the two agree without renaming files that are cited in captions, so
this table is the mapping. It is the authority: if a docstring elsewhere
disagrees with it, the docstring is stale.

| Manuscript | Script | CSV | Figure script |
| --- | --- | --- | --- |
| Fig 2, dispersion sweep | `experiments/e1_dispersion.py` | `e1_dispersion.csv` | `figures/make_fig_e1.py` |
| Fig 2, theory lines | `experiments/e1_theory.py` | `e1_theory.csv` | `figures/make_fig_e1.py` |
| Fig 3a–c, codable interval | `experiments/e2_aliasing.py` | `e2_aliasing.csv` | `figures/make_fig_e2.py` |
| Fig 3d–f, precision | `experiments/e6_precision.py` | `e6_precision.csv` | `figures/make_fig_e6.py` |
| S2 Text, precision time-step convergence | `experiments/e6_precision.py` at 1, 4, 10 and 20 steps per ms | `e6_convergence_r*.csv` | `figures/report_e6_convergence.py` |
| S2 Text, why slowing costs less than speeding | `experiments/e6_alignment.py` | `e6_alignment.csv` | *(printed by the script)* |
| Fig 4, column diameter | *(none: an analytic relation)* | — | `../figures/make_fig_column_diameter.py --geometry` |
| S2 Fig, population coding | `experiments/e5_population.py` | `e5_population.csv` | `figures/make_fig_e5.py` |
| S3 Fig a–c, worked example | *(none: hand-set example)* | — | `../figures/make_fig4_lif_vs_integrator.py` |
| S3 Fig d–f, hard versus soft gating | `experiments/e3_discriminability.py` | `e3_discriminability.csv` | `figures/make_fig_e3.py` |
| S5 Text, conjunctive soma | `experiments/e7_verification.py` | `e7_verification.csv` | `figures/make_fig_e7.py` |
| *(cut from the manuscript)* | `experiments/e4_noise.py` | `e4_noise.csv` | `figures/make_fig_e4.py` |

The worked example of S3 Fig a–c has no experiment script, because it is a
hand-set demonstration rather than a measurement: one branch, two inputs, and
delays set to (5, 0). Its figure is drawn by a standalone script with
illustrative traces. The claim it makes is still checked against the production
simulator, by
`tests/test_core.py::test_experiment1_worked_example_under_the_production_model`,
which runs the same configuration through `simulate_population`.

The theory lines of Fig 2 come from `delaykey/theory.py`, which computes the
class fractions without simulating a neuron. They are checked against the
simulator on fresh seeds by `tests/test_theory.py`.

Four experiments are also run at `theta = 4.5`, the dispersion sweep's
operating point, and write to `*_theta45.csv`: the gating comparison, the
aliasing and precision sweeps, and the S5 Text experiment. Those arms use
larger populations, because at the stricter gate the detector count falls by
roughly thirtyfold.

## Reproducing the reported grids

The argparse defaults do **not** reproduce the reported grids — `e1_dispersion`
defaults to dispersions 0–30 where the manuscript sweeps 0–50, and every
experiment defaults to fewer seeds than were run. `run_all.sh` records the exact
command line behind every committed CSV:

```bash
cd delaykey && bash run_all.sh          # full grids, ~13 min on 16 cores
cd delaykey && bash run_all.sh --quick  # smoke-test grids, ~10 s
```

`--quick` writes to `results/quick/` and `figures/out/quick/` rather than
alongside the reported output, and it runs the figure scripts too, so it
exercises the whole pipeline without being able to overwrite any of it. The
experiment scripts' own `--quick` flag reduces the grid but leaves `--out` at
its default, so calling one directly with `--quick` and no `--out` *will*
overwrite the corresponding reported CSV.

## Reproducing

```bash
python3 tests/test_core.py                 # verify the simulator
python3 tests/test_detectors.py            # verify the classification

python3 experiments/e1_dispersion.py --seeds 20      # ~5 min on 60 cores
python3 figures/make_fig_e1.py                       # writes figures/out/
```

`--quick` on any experiment runs a reduced grid as a smoke test.

Every experiment is seeded from `--seed-base` and reports its seed in each CSV
row, so a single row can be reproduced in isolation.

## Verification

The simulator exists in two forms. `simulate_population` is vectorised over
neurons and branches and is what the experiments call. `simulate_reference`
advances one neuron at a time with delays applied feature by feature; it is
written to be read rather than to be fast.

`tests/test_core.py` asserts the two produce **identical** spike counts across
randomised branch counts, dispersions, connection densities, plateau durations
and refractory periods, for both binary and graded input, under both gating
rules, and with per-neuron proximal drive and inhibition — the last of these
because the S5 Text result rests entirely on that path, and an equivalence
guarantee that stopped short of it would not cover the code that produced the
numbers. The remaining tests pin behaviour that has an answer independent of
either implementation: delay compensation discriminating sequence order, the
Experiment 1 worked example, silence without input, the refractory ceiling on
spike count, and independence of neurons within a population.

## Notes on the model

Two deliberate departures from the exploratory notebook that preceded this code:

- **Plateau termination is per neuron.** The notebook cleared every plateau in
  the batch whenever any batch element spiked, because the reset was guarded by
  a scalar `if spike.sum() > 0`. Neurons in a population are independent.
- **Connection weights default to 1.0.** This makes the calcium threshold
  directly interpretable as the number of coincident inputs required, and keeps
  branch drive on the same scale as the threshold regardless of how large the
  upstream population is.

Detector class is assigned functionally, from responses to a two-event stimulus
and to each event alone, all sharing the same spike times. A neuron counts as a
sequence detector only if it fires for the conjunction and for neither event by
itself.

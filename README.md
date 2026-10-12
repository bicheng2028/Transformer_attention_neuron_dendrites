# Companion code

Code for *From anatomy to computation: axonal delay dispersion shifts neurons
from event to sequence detection* (Bi and Sun). It holds the integrator-neuron
model, one script per experiment and per figure, the exact command lines behind
every result table, and the result tables themselves.

## Start here

```bash
cd delaykey-main
pip install -r requirements.txt
python3 tests/test_core.py          # the simulator against a reference implementation
python3 tests/test_detectors.py     # the detector classification
python3 tests/test_theory.py        # the Fig 2 theory against fresh simulations
bash run_all.sh --quick             # every experiment and figure on a reduced grid
```

`delaykey-main/README.md` is the documentation. Its table maps each figure and
reported number to the script and the CSV behind it.

## Layout

| Path | What it is |
| --- | --- |
| `delaykey-main/` | The model (`delaykey/`), the experiments, the figure scripts, the tests, and `run_all.sh`. |
| `delaykey-main/results/` | The CSVs behind the reported numbers. Any figure redraws from them without re-running a simulation. A full re-run is `bash run_all.sh`. |
| `figures/` | The shared figure style and the two generators that sit outside the package: S3 Fig a–c and Fig 4. Run them from the repository root. |

Figures are typeset in Liberation Sans. Run `bash delaykey-main/fonts/fetch.sh`
before regenerating any; without the fonts, matplotlib falls back to DejaVu
Sans with only a warning.

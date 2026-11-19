# Process and Data Science on the BPIC-17 Event Log

Code for the first assignment of *Business Process Prediction, Simulation, and Optimization*
(TUM, winter semester 2026/27). The report is in [`report/`](report/).

The analysis uses the BPI Challenge 2017 loan application log and covers:

1. basic statistics of the event log,
2. an event log with 16 activities (W\_ work items, O\_ offers, A\_ application outcome),
   process discovery (Inductive Miner, Heuristics Miner) and an iteratively refined final process model,
3. decision mining on the XOR gateways of the final model,
4. trace clustering (K-means) with per-cluster process models.

## Repository structure

| Path | Content |
|---|---|
| `01_simple_event_log_analysis.ipynb` | basic statistics (Table 1), variant coverage and case outcome (Table 2) |
| `02_discovery.ipynb` | event log with 16 activities, train/test split check, discovered models and their quality (Table 3) |
| `03_decision_mining.ipynb` | decision mining on the final model (Table 5) and the drawings of the final model |
| `04_clustering.ipynb` | trace clustering, global vs. cluster models (Table 6, silhouette, ARI stability) |
| `src/prep.py` | loading the XES log, closed cases, train/test split, variant filter |
| `src/abstraction_w.py` | event log with 16 activities (one event per W\_ work item) |
| `src/evaluation.py` | fitness, precision, generalization, simplicity, soundness |
| `src/simplicity.py` | own simplicity metrics: size, control-flow complexity (CFC), cyclicity |
| `src/final_model.py` | builds and evaluates one version of the final model (`models/final/<version>/`) |
| `src/draw_bpmn.py` | readable BPMN drawing of the final model (Graphviz) |
| `src/decision_mining.py` | alignment-based decision mining (de Leoni & van der Aalst, 2013) |
| `src/report_assets.py` | LaTeX tables and figures for the report (`report/tables/`, `report/figures/`) |
| `splits/` | case ids of the training (70 %) and test (30 %) set |
| `models/discovery/` | discovered models (PNML, BPMN) |
| `models/final/v00` … `v04` | versions of the final model: definition (`model.edges` or `model.tree`), `notes.md` with the change, PNML/BPMN, drawing (`bpmn.pdf`), metrics and deviations; `history.csv` compares all versions (Table 4) |
| `tables/`, `figures/` | results written by the notebooks |
| `report/` | LaTeX source of the report (TUM template) |

## Setup

Python 3.11 and [Graphviz](https://graphviz.org) (the `dot` program, used for all model drawings) are required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Graphviz is not a Python package: install it with your system package manager
(e.g. `brew install graphviz`, `apt install graphviz` or `conda install graphviz`).

## Data

Download the BPI Challenge 2017 event log (van Dongen, 2017) from 4TU.ResearchData,
DOI [10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b),
and place the XES file at

```
BPIC-17 Data/BPI Challenge 2017.xes
```

On the first run, the log is cached as `BPIC-17 Data/bpic17_eventlog.pkl`.

## Running the analysis

Run everything from the project root in this order. `PYTHONHASHSEED=0` makes the results identical
between runs: PM4Py breaks ties between equally good alignments by set order, which depends on
Python's string hashing (this affects a few decision-mining observations).

```bash
export PYTHONHASHSEED=0
jupyter nbconvert --to notebook --execute --inplace 01_simple_event_log_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_discovery.ipynb
for v in v00 v01 v02 v03 v04; do python -m src.final_model $v; done
jupyter nbconvert --to notebook --execute --inplace 03_decision_mining.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_clustering.ipynb
python -m src.report_assets
```

The notebooks can also be run cell by cell in Jupyter (start it with `PYTHONHASHSEED=0 jupyter lab`). `02_discovery.ipynb` takes about 10 minutes
(alignments for the model without noise filtering); the other steps take a few minutes each.

The report is compiled from `report/main.tex` (e.g. `latexmk -pdf main.tex` in `report/`, or by
uploading the folder to Overleaf).

## Reproducibility

* Random seed 42 for the train/test split, K-means and all sampling; K-means stability is checked with seeds 1–5.
* The train/test split is stored in `splits/`; `02_discovery.ipynb` checks that it is reproduced exactly.
* All models are discovered on the training cases (variants covering 90 % of them) and evaluated on all test cases.
* Python's hash seed is fixed (`PYTHONHASHSEED=0`, see above); `src/final_model.py` sets it itself,
  `03_decision_mining.ipynb` prints a warning if it is not set.
* Package versions are pinned in `requirements.txt`.

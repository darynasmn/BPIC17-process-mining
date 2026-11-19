"""Fitting cases and precision on the training and on the test log (Appendix of the report).

For the discovered models (models/discovery/) and the versions v00-v04 of the final model
(models/final/), with the same computation as for the test set (src/evaluation.py: alignments,
align-ETC precision; token-based replay for the unsound Heuristics Miner model). The models are
not changed. All training cases are used, not only the variants the models were discovered from.

Run from the project root (takes about 30 minutes, mostly IM 0.0):
    PYTHONHASHSEED=0 python -m src.train_vs_test
Writes tables/train_vs_test.csv.
"""
import os
import warnings

os.environ["TQDM_DISABLE"] = "1"  # no progress bars (must be set before tqdm is imported)
warnings.filterwarnings("ignore")

import pandas as pd
import pm4py

from src.evaluation import evaluate
from src.final_model import load_logs
from src.prep import ACT, CASE, TS

MODELS = {**{f"IM {t}": f"models/discovery/L3_WOA__IM_{t}.pnml" for t in ("0.0", "0.2", "0.4")},
          "HM": "models/discovery/L3_WOA__HM.pnml",
          **{v: f"models/final/{v}/model.pnml" for v in ("v00", "v01", "v02", "v03", "v04")}}


def main(out="tables/train_vs_test.csv"):
    train, test = load_logs()
    rows = []
    for name, path in MODELS.items():
        net, im, fm = pm4py.read_pnml(path)
        row = {"model": name}
        for part, log in (("train", train), ("test", test)):
            r, _ = evaluate(net, im, fm, log[[CASE, ACT, TS]])
            row.update({f"{part} method": r["method"], f"{part} fitting cases %": r["fitting cases %"],
                        f"{part} avg fitness": r["avg fitness"], f"{part} precision": r["precision"]})
        print(row, flush=True)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()

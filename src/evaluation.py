"""Quality metrics of a Petri net on a test log (Section 2.3), same computation as for the
previous model versions:

* fitness: share of perfectly fitting cases and average trace fitness (alignments),
* precision: alignment-based ETConformance,
* generalization: PM4Py formula 1 - mean over transitions of 1/sqrt(#firings), with the firings
  counted in the alignments (token replay would credit duplicated labels to one transition),
* simplicity: PM4Py arc degree and our own size, CFC and cyclicity (src/simplicity.py),
* soundness (WF-net check of PM4Py).

If alignments do not finish for every trace within the time limits (possible for unsound nets),
all behavioural metrics fall back to token-based replay and the row is marked `method = TBR`.
"""
import math
import time
from collections import Counter

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

from src.prep import ACT, CASE
from src.simplicity import all_metrics

ALIGN_PARAMS = {"ret_tuple_as_trans_desc": True,
                "max_align_time_trace": 20,    # seconds per trace variant
                "max_align_time": 3600}        # seconds for the whole log


def generalization(net, fired: Counter) -> float:
    s = sum(1 / math.sqrt(fired[t]) if fired[t] else 1 for t in net.transitions)
    return 1 - s / len(net.transitions)


def _variants(test: pd.DataFrame) -> pd.Series:
    return test.groupby(CASE, sort=False)[ACT].agg(tuple)


def evaluate(net, im, fm, test: pd.DataFrame, top_unfit: int = 5) -> tuple[dict, pd.DataFrame]:
    """Metrics row and the `top_unfit` most frequent non-fitting test variants."""
    t0 = time.time()
    row = {"sound": bool(pm4py.check_soundness(net, im, fm)[0]), **all_metrics(net),
           "simplicity (arc degree)": round(pm4py.simplicity_petri_net(net, im, fm), 3)}
    var = _variants(test)
    raw = None
    try:
        raw = alignments.apply(pm4py.convert_to_event_log(test), net, im, fm, parameters=ALIGN_PARAMS)
        if any(a is None for a in raw):
            raw = None
    except Exception as e:  # noqa: BLE001 - any failure -> token-based fallback
        row["note"] = f"alignments failed: {type(e).__name__}"
    if raw is not None:
        fit = [a["fitness"] for a in raw]
        by_name = {t.name: t for t in net.transitions}
        fired = Counter(by_name[m[0][1]] for a in raw for m in a["alignment"] if m[0][1] != ">>")
        row.update({
            "method": "alignments",
            "fitting cases %": round(100 * sum(f == 1 for f in fit) / len(fit), 1),
            "avg fitness": round(sum(fit) / len(fit), 3),
            "precision": round(pm4py.precision_alignments(test, net, im, fm, multi_processing=False), 3),
            "generalization": round(generalization(net, fired), 3),
        })
        is_fit = pd.Series([f == 1 for f in fit], index=var.index)
    else:
        tbr = pm4py.conformance_diagnostics_token_based_replay(test, net, im, fm)
        fit = [r["trace_fitness"] for r in tbr]
        row.update({
            "method": "TBR",
            "fitting cases %": round(100 * sum(r["trace_is_fit"] for r in tbr) / len(tbr), 1),
            "avg fitness": round(sum(fit) / len(fit), 3),
            "precision": round(pm4py.precision_token_based_replay(test, net, im, fm), 3),
            "generalization": round(pm4py.generalization_tbr(test, net, im, fm), 3),
        })
        row.setdefault("note", "alignments did not finish within the time limit")
        is_fit = pd.Series([r["trace_is_fit"] for r in tbr], index=var.index)
    row["seconds"] = round(time.time() - t0)
    unfit = var[~is_fit].value_counts().head(top_unfit)
    top = pd.DataFrame({"cases": unfit.values,
                        "% of test cases": (100 * unfit.values / len(var)).round(2),
                        "variant": [" > ".join(v) for v in unfit.index]})
    return row, top

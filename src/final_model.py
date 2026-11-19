"""Iterative construction of the final process model (Section 2.3).

Event log: W_ work items + O_ offers + A_ (16 activities), built by src/abstraction_w.py;
same train/test split (splits/) and the same metric computation as for the discovered models
(src/evaluation.py: alignments, align-ETC precision, alignment-based generalization).

Every version lives in its own folder models/final/<version>/ and is defined by ONE input file:
  * model.edges - a BPMN written as a list of `source -> target` flows (see bpmn_from_edges), or
  * model.tree  - a process tree in PM4Py notation (always sound, but an activity may be duplicated), or
  * model.bpmn  - a BPMN file edited by hand (e.g. in bpmn.io), for non block-structured changes.
An optional notes.md describes what was changed compared to the previous version.

Running  `python -m src.final_model v01`  (from the project root) writes into the same folder:
  model.bpmn / model.pnml, bpmn.pdf (drawing of the model), metrics.json, deviations.csv
and adds (or replaces) one row in models/final/history.csv.

Process tree operators: ->( ) sequence, X( ) exclusive choice, +( ) parallel,
*( do, redo ) loop, tau = silent step.
"""
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    # PM4Py iterates over sets of places/transitions; with Python's random string hashing,
    # ties between equally good alignments are broken differently on every run, which
    # changes align-ETC precision for models with parallelism. Fix the hash seed.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-m", "src.final_model"] + sys.argv[1:])

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"  # no progress bars
if shutil.which("dot") is None:  # Graphviz from the conda env (see 02_discovery.ipynb)
    os.environ["PATH"] = "/opt/anaconda3/envs/graphviz/bin:" + os.environ["PATH"]

import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.bpmn.obj import BPMN
from pm4py.objects.process_tree.utils.generic import parse

from src.abstraction_w import build_log
from src.draw_bpmn import draw
from src.evaluation import generalization
from src.prep import ACT, CASE, TS, drop_open_cases, load_event_log
from src.simplicity import all_metrics

ROOT = Path("models/final")
XES = Path("BPIC-17 Data/BPI Challenge 2017.xes")
CACHE = Path("BPIC-17 Data/bpic17_eventlog.pkl")


def load_logs():
    df, _ = drop_open_cases(load_event_log(XES, CACHE))
    log = build_log(df)[0][[CASE, ACT, TS]]
    train = set(Path("splits/train_cases.txt").read_text().split("\n"))
    test = set(Path("splits/test_cases.txt").read_text().split("\n"))
    return log[log[CASE].isin(train)], log[log[CASE].isin(test)]


def bpmn_from_edges(text: str) -> BPMN:
    """Build a BPMN model from a list of sequence flows, one `source -> target` per line.

    Node names: `start` / `end` are the events, names starting with `xor_` are exclusive
    gateways, names starting with `and_` are parallel gateways, everything else is a task
    with that activity label. A second node with the same label is written as
    `label @2` (the part from " @" on is dropped from the label).
    Lines starting with # are comments.
    """
    bpmn, nodes = BPMN(), {}
    flows = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            src, dst = (s.strip() for s in line.split("->"))
            flows.append((src, dst))
    names = list(dict.fromkeys(n for f in flows for n in f))
    n_in = Counter(d for _, d in flows)
    n_out = Counter(s for s, _ in flows)
    for n in names:
        if n == "start":
            node = BPMN.StartEvent(id=n, name="start")
        elif n == "end":
            node = BPMN.NormalEndEvent(id=n, name="end")
        elif n.startswith(("xor_", "and_")):
            if n_in[n] > 1 and n_out[n] > 1:
                raise ValueError(f"gateway {n} both merges and splits; use two gateways")
            d = BPMN.Gateway.Direction
            direction = d.CONVERGING if n_in[n] > 1 else d.DIVERGING if n_out[n] > 1 else d.UNSPECIFIED
            cls = BPMN.ExclusiveGateway if n.startswith("xor_") else BPMN.ParallelGateway
            node = cls(id=n, name="", gateway_direction=direction)
        else:
            node = BPMN.Task(id="t" + str(len(nodes)), name=n.split(" @")[0])
        nodes[n] = node
        bpmn.add_node(node)
    # fixed flow ids: PM4Py names silent transitions after them, and those names break ties
    # between equally good alignments (random ids would make precision vary between runs)
    for i, (s, d) in enumerate(flows):
        bpmn.add_flow(BPMN.SequenceFlow(nodes[s], nodes[d], id=f"f{i:03d}"))
    return bpmn


def build_model(folder: Path):
    """Read the version definition and return (net, im, fm, bpmn)."""
    tree_file, edges_file = folder / "model.tree", folder / "model.edges"
    if edges_file.exists():
        bpmn = bpmn_from_edges(edges_file.read_text())
        pm4py.write_bpmn(bpmn, str(folder / "model.bpmn"))
        draw(folder)  # readable drawing with blocks: bpmn.pdf
        net, im, fm = pm4py.convert_to_petri_net(bpmn)
        # parallel joins get a random uuid name; name them after their input places
        for t in net.transitions:
            if t.label is None and "-" in t.name:
                t.name = "join_" + "+".join(sorted(a.source.name for a in t.in_arcs))
    elif tree_file.exists():
        tree = parse(tree_file.read_text())
        bpmn = pm4py.convert_to_bpmn(tree)
        pm4py.write_bpmn(bpmn, str(folder / "model.bpmn"))
        net, im, fm = pm4py.convert_to_petri_net(tree)
    else:
        bpmn = pm4py.read_bpmn(str(folder / "model.bpmn"))
        net, im, fm = pm4py.convert_to_petri_net(bpmn)
    if not edges_file.exists():
        pm4py.save_vis_bpmn(bpmn, str(folder / "bpmn.pdf"))
    pm4py.write_pnml(net, im, fm, str(folder / "model.pnml"))
    return net, im, fm, bpmn


def deviations(aligned) -> pd.DataFrame:
    """Per activity: how often the log has it where the model cannot (log move) and how
    often the model needs it although it did not happen (model move)."""
    log_moves, model_moves = Counter(), Counter()
    for al in aligned:
        for log_lab, mod_lab in al["alignment"]:
            if mod_lab == ">>":
                log_moves[log_lab] += 1
            elif log_lab == ">>" and mod_lab is not None:
                model_moves[mod_lab] += 1
    acts = sorted(set(log_moves) | set(model_moves))
    out = pd.DataFrame({"activity": acts,
                        "log moves": [log_moves[a] for a in acts],
                        "model moves": [model_moves[a] for a in acts]})
    out["total"] = out["log moves"] + out["model moves"]
    return out.sort_values("total", ascending=False)


def top_unfit_variants(test, aligned, n=5):
    seqs = test.groupby(CASE, sort=False)[ACT].agg(tuple)
    unfit = Counter(seq for seq, al in zip(seqs, aligned) if al["fitness"] < 1)
    return [{"cases": c, "variant": " > ".join(v)} for v, c in unfit.most_common(n)]


def evaluate(version: str):
    folder = ROOT / version
    t0 = time.time()
    net, im, fm, _ = build_model(folder)
    train, test = load_logs()
    sound = pm4py.check_soundness(net, im, fm)[0]

    # case order of `aligned` follows the order of cases in `test`;
    # each move is ((log name, model transition name), (log label, model label))
    raw = alignments.apply(pm4py.convert_to_event_log(test), net, im, fm,
                           parameters={"ret_tuple_as_trans_desc": True})
    aligned = [{"fitness": a["fitness"], "alignment": [m[1] for m in a["alignment"]]} for a in raw]
    fit = [a["fitness"] for a in aligned]
    by_name = {t.name: t for t in net.transitions}
    fired = Counter(by_name[m[0][1]] for a in raw for m in a["alignment"] if m[0][1] != ">>")
    tbr_train = pm4py.fitness_token_based_replay(train, net, im, fm)

    m = {
        "version": version,
        "sound": bool(sound),
        **all_metrics(net),
        "arc degree": round(pm4py.simplicity_petri_net(net, im, fm), 3),
        "train fitting cases % (TBR)": round(tbr_train["perc_fit_traces"], 1),
        "test fitting cases % (align)": round(100 * sum(f == 1 for f in fit) / len(fit), 1),
        "test avg fitness (align)": round(sum(fit) / len(fit), 3),
        "test precision (align-ETC)": round(pm4py.precision_alignments(
            test, net, im, fm, multi_processing=False), 3),
        "test generalization (TBR)": round(pm4py.generalization_tbr(test, net, im, fm), 3),
        "test generalization (align)": round(generalization(net, fired), 3),
        "seconds": round(time.time() - t0),
    }
    m["top unfit variants (test)"] = top_unfit_variants(test, aligned)
    (folder / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    deviations(aligned).to_csv(folder / "deviations.csv", index=False)

    row = {k: v for k, v in m.items() if k != "top unfit variants (test)"}
    notes = folder / "notes.md"
    row["change"] = notes.read_text().strip().splitlines()[0] if notes.exists() else ""
    hist_file = ROOT / "history.csv"
    hist = pd.read_csv(hist_file) if hist_file.exists() else pd.DataFrame()
    if len(hist):
        hist = hist[hist["version"] != version]
    hist = pd.concat([hist, pd.DataFrame([row])]).sort_values("version")
    hist.to_csv(hist_file, index=False)
    return m


if __name__ == "__main__":
    res = evaluate(sys.argv[1])
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n" + pd.read_csv(ROOT / "history.csv").drop(columns=["seconds"]).to_string(index=False))

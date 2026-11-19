"""Decision mining on the final model (Section 2.3).

Approach of de Leoni & van der Aalst (SAC 2013), "Data-aware process mining: discovering
decisions in processes using alignments":
1. align every case with the Petri net; the alignment tells which output transition of every
   decision point (place with more than one outgoing transition = XOR-split) the case took,
   also for silent transitions that are invisible in the log;
2. every visit of a decision point becomes one observation: the attribute values known at that
   moment (latest value written by the log events aligned so far) -> the branch taken;
3. one decision tree per decision point (CART from scikit-learn instead of C4.5/J48).
Moves on log only are skipped for the routing, but their attribute values are still read.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from src.prep import ACT, CASE, SEED, TS

CASE_ATTRS = ["case:LoanGoal", "case:ApplicationType", "case:RequestedAmount"]
OFFER_ATTRS = ["OfferedAmount", "MonthlyCost", "NumberOfTerms", "FirstWithdrawalAmount",
               "Accepted", "Selected", "CreditScore"]
# written on O_Create Offer, but they encode what happens later to the offer:
# Selected=True <=> the offer is returned, CreditScore>0 <=> the offer is accepted
LEAKING = ["Selected", "CreditScore"]
# case history at the moment of the decision (W_ + O_ + A_ log, src/abstraction_w.py)
COUNTS = {"n_offers": "O_Create Offer", "n_validate": "W_Validate application",
          "n_incomplete": "W_Call incomplete files"}
# n_later_offers: offers created after the first O_Sent; returned: O_Returned has occurred (0/1);
# open_offers: offers not closed yet (no O_Cancelled / O_Refused / O_Accepted)
DERIVED = ["n_later_offers", "returned", "open_offers", "days_since_start"]
INTEGER_FEATURES = list(COUNTS) + ["n_later_offers", "returned", "open_offers"]
# Attributes of W_ instances (w_duration_h, w_end, resource) are never used: they reveal the later outcome.
# readable names for branches that end at another split gateway (final model v04)
GATEWAY_NAMES = {"xor_out": "to outcome", "xor_bypass": "continue (validation / outcome)"}
# names for silent branches, by the flow (split gateway, next node) of the final model v04
BRANCH_NAMES = {("xor_sub", "xor_wca_in"): "not online", ("xor_hl", "xor_hl_j"): "no W_Handle leads",
                ("xor_wca", "xor_wca_j"): "skip W_Complete application", ("xor_ret", "xor_ret_j"): "no O_Returned"}
CATEGORICAL = ["case:LoanGoal", "case:ApplicationType", "Accepted", "Selected"]


def parse_flows(edges_text: str):
    out = []
    for line in edges_text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(tuple(s.strip() for s in line.split("->")))
    return out


def branch_names(net, flows):
    """Readable name for every output transition of every decision point."""
    succ = Counter(s for s, _ in flows)
    nxt = {s: d for s, d in flows if succ[s] == 1}

    def resolve(node):
        seen = set()
        while node.startswith("xor_") and node in nxt and node not in seen and node not in GATEWAY_NAMES:
            seen.add(node)
            node = nxt[node]
        if node in GATEWAY_NAMES:
            return GATEWAY_NAMES[node]
        if node == "end":
            return "stop"
        label = node.split(" @")[0]
        return label + " (later offer)" if " @" in node else label

    names = {}
    for p in net.places:
        if len(p.out_arcs) < 2:
            continue
        gw = p.name.removeprefix("exi_")
        for a in p.out_arcs:
            t = a.target
            if t.label:
                names[(gw, t.name)] = t.label
            elif t.name.startswith("sfl_f"):
                flow = flows[int(t.name[5:8])]
                names[(gw, t.name)] = BRANCH_NAMES.get(flow) or resolve(flow[1])
            else:
                names[(gw, t.name)] = t.name
    return names


def decision_instances(log: pd.DataFrame, net, im, fm, flows) -> pd.DataFrame:
    """One row per visit of a decision point: gateway, branch, case id and the attribute state."""
    names = branch_names(net, flows)
    dp_of = {}  # transition name -> decision point gateway
    for p in net.places:
        if len(p.out_arcs) > 1:
            for a in p.out_arcs:
                dp_of[a.target.name] = p.name.removeprefix("exi_")
    log = log.sort_values([CASE, TS, "_order"]) if "_order" in log else log
    cases = list(log.groupby(CASE, sort=False))
    ev_log = pm4py.convert_to_event_log(log[[CASE, ACT, TS]])
    aligned = alignments.apply(ev_log, net, im, fm, parameters={"ret_tuple_as_trans_desc": True})
    rows = []
    for (cid, events), al in zip(cases, aligned):
        events = events.reset_index(drop=True)
        start = events[TS].iloc[0]
        state = {c: events[c].iloc[0] for c in CASE_ATTRS}
        state.update({c: np.nan for c in OFFER_ATTRS})
        counts, closed, sent, returned = Counter(), 0, False, 0
        k = 0  # next log event
        last_ts = start
        for (log_name, model_name), _ in al["alignment"]:
            if model_name in dp_of:  # decision point visited: record the state BEFORE the move
                rows.append({"gateway": dp_of[model_name], "branch": names[(dp_of[model_name], model_name)],
                             CASE: cid, **state, **{c: counts[c] for c in COUNTS},
                             "n_later_offers": counts["n_later_offers"], "returned": returned,
                             "open_offers": counts["n_offers"] - closed,
                             "days_since_start": (last_ts - start).total_seconds() / 86400})
            if log_name != ">>":  # the move consumes a log event: update the attribute state
                ev = events.iloc[k]
                k += 1
                last_ts = ev[TS]
                for c in OFFER_ATTRS:
                    if pd.notna(ev[c]):
                        state[c] = ev[c]
                for c, a in COUNTS.items():
                    counts[c] += ev[ACT] == a
                counts["n_later_offers"] += sent and ev[ACT] == "O_Create Offer"
                sent = sent or ev[ACT] == "O_Sent"
                returned = max(returned, int(ev[ACT] == "O_Returned"))
                closed += ev[ACT] in ("O_Cancelled", "O_Refused", "O_Accepted")
    return pd.DataFrame(rows)


def features(df: pd.DataFrame, drop_leaking: bool):
    cols = CASE_ATTRS + OFFER_ATTRS + list(COUNTS) + DERIVED
    if drop_leaking:
        cols = [c for c in cols if c not in LEAKING]
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("string").fillna("missing")
    num = [c for c in X if c not in CATEGORICAL]
    X[num] = X[num].astype(float).fillna(-1)  # -1 = no offer yet
    return pd.get_dummies(X, columns=[c for c in CATEGORICAL if c in X], dtype=float)


def mine(train_inst: pd.DataFrame, test_inst: pd.DataFrame, drop_leaking: bool, max_depth=3):
    """Fit one tree per gateway on train, evaluate on test against the majority baseline."""
    results, trees = [], {}
    for gw in sorted(train_inst["gateway"].unique()):
        tr = train_inst[train_inst.gateway == gw]
        te = test_inst[test_inst.gateway == gw]
        Xtr = features(tr, drop_leaking)
        Xte = features(te, drop_leaking).reindex(columns=Xtr.columns, fill_value=0.0)
        majority = tr["branch"].value_counts().idxmax()
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(20, len(tr) // 100),
                                      random_state=SEED).fit(Xtr, tr["branch"])
        pred = tree.predict(Xte)
        results.append({
            "gateway": gw,
            "branches": " | ".join(f"{b} ({n})" for b, n in tr["branch"].value_counts().items()),
            "n train": len(tr), "n test": len(te),
            "baseline acc": round(accuracy_score(te["branch"], [majority] * len(te)), 3),
            "tree acc": round(accuracy_score(te["branch"], pred), 3),
            "tree macro-F1": round(f1_score(te["branch"], pred, average="macro"), 3),
            "baseline macro-F1": round(f1_score(te["branch"], [majority] * len(te), average="macro"), 3),
            "top features": ", ".join(f"{c} {v:.2f}" for c, v in sorted(
                zip(Xtr.columns, tree.feature_importances_), key=lambda x: -x[1])[:3] if v > 0),
        })
        trees[gw] = (tree, list(Xtr.columns))
    return pd.DataFrame(results), trees


def rules_text(tree, columns) -> str:
    return export_text(tree, feature_names=columns, show_weights=False, decimals=1)


def _condition(col: str, thr: float, left: bool) -> str:
    col = col.removeprefix("case:")
    for cat in CATEGORICAL:  # one-hot column "<attr>_<value>", threshold 0.5
        c = cat.removeprefix("case:")
        if col.startswith(c + "_"):
            value = col[len(c) + 1:]
            if left and value in ("True", "False"):  # "!= False" reads better as "= True"
                return f"{c} = {'True' if value == 'False' else 'False'}"
            return f"{c} {'!=' if left else '='} {value}"
    if col == "returned":
        return "no O_Returned yet" if left else "O_Returned occurred"
    if col in INTEGER_FEATURES:  # integer counts: threshold k + 0.5
        k = int(np.floor(thr))
        if k == 0:
            return f"{col} {'= 0' if left else '>= 1'}"
        return f"{col} {'<=' if left else '>'} {k}"
    return f"{col} {'<=' if left else '>'} {thr:g}"


def short_rule(tree, columns) -> str:
    """Root split of the tree with the majority branch (and its share) on each side;
    the second branch is added when it has at least 25 %."""
    t = tree.tree_
    classes = tree.classes_

    def majority(node):
        v = t.value[node][0]
        v = v / v.sum()
        top = np.argsort(v)[::-1]
        out = f"{classes[top[0]]} ({100 * v[top[0]]:.0f}%)"
        if len(top) > 1 and v[top[1]] >= 0.25:
            out += f", {classes[top[1]]} ({100 * v[top[1]]:.0f}%)"
        return out

    if t.children_left[0] == -1:
        return f"always {majority(0)}"
    col, thr = columns[t.feature[0]], t.threshold[0]
    return (f"if {_condition(col, thr, True)}: {majority(t.children_left[0])}\n"
            f"else: {majority(t.children_right[0])}")


def offer_fate_table(log: pd.DataFrame) -> pd.DataFrame:
    """Per offer: the attributes written on O_Create Offer and what later happens to that offer.
    O_Create Offer has no OfferID; it is always directly followed by O_Created, which has it.
    Needs the ORIGINAL log (with O_Created)."""
    log = log.reset_index(drop=True)
    co = log.index[log[ACT] == "O_Create Offer"]
    assert (log.loc[co + 1, ACT] == "O_Created").all()
    off = pd.DataFrame({"OfferID": log.loc[co + 1, "OfferID"].values,
                        **{c: log.loc[co, c].values for c in ["Accepted", "Selected", "CreditScore"]}})
    later = log[log["OfferID"].notna()].groupby("OfferID")[ACT].agg(set)
    off["returned"] = off.OfferID.map(lambda o: "O_Returned" in later[o])
    off["fate"] = off.OfferID.map(lambda o: next(
        (a for a in ("O_Accepted", "O_Refused", "O_Cancelled") if a in later[o]), "open"))
    off["CreditScore > 0"] = off["CreditScore"] > 0
    return off

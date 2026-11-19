"""Event log abstraction for Section 2.3: work of the bank (W_) + offers (O_) + application outcome (A_).

The final model is the base model for a simulation, so it shows the work items of the bank.
A_ activities that only duplicate W_ work are removed; 16 activities remain:

* W_ (5): one event per work item instance ("one task = one event");
* O_ (6): O_Sent (mail and online) and O_Sent (online only) are merged into O_Sent;
* A_ (5): A_Create Application, A_Submitted (online application) and the three outcomes.

Work item instances
-------------------
The log has no work item id. An instance is reconstructed per case and W_ activity from the
ordered lifecycle transitions: it opens with `schedule` (or with `start` if no instance is open),
collects start / suspend / resume and closes with `complete`, `ate_abort` or `withdraw`.
The instance becomes one event with
* timestamp = first `start` (ties with other events are broken by the file order of that start),
* `w_duration_h` = first start -> end (hours), `w_end` = complete / ate_abort / withdraw,
* `org:resource` = resource of the first start.
Instances that were never started are dropped. Irregular sequences are counted in `stats`.
"""
from collections import Counter

import numpy as np
import pandas as pd

from src.prep import ACT, CASE, LC, ORIGIN, RES, TS

LEVEL = "L3_WOA"
W_KEEP = ["W_Handle leads", "W_Complete application", "W_Call after offers",
          "W_Validate application", "W_Call incomplete files"]
O_KEEP = ["O_Create Offer", "O_Sent", "O_Returned", "O_Accepted", "O_Refused", "O_Cancelled"]
A_KEEP = ["A_Create Application", "A_Submitted", "A_Pending", "A_Denied", "A_Cancelled"]
ACTIVITIES = W_KEEP + O_KEEP + A_KEEP
O_SENT = {"O_Sent (mail and online)": "mail and online", "O_Sent (online only)": "online only"}
# rare W_ activities that are excluded (all other removed activities are simply not kept above);
# note: in the original log the activity name is "W_Shortened completion " (trailing space)
W_REMOVED = ["W_Assess potential fraud", "W_Shortened completion ", "W_Personal Loan collection"]
END_TRANSITIONS = ("complete", "ate_abort", "withdraw")


def w_instances(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """One row per work item instance of the kept W_ activities (also never-started ones)."""
    w = df[df[ACT].isin(W_KEEP)].sort_values([CASE, ACT, TS, "_order"], kind="stable")
    stats = Counter()
    rows = []
    for (case, act), g in w.groupby([CASE, ACT], sort=False):
        cur = None

        def close(end, end_ts):
            rows.append({CASE: case, ACT: act, "first_start": cur["start"], "_order": cur["order"],
                         "end_ts": end_ts, "w_end": end, RES: cur["res"], "scheduled": cur["scheduled"]})

        for lc, ts, order, res in zip(g[LC], g[TS], g["_order"], g[RES]):
            if lc == "schedule":
                if cur is not None:  # new schedule before the previous instance ended
                    stats["schedule while an instance is open (previous closed as 'unfinished')"] += 1
                    close("unfinished", cur["last"])
                cur = {"start": None, "order": None, "res": None, "scheduled": True, "last": ts}
            elif lc == "start":
                if cur is None:
                    stats["start without schedule (new instance)"] += 1
                    cur = {"start": None, "order": None, "res": None, "scheduled": False, "last": ts}
                if cur["start"] is None:
                    cur.update(start=ts, order=order, res=res)
                else:
                    stats["repeated start within an instance (ignored)"] += 1
                cur["last"] = ts
            elif lc in ("suspend", "resume"):
                if cur is None:
                    stats[f"{lc} without open instance (ignored)"] += 1
                else:
                    cur["last"] = ts
            elif lc in END_TRANSITIONS:
                if cur is None:
                    stats[f"{lc} without open instance (ignored)"] += 1
                    continue
                close(lc, ts)
                cur = None
        if cur is not None:  # case ends while the instance is open
            stats["instance open at the end of the case ('unfinished')"] += 1
            close("unfinished", cur["last"])
    inst = pd.DataFrame(rows)
    return inst, stats


def build_log(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Event log of the 16 activities (sorted by case, time, file order) and a report dict."""
    inst, irregular = w_instances(df)
    started = inst[inst["first_start"].notna()].copy()
    report = {
        "W instances": len(inst),
        "W instances never started (dropped)": int(inst["first_start"].isna().sum()),
        "W instances never started, by end": inst.loc[inst["first_start"].isna(), "w_end"].value_counts().to_dict(),
        "W events (started instances)": len(started),
        "W events by end type": started["w_end"].value_counts().to_dict(),
        "irregular lifecycle sequences": dict(irregular),
    }
    started[TS] = started["first_start"]
    started["w_duration_h"] = (started["end_ts"] - started["first_start"]).dt.total_seconds() / 3600
    started[LC] = "complete"
    started[ORIGIN] = "Workflow"
    w_events = started.drop(columns=["first_start", "end_ts", "scheduled"])

    ao = df[df[ACT].isin(O_KEEP + A_KEEP + list(O_SENT))].copy()
    ao["channel"] = ao[ACT].map(O_SENT)
    ao.loc[ao[ACT].isin(list(O_SENT)), ACT] = "O_Sent"
    log = pd.concat([ao, w_events], ignore_index=True)
    # case attributes for the W_ events (taken from the case's first event)
    case_cols = [c for c in df.columns if c.startswith("case:") and c != CASE]
    first = df.drop_duplicates(CASE).set_index(CASE)[case_cols]
    for c in case_cols:
        log[c] = log[CASE].map(first[c])
    log = log.sort_values([CASE, TS, "_order"], kind="stable").reset_index(drop=True)
    assert set(log[ACT]) <= set(ACTIVITIES), set(log[ACT]) - set(ACTIVITIES)
    return log, report


def coverage_stats(log: pd.DataFrame) -> dict:
    """Same columns as Table 2a: events, variants, top-10 variants (% cases), variants for 80 %."""
    tr = log.groupby(CASE, sort=False)[ACT].agg(tuple)
    vc = tr.value_counts()
    cum = 100 * vc.cumsum().to_numpy() / len(tr)
    return {"cases": len(tr), "events": len(log), "variants": len(vc),
            "top-10 variants % cases": round(cum[min(9, len(cum) - 1)], 1),
            "variants for 80 % cases": int(np.searchsorted(cum, 80) + 1)}


# --- checks on the original log ----------------------------------------------------------------
PAIRS = [("A_Submitted", "W_Handle leads"), ("A_Concept", "W_Complete application"),
         ("A_Accepted", "W_Complete application"), ("A_Complete", "W_Call after offers"),
         ("A_Validating", "W_Validate application"), ("A_Incomplete", "W_Call incomplete files")]


def pair_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per replaced pair: cases with both / only A / only W, and the order of the first A event
    vs. the first `start` of the W activity (cases with both and a started W)."""
    n_cases = df[CASE].nunique()
    rows = []
    for a, w_act in PAIRS:
        first_a = df[df[ACT] == a].groupby(CASE)[TS].min()
        w = df[df[ACT] == w_act]
        any_w = set(w[CASE])
        first_start = w[w[LC] == "start"].groupby(CASE)[TS].min()
        both = set(first_a.index) & any_w
        common = first_a.index.intersection(first_start.index)
        d = first_a[common] - first_start[common]
        rows.append({"A activity": a, "W activity": w_act, "cases": n_cases,
                     "both": len(both), "only A": len(set(first_a.index) - any_w),
                     "only W": len(any_w - set(first_a.index)),
                     "neither": n_cases - len(set(first_a.index) | any_w),
                     "A before W start %": round(100 * (d < pd.Timedelta(0)).mean(), 1),
                     "same time %": round(100 * (d == pd.Timedelta(0)).mean(), 1),
                     "A after W start %": round(100 * (d > pd.Timedelta(0)).mean(), 1),
                     "median A - W start (h)": round(d.dt.total_seconds().median() / 3600, 2)})
    return pd.DataFrame(rows)


def classify_o_cancelled(df: pd.DataFrame) -> pd.DataFrame:
    """Class of every O_Cancelled event: 'after outcome' (an A_Pending / A_Denied / A_Cancelled
    happened before it in the case), else 'replacement' (an O_Create Offer follows later in the
    case), else 'other'. Also flags events that would match both definitions."""
    out = []
    for case, g in df[df[ACT].isin(["O_Cancelled", "O_Create Offer", "A_Pending", "A_Denied",
                                    "A_Cancelled"])].groupby(CASE, sort=False):
        acts = g[ACT].to_numpy()
        for i in np.flatnonzero(acts == "O_Cancelled"):
            after_outcome = bool(np.isin(acts[:i], ["A_Pending", "A_Denied", "A_Cancelled"]).any())
            replacement = bool((acts[i + 1:] == "O_Create Offer").any())
            cls = "after outcome" if after_outcome else ("replacement" if replacement else "other")
            out.append({CASE: case, "class": cls, "both definitions": after_outcome and replacement})
    return pd.DataFrame(out)

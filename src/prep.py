"""Shared data preparation for the BPIC-17 notebooks.

All notebooks import these functions so that every stage works on exactly the same
event log, the same train/test split and the same discovery log.
The process-model log of Section 2.3 (16 activities) is built in src/abstraction_w.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# XES standard keys
CASE, ACT, TS = "case:concept:name", "concept:name", "time:timestamp"
LC, RES, ORIGIN = "lifecycle:transition", "org:resource", "EventOrigin"

SEED = 42
OUTCOME_ACTIVITIES = ("A_Pending", "A_Denied", "A_Cancelled")

def load_event_log(xes_path: Path, cache_path: Path) -> pd.DataFrame:
    """Read the XES log (or its pickle cache) and sort events by case, time and file order."""
    if Path(cache_path).exists():
        df = pd.read_pickle(cache_path)
    else:
        import pm4py
        df = pm4py.read_xes(str(xes_path))
        df.to_pickle(cache_path)
    df[TS] = pd.to_datetime(df[TS], utc=True)
    df["_order"] = np.arange(len(df))  # original file order, breaks timestamp ties
    return df.sort_values([CASE, TS, "_order"], kind="stable").reset_index(drop=True)


def drop_open_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep only cases that reach an outcome (A_Pending / A_Denied / A_Cancelled)."""
    closed = df.loc[df[ACT].isin(OUTCOME_ACTIVITIES), CASE].unique()
    n_open = df[CASE].nunique() - len(closed)
    return df[df[CASE].isin(closed)], n_open


def build_abstraction(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Event log at an abstraction level. Only "L2_AO" (A_ and O_ events, W_ removed) is
    defined; it is used for the case features of 03_decision_mining and 04_clustering."""
    if name == "L2_AO":
        return df[df[ORIGIN].isin(["Application", "Offer"])]
    raise ValueError(f"unknown abstraction {name!r}")


def split_cases(case_ids, test_share: float = 0.3, seed: int = SEED) -> tuple[list, list]:
    """Random case-level train/test split with a fixed seed. Returns (train_ids, test_ids)."""
    cases = np.array(sorted(set(case_ids)))
    rng = np.random.default_rng(seed)
    test = set(rng.choice(cases, size=round(test_share * len(cases)), replace=False))
    train = [c for c in cases if c not in test]
    return train, sorted(test)


def traces(log: pd.DataFrame) -> pd.Series:
    """Activity sequence (tuple) per case; the log must already be sorted."""
    return log.groupby(CASE, sort=False)[ACT].agg(tuple)


def filter_top_variants(log: pd.DataFrame, coverage: float) -> tuple[pd.DataFrame, int]:
    """Keep the cases of the most frequent variants that together cover `coverage` of the cases.

    Returns the filtered log and the number of variants kept.
    """
    tr = traces(log)
    vc = tr.value_counts()
    cum = vc.cumsum().to_numpy() / len(tr)
    k = int(np.searchsorted(cum, coverage) + 1)
    keep = set(vc.index[:k])
    kept_cases = tr.index[tr.isin(keep)]
    return log[log[CASE].isin(kept_cases)], k

"""Collect tables and figures for the LaTeX report (folder report/).

Run from the project root after all notebooks/scripts:  python -m src.report_assets
* tables come from the files written by the notebooks / scripts (tables/*.csv, models/final/history.csv)
  and, for Section 2.4, from the stored outputs of the executed notebook 04_clustering.ipynb
  (the notebook does not write result files);
* figures are copied from figures/ to report/figures/ (the final model figures are drawn by
  03_decision_mining.ipynb); the metric comparison of Section 2.4 is redrawn here with short
  axis labels from the stored notebook outputs.
Every table is written to report/tables/<name>.tex (tabular only; caption and label are in main.tex).
Row labels are shortened so that every row fits on one line of a single report column.
"""
import os
import re
import shutil
from pathlib import Path

import nbformat
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if shutil.which("dot") is None:  # Graphviz installed in a separate conda environment
    os.environ["PATH"] = "/opt/anaconda3/envs/graphviz/bin:" + os.environ["PATH"]

REPORT = Path("report")
TAB_OUT = REPORT / "tables"
FIG_OUT = REPORT / "figures"


def tex(s) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#").replace("→", r"$\rightarrow$"))


def write(name: str, lines: list[str]):
    (TAB_OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# --- 2.2 ------------------------------------------------------------------------------------
SHORT_STATS = {
    "Number of cases": "Cases",
    "Number of events": "Events",
    "Number of process variants": "Variants",
    "Number of distinct activities": "Activities",
    "Number of distinct case attribute labels": "Case attribute labels",
    "Number of distinct event attribute labels": "Event attribute labels",
    "Case length (events): mean / std": "Case length, mean / std",
    "Case length (events): median [min, max]": "Case length, median [range]",
    "Case duration (days): mean / std": "Duration (d), mean / std",
    "Case duration (minutes): mean / std": "Duration (min), mean / std",
    "Case duration (seconds): mean / std": "Duration (s), mean / std",
    "Case duration (days): median": "Duration (d), median",
    "Number of categorical event attributes": "Categorical event attr.",
    "Number of resources": "Resources",
    "Observation period": "Observation period",
}


def basic_stats():
    t = pd.read_csv("tables/basic_stats.csv")
    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Statistic & Value \\", r"\midrule"]
    lines += [f"{tex(SHORT_STATS.get(r.Statistic, r.Statistic))} & {tex(r.Value)} \\\\" for r in t.itertuples()]
    write("basic_stats", lines + [r"\bottomrule", r"\end{tabular}"])


def additional_stats():
    cov = pd.read_csv("tables/variant_coverage.csv")
    out = pd.read_csv("tables/outcome_throughput.csv")
    names = {"Full log": "Full log", "Only 'complete' events": "Complete events", "Only A_ and O_ events": "A\\_ and O\\_ events",
             "W_, O_ and A_ (16 activities)": "16 activities"}
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"\multicolumn{5}{@{}l}{\textbf{(a) Variant coverage by abstraction level}} \\",
             r"Abstraction & Events & Variants & \makecell[r]{Top-10\\var.\ (\%)} & \makecell[r]{Var.\ for\\80\,\%} \\",
             r"\midrule"]
    for r in cov.itertuples():
        lines.append(f"{names[r.abstraction]} & {r.events:,} & {r.variants:,} & {r._6:.1f} & {r._8:,} \\\\")
    lines += [r"\midrule",
              r"\multicolumn{5}{@{}l}{\textbf{(b) Case outcome and throughput time}} \\",
              r"Outcome & Cases & \makecell[r]{\% of\\cases} & \makecell[r]{Median\\(days)} & \makecell[r]{Mean / std\\(days)} \\",
              r"\midrule"]
    for r in out.itertuples():
        lines.append(f"{tex(r.outcome)} & {r.cases:,} & {r._3:.1f} & {r.median:.1f} & "
                     f"{r.mean:.1f} / {r.std:.1f} \\\\")
    write("additional_stats", lines + [r"\bottomrule", r"\end{tabular}"])


# --- 2.3 ------------------------------------------------------------------------------------
def discovery_quality():
    t = pd.read_csv("tables/discovery_quality.csv")
    lines = [r"\begin{tabular}{@{}lcrrrrrrr@{}}", r"\toprule",
             r"Model & Sound & \makecell[r]{Fit.\\\%} & \makecell[r]{Avg\\fit.} & Prec. & Gen. & "
             r"\makecell[r]{Simpl.\\(arc)} & Size & CFC \\", r"\midrule"]
    for r in t.itertuples():
        star = "" if r.method == "alignments" else "$^{*}$"  # token-based replay (alignments impossible)
        lines.append(f"{tex(r.model)} & {'yes' if r.sound else 'no'} & {r._4:.1f}{star} & {r._5:.3f}{star} & "
                     f"{r.precision:.3f}{star} & {r.generalization:.3f}{star} & {r._8:.3f} & {r.size} & {r.CFC} \\\\")
    write("discovery_quality", lines + [r"\bottomrule", r"\end{tabular}"])


SHORT_CHANGE = {
    "v00": "IM 0.2 (baseline)",
    "v01": "sequential validation",
    "v02": "later-offer block",
    "v03": "one outcome (XOR)",
    "v04": "W\\_Complete appl.\\ optional",
}


def model_versions():
    t = pd.read_csv("models/final/history.csv")
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Ver. & Change & Fit.\ \% & Prec. & Gen. & Size & CFC \\", r"\midrule"]
    for r in t.itertuples(index=False):
        d = r._asdict()
        lines.append(f"{r.version} & {SHORT_CHANGE[r.version]} & {d['_7']:.1f} & {d['_9']:.3f} & "
                     f"{d['_11']:.3f} & {r.size} & {r.CFC} \\\\")
    write("model_versions", lines + [r"\bottomrule", r"\end{tabular}"])


SHORT_DECISION = {
    "D1": "online submission", "D2": "W\\_Handle leads (online)", "D3": "W\\_Complete appl.\\ (online)",
    "D4": "create one more offer", "D5": "send one more offer", "D6": "later offer or continue",
    "D7": "one more later offer", "D8": "send one more later offer", "D9": "validation or outcome",
    "D10": "O\\_Returned in validation", "D11": "after validation", "D12": "after missing documents",
    "D13": "outcome", "D14": "cancel open offers", "D15": "cancel one more offer",
    "D16": "refuse open offers", "D17": "refuse one more offer",
}


def decision_mining():
    t = pd.read_csv("tables/decision_mining.csv")
    lines = [r"\begin{tabular}{@{}llccl@{}}", r"\toprule",
             r"ID & Decision & \makecell{Accuracy\\base / tree} & \makecell{Macro-F1\\base / tree} & Expl. \\",
             r"\midrule"]
    for r in t.itertuples():
        lines.append(f"{r.id} & {SHORT_DECISION[r.id]} & {r._7:.2f} / {r._8:.2f} & "
                     f"{r._9:.2f} / {r._10:.2f} & {r.explained} \\\\")
    write("decision_mining", lines + [r"\bottomrule", r"\end{tabular}"])


# --- 2.4: stored outputs of the executed clustering notebook ---------------------------------
def _nb_text(nb, marker: str, index: int = 0) -> str:
    """index-th printed text/plain output of the code cell whose source contains `marker`."""
    for c in nb.cells:
        if c.cell_type == "code" and marker in c.source:
            outs = [o.get("data", {}).get("text/plain") or "" for o in c.outputs]
            outs = [t for t in outs if t and not t.startswith("<")]
            return outs[index]
    raise KeyError(marker)


def _read_wrapped(txt: str) -> pd.DataFrame:
    """Parse a pandas text repr that may be wrapped into several blocks (index = first column)."""
    blocks = [b for b in re.split(r"\n\s*\n", txt.strip()) if b.strip()]
    parts = []
    for b in blocks:
        rows = [l for l in b.splitlines() if l.strip()]
        header = re.split(r"\s{2,}", rows[0].strip().rstrip("\\").strip())
        # data rows start with the (integer) cluster id; all values are numeric
        data = [l.rstrip("\\").split() for l in rows[1:] if re.match(r"^\d+\s", l)]
        assert all(len(d) == len(header) + 1 for d in data), b
        parts.append(pd.DataFrame([d[1:] for d in data], index=[d[0] for d in data], columns=header))
    return pd.concat(parts, axis=1)


def cluster_profile(nb):
    p = _read_wrapped(_nb_text(nb, "for c in CLUSTERS:\n    ids = train_cluster.index")).apply(pd.to_numeric)
    lines = [r"\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
             r"Cl. & \makecell[r]{Share\\\%} & Off. & Inc. & Val. & Sub. & Ret. & \makecell[r]{Days\\(med.)} & "
             r"\makecell[r]{Pend.\\\%} & \makecell[r]{Den.\\\%} & \makecell[r]{Canc.\\\%} \\", r"\midrule"]
    for c, r in p.iterrows():
        lines.append(" & ".join([c, f"{100 * r['train share']:.0f}", f"{r['mean n_offers']:.1f}",
                                 f"{r['mean n_incomplete']:.1f}", f"{r['mean n_validating']:.1f}",
                                 f"{r['mean submitted']:.1f}", f"{r['mean returned']:.1f}",
                                 f"{r['median duration_days']:.0f}", f"{100 * r['Pending share']:.0f}",
                                 f"{100 * r['Denied share']:.0f}", f"{100 * r['Cancelled share']:.0f}"]) + r" \\")
    write("cluster_profile", lines + [r"\bottomrule", r"\end{tabular}"])


def _comparison_values(nb) -> dict:
    txt = _nb_text(nb, "glob = results[(results.kind", 0)
    vals = {}
    for l in txt.splitlines()[2:]:
        m = re.match(r"^(.*?)\s{2,}(\S+)\s+(\S+)\s+(\S+)$", l.strip())
        if m:
            vals[m.group(1).strip()] = [float(x) for x in m.groups()[1:]]
    return vals


def model_comparison_figure(nb):
    """Fitting share, precision and size: global model, cluster models, weighted mean, final model."""
    v = _comparison_values(nb)
    clusters = sorted(k for k in v if re.fullmatch(r"cluster \d+", k))
    names = ["global"] + clusters + ["clusters, weighted mean", "final model v04 (reference)"]
    labels = ["G"] + [f"C{k.split()[1]}" for k in clusters] + ["W", "F"]
    size = [v[n][2] for n in names]
    size[-2] = v["clusters, total size"][2]
    colors = ["#5f6368"] + ["#4a7ebb"] * len(clusters) + ["#d9822b", "#b9b9b9"]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 4.6), sharex=True)
    panels = [("Fitting cases (share)", [v[n][0] for n in names], v["global"][0]),
              ("Precision", [v[n][1] for n in names], v["global"][1]),
              ("Size (W = total)", size, v["global"][2])]
    for ax, (title, ys, ref) in zip(axes, panels):
        bars = ax.bar(labels, ys, color=colors, width=0.7)
        bars[-1].set_hatch("///"); bars[-1].set_edgecolor("#666666")
        ax.axhline(ref, color="#5f6368", lw=0.8, ls="--")
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}" if y <= 1 else f"{y:.0f}",
                    ha="center", va="bottom", fontsize=5.5)
        ax.margins(y=0.12)
    axes[0].set_ylim(0, 1.05); axes[1].set_ylim(0, 1.05)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_OUT / "model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def ari_matrix(nb):
    txt = _nb_text(nb, "labelings = {")
    rows = [l for l in txt.splitlines() if l.strip()]
    names = re.split(r"\s{2,}", rows[0].strip())
    data = []
    for l in rows[1:]:
        m = re.match(r"^(.*?)\s{2,}([\d.\s]+)$", l)
        data.append([m.group(1).strip()] + m.group(2).split())
    short = [n.replace(" (main)", "*").replace("seed ", "") for n in names]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(names) + r"@{}}", r"\toprule",
             "Seed & " + " & ".join(short) + r" \\", r"\midrule"]
    lines += [" & ".join([d[0].replace(" (main)", "*").replace("seed ", "")] + d[1:]) + r" \\" for d in data]
    write("ari_matrix", lines + [r"\bottomrule", r"\end{tabular}"])


# --- figures ---------------------------------------------------------------------------------
FIGURES = {
    "figures/final_model.pdf": "final_model.pdf",                       # Figure 1 (03_decision_mining)
    "figures/final_model_decisions.pdf": "final_model_decisions.pdf",   # Figure 2 (03_decision_mining)
    "figures/clustering/silhouette.pdf": "silhouette.pdf",
    **{f"figures/discovery/L3_WOA__IM_{t}_bpmn.pdf": f"discovered_IM_{t}.pdf" for t in ("0.0", "0.2", "0.4")},
    "figures/discovery/L3_WOA__HM_petri.pdf": "discovered_HM.pdf",
}

if __name__ == "__main__":
    for d in (TAB_OUT, FIG_OUT):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    basic_stats(); additional_stats(); discovery_quality(); model_versions(); decision_mining()
    nb = nbformat.read("04_clustering.ipynb", as_version=4)
    cluster_profile(nb); ari_matrix(nb)
    model_comparison_figure(nb)
    for src, dst in FIGURES.items():
        shutil.copy(src, FIG_OUT / dst)
    print("tables:", sorted(p.name for p in TAB_OUT.iterdir()))
    print("figures:", sorted(p.name for p in FIG_OUT.iterdir()))

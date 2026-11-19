"""Readable BPMN drawing for models written as `model.edges` (see src/final_model.py).

Graphviz with orthogonal edges, one framed cluster per block of the process.
Blocks come from an optional blocks.txt, one block per line:
    Block title: node, node, node
Nodes that are not listed are drawn outside the frames.

For the report the model can be wrapped into several rows (`rows`): flows between rows are
replaced by BPMN link events (a throwing link at the source, a catching link with the same
letter at the target), as is usual for large BPMN diagrams.
"""
import subprocess
import textwrap
from pathlib import Path

TASK = 'shape=box, style="rounded,filled", fillcolor="#ffffff", color="#333333", fontname="Helvetica", fontsize={fs}, margin="0.10,0.05"'
XOR = 'shape=diamond, label="×", width=0.34, height=0.34, fixedsize=true, style=filled, fillcolor="#ffffff", color="#333333", fontsize=12, fontname="Helvetica"'
AND = 'shape=diamond, label="+", width=0.34, height=0.34, fixedsize=true, style=filled, fillcolor="#ffffff", color="#333333", fontsize=14, fontname="Helvetica"'
START = 'shape=circle, label="", width=0.28, style=filled, fillcolor="#e8f5e9", color="#2e7d32", penwidth=1.5'
END = 'shape=circle, label="", width=0.28, style=filled, fillcolor="#ffebee", color="#c62828", penwidth=3'
LINK_OUT = 'shape=doublecircle, width=0.22, fixedsize=true, style=filled, fillcolor="#333333", fontcolor="#ffffff", color="#333333", fontname="Helvetica-Bold", fontsize=9'
LINK_IN = 'shape=doublecircle, width=0.22, fixedsize=true, style=filled, fillcolor="#ffffff", fontcolor="#333333", color="#333333", fontname="Helvetica-Bold", fontsize=9'
NOTE = 'shape=note, style=filled, fillcolor="#fff8e1", color="#b08900", fontname="Helvetica", fontsize={fs}, margin="0.06,0.03"'


def _flows(text):
    out = []
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.append(tuple(s.strip() for s in line.split("->")))
    return out


def _blocks(path: Path):
    blocks = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and ":" in line:
                title, nodes = line.split(":", 1)
                blocks.append((title.strip(), [n.strip() for n in nodes.split(",") if n.strip()]))
    return blocks


def _esc(text: str) -> str:
    """Graphviz label: left-aligned lines."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\l") + "\\l"


def _wrap(label: str, width: int) -> str:
    """Break a long activity label once, before '(' or at the space nearest to the middle."""
    if len(label) <= width or " " not in label:
        return label
    if " (" in label:
        a, b = label.split(" (", 1)
        return a + "\\n(" + b
    spaces = [i for i, c in enumerate(label) if c == " "]
    i = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return label[:i] + "\\n" + label[i + 1:]


def to_dot(edges_text, blocks, notes=None, gateway_ids=None, rows=None, compact=False, tight=False):
    """notes: {gateway: text} decision blocks next to the gateway (same rank, dashed link);
    gateway_ids: {gateway: 'D1'} short id next to the gateway;
    rows: list of lists of block titles, one list per row of the drawing;
    tight: smaller gaps between nodes and blocks and no outer margin (figure for the report)."""
    notes, gateway_ids = notes or {}, gateway_ids or {}
    fs_task, fs_note = (10, 8) if compact else (11, 9)
    flows = _flows(edges_text)
    nodes = list(dict.fromkeys(n for f in flows for n in f))
    block_of = {n: t for t, ns in blocks for n in ns}

    # --- wrap into rows: cut flows between rows and insert link events ---------------------
    extra_nodes = {}  # link node -> (attrs, label, block of the node it belongs to)
    if rows:
        row_of_block = {t: r for r, ts in enumerate(rows) for t in ts}
        row = {n: row_of_block[block_of[n]] for n in nodes if n in block_of}
        for _ in range(len(nodes)):  # nodes outside blocks: row of their successor
            for s, d in flows:
                if s not in row and d in row:
                    row[s] = row[d]
        letters, new_flows = {}, []
        cross = sorted({d for s, d in flows if row[s] != row[d]},
                       key=lambda d: (row[d], [x for _, x in flows].index(d)))
        letters = {d: chr(ord("A") + i) for i, d in enumerate(cross)}
        for s, d in flows:
            if row[s] == row[d]:
                new_flows.append((s, d))
                continue
            letter = letters[d]
            out_node, in_node = f"link_out_{s}_{letter}", f"link_in_{letter}"
            extra_nodes[out_node] = (LINK_OUT, letter, block_of.get(s))
            extra_nodes[in_node] = (LINK_IN, letter, block_of.get(d))
            new_flows += [(s, out_node), (in_node, d)]
        flows = list(dict.fromkeys(new_flows))

    def node_line(n):
        q = '"' + n.replace('"', '\\"') + '"'
        if n in extra_nodes:
            attrs, label, _ = extra_nodes[n]
            return f'{q} [{attrs}, label="{label}"];'
        if n == "start":
            return f"{q} [{START}];"
        if n == "end":
            return f"{q} [{END}];"
        if n.startswith(("xor_", "and_")):
            attrs = XOR if n.startswith("xor_") else AND
            if n in gateway_ids:
                attrs += f', xlabel="{gateway_ids[n]}"'
            return f"{q} [{attrs}];"
        label = n.split(" @")[0]
        if compact:
            label = _wrap(label, 14)
        return f'{q} [{TASK.format(fs=fs_task)}, label="{label}"];'

    def node_lines(ns, indent):
        out = []
        for n in ns:
            out.append(indent + node_line(n))
            if n in notes:
                text = notes[n]
                if compact:  # narrow notes: every note widens its whole column
                    text = "\n".join(w for line in text.split("\n")
                                      for w in textwrap.wrap(line, 30, subsequent_indent="  "))
                out.append(f'{indent}"note_{n}" [{NOTE.format(fs=fs_note)}, label="{_esc(text)}"];')
                out.append(f'{indent}{{ rank=same; "{n}"; "note_{n}"; }}')
        return out

    all_nodes = list(dict.fromkeys(n for f in flows for n in f))
    sep = 'nodesep=0.22; ranksep=0.28;' if compact else 'nodesep=0.35; ranksep=0.45;'
    if tight:  # small vertical gaps inside and between the blocks (figure in the report)
        sep = 'nodesep=0.05; ranksep=0.22; pad=0;'
    cluster_margin = ' margin=2; fontsize=9;' if tight else ''
    lines = [
        "digraph BPMN {",
        f'  rankdir=LR; splines=ortho; {sep} newrank=true; forcelabels=true;',
        '  graph [fontname="Helvetica", fontsize=11, labeljust=l];',
        '  edge [color="#444444", arrowsize=0.55, penwidth=0.9];',
    ]
    members = {t: [n for n in ns if n in all_nodes] for t, ns in blocks}
    for n, (_, _, b) in extra_nodes.items():
        if b is not None:
            members[b].append(n)
    in_block = {n for ns in members.values() for n in ns}
    order = list(range(len(blocks)))
    if rows:
        order.sort(key=lambda i: -row_of_block.get(blocks[i][0], 0))
    for i in order:
        title = blocks[i][0]
        lines.append(f'  subgraph cluster_{i} {{ label="{title}"; style="rounded,dashed"; color="#9e9e9e"; fontcolor="#555555";{cluster_margin}')
        lines += node_lines(members[title], "    ")
        lines.append("  }")
    lines += node_lines([n for n in all_nodes if n not in in_block], "  ")
    for s, d in flows:
        lines.append(f'  "{s}" -> "{d}";')
    for n in notes:
        lines.append(f'  "{n}" -> "note_{n}" [style=dashed, color="#b08900", arrowhead=none, constraint=false];')
    lines.append("}")
    return "\n".join(lines)


def render(dot: str, out_base: Path):
    """Write <out_base>.pdf."""
    pdf = Path(out_base).with_suffix(".pdf")
    subprocess.run(["dot", "-Tpdf", "-o", str(pdf)], input=dot, text=True, check=True)


def svg(dot: str) -> str:
    """SVG text of the drawing, to show it in a notebook (IPython.display.SVG)."""
    return subprocess.run(["dot", "-Tsvg"], input=dot, text=True, capture_output=True, check=True).stdout


def _dot(folder: Path, notes, gateway_ids, rows, compact, tight) -> str:
    folder = Path(folder)
    return to_dot((folder / "model.edges").read_text(), _blocks(folder / "blocks.txt"),
                  notes, gateway_ids, rows, compact, tight)


def draw(folder: Path, notes=None, gateway_ids=None, out_base=None, rows=None, compact=False, tight=False) -> str:
    """Draw model.edges of a version folder to <out_base>.pdf (default <folder>/bpmn.pdf);
    returns the SVG text of the same drawing."""
    dot = _dot(folder, notes, gateway_ids, rows, compact, tight)
    render(dot, out_base or Path(folder) / "bpmn")
    return svg(dot)


def preview(folder: Path, **kwargs) -> str:
    """SVG text of the drawing without writing any file."""
    return svg(_dot(folder, kwargs.get("notes"), kwargs.get("gateway_ids"), kwargs.get("rows"),
                    kwargs.get("compact", False), kwargs.get("tight", False)))

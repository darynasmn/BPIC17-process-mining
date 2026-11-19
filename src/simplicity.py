"""Own simplicity metrics for Petri nets (Section 2.3, Table 3).

Mapping of gateway concepts to Petri nets (used by CFC):
* a place with more than one outgoing arc is an exclusive choice (XOR-split);
* a transition with more than one outgoing arc is a parallel split (AND-split).

References
* Size and cyclicity: J. Mendling, Metrics for Process Models, LNBIP 6, Springer, 2008.
* CFC: J. Cardoso, "How to measure the control-flow complexity of web processes and workflows",
  Workflow Handbook, 2005. Workflow-net version: K. B. Lassen, W. M. P. van der Aalst,
  "Complexity metrics for Workflow nets", Information and Software Technology 51(3), 2009.
"""
import networkx as nx
from pm4py.objects.petri_net.obj import PetriNet


def size(net: PetriNet) -> int:
    """Number of nodes: places + transitions (silent transitions included, they are routing nodes)."""
    return len(net.places) + len(net.transitions)


def cfc(net: PetriNet) -> int:
    """Control-flow complexity: every XOR-split adds its number of outgoing branches,
    every AND-split adds 1 (all branches are always taken, only one combined state)."""
    xor = sum(len(p.out_arcs) for p in net.places if len(p.out_arcs) > 1)
    and_ = sum(1 for t in net.transitions if len(t.out_arcs) > 1)
    return xor + and_


def _as_graph(net: PetriNet) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(net.places)
    g.add_nodes_from(net.transitions)
    g.add_edges_from((a.source, a.target) for a in net.arcs)
    return g


def cyclicity(net: PetriNet) -> float:
    """Share of nodes that lie on at least one cycle (0 = acyclic, 1 = everything in loops).

    A node is on a cycle if its strongly connected component has more than one node
    or it has a self-loop.
    """
    g = _as_graph(net)
    on_cycle = set()
    for comp in nx.strongly_connected_components(g):
        if len(comp) > 1:
            on_cycle |= comp
        else:
            (n,) = comp
            if g.has_edge(n, n):
                on_cycle.add(n)
    return len(on_cycle) / g.number_of_nodes()


def all_metrics(net: PetriNet) -> dict:
    return {"size": size(net), "CFC": cfc(net), "cyclicity": round(cyclicity(net), 3)}

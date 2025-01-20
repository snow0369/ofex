from typing import Optional, List, Any, Tuple, Union

import networkx as nx
import numpy as np
from networkx import Graph

from ofex.utils.graph import graph_from_edge_list

# https://arxiv.org/abs/2001.05983
__all__ = ["find_max_clique"]


def find_max_clique(edges: Union[List[Tuple[Any, Any]], np.ndarray],
                    nodes: Optional[List[Any]] = None) -> set:
    g = graph_from_edge_list(edges, nodes)
    return bb_maximum_clique(g)


def bronk(graph: Graph, p, r: Optional[set] = None, x: Optional[set] = None):
    """
    Implementation of Bron–Kerbosch algorithm for finding all maximal cliques in graph
    """
    r = r if r is not None else set()
    x = x if x is not None else set()
    if not any((p, x)):
        yield r
    for node in p.copy():
        for r in bronk(graph, p.intersection(graph.neighbors(node)),
                       r=r.union(node), x=x.intersection(graph.neighbors(node))):
            yield r
        p.remove(node)
        x.add(node)


def greedy_clique_heuristic(graph: Graph):
    """
    Greedy search for clique iterating by nodes
    with highest degree and filter only neighbors
    """
    k = set()
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    while len(nodes) != 0:
        neigh = list(graph.neighbors(nodes[0]))
        k.add(nodes[0])
        nodes.remove(nodes[0])
        nodes = list(filter(lambda x: x in neigh, nodes))
    return k


def greedy_coloring_heuristic(graph: Graph):
    """
    Greedy graph coloring heuristic with degree order rule
    """
    color_num = iter(range(0, len(graph)))
    color_map = {}
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    color_map[nodes.pop(0)] = next(color_num)  # color node with color code
    used_colors = {i for i in color_map.values()}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in
                            list(filter(lambda x: x in color_map, graph.neighbors(node)))}
        if len(neighbors_colors) == len(used_colors):
            color = next(color_num)
            used_colors.add(color)
            color_map[node] = color
        else:
            color_map[node] = next(iter(used_colors - neighbors_colors))
    return len(used_colors)


def branching(graph, cur_max_clique_len):
    """
    Branching procedure
    """
    g1, g2 = graph.copy(), graph.copy()
    max_node_degree = len(graph) - 1
    nodes_by_degree = [node for node in sorted(nx.degree(graph),  # All graph nodes sorted by degree (node, degree)
                                               key=lambda x: x[1], reverse=True)]
    # Nodes with (current clique size < degree < max possible degree)
    partial_connected_nodes = list(filter(
        lambda x: x[1] != max_node_degree and x[1] <= max_node_degree, nodes_by_degree))
    # graph without partial connected node with highest degree
    g1.remove_node(partial_connected_nodes[0][0])
    # graph without nodes which is not connected with partial connected node with highest degree
    g2.remove_nodes_from(
        graph.nodes() -
        graph.neighbors(
            partial_connected_nodes[0][0]) - {partial_connected_nodes[0][0]}
    )
    return g1, g2


def bb_maximum_clique(graph):
    max_clique = greedy_clique_heuristic(graph)
    chromatic_number = greedy_coloring_heuristic(graph)
    if len(max_clique) == chromatic_number:
        return max_clique
    else:
        g1, g2 = branching(graph, len(max_clique))
        return max(bb_maximum_clique(g1), bb_maximum_clique(g2), key=lambda x: len(x))

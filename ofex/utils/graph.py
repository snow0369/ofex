from itertools import product
from typing import List, Tuple, Optional, Any, Union

import numpy as np
from networkx import Graph

__all__ = ["graph_from_edge_list"]

def graph_from_edge_list(edges: Union[np.ndarray, List[Tuple[Any, Any]]],
                         nodes: Optional[List[Any]] = None):
    if isinstance(edges, np.ndarray):
        if len(edges.shape) != 2 or edges.shape[0] != edges.shape[1]:
            raise ValueError(edges)
        n = edges.shape[0]
        nodes = list(range(n))
        edges = [(i, j) for i, j in product(range(n), repeat=2) if i < j and edges[i, j]]

    if nodes is None:
        nodes = set(sum([list(e) for e in edges], []))
    g = Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g

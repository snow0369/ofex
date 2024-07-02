import numpy as np
from openfermion import FenwickTree

from ofex.clifford.clifford_tools import gf
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import SparseStateDict


def _depth_first_search(node_now, visited, func, args):
    for child_node in node_now.children:
        if child_node.index not in visited:
            _depth_first_search(child_node, visited, func, args)
    else:
        func(node_now, args)
        visited.append(node_now)


def _update_descendent_mat(node, mat):
    idx = node.index
    mat[idx, idx] = 1
    for c in node.children:
        mat[idx, :] += mat[c.index, :]


def _update_children_mat(node, mat):
    idx = node.index
    mat[idx, idx] = 1
    for c in node.children:
        mat[idx, c.index] = 1


def _bk_tree_state_transform(state: SparseStateDict, func) -> SparseStateDict:
    num_qubits = get_num_qubits(state)
    fenwick_tree = FenwickTree(num_qubits)
    mat = gf(np.zeros((num_qubits, num_qubits), dtype=int))
    # By visiting the nodes, in the dfs manner, update the conversion matrix.
    _depth_first_search(fenwick_tree.root, list(), func, mat)
    new_state = dict()
    for occ, coeff in state.items():
        occ = gf(occ)
        new_state[BinaryFockVector(mat @ occ)] = coeff
    return new_state


def bravyi_kitaev_tree_state(fermion_state: SparseStateDict) -> SparseStateDict:
    return _bk_tree_state_transform(fermion_state, _update_descendent_mat)


def inv_bravyi_kitaev_tree_state(qubit_state: SparseStateDict) -> SparseStateDict:
    return _bk_tree_state_transform(qubit_state, _update_children_mat)

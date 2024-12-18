"""
This module implements Bravyi-Kitaev tree transformations for Fermionic quantum states, 
as described in https://arxiv.org/abs/1701.07072. This transformation maps Fermionic states to qubit
states while maintaining computational efficiency based on Fenwick-tree. Key features include the
conversion to and from the tree-based representation.

Functions:
- bravyi_kitaev_tree_state: Converts Fermionic states to the Bravyi-Kitaev tree representation.
- inv_bravyi_kitaev_tree_state: Converts Bravyi-Kitaev tree states back to Fermionic states.
"""
import numpy as np
from openfermion import FenwickTree

from ofex.clifford.clifford_tools import gf
from ofex.state import BinaryFockVector
from ofex.state.state_tools import get_num_qubits
from ofex.state.types import SparseStateDict

__all__ = []

def _depth_first_search(node_now, visited, func, args):
    """
    Perform a depth-first traversal of a tree structure.

    Args:
        node_now: The current node being visited in the tree.
        visited: A list of nodes that have already been traversed.
        func: A function to apply to each node during the traversal.
        args: Additional arguments required by the `func`.

    Returns:
        None. Modifies `visited` and applies `func` to each node in the depth-first search order.
    """
    for child_node in node_now.children:
        if child_node.index not in visited:
            _depth_first_search(child_node, visited, func, args)
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
    """
    Convert a Fermionic Fock state to its Bravyi-Kitaev tree representation.
    Refer to https://arxiv.org/abs/1701.07072 for more details.

    Args:
        fermion_state (SparseStateDict): A dictionary representing the input Fermionic Fock state.

    Returns:
        SparseStateDict: The Bravyi-Kitaev tree representation of the input state.
    """
    return _bk_tree_state_transform(fermion_state, _update_descendent_mat)


def inv_bravyi_kitaev_tree_state(qubit_state: SparseStateDict) -> SparseStateDict:
    """
    Convert a Bravyi-Kitaev tree state back to its original Fermionic Fock representation.
    Refer to https://arxiv.org/abs/1701.07072 for more details.

    Args:
        qubit_state (SparseStateDict): A dictionary representing the input Bravyi-Kitaev tree state.

    Returns:
        SparseStateDict: The Fermionic Fock representation of the input Bravyi-Kitaev tree state.
    """
    return _bk_tree_state_transform(qubit_state, _update_children_mat)

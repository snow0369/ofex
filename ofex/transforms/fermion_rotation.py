import numpy as np
from openfermion import FermionOperator, normal_ordered

from ofex.operators.fermion_operator_tools import cre_ann
from ofex.state.binary_fock import BinaryFockVector
from ofex.state.state_tools import to_sparse_dict, get_num_qubits, state_type_transform
from ofex.state.types import State, type_state
from ofex.utils.dict_utils import add_values


def fermion_rotation_operator(fham: FermionOperator, v_mat: np.ndarray,
                              spatial_v: bool = False) -> FermionOperator:
    """
    Transforms the hamiltonian by the matrix mixing the orbitals.

    Args:
        fham: Original Hamiltonian
        v_mat: Mixing matrix
        spatial_v:

    Returns:
        trans_fham: Transformed hamiltonian

    """
    ret = FermionOperator()
    if spatial_v:
        v_mat = np.kron(v_mat, np.eye(2))
    v_mat = np.linalg.inv(v_mat)
    fham = normal_ordered(fham)
    for op, coeff in fham.terms.items():
        op_cre, op_ann = cre_ann(op)
        tmp_fsum_1 = FermionOperator() + coeff
        for i, c in enumerate(op_cre + op_ann):
            is_ann = i >= len(op_cre)
            tmp_fsum_2 = FermionOperator()
            for j in np.argwhere(np.logical_not(np.isclose(v_mat[c], 0.0))).ravel():
                j = int(j)
                tmp_fsum_2 += FermionOperator(((j, int(not is_ann)),), v_mat[c][j])
            tmp_fsum_1 = tmp_fsum_1 * tmp_fsum_2
        tmp_fsum_1 = normal_ordered(tmp_fsum_1)
        ret = ret + tmp_fsum_1
    return ret


def fermion_rotation_state(state: State,
                           v_mat: np.ndarray,
                           spatial_v: bool) -> State:
    num_qubits = get_num_qubits(state)
    input_type = type_state(state)
    state = to_sparse_dict(state)
    if spatial_v:
        v_mat = np.kron(v_mat, np.eye(2))
    v_mat = np.linalg.inv(v_mat)
    coeff_dict = dict()

    for f, c in state.items():
        cre = FermionOperator.identity()
        for i, occ in enumerate(f):
            if not occ:
                continue
            tmp_cre = FermionOperator()
            for j in range(v_mat.shape[1]):
                if np.isclose(v_mat[i][j], 0.0):
                    continue
                tmp_cre = tmp_cre + FermionOperator(((j, 1),), v_mat[i][j])
            cre = cre * tmp_cre
        tmp_dict = dict()
        for op, c1 in cre.items():
            cre_idx, _ = op.operator
            new_f = BinaryFockVector([0 if i not in cre_idx else 1 for i in range(num_qubits)])
            assert new_f not in tmp_dict
            tmp_dict[new_f] = c * c1
        coeff_dict = add_values(coeff_dict, tmp_dict)
    return state_type_transform(state, input_type)

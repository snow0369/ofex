import numpy as np
from openfermion import FermionOperator, normal_ordered

from ofex.operators.fermion_operator_tools import cre_ann
from ofex.state import BinaryFockVector
from ofex.state.state_tools import to_sparse_dict, get_num_qubits, state_type_transform
from ofex.state.types import State, type_state
from ofex.utils.dict_utils import add_values


__all__ = ["fermion_rotation_operator", "fermion_rotation_state"]

def fermion_rotation_operator(fop: FermionOperator,
                              v_mat: np.ndarray,
                              spatial_v: bool = False) -> FermionOperator:
    """
    Transforms the FermionOperator using an orbital mixing matrix.
    
    This function takes a FermionOperator and applies a transformation based on the provided
    orbital mixing matrix (v_mat). The matrix may optionally account for spin-orbital mixing
    when spatial_v is set to True.
    
    Args:
        fop (FermionOperator): The original FermionOperator.
        v_mat (np.ndarray): The orbital mixing matrix. Each element represents the transformation
            coefficients between orbitals.
        spatial_v (bool): Whether v_mat is in spatial orbital. If True, considers spin-orbital
            mixing by applying the Kronecker product of v_mat with the identity matrix for spin
            states. Defaults to False.
    
    Returns:
        FermionOperator: A new FermionOperator representing the transformed Hamiltonian after
        applying the orbital mixing defined by v_mat.
    """
    ret = FermionOperator()
    if spatial_v:
        v_mat = np.kron(v_mat, np.eye(2))
    v_mat = np.linalg.inv(v_mat)
    fop = normal_ordered(fop)
    for op, coeff in fop.terms.items():
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
    """
    Transforms a Fermionic state represented in the Fock basis using
    an orbital mixing matrix.

    This function applies a rotation on the input state according to the
    orbital mixing matrix `v_mat`. If `spatial_v` is set to True, the mixing
    matrix is expanded to include spin-orbital interactions.

    The function maintains type consistency by converting the final state back
    to its original representation.

    Args:
        state (State): The input quantum state in Fock representation.
        v_mat (np.ndarray): The orbital mixing matrix. Each element represents the transformation
            coefficients between orbitals.
        spatial_v (bool): Whether v_mat is in spatial orbital. If True, considers spin-orbital
            mixing by applying the Kronecker product of v_mat with the identity matrix for spin
            states. Defaults to False.

    Returns:
        State: The rotated quantum state in its original representation.
    """
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

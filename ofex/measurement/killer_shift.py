from copy import deepcopy
from itertools import product
from typing import Tuple, Any, Dict, Optional

import numpy as np
import scipy.optimize
from openfermion import FermionOperator, QubitOperator, normal_ordered

from ofex.linalg.sparse_tools import sparse_apply_operator
from ofex.measurement.pauli_grouping import sorted_insertion
from ofex.operators.fermion_operator_tools import cre_ann
from ofex.state import BinaryFockVector
from ofex.state.state_tools import compress_sparse
from ofex.transforms import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.transforms.fermion_factorization import ham_to_ei_spin


__all__ = ["killer_shift_opt_fermion_hf"]

def killer_shift_opt_fermion_hf(fham: FermionOperator,
                                hf_vector: BinaryFockVector,
                                transform: str,
                                optimization_level: int = 1,
                                repeat_opt: int = 1,
                                f2q_kwargs: Optional[Dict[str, Any]] = None) \
        -> Tuple[FermionOperator, QubitOperator, float]:
    """
    Perform the killer shift optimization for a given fermionic Hamiltonian 
    and Hartree-Fock (HF) vector.
    
    This method applies the killer shift algorithm to reduce the measurement cost 
    in the transitional amplitude <φ0|H|φ1>, where φ0 corresponds to a single 
    Slater determinant. The algorithm identifies a Hermitian operator T such that 
    T|φ0> = t|φ0>, where t is a real constant, minimizing the norm of H - T. It 
    supports varying levels of optimization and is based on the methodologies 
    described in the paper [Arxiv:2409.02504](https://arxiv.org/abs/2409.02504).
    
    Args:
        fham (FermionOperator): The input fermionic Hamiltonian to which the 
            killer shift optimization is applied.
        hf_vector (BinaryFockVector): The Hartree-Fock state represented as a 
            binary vector, serving as the reference state for the killer shift.
        transform (str): Specifies the transformation method for mapping fermionic 
            operators to qubit operators (e.g., 'Jordan-Wigner' or 'Bravyi-Kitaev'), 
            supported in the function `ofex.transform.fermion_to_qubit_operator`.
        optimization_level (int, optional): The optimization level used in the killer 
            shift procedure. Available options (default: 1):
            - 0: Use the number operator only.
            - 1: Include contributions from both one- and two-body operators with efficient optimization.
            - 2: Include contributions from both one- and two-body operators with numerical optimization.
        repeat_opt (int, optional): The number of optimization iterations to 
            perform for refining the solution (default: 1).
        f2q_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments 
            passed to the fermion-to-qubit transformation functions (default: None).
    
    Returns:
        Tuple[FermionOperator, QubitOperator, float]: A tuple containing:
            - FermionOperator: The optimized fermionic Hamiltonian with reduced 
              norm after applying the killer shift.
            - QubitOperator: The corresponding qubit Hamiltonian obtained after 
              the fermion-to-qubit mapping and applying the killer shift.
            - float: The energy constant resulting from the killer shift 
              transformation.
    """
    if f2q_kwargs is None:
        f2q_kwargs = dict()
    if optimization_level not in [0, 1, 2]:
        raise ValueError
    const = fham.constant
    fham = fham - const
    fham = normal_ordered(fham)
    n_spinorb = len(hf_vector)

    occ = [i for i, f in enumerate(hf_vector) if f]

    pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
    const -= pham.constant
    hf_qubit_state = fermion_to_qubit_state({hf_vector: 1.0}, transform, **f2q_kwargs)
    hf_qubit_fock = list(hf_qubit_state.keys())[0]
    assert np.isclose(hf_qubit_state[hf_qubit_fock], 1.0)
    assert len(hf_qubit_state) == 1

    # Find Z-Only
    """
    shift_0 = PauliSum()
    for op in pham.terms():
        if op.is_z_only():
            shift_0 = shift_0 + op
    """
    shift = FermionOperator()
    for f_term in fham.get_operators():
        cre, ann = cre_ann(f_term)
        if sorted(cre) == sorted(ann):  # Number operator
            shift += f_term

    # Objects for opt_level=2
    # Number operator(p) * one-body operator(r,s)
    g_prs = np.zeros((n_spinorb, n_spinorb, n_spinorb), dtype=float)
    # Targets to be optimized :: operators corresponds to the occupied orbitals (n_p = 1)
    dof = list()  # degree of freedom to be optimized (p, r, s)
    dof_idx = dict()  # idx of (p, r, s) in the dof list.
    shift_op = list()  # Shift operators of corresponding dof.

    if optimization_level == 0:  # one-body number operator only
        pass
    elif optimization_level in [1, 2]:
        shifted_fham = fham - shift
        _, tei, c = ham_to_ei_spin(shifted_fham, n_spinorb)
        tei_diag_1 = np.zeros(tei.shape[:3], dtype=complex)
        tei_diag_2 = np.zeros(tei.shape[:3], dtype=complex)
        for p in range(tei.shape[0]):
            tei_diag_1[p] = tei[p, p, :, :]
            tei_diag_2[p] = tei[:, :, p, p]
        assert np.allclose(tei_diag_1, tei_diag_2)
        assert np.isclose(c, 0.0)
        for p in range(n_spinorb):
            if p in occ:
                n_op = FermionOperator(((p, 1), (p, 0)), 1.0) - FermionOperator.identity()
            else:
                n_op = FermionOperator(((p, 1), (p, 0)), 1.0)

            for r, s in product(range(n_spinorb), repeat=2):
                if p in {r, s}:  # a^2
                    continue
                if r <= s:  # Lower triangular part
                    continue
                # op1 + op2 = n_p (a†_r a_s + c.c), normal ordered

                if not np.isclose(tei_diag_1[p, r, s], 0.0):
                    assert np.isclose(tei_diag_1[p, r, s], tei_diag_1[p, s, r])
                    ex_op = FermionOperator(((r, 1), (s, 0)), 1.0) + FermionOperator(((s, 1), (r, 0)), 1.0)
                    tmp_shift = n_op * ex_op
                    tmp_shift += ex_op * n_op
                    no_tmp_shift = normal_ordered(tmp_shift)
                    if optimization_level == 1 or p not in occ:
                        c = tei_diag_1[p, r, s]
                        shift = shift + no_tmp_shift * (c / 2)
                        continue

                    # optimization_level == 2 and p in occ:
                    g_prs[p, r, s] = c / 2
                    dof_idx[(p, r, s)] = len(dof)
                    dof.append((p, r, s))
                    shift_op.append(no_tmp_shift)

    if optimization_level == 2:
        # Optimization
        x_initial = np.zeros(len(dof), dtype=float)
        for r, s in product(range(n_spinorb), repeat=2):
            if r <= s:
                continue
            for p in occ:
                if (p, r, s) in dof_idx:
                    x_initial[dof_idx[(p, r, s)]] = g_prs[p, r, s]

        def cost(_x):
            _shift = FermionOperator()
            _shift.terms = deepcopy(shift.terms)
            for _coeff, _op in zip(_x, shift_op):
                _shift += _op * _coeff
            _shift_pauli = fermion_to_qubit_operator(_shift, transform, **f2q_kwargs)
            return sum([_h.induced_norm(order=2) for _h in sorted_insertion(pham - _shift_pauli)])

        res_list = list()
        for _ in range(repeat_opt):
            res = scipy.optimize.minimize(cost, x_initial, method="powell", options={"maxiter": 1000})
            res_list.append(res)
        res = sorted(res_list, key=lambda opt_res: opt_res.fun)[0]
        for x, op in zip(res.x, shift_op):
            shift += op * x

    shift_pauli = fermion_to_qubit_operator(shift, transform, **f2q_kwargs)

    tilde_qubit_state = sparse_apply_operator(shift_pauli, hf_qubit_state)
    tilde_qubit_state = compress_sparse(tilde_qubit_state, atol=1e-7)
    assert len(tilde_qubit_state) == 1  # If bug here, fix atol in compress_sparse().
    tilde_fock = list(tilde_qubit_state.keys())[0]
    assert hf_qubit_fock == tilde_fock, (hf_qubit_fock, tilde_fock)

    eig = list(tilde_qubit_state.values())[0]
    const += eig

    assert np.isclose(const.imag, 0.0)
    const = const.real

    return fham - shift, pham - shift_pauli, const

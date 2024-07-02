from itertools import product
from typing import Tuple, Any, Dict, Optional

import numpy as np
import scipy.optimize
from openfermion import FermionOperator, QubitOperator, normal_ordered, get_fermion_operator

from ofex.linalg.sparse_tools import sparse_apply_operator
from ofex.operators.fermion_operator_tools import cre_ann, normal_ordered_single, one_body_excitation
from ofex.operators.qubit_operator_tools import dict_to_operator
from ofex.state.binary_fock import BinaryFockVector
from ofex.transforms.fermion_qubit import fermion_to_qubit_operator, fermion_to_qubit_state
from ofex.operators import symbolic_operator_tools


def killer_shift_opt_fermion_hf(fham: FermionOperator,
                                hf_vector: BinaryFockVector,
                                transform: str,
                                optimization_level: int = 1,
                                repeat_opt: int = 1,
                                f2q_kwargs: Optional[Dict[str, Any]] = None) \
        -> Tuple[FermionOperator, QubitOperator, float]:
    if f2q_kwargs is None:
        f2q_kwargs = dict()
    if optimization_level not in [0, 1, 2]:
        raise ValueError
    const = fham.constant
    fham = fham - const
    fham = normal_ordered(fham)
    n_spinorb = len(hf_vector)

    # Find Z-Only
    """    shift_0 = PauliSum()
    for op in pham.terms():
        if op.is_z_only():
            shift_0 = shift_0 + op"""
    shift = FermionOperator()
    for f_term in fham.get_operators():
        cre, ann = cre_ann(f_term)
        if sorted(cre) == sorted(ann):  # Number operator
            shift += f_term
    if optimization_level == 0:  # one-body number operator only
        pass
    elif optimization_level == 1:
        for p in range(n_spinorb):
            for r, s in product(range(n_spinorb), repeat=2):
                if p == r or p == s:  # a^2
                    continue
                if r <= s:  # Lower triangular part
                    continue
                # op1 + op2 = n_p (a†_r a_s + c.c), normal ordered
                op1 = normal_ordered_single(cre=[p, r], ann=[p, s])
                op2 = normal_ordered_single(cre=[p, s], ann=[p, r])
                if op1 in fham.terms:
                    assert op2 in fham.terms, (op1, op2, fham.terms[op1])
                    assert np.isclose(fham.terms[op1], fham.terms[op2])
                    coeff = fham.terms[op1]
                    tmp_shift = dict_to_operator({op1: coeff, op2: coeff}, base=FermionOperator)
                    shift = shift + tmp_shift
    elif optimization_level == 2:  # Include number * excitation
        # Recover one-body tensor
        h_rs = np.zeros((n_spinorb, n_spinorb), dtype=float)
        for r, s in product(range(n_spinorb), repeat=2):
            if r <= s:
                continue
            one_op0 = ((r, 1), (s, 0))
            one_op1 = ((s, 1), (r, 0))
            if one_op0 in fham.terms:
                h_rs[r, s] = fham.terms[one_op0]
                assert np.isclose(h_rs[r, s], fham.terms[one_op1])

        # Number operator(p) * one-body operator(r,s)
        g_prs = np.zeros((n_spinorb, n_spinorb, n_spinorb), dtype=float)
        # Targets to be optimized :: operators corresponds to the occupied orbitals (n_p = 1)
        dof = list()  # degree of freedom to be optimized (p, r, s)
        dof_idx = dict()  # idx of (p, r, s) in the dof list.
        occ_set = set()  # Set of occupied orbitals
        shift_op = list()  # Shift operators of corresponding dof.

        for p in range(n_spinorb):
            for r, s in product(range(n_spinorb), repeat=2):
                if p == r or p == s:  # a^2
                    continue
                if r <= s:  # Lower triangular part
                    continue
                # op1 + op2 = n_p (a†_r a_s + c.c), normal ordered
                op1 = normal_ordered_single(cre=[p, r], ann=[p, s])
                op2 = normal_ordered_single(cre=[p, s], ann=[p, r])
                if op1 in fham.terms:
                    assert op2 in fham.terms, (op1, op2, fham.terms[op1])
                    assert np.isclose(fham.terms[op1], fham.terms[op2])
                    coeff = fham.terms[op1]
                    if hf_vector[p]:  # n_p = 1
                        occ_set.add(p)
                        g_prs[p, r, s] = coeff
                        dof_idx[(p, r, s)] = len(dof)
                        dof.append((p, r, s))
                        ph = symbolic_operator_tools.coeff(FermionOperator(((r, 1), (s, 0))) * FermionOperator(((p, 1), (p, 0))))
                        shift_op.append(dict_to_operator({op1: 1.0, op2: 1.0}, FermionOperator)
                                        - one_body_excitation(r, s) * ph)
                        """
                        coeff = 0.5 * (coeff + one_coeff)
                        tmp_shift = FermionSum({op1: coeff, op2: coeff})
                        ph = (FermionOperator(((q,), (r,))) * FermionOperator(((p,), (p,)))).coeff
                        tmp_shift = tmp_shift + (one_body_excitation(q,r)* -coeff * ph)
                        """
                    else:
                        tmp_shift = dict_to_operator({op1: coeff, op2: coeff}, base=FermionOperator)
                        shift = shift + tmp_shift

        # Optimization
        x_initial = np.zeros(len(dof), dtype=float)
        for r, s in product(range(n_spinorb), repeat=2):
            if r <= s:
                continue
            for p in list(occ_set):
                if (p, r, s) in dof_idx:
                    x_initial[dof_idx[(p, r, s)]] = - h_rs[r, s] / len(occ_set)

        def cost(_x):
            _c = 0.0
            for _r, _s in product(range(n_spinorb), repeat=2):
                if _r <= _s:
                    continue
                # One-body cost
                _c += abs(h_rs[_r, _s] + sum([_x[dof_idx[(_p, _r, _s)]] for _p in list(occ_set)
                                              if (_p, _r, _s) in dof_idx]))

                # Two-body cost
                for _p in list(occ_set):
                    if (_p, _r, _s) in dof_idx:
                        _c += abs(g_prs[(_p, _r, _s)] - _x[dof_idx[(_p, _r, _s)]])
            return _c

        res_list = list()
        for _ in range(repeat_opt):
            res = scipy.optimize.minimize(cost, x_initial, method="powell", options={"maxiter": 1000})
            res_list.append(res)
        res = sorted(res_list, key=lambda opt_res: opt_res.fun)[0]
        for x, op in zip(res.x, shift_op):
            shift += op * x
    else:
        raise ValueError

    pham = fermion_to_qubit_operator(fham, transform, **f2q_kwargs)
    hf_qubit_state = fermion_to_qubit_state({hf_vector: 1.0}, transform, **f2q_kwargs)
    hf_qubit_fock = list(hf_qubit_state.keys())[0]

    shift_pauli = fermion_to_qubit_operator(shift, transform, **f2q_kwargs)

    tilde_qubit_state = sparse_apply_operator(shift_pauli, hf_qubit_state)
    tilde_fock = list(tilde_qubit_state.keys())[0]

    if tilde_fock is not None:
        assert hf_qubit_fock == tilde_fock, (hf_qubit_fock, tilde_fock)
        eig = list(tilde_qubit_state.values())[0]
        const += eig
    assert np.isclose(const.imag, 0.0)
    const = const.real

    return fham - shift, pham - shift_pauli, const


if __name__ == "__main__":
    from ofex.utils.chem import molecule_example
    from ofex.state.chem_ref_state import hf_ground
    from ofex.measurement.sorted_insertion import sorted_insertion


    def betas(pham: QubitOperator):
        fh = sorted_insertion(pham, anticommute=False)
        lcu = sorted_insertion(pham, anticommute=True)
        return sum([x.induced_norm(order=2) for x in fh]), sum([x.induced_norm(order=2) for x in lcu])


    def shift_test():
        mol_list = ["H2", "LiH", "BeH2", "H2O"]
        tr_list = ["jordan_wigner", "bravyi_kitaev", "symmetry_conserving_bravyi_kitaev"]
        for mol_name, transform in product(mol_list, tr_list):
            print(f"mol = {mol_name}, transform = {transform}")
            mol = molecule_example(mol_name)
            hf = hf_ground(mol)
            assert len(hf) == 1
            hf = list(hf.keys())[0]
            fham = mol.get_molecular_hamiltonian()
            fham = get_fermion_operator(fham)
            if transform == "jordan_wigner":
                kwargs = dict()
            elif transform == "bravyi_kitaev":
                kwargs = {'n_qubits': mol.n_qubits}
            elif transform == 'symmetry_conserving_bravyi_kitaev':
                kwargs = {'active_fermions': mol.n_electrons,
                          'active_orbitals': mol.n_qubits}
            else:
                raise AssertionError
            pham = fermion_to_qubit_operator(fham, transform, **kwargs)
            pham = pham - pham.constant
            beta, antibeta = betas(pham)
            print("\tOriginal")
            print(f"\t\tnorm     = {pham.induced_norm(order=1)}")
            print(f"\t\tSI norm  = {beta}")
            print(f"\t\tASI norm = {antibeta}")
            # print(pham.pretty_string(num_tabs=2))

            _, shifted_pham_0, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                                   optimization_level=0,
                                                                   f2q_kwargs=kwargs)
            beta, antibeta = betas(shifted_pham_0)
            print("\tShifted 0")
            print(f"\t\tnorm     = {shifted_pham_0.induced_norm(order=1)}")
            print(f"\t\tSI norm  = {beta}")
            print(f"\t\tASI norm = {antibeta}")
            print(f"\t\tshift_const = {const}")
            # print(shifted_pham_0.pretty_string(num_tabs=2))

            _, shifted_pham_1, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                                   optimization_level=1,
                                                                   f2q_kwargs=kwargs)
            beta, antibeta = betas(shifted_pham_1)
            print("\tShifted 1")
            print(f"\t\tnorm     = {shifted_pham_1.induced_norm(order=1)}")
            # print(shifted_pham_1.pretty_string(num_tabs=2))
            print(f"\t\tSI norm  = {beta}")
            print(f"\t\tASI norm = {antibeta}")
            print(f"\t\tshift_const = {const}")

            _, shifted_pham_2, const = killer_shift_opt_fermion_hf(fham, hf, transform,
                                                                   optimization_level=2,
                                                                   repeat_opt=5,
                                                                   f2q_kwargs=kwargs)
            beta, antibeta = betas(shifted_pham_2)
            print("\tShifted 2")
            print(f"\t\tnorm     = {shifted_pham_2.induced_norm(order=1)}")
            # print(shifted_pham_1.pretty_string(num_tabs=2))
            print(f"\t\tSI norm  = {beta}")
            print(f"\t\tASI norm = {antibeta}")
            print(f"\t\tshift_const = {const}")


    shift_test()

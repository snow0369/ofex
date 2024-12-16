import os
import pickle
from typing import List, Optional

from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.measurement.measurement_variance import pauli_covariance
from ofex.measurement.pauli_grouping import pauli_split_group
from ofex.operators.symbolic_operator_tools import compare_operators
from ofex.state.types import State


def init_ics(ham: QubitOperator,
             ref1: State,
             ref2: Optional[State] = None,
             num_workers: int = 1,
             anticommute: bool = False,
             method="even",
             cov_buf_dir: Optional[str] = None,
             phase_list: Optional[List[float]] = None,
             debug: bool = False):
    """
    Produce initial coefficient splitting and covariance matrices for Iterative Coefficient Splitting (ICS).
    
    Args:
        ham (QubitOperator): The Hamiltonian operator with zero constant term.
        ref1 (State): The reference quantum state for covariance computation.
        ref2 (Optional[State]): An optional second reference state for covariance computation.
        num_workers (int, optional): The number of worker processes for parallel computation. Defaults to 1.
        anticommute (bool, optional): If True, enables anti-commutation partitioning. Defaults to False.
        method (str, optional): Method for initial coefficient assignment. Options are "even" or "si". Defaults to "even".
        cov_buf_dir (Optional[str], optional): Directory to store or load pre-computed covariance matrices. Defaults to None.
        phase_list (Optional[List[float]], optional): List of phase factors for the states. Defaults to None.
        debug (bool, optional): If True, enables debug information during execution. Defaults to False.
    
    Returns:
        tuple: A tuple containing:
            - initial_grp (tuple): Initial grouping of the Hamiltonian into compatible groups:
                - pauli_list: List of all Pauli operators from the Hamiltonian.
                - grp_pauli_list: List of Pauli operators in each group.
                - pauli_grp_list: List mapping Pauli operators to their respective group indices.
            - cov_dict (dict): Covariance matrix data for Pauli operators.
    
    Raises:
        ValueError: If the Hamiltonian operator has a non-zero constant term or if the checksum validation fails.
        AssertionError: If cov_buf_dir is not valid.
    """
    if ham.constant != 0.0:
        raise ValueError("Hamiltonian should have zero trace.")

    if cov_buf_dir is None or not os.path.exists(cov_buf_dir):
        _, initial_grp = pauli_split_group(ham, anticommute, method, debug)
        pauli_list, grp_pauli_list, pauli_grp_list = initial_grp

        if cov_buf_dir is not None:
            if not os.path.isdir(cov_buf_dir):
                os.mkdir(cov_buf_dir)
            save_pauli_list = [p.terms for p in pauli_list]
            with open(os.path.join(cov_buf_dir, "init_frag.pkl"), "wb") as f:
                pickle.dump((save_pauli_list, grp_pauli_list, pauli_grp_list), f)

    elif os.path.isdir(cov_buf_dir):
        with open(os.path.join(cov_buf_dir, "init_frag.pkl"), "rb") as f:
            load_pauli_list, grp_pauli_list, pauli_grp_list = pickle.load(f)
        pauli_list = list()
        for p in load_pauli_list:
            tmp_pauli = QubitOperator()
            tmp_pauli.terms = p
            pauli_list.append(tmp_pauli)
        checksum = QubitOperator.accumulate(pauli_list)
        if not ham.isclose(checksum, tol=EQ_TOLERANCE):
            raise ValueError(compare_operators(ham, checksum))

    else:
        raise AssertionError

    initial_grp = pauli_list, grp_pauli_list, pauli_grp_list

    cov_dict = pauli_covariance(pauli_list, grp_pauli_list, ref1, ref2, num_workers, anticommute, cov_buf_dir, phase_list, debug)
    return initial_grp, cov_dict


def init_efficient_ics(ham,
                       anticommute: bool = False,
                       method: str = "even",
                       debug: bool = False):
    """
    Produce initial coefficient splitting for efficient_ics
    
    Args:
        ham (QubitOperator): The Hamiltonian operator to partition.
        anticommute (bool, optional): If True, utilizes anti-commutation partitioning. Defaults to False.
        method (str, optional): Method used for initial coefficient assignment. Options include "even". Defaults to "even".
        debug (bool, optional): If True, enables debug information during execution. Defaults to False.

    Returns:
        grp_pauli_list (list): A list of groups containing Pauli operators in each group.
    """
    return pauli_split_group(ham, anticommute, method, debug)[1]

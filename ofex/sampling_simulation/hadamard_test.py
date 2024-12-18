from numbers import Number
from typing import Tuple, Optional

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.linalg.sparse_tools import transition_amplitude, state_dot
from ofex.sampling_simulation.sampling_base import ProbDist
from ofex.state.types import State

__all__ = ["hadamard_test_general", "hadamard_test_qubit_operator", "hadamard_test_fermion_operator"]

def hadamard_test_general(overlap: complex,
                          imaginary: bool,
                          coeff: Number = 1.0,) \
        -> ProbDist:
    r"""
    Computes the probability distribution for a Hadamard Test.

    This function generates the probability distribution for a Hadamard Test, where the expectation value
    is computed as :math:``c \braket{\phi_1 | \phi_2}``. It supports computation for both the real and
    imaginary components of the overlap.

    Args:
        overlap (complex): The overlap value :math:``\braket{\phi_1 | \phi_2}``, provided as a complex number.
        imaginary (bool): If True, computes the probability distribution for the imaginary component of
            the overlap. If False, computes the real component.
        coeff (Number, optional): A scaling coefficient :math:``c`` applied to the probabilities.
            Defaults to 1.0.

    Returns:
        ProbDist: The probability distribution for the chosen component (real or imaginary) of the
            expectation value.

    Raises:
        ValueError: If the absolute value of the overlap exceeds 1.0, which indicates a non-unitary operator.

    """
    ab_overlap = abs(overlap)
    if not (np.isclose(ab_overlap, 1.0, atol=EQ_TOLERANCE) or ab_overlap < 1.0):
        raise ValueError(f"Operator seems non-unitary (abs(overlap)={ab_overlap} exceeds 1.0).")
    if not imaginary:
        p0 = 0.5 * (overlap.real + 1)
    else:
        p0 = 0.5 * (overlap.imag + 1)
    probdist = ProbDist({coeff: p0, -coeff: 1 - p0})
    return probdist


def hadamard_test_qubit_operator(ref_state_1: State,
                                 ref_state_2: State,
                                 unitary: Optional[QubitOperator] = None,
                                 coeff=1.0,
                                 sparse_1: bool = False,
                                 sparse_2: bool = False,) \
        -> Tuple[ProbDist, ProbDist]:
    r"""
    Computes probability distributions for the Hadamard Test expectation value of a qubit operator.

    This function calculates the real and imaginary components of the expectation value:

    .. math::

        c \braket{\phi_1 | U | \phi_2},

    where :math:`U` is an optional unitary operator applied to the second reference state :math:`\ket{ \phi_2 }`.
    If the scaling coefficient :math:`c` is zero, the function returns trivial distributions.

    Args:
        ref_state_1 (State): The first reference state :math:`\ket{ \phi_1 }`.
        ref_state_2 (State): The second reference state :math:`\ket{ \phi_2 }`.
        unitary (Optional[QubitOperator]): An optional unitary operator :math:`U` applied to :math:`\ket{ \phi_2 }`.
            If not provided, no unitary is applied.
        coeff (float, optional): A scaling coefficient applied to the probabilities. Defaults to 1.0.
        sparse_1 (bool, optional): A flag indicating if `ref_state_1` is in sparse format. Defaults to False.
        sparse_2 (bool, optional): A flag indicating if `ref_state_2` is in sparse format. Defaults to False.

    Returns:
        Tuple[ProbDist, ProbDist]: A tuple containing two probability distributions:
            - The first distribution corresponds to the real component of the expectation value.
            - The second distribution corresponds to the imaginary component of the expectation value.

    Raises:
        ValueError: If the scaling coefficient :math:`\text{coeff}` is not a finite number.
    """

    if np.isclose(coeff, 0, atol=EQ_TOLERANCE):
        probdist_real = ProbDist({coeff: 1})
        probdist_imag = ProbDist({coeff: 1})
        return probdist_real, probdist_imag
    if unitary is not None:
        overlap = transition_amplitude(unitary, ref_state_1, ref_state_2, sparse_1, sparse_2)
    else:
        overlap = state_dot(ref_state_1, ref_state_2)
    prob_re = hadamard_test_general(overlap, False, coeff)
    prob_im = hadamard_test_general(overlap, True, coeff)
    return prob_re, prob_im


def hadamard_test_fermion_operator(ref_state_1: State,
                                   ref_state_2: State,
                                   unitary: Optional[QubitOperator] = None,
                                   coeff=1.0) \
        -> Tuple[ProbDist, ProbDist]:
    raise NotImplementedError

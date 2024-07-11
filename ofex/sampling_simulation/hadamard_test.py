from numbers import Number
from typing import Tuple, Optional

import numpy as np
from openfermion import QubitOperator
from openfermion.config import EQ_TOLERANCE

from ofex.linalg.sparse_tools import transition_amplitude, state_dot
from ofex.sampling_simulation.sampling_base import ProbDist
from ofex.state.types import State


def hadamard_test_general(overlap: complex,
                          imaginary: bool,
                          coeff: Number = 1.0,) \
        -> ProbDist:
    """
    Generate probability distributions for a Hadamard Test whose expectation value is c * <φ1|φ2>.

    :param overlap: The value of <φ1|φ2>
    :param coeff: (Optional) Additional coefficient
    :return:
        probdist_real, probdist_imag: Probability distributions for real and imaginary parts.
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
    """
    Generate probability distributions for a Hadamard Test whose expectation value is c * <φ1|U|φ2>.

    :param ref_state_1: Reference state φ1
    :param ref_state_2: Reference state φ2
    :param unitary: (Optional) Unitary applied to φ2
    :param coeff: (Optional) Additional coefficient
    :return:
        probdist_real, probdist_imag: Probability distributions for real and imaginary parts.
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

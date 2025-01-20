from copy import deepcopy
from typing import Union, Optional

import numpy as np
from openfermion import QubitOperator
from scipy.sparse import spmatrix, issparse, identity

from ofex.exceptions import OfexTypeError
from ofex.propagator import exact_rte

__all__ = ["hamiltonian_fourier_series"]


def hamiltonian_fourier_series(coeff: np.ndarray,
                               freqs: np.ndarray,
                               real_time_propagator: Optional[Union[spmatrix, np.ndarray]] = None,
                               normalized_pham: Optional[Union[QubitOperator, np.ndarray, spmatrix]] = None,
                               n_qubits: Optional[int] = None,
                               exact_sparse: bool = False, ) \
        -> Union[spmatrix, np.ndarray]:
    r"""Computes a Fourier series representation of a function of the Hamiltonian,
    :math:`F(H) = \sum_{k=-N}^{N} c_k e^{i \omega_k t}`.
    
    This function calculates the Fourier series representation of a function of the 
    Hamiltonian, using user-defined coefficients and frequencies. The representation 
    can be constructed by directly using a precomputed real-time propagator or by 
    providing the normalized Hamiltonian and system specifications.
    
    Args:
        coeff (np.ndarray): Fourier series coefficients, where the middle element
            corresponds to the zero frequency.
        freqs (np.ndarray): Frequencies corresponding to the Fourier coefficients.
            Must be a 1-dimensional array with an odd number of equally spaced
            elements, and the middle value must be zero.
        real_time_propagator (Optional[Union[spmatrix, np.ndarray]]): Precomputed
            propagator for evolving the function of the Hamiltonian in real-time. 
            If not provided, both `normalized_pham` and `n_qubits` must be specified.
        normalized_pham (Optional[Union[QubitOperator, np.ndarray, spmatrix]]): 
            Normalized Hamiltonian with the spectral norm less than or equal to one
            to used to compute the propagator if `real_time_propagator` is not provided.
        n_qubits (Optional[int]): Number of qubits for the normalized Hamiltonian, 
            required when `real_time_propagator` is not specified.
        exact_sparse (bool): Whether to use sparse matrix operations when building
            the propagator internally. Only used when `real_time_propagator` is not provided.
    
    Returns:
        Union[spmatrix, np.ndarray]: The Fourier series representation of a function of 
        the Hamiltonian, as either a sparse or dense matrix, depending on the input 
        type of `real_time_propagator`.
    
    Raises:
        ValueError: If `coeff` and `freqs` have mismatched shapes, `freqs` is not 1D,
            `freqs` does not have equally spaced values, or its middle value is not zero.
        OfexTypeError: If `real_time_propagator` is not a supported type (e.g., sparse
            matrix or dense ndarray).
    """

    if coeff.shape != freqs.shape:
        raise ValueError("coeff and freqs must have the same shape.")
    if freqs.ndim != 1:
        raise ValueError("freqs must be a 1-dimensional array.")
    if len(freqs) % 2 == 0:
        raise ValueError("freqs must have an odd number of elements.")

    df = np.diff(freqs)
    if not np.all(df == df[0]):
        raise ValueError("freqs must be equally spaced.")
    df = float(df[0])

    if real_time_propagator is None:
        if normalized_pham is None or n_qubits is None:
            raise ValueError(
                "Both 'normalized_pham' and 'n_qubits' cannot be None when 'real_time_propagator' is not provided.")
        real_time_propagator = exact_rte(normalized_pham, df, n_qubits, exact_sparse)

    if issparse(real_time_propagator):
        series = real_time_propagator.__class__(real_time_propagator.shape)
        out_type = "spmatrix"
    elif isinstance(real_time_propagator, np.ndarray):
        series = np.zeros_like(real_time_propagator)
        out_type = "ndarray"
    else:
        raise OfexTypeError(real_time_propagator)

    half_idx = len(freqs) // 2
    if not np.isclose(freqs[half_idx], 0.0):
        raise ValueError("Middle frequency should be zero.")
    elif out_type == "spmatrix":
        series += identity(series.shape[0]) * coeff[half_idx]
    elif out_type == "ndarray":
        series += np.eye(series.shape[0]) * coeff[half_idx]

    neg_freqs = freqs[:half_idx][::-1]
    neg_coeff = coeff[:half_idx][::-1]
    pos_freqs = freqs[half_idx + 1:]
    pos_coeff = coeff[half_idx + 1:]

    rte_k = deepcopy(real_time_propagator)
    for idx, (neg_freq, neg_coeff) in enumerate(zip(neg_freqs, neg_coeff)):
        series += rte_k * neg_coeff
        if idx < len(neg_freqs) - 1:
            rte_k = rte_k @ real_time_propagator

    rte_dag = deepcopy(real_time_propagator.conj().T)
    rte_k = deepcopy(rte_dag)
    for idx, (pos_freq, pos_coeff) in enumerate(zip(pos_freqs, pos_coeff)):
        series += rte_dag * pos_coeff
        if idx < len(pos_freqs) - 1:
            rte_dag = rte_k @ rte_dag

    return series

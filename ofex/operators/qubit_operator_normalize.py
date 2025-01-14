from typing import Tuple

import numpy as np
from openfermion import QubitOperator

__all__ = ["normalize_by_lcu_norm"]


def normalize_by_lcu_norm(ham: QubitOperator,
                          level: int = 1,
                          **kwargs) -> Tuple[QubitOperator, float]:
    """
    Normalize a QubitOperator using different methods to calculate its Linear Combination of Unitaries (LCU) norm.

    The eigenspectrum of the normalized operator is guaranteed to be less than one, while this computation is
    much more efficient than normalization by spectral norm.
    Better normalization makes the spectrum more separated within the range [-1, 1], which corresponds to
    normalization using smaller norms.

    This function supports multiple levels of normalization strategies. Before normalization, it ensures that
    any constant term in the Hamiltonian is removed.

    Args:
        ham (QubitOperator): The Hamiltonian to normalize.
        level (int, optional): Specifies the normalization strategy:
            - level = 0: Uses the Pauli 1-norm for normalization.
            - level = 1: Applies the Sorted Insertion (SI) method based on commutation relationships.
            - level = 2: Uses the optimally sorted insertion (SI) method that minimizes norms.
        **kwargs: Additional arguments for optimization when `level=2`.

    Returns:
        Tuple[QubitOperator, float]: A tuple where the first element is the normalized Hamiltonian,
                                     and the second element is the calculated norm.

    Raises:
        ValueError: If the Hamiltonian contains a constant term or if the specified level is invalid.
    """
    from ofex.measurement import sorted_insertion, optimal_sorted_insertion

    if () in ham.terms:
        if np.isclose(ham.terms[()], 0.0):
            del ham.terms[()]
        else:
            raise ValueError("Make trace zero before the normalization (Remove the constant term).")

    if level == 0:
        norm = ham.induced_norm(order=1)
    elif level in (1, 2):
        if level == 1:
            si_out = sorted_insertion(ham, anticommute=True)
        else:
            si_out = optimal_sorted_insertion(ham, anticommute=True, **kwargs)
        l2_norm_list = [frag.induced_norm(order=2) for frag in si_out]
        norm = sum(l2_norm_list)
    else:
        raise ValueError
    return ham / norm, norm

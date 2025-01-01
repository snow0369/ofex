from typing import Tuple

import galois
import numpy as np
from galois import FieldArray

gf = galois.GF(2)

__all__ = ["gf_is_zero", "gf_is_equal", "gf_concatenate", "gf_eye"]

def gf_is_zero(a: FieldArray) -> bool:
    """
    Checks if all elements of a given FieldArray are zero.

    Args:
        a (FieldArray): An input FieldArray to check.

    Returns:
        bool: True if all elements in the FieldArray are zero, otherwise False.
    """
    return np.allclose(np.array(a), 0)


def gf_is_equal(a: FieldArray, b: FieldArray) -> bool:
    """
    Checks if two FieldArray objects are equal in GF(2).

    Args:
        a (FieldArray): The first input FieldArray.
        b (FieldArray): The second input FieldArray.

    Returns:
        bool: True if the two FieldArrays are equal, otherwise False.
    """
    if not isinstance(a, FieldArray) or not isinstance(b, FieldArray):
        raise TypeError("Both inputs must be of type FieldArray")
    return np.allclose(np.array(a + b), 0)


def gf_concatenate(arrays: Tuple[FieldArray, ...], axis:int=0) -> FieldArray:
    arrays = tuple(np.array(a) for a in arrays)
    return gf(np.concatenate(arrays, axis=axis))


def gf_eye(n:int) -> FieldArray:
    return gf(np.eye(n, dtype=int))

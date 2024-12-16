"""
This module provides utility functions for binary manipulation and calculations.

Functions:
- int_to_binary: Converts an integer to a binary representation as a tuple.
- binary_to_int: Converts a binary tuple back into an integer.
- hamming_weight: Computes the Hamming weight (number of 1 bits) in an integer.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["int_to_binary", "binary_to_int", "hamming_weight"]


def int_to_binary(integer, length: int, lsb_first: bool) -> Tuple[int, ...]:
    """
    Convert an integer to its binary representation as a tuple of bits.

    Args:
        integer (int): The integer to be converted.
        length (int): The length of the resulting binary tuple, padded with zeros if needed.
        lsb_first (bool): If True, the least significant bit (LSB) is placed at the start of the tuple.

    Returns:
        Tuple[int, ...]: A tuple representing the binary form of the integer.
    """
    if lsb_first:
        return tuple([int(x) for x in list(bin(integer)[2:].rjust(length, "0"))][::-1])
    else:
        return tuple([int(x) for x in list(bin(integer)[2:].rjust(length, "0"))])


def binary_to_int(binary: Tuple[int, ...], lsb_first: bool) -> int:
    """
    Convert a tuple of binary digits into its integer representation.

    Args:
        binary (Tuple[int, ...]): A tuple containing binary digits (0s and 1s).
        lsb_first (bool): If True, the least significant bit (LSB) is placed at the start of the tuple.

    Returns:
        int: The integer value of the binary input.
    """
    if lsb_first:
        return sum([2 ** i for i in range(len(binary)) if binary[i]])
    else:
        return sum([2 ** i for i in range(len(binary)) if binary[::-1][i]])


def hamming_weight(x: int) -> int:
    """
    Calculate the Hamming weight (number of 1s in the binary representation) of an integer.

    Args:
        x (int): The integer whose Hamming weight is to be calculated.

    Returns:
        int: The number of 1s in the binary representation of x.

    Raises:
        ValueError: If the input integer is negative.
    """
    c = 0
    if x < 0:
        raise ValueError("Negative integers are not allowed.")
    while x != 0:
        c += x & 1
        x >>= 1
    return c

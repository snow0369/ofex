from __future__ import annotations

from typing import Sequence, Union, Tuple

from ofex.operators.types import SinglePauli
from ofex.utils.binary import int_to_binary, binary_to_int

STATE_LSB_FIRST = False
STATE_PRINT_LSB_FIRST = True

__all__ = ["BinaryFockVector"]

class BinaryFockVector(tuple):
    """
    Represents a binary Fock vector, a tuple-like structure used to describe quantum states
    in terms of occupation numbers of either qubits or spin orbitals.
    """
    @property
    def num_qubits(self) -> int:
        """
        Returns:
            int: The number of qubits (or bits) represented by this BinaryFockVector.
        """
        return len(self)

    @property
    def num_spin_orbitals(self) -> int:
        """
        Returns:
            int: The number of spin orbitals, equal to the number of qubits.
        """
        return len(self)

    @property
    def num_spatial_orbitals(self) -> int:
        """
        Returns:
            int: The number of spatial orbitals, calculated as half the number of spin orbitals.
            Assumes that the total number of spin orbitals is even.
        """
        assert len(self) % 2 == 0
        return len(self) // 2

    def __new__(cls,
                inp: Sequence[Union[int, bool]], ):
        """
        Converts the input sequence to a BinaryFockVector instance.
    
        Args:
            inp (Sequence[Union[int, bool]]): A sequence representing binary occupation states.
    
        Returns:
            BinaryFockVector: An instance of the BinaryFockVector class.
        """
        inp = [int(bool(x)) for x in inp]
        return super(BinaryFockVector, cls).__new__(cls, tuple(inp))

    def to_int(self) -> int:
        """
        Converts the BinaryFockVector to its integer representation.
    
        Returns:
            int: Integer equivalent of the binary Fock vector.
        """
        return fock_to_int(self)

    def __getitem__(self, item) -> Union[BinaryFockVector, int]:
        """
        Retrieves an element or subvector from the BinaryFockVector.
    
        Args:
            item : A single index or a slice object.
    
        Returns:
            Union[BinaryFockVector, int]: The specific bit (as an integer) or a subvector
            (as a BinaryFockVector) based on the item provided.
        """
        if isinstance(item, slice):
            return BinaryFockVector(super(BinaryFockVector, self).__getitem__(item))
        elif hasattr(item, "__iter__"):
            return BinaryFockVector([super(BinaryFockVector, self).__getitem__(i) for i in item])
        x = super(BinaryFockVector, self).__getitem__(item)
        if isinstance(x, tuple):  # Re-wrap tuple as BinaryFockVector if needed
            x = BinaryFockVector(x)
        return x

    def marginal_check(self,
                       qubits: Sequence[int],
                       idx: Union[int, Sequence[Union[int, bool]]]) -> bool:
        """
        Checks if the marginal state of the given qubits matches the provided index.
    
        Args:
            qubits (Sequence[int]): Indices of qubits to check against.
            idx (Union[int, Sequence[Union[int, bool]]]): Expected marginal state, either as
                an integer or a sequence of binary values.
    
        Returns:
            bool: True if the given qubits' state matches the index, else False.
        """
        if isinstance(idx, int):
            return BinaryFockVector(self[qubits]).to_int() == idx
        return tuple(self[qubits]) == tuple(idx)

    def apply_pauli(self, pauli: SinglePauli) -> Tuple[BinaryFockVector, complex]:
        """
        Applies a Pauli operator to the BinaryFockVector, modifying its state and calculating
        any associated phase change.
    
        Args:
            pauli (SinglePauli): A Pauli operator to apply, which contains indices and
                                 operation types (X, Y, Z, or I) (e.g. ((0, "X"), (1, "Y"))).
    
        Returns:
            Tuple[BinaryFockVector, complex]: A new BinaryFockVector representing the updated
            state and the complex phase factor introduced by the Pauli operations.
        """
        new_fock = list(self)
        new_phase = 0
        for idx, p in pauli:
            # idx = p_idx if STATE_LSB_FIRST else (-p_idx-1)
            if p == "X":
                new_fock[idx] = not new_fock[idx]
            elif p == "Y":
                new_phase += 3 if new_fock[idx] else 1
                new_fock[idx] = not new_fock[idx]
            elif p == "Z":
                new_phase += 2 if new_fock[idx] else 0
            elif p == "I":
                continue
            else:
                raise AssertionError
        phase = 1j ** new_phase
        return BinaryFockVector(new_fock), phase

    def pretty_string(self, fermion: bool = False) -> str:
        """
        Generates a human-readable string representation of the BinaryFockVector.
    
        Args:
            fermion (bool): If True, represents the state in a fermionic format. Otherwise,
                            uses a raw binary representation.
    
        Returns:
            str: A string representation of the BinaryFockVector.
        """
        fock = tuple(self) if STATE_LSB_FIRST == STATE_PRINT_LSB_FIRST else tuple(self)[::-1]
        if fermion:
            n_fermion_mode = len(fock)
            if n_fermion_mode % 2 != 0:
                raise ValueError(f"Even number of orbitals are required to represent with spins.")
            ret_str = ""
            for k in range(n_fermion_mode // 2):
                if fock[2 * k] and fock[2 * k + 1]:
                    ret_str += "2"
                elif fock[2 * k] and (not fock[2 * k + 1]):
                    ret_str += "↑"
                elif (not fock[2 * k]) and fock[2 * k + 1]:
                    ret_str += "↓"
                else:
                    ret_str += "0"
            return "|" + ret_str + ">"
        else:
            return "|" + "".join([str(x) for x in fock]) + ">"


def int_to_fock(idx: int,
                num_qubits: int) -> BinaryFockVector:
    """
    Converts an integer to a BinaryFockVector, suited for a specified number of qubits.

    Args:
        idx (int): The integer index representing the state.
        num_qubits (int): The number of qubits the state should span.

    Returns:
        BinaryFockVector: The corresponding binary Fock vector representation.

    Raises:
        IndexError: If the integer index exceeds the state space for the given number of qubits.
    """
    if idx > 2 ** num_qubits:
        raise IndexError(f"index {idx} is out of bounds with qubits of {num_qubits} (n_dim={2**num_qubits})")
    return BinaryFockVector(int_to_binary(idx, num_qubits, lsb_first=STATE_LSB_FIRST))


def fock_to_int(fock: BinaryFockVector) -> int:
    """
    Converts a BinaryFockVector to its integer representation.

    Args:
        fock (BinaryFockVector): The binary Fock vector to convert.

    Returns:
        int: The corresponding integer representation.
    """
    return binary_to_int(fock, lsb_first=STATE_LSB_FIRST)

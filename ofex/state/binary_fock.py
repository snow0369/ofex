from __future__ import annotations

from typing import Sequence, Union, Tuple

from ofex.operators.types import SinglePauli
from ofex.utils.binary import int_to_binary, binary_to_int

STATE_LSB_FIRST = False
STATE_PRINT_LSB_FIRST = True


class BinaryFockVector(tuple):
    @property
    def num_qubits(self) -> int:
        return self.__len__()

    @property
    def num_spin_orbitals(self) -> int:
        return self.__len__()

    @property
    def num_spatial_orbitals(self) -> int:
        assert self.__len__() % 2 == 0
        return self.__len__() // 2

    def __new__(cls,
                inp: Sequence[Union[int, bool]], ):
        inp = [int(bool(x)) for x in inp]
        return super(BinaryFockVector, cls).__new__(cls, tuple(inp))

    def to_int(self) -> int:
        return fock_to_int(self)

    def __getitem__(self, item) -> Union[BinaryFockVector, int]:
        if hasattr(item, "__getitem__"):
            return BinaryFockVector([super(BinaryFockVector, self).__getitem__(i) for i in item])
        x = (super(BinaryFockVector, self).__getitem__(item))
        if isinstance(x, tuple):
            x = BinaryFockVector(x)
        return x

    def marginal_check(self,
                       qubits: Sequence[int],
                       idx: Union[int, Sequence[Union[int, bool]]]) -> bool:
        if isinstance(idx, int):
            return BinaryFockVector(self[qubits]).to_int() == idx
        return tuple(self[qubits]) == tuple(idx)

    def apply_pauli(self, pauli: SinglePauli) -> Tuple[BinaryFockVector, complex]:
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

    def pretty_string(self, fermion: bool = False):
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
    if idx > 2 ** num_qubits:
        raise IndexError(f"index {idx} is out of bounds with qubits of {num_qubits} (n_dim={2**num_qubits})")
    return BinaryFockVector(int_to_binary(idx, num_qubits, lsb_first=STATE_LSB_FIRST))


def fock_to_int(fock: BinaryFockVector) -> int:
    return binary_to_int(fock, lsb_first=STATE_LSB_FIRST)

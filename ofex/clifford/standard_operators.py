from typing import Tuple, Optional, List

from galois import FieldArray

from ofex.clifford.clifford_tools import gf


def clifford_op_str(op, *args) -> str:
    """
    Converts a Clifford operation and its arguments into a string representation.
    
    Args:
        op (str): The operation to be represented. Supported operations are:
            - "H": Hadamard gate (applies to 1 qubit)
            - "S": Phase gate (applies to 1 qubit)
            - "CX" (or "CNOT"): Controlled-X gate (applies to 2 qubits)
            - "CZ": Controlled-Z gate (applies to 2 qubits)
            - "QSW": Qubit swap operation (applies to 2 qubits)
        *args: The integers representing the qubits the operation applies to.
    
    Returns:
        str: The string representation of the operation and its target qubits.
    
    Raises:
        ValueError: If the operation is not supported or the number of arguments does not match the required format.
    """
    if op in ["H", "S"]:
        if len(args) != 1:
            raise ValueError
        return f"{op}_{args[0]}"
    elif op in ["CX", "CZ", "QSW"]:
        if len(args) != 2:
            raise ValueError
        return f"{op}_{args[0]}_{args[1]}"
    else:
        raise ValueError


def hadamard(arr: FieldArray, ph: FieldArray, idx: int, cliff_hist: Optional[List[str]] = None) \
        -> Tuple[FieldArray, FieldArray]:
    """
    Applies the Hadamard gate to the specified qubit.

    Args:
        arr (FieldArray): The input Pauli tableau (a binary symplectic matrix) which is being transformed.
        ph (FieldArray): The phase vector.
        idx (int): The qubit index to apply the Hadamard gate.
        cliff_hist (Optional[List[str]]): History of Clifford operations (if provided).

    Returns:
        Tuple[FieldArray, FieldArray]: Updated operator matrix and phase vector.

    Raises:
        AssertionError: If the input matrix does not have the correct dimensions.
    """
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    ph[:] = ph[:] + arr[idx, :] * arr[idx + n_qubits, :]
    arr[[idx, n_qubits + idx], :] = arr[[n_qubits + idx, idx], :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("H", idx))
    return arr, ph


def s_gate(arr: FieldArray, ph: FieldArray, idx: int, cliff_hist: Optional[List[str]] = None) \
        -> Tuple[FieldArray, FieldArray]:
    """
    Applies the S (phase) gate to the specified qubit.

    Args:
        arr (FieldArray): The input Pauli tableau (a binary symplectic matrix) which is being transformed.
        ph (FieldArray): The phase vector.
        idx (int): The qubit index to apply the S gate.
        cliff_hist (Optional[List[str]]): History of Clifford operations (if provided).

    Returns:
        Tuple[FieldArray, FieldArray]: Updated operator matrix and phase vector.

    Raises:
        AssertionError: If the input matrix does not have the correct dimensions.
    """
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    ph[:] = ph[:] + arr[idx, :] * arr[idx + n_qubits, :]
    arr[n_qubits + idx, :] = arr[idx, :] + arr[n_qubits + idx, :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("S", idx))
    return arr, ph


def cx(arr: FieldArray, ph: FieldArray, idx_c: int, idx_t: int, cliff_hist: Optional[List[str]] = None) \
        -> Tuple[FieldArray, FieldArray]:
    """
    Applies the CX (CNOT) gate to a pair of qubits.

    Args:
        arr (FieldArray): The input Pauli tableau (a binary symplectic matrix) which is being transformed.
        ph (FieldArray): The phase vector.
        idx_c (int): The control qubit index.
        idx_t (int): The target qubit index.
        cliff_hist (Optional[List[str]]): History of Clifford operations (if provided).

    Returns:
        Tuple[FieldArray, FieldArray]: Updated operator matrix and phase vector.

    Raises:
        AssertionError: If the input matrix does not have the correct dimensions.
        ValueError: If the control and target qubit indices are the same.
    """
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    if idx_c == idx_t:
        raise ValueError
    # r = r + x_c z_t (x_t + z_c + 1)
    ph[:] = ph[:] + arr[idx_c, :] * arr[idx_t + n_qubits, :] * \
            (arr[idx_t, :] + arr[idx_c + n_qubits, :] + gf([1 for _ in range(ph.shape[0])]))
    arr[idx_t, :] = arr[idx_t, :] + arr[idx_c, :]
    arr[idx_c + n_qubits, :] = arr[idx_c + n_qubits, :] + arr[idx_t + n_qubits, :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("CX", idx_c, idx_t))
    return arr, ph


def cz(arr: FieldArray, ph: FieldArray, idx_1: int, idx_2: int, cliff_hist: Optional[List[str]] = None) \
        -> Tuple[FieldArray, FieldArray]:
    """
    Applies the CZ (controlled-Z) gate to a pair of qubits.

    Args:
        arr (FieldArray): The input Pauli tableau (a binary symplectic matrix) which is being transformed.
        ph (FieldArray): The phase vector.
        idx_1 (int): The index of the first qubit.
        idx_2 (int): The index of the second qubit.
        cliff_hist (Optional[List[str]]): History of Clifford operations (if provided).

    Returns:
        Tuple[FieldArray, FieldArray]: Updated operator matrix and phase vector.

    Raises:
        AssertionError: If the input matrix does not have the correct dimensions.
        ValueError: If the two qubit indices are the same.
    """
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    if idx_1 == idx_2:
        raise ValueError
    ph[:] = ph[:] + arr[idx_1, :] * arr[idx_2, :] * \
            (arr[idx_2 + n_qubits, :] + arr[idx_1 + n_qubits, :])
    arr[idx_2 + n_qubits, :] = arr[idx_2 + n_qubits, :] + arr[idx_1, :]
    arr[idx_1 + n_qubits, :] = arr[idx_1 + n_qubits, :] + arr[idx_2, :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("CZ", idx_1, idx_2))
    return arr, ph


def qsw(arr: FieldArray, ph: FieldArray, idx_1: int, idx_2: int, cliff_hist: Optional[List[str]] = None) \
        -> Tuple[FieldArray, FieldArray]:
    """
    Swaps the positions of two qubits.

    Args:
        arr (FieldArray): The input Pauli tableau (a binary symplectic matrix) which is being transformed.
        ph (FieldArray): The phase vector.
        idx_1 (int): The index of the first qubit.
        idx_2 (int): The index of the second qubit.
        cliff_hist (Optional[List[str]]): History of Clifford operations (if provided).

    Returns:
        Tuple[FieldArray, FieldArray]: Updated operator matrix and phase vector.

    Raises:
        AssertionError: If the input matrix does not have the correct dimensions.
    """
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    arr[[idx_1, idx_2], :] = arr[[idx_2, idx_1], :]
    arr[[idx_1 + n_qubits, idx_2 + n_qubits], :] = arr[[idx_2 + n_qubits, idx_1 + n_qubits], :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("QSW", idx_1, idx_2))
    return arr, ph

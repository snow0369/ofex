from typing import Tuple, Optional, List

from galois import FieldArray

from ofex.clifford.clifford_tools import gf


def clifford_op_str(op, *args) -> str:
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
    arr, ph = gf(arr), gf(ph)
    assert arr.shape[0] % 2 == 0
    n_qubits = arr.shape[0] // 2
    arr[[idx_1, idx_2], :] = arr[[idx_2, idx_1], :]
    arr[[idx_1 + n_qubits, idx_2 + n_qubits], :] = arr[[idx_2 + n_qubits, idx_1 + n_qubits], :]
    if cliff_hist is not None:
        cliff_hist.append(clifford_op_str("QSW", idx_1, idx_2))
    return arr, ph

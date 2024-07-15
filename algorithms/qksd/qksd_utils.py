from itertools import product
from typing import Optional

import numpy as np
import scipy
from openfermion.config import EQ_TOLERANCE
from scipy.linalg import eigh


def toeplitz_arr_to_mat(toeplitz_arr: np.ndarray) -> np.ndarray:
    if not np.allclose(toeplitz_arr[..., 0].imag, 0.0, atol=EQ_TOLERANCE):
        raise ValueError(toeplitz_arr)
    n = toeplitz_arr.shape[-1]
    mat = np.zeros((*toeplitz_arr.shape[:-1], n, n), dtype=toeplitz_arr.dtype)
    for i, j in product(range(n), repeat=2):
        val = toeplitz_arr[..., abs(i - j)]
        mat[..., i, j] = val if i < j else val.conj()
    return mat


def diag_s(smat: np.ndarray):
    s_diag, v_mat = eigh(smat)
    assert np.allclose(s_diag.imag, 0.0)
    s_diag = s_diag.real
    sort_idx = s_diag.argsort()[::-1]
    s_diag = s_diag[sort_idx]
    v_mat = v_mat[:, sort_idx]
    # assert np.allclose(v_mat.T.conj() @ v_mat, np.eye(v_mat.shape[0]), atol=1e-3), (v_mat, v_mat.T.conj() @ v_mat,
    # s_diag)
    assert np.allclose(smat, v_mat @ np.diag(s_diag) @ v_mat.T.conj())
    return s_diag, v_mat


def condition_number(a: np.ndarray, b: np.ndarray, atol=None, verbose=False):
    if atol is None:
        eigval, eigvec = eigh(a, b)
    else:
        eigval, eigvec, _, _, a, b = trunc_eigh_verbose(a, b, atol)
    assert a.shape == b.shape
    d = a.shape[0]
    c = a + 1.0j * b
    for j in range(d):
        norm = np.sqrt(np.vdot(eigvec[:, j], eigvec[:, j]))
        eigvec[:, j] /= norm
    conj_c = eigvec.T.conj() @ c @ eigvec
    assert np.allclose(np.diag(np.diag(conj_c)), conj_c)
    conj_c = np.diag(conj_c)
    eigval_conj = conj_c.real / conj_c.imag
    idx_sort = np.argsort(eigval_conj)
    # assert np.allclose(eigval, eigval_conj, atol=1e-4), np.linalg.norm(eigval-eigval_conj)
    cond = np.abs(conj_c)[idx_sort]
    if verbose:
        return eigval, cond ** -1
    else:
        return cond ** -1


def trunc_s(hmat, smat, epsilon: Optional[float] = None, n_lambda: Optional[int] = None):
    if (epsilon is None) == (n_lambda is None):
        raise ValueError
    s_diag, v_mat = diag_s(smat)
    if epsilon is not None:
        k = int(max(np.where((s_diag >= epsilon))[0]) + 1)
    else:
        k = min(n_lambda, int(max(np.where((s_diag >= 0.0))[0]) + 1))
    v_mat_tr = v_mat[:, :k]
    smat = v_mat_tr.T.conj() @ smat @ v_mat_tr
    hmat = v_mat_tr.T.conj() @ hmat @ v_mat_tr
    return hmat, smat, s_diag, v_mat_tr


# Truncated eig function for QKSD method in the Chebyshev basis.

def trunc_eig(hmat: np.ndarray,
              smat: np.ndarray,
              epsilon: Optional[float] = None,
              n_lambda: Optional[int] = None):
    if not (epsilon is None and n_lambda is None):
        amat, bmat, s_diag, v_mat = trunc_s(hmat, smat, epsilon, n_lambda)
        eigen_values, eigen_vectors = scipy.linalg.eig(amat, bmat)
    else:
        eigen_values, eigen_vectors = scipy.linalg.eig(hmat, smat)
    return eigen_values, eigen_vectors


def trunc_eigh_verbose(hmat: np.ndarray,
                       smat: np.ndarray,
                       epsilon: Optional[float] = None,
                       n_lambda: Optional[int] = None,
                       hermitian=True):
    amat, bmat, s_diag, v_mat = trunc_s(hmat, smat, epsilon, n_lambda)
    if hermitian:
        eigen_values, eigen_vectors = eigh(amat, bmat, eigvals_only=False)
    else:
        eigen_values, eigen_vectors = scipy.linalg.eig(amat, bmat)
    return eigen_values, eigen_vectors, s_diag, v_mat, amat, bmat


def trunc_eigh(hmat: np.ndarray,
               smat: np.ndarray,
               epsilon: Optional[float] = None,
               n_lambda: Optional[int] = None,
               hermitian=True):
    if not (epsilon is None and n_lambda is None):
        val, vec, _, _, _, _ = trunc_eigh_verbose(hmat, smat, epsilon, n_lambda, hermitian)
    else:
        val, vec = eigh(hmat, smat, eigvals_only=False)
    return val, vec

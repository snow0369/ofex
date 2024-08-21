from itertools import product
from typing import Optional

import numpy as np
import scipy
from openfermion.config import EQ_TOLERANCE
from scipy.linalg import eigh, fractional_matrix_power


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
    check = v_mat @ np.diag(s_diag) @ v_mat.T.conj()
    assert np.allclose(smat, check)
    return s_diag, v_mat


def condition_number(a: np.ndarray, b: np.ndarray, epsilon=None, n_lambda=None):
    _, d, _ = gevp_pert_analysis(a, b, epsilon, n_lambda)
    return d ** -1


def gevp_pert_analysis(a: np.ndarray, b: np.ndarray, epsilon=None, n_lambda=None,
                       check_lowest_eig_only=False):
    if epsilon is not None or n_lambda is not None:
        _, _, _, _, a, b = trunc_eigh_verbose(a, b, epsilon, n_lambda)

    # Check Proposition 2.4 in Mathias and Li
    n = a.shape[0]
    b_inv = np.linalg.inv(b)
    b_inv_half = fractional_matrix_power(b_inv, 0.5)
    _, u = eigh(b_inv_half @ a @ b_inv_half)
    check_u_unitary = u.T.conj() @ u
    assert np.allclose(check_u_unitary, np.eye(n))
    s = np.diag(np.sqrt(np.diagonal(u.T.conj() @ b_inv @ u)) ** -1)
    x = b_inv_half @ u @ s

    # Check unit-column of x:
    cond_b = np.linalg.cond(b)
    for j in range(n):
        norm = np.linalg.norm(x[:, j])
        assert np.isclose(norm, 1.0, atol=1e-12 * cond_b)

    # Obtain z_i = α_i + i β_i
    z = x.T.conj() @ (a + 1j * b) @ x
    check_z_diagonal = np.diag(np.diagonal(z))
    assert np.allclose(z, check_z_diagonal)
    z = np.diag(z)
    eigval = z.real / z.imag
    eig_order = np.argsort(eigval)
    eigval = eigval[eig_order]
    z = z[eig_order]
    theta = np.angle(z)

    # distance
    d = np.abs(z)

    # Check eigvals
    check_eigvals, _ = eigh(a, b)
    if check_lowest_eig_only:
        assert np.isclose(eigval[0], check_eigvals[0], atol=1e-12 / np.min(d))
    else:
        assert np.allclose(eigval, check_eigvals, atol=1e-12 / np.min(d))
    return z, d, theta


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
    eigen_vectors = v_mat @ eigen_vectors @ v_mat.T.conj()
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


def tikhonov_eigh(hmat: np.ndarray,
                  smat: np.ndarray,
                  epsilon: float = 0.0,
                  n_lambda: Optional[int] = None,
                  hermitian=True):
    n = smat.shape[0]
    hmat_new = (smat.T.conj()) @ hmat
    smat_new = smat.T.conj() @ smat + np.eye(n) * (epsilon ** 2)
    if n_lambda is not None:
        return trunc_eigh(hmat_new, smat_new, n_lambda=n_lambda, hermitian=hermitian)
    elif hermitian:
        return eigh(hmat_new, smat_new, eigvals_only=False)
    else:
        return scipy.linalg.eig(hmat_new, smat_new)


if __name__ == "__main__":
    def random_hermitian(n, positive):
        h = np.random.normal(size=(n, n)) + 1j * np.random.normal(size=(n, n))
        h = h + h.conj().T
        if positive:
            h = h @ h
            for i in range(n):
                h[i, i] = h[i, i].real
            return h
        else:
            return h


    np.random.seed(10)
    a, b = random_hermitian(10, False), random_hermitian(10, True)
    gevp_pert_analysis(a, b)

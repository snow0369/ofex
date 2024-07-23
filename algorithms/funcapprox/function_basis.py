from functools import partial
from itertools import product
from numbers import Number
from typing import List, Callable, Optional, Tuple, Union, Sequence

import numpy as np
import sympy as sp
from scipy.integrate import quad


def _uniform_inner_product(x_max, numerical_integ, a: sp.Expr, b: sp.Expr) -> complex:
    prod = sp.simplify((a.conjugate() * b))
    syms = prod.free_symbols
    if len(syms) == 0:
        return complex(prod)
    elif len(syms) > 1:
        raise ValueError(f"There are redundant symbols {prod}, {syms}")
    x = list(syms)[0]
    if not numerical_integ:
        try:
            return complex(sp.integrate(prod, (x, -x_max, x_max)).evalf() / (2 * x_max))
        except TypeError:
            pass
    integrand = sp.lambdify(x, prod, "numpy")
    integral, _ = quad(integrand, -x_max, x_max, complex_func=True)
    return complex(integral / (2 * x_max))


def uniform_inner_product(x_max: float,
                          numerical_integ: bool) -> Callable[[sp.Expr, sp.Expr], complex]:
    return partial(_uniform_inner_product, x_max, numerical_integ)


class FunctionBasis(object):
    def __init__(self,
                 input_basis: List[sp.Expr],
                 inner_product: Optional[Callable[[sp.Expr, sp.Expr], complex]] = None, ):
        self.basis = input_basis
        self.inner_product = inner_product
        self._overlap_matrix = None

    @property
    def n_basis(self) -> int:
        return len(self.basis)

    @property
    def overlap_matrix(self) -> np.ndarray:
        if self._overlap_matrix is None:
            self._overlap_matrix = np.zeros((self.n_basis, self.n_basis), dtype=complex)
            for (idx1, b1), (idx2, b2) in product(enumerate(self.basis), repeat=2):
                self._overlap_matrix[idx1, idx2] = self.inner_product(b1, b2)
        return self._overlap_matrix

    def projection(self, func: sp.Expr) -> np.ndarray:
        return np.array([self.inner_product(b, func) for b in self.basis], dtype=complex)

    def l2_minimization(self, func: sp.Expr) -> Tuple[sp.Expr, np.ndarray, float]:
        s = self.overlap_matrix
        mu = self.projection(func)
        c = np.linalg.solve(s, mu)
        opt_g: sp.Expr = sum([cf * b for cf, b in zip(c, self.basis)])
        func_norm_2 = self.inner_product(func, func)

        diff = (func_norm_2 - np.dot(mu.conj(), c))
        assert np.isclose(diff.imag, 0.0), diff
        if diff < 0 and np.isclose(diff, 0.0):
            diff = 0.0
        return opt_g, c, np.sqrt(diff.real)

    def l2_minimization_regularized_1(self,
                                      func: sp.Expr,
                                      reg_coeff: Union[Sequence[float], float],
                                      max_iter: int = 1000,
                                      conv_atol: float = 1e-8) \
            -> Tuple[sp.Expr, np.ndarray, float]:
        if isinstance(reg_coeff, Number):
            reg_coeff = [reg_coeff for _ in range(self.n_basis)]
        func_norm_2 = self.inner_product(func, func)
        reg_coeff = np.array(reg_coeff)
        _, c0, diff0 = self.l2_minimization(func)
        c0_phase = c0 / np.abs(c0)

        s = self.overlap_matrix
        s_tilde = np.real(np.diag(c0_phase).conj() @ s @ np.diag(c0_phase))

        mu = self.projection(func)
        mu_tilde_0 = np.real(np.diag(c0_phase).conj() @ mu)

        diff, c = diff0, c0
        for n_it in range(max_iter):
            mu_tilde = mu_tilde_0 - diff0 * reg_coeff
            c_abs = np.linalg.solve(s_tilde, mu_tilde)
            c = c_abs * c0_phase

            dot = np.dot(c.conj(), mu)
            diff = (func_norm_2 + c.conj().T @ s @ c - dot - dot.conjugate()) ** (1 / 2) + np.dot(c_abs, reg_coeff)
            if abs(diff - diff0) < conv_atol:
                break
        else:
            print("Reached max iter")

        assert np.isclose(diff.imag, 0.0), diff
        opt_g: sp.Expr = sum([cf * b for cf, b in zip(c, self.basis)])
        return opt_g, c, np.sqrt(diff.real)


def uniform_fourier_basis(x_max: float, n_harmonics: int, deriv_order: int = 0,
                          numerical_integ: bool = False,
                          sym_x: Optional[sp.Symbol] = None,
                          debug=False) \
        -> Tuple[FunctionBasis, sp.Symbol, np.ndarray]:
    inner_product = uniform_inner_product(x_max, numerical_integ)
    if sym_x is None:
        sym_x = sp.Symbol('x')
    delta_omega = np.pi / x_max
    omega_list = [delta_omega * k for k in range(-n_harmonics, n_harmonics + 1)]
    basis = [sp.exp(1j * omega * sym_x) for omega in omega_list]
    for d in range(deriv_order):
        basis += [(sym_x ** (d + 1)) * sp.exp(1j * omega * sym_x) for omega in omega_list]

    w_len = len(omega_list)
    overlap_matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    for (idx_w1, w1), (idx_w2, w2) in product(enumerate(omega_list), repeat=2):
        for d1, d2 in product(range(deriv_order + 1), repeat=2):
            idx_b1 = idx_w1 + d1 * w_len
            idx_b2 = idx_w2 + d2 * w_len
            if idx_b1 > idx_b2:
                continue
            overlap_matrix[idx_b1, idx_b2] = _monomial_fourier_integral(w2 - w1, d1 + d2, -x_max, x_max) / (2 * x_max)

    for idx_b1, idx_b2 in product(range(len(basis)), repeat=2):
        if idx_b1 > idx_b2:
            overlap_matrix[idx_b1, idx_b2] = overlap_matrix[idx_b2, idx_b1].conjugate()

    fb = FunctionBasis(basis, inner_product)
    if debug:
        assert np.allclose(fb.overlap_matrix, overlap_matrix)
    else:
        fb._overlap_matrix = overlap_matrix
    return fb, sym_x, np.array(omega_list)


def _monomial_fourier_integral(omega, n, a, b):
    # integrate x^n e^{iωx} from x=-a to b.
    if np.isclose(omega, 0):
        return (b ** (n + 1) - a ** (n + 1)) / (n + 1)
    elif n == 0:
        return (np.exp(1j * omega * b) - np.exp(1j * omega * a)) / (1j * omega)
    else:
        res = (np.exp(1j * omega * b) * b ** n - np.exp(1j * omega * a) * a ** n) / (1j * omega)
        return res - _monomial_fourier_integral(omega, n - 1, a, b) * (n / (1j * omega))


if __name__ == '__main__':
    from algorithms.funcapprox.func_plt import plot_functions


    def function_basis_test():
        f_basis_0, x, omega_list = uniform_fourier_basis(x_max=1.0, n_harmonics=4, deriv_order=0,
                                                         numerical_integ=True,
                                                         debug=False)
        f_basis_1, x, omega_list = uniform_fourier_basis(x_max=1.0, n_harmonics=4, deriv_order=1,
                                                         numerical_integ=True,
                                                         sym_x=x,
                                                         debug=False)
        gaussian = sp.exp(- x ** 2)
        opt_g_0, c_0, df_0 = f_basis_0.l2_minimization(gaussian)
        opt_g_1, c_1, df_1 = f_basis_1.l2_minimization(gaussian)
        plot_functions({'target': gaussian, 'approx_0': opt_g_0, 'qpprox_1': opt_g_1}, x)
        print(c_0)
        print(c_1)
        print(df_0, df_1)

        for samp_err in [0.1, 0.01, 0.001]:
            opt_g_0, c_0, df_0 = f_basis_0.l2_minimization_regularized_1(gaussian, samp_err)
            opt_g_1, c_1, df_1 = f_basis_1.l2_minimization_regularized_1(gaussian, samp_err)
            plot_functions({'target': gaussian, 'approx_0': opt_g_0, 'qpprox_1': opt_g_1}, x, title=f"{samp_err}")
            print(c_0)
            print(c_1)
            print(df_0, df_1)


    function_basis_test()

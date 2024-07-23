from functools import partial
from itertools import product
from numbers import Number
from typing import List, Callable, Optional, Tuple, Union, Sequence

import numpy as np
import sympy as sp


def _uniform_inner_product(x_max, a: sp.Expr, b: sp.Expr) -> complex:
    prod = sp.simplify((a.conjugate() * b))
    syms = prod.free_symbols
    if len(syms) == 0:
        return complex(prod)
    elif len(syms) > 1:
        raise ValueError(f"There are redundant symbols {prod}, {syms}")
    x = list(syms)[0]
    return complex(sp.integrate(prod, (x, -x_max, x_max)).evalf() / (2 * x_max))


def uniform_inner_product(x_max: float) -> Callable[[sp.Expr, sp.Expr], complex]:
    return partial(_uniform_inner_product, x_max)


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
        diff = self.inner_product(func, func) - c.T.conj() @ mu
        assert np.isclose(diff.imag, 0.0), diff
        opt_g: sp.Expr = sum([cf * b for cf, b in zip(c, self.basis)])
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
            assert np.allclose(c_abs / np.abs(c_abs), 1.0)
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


def uniform_fourier_basis(x_max: float, n_harmonics: int) -> Tuple[FunctionBasis, sp.Symbol, np.ndarray]:
    inner_product = uniform_inner_product(x_max)
    x = sp.Symbol('x')
    delta_omega = np.pi / x_max
    omega_list = [delta_omega * k for k in range(-n_harmonics, n_harmonics + 1)]
    basis = [sp.exp(1j * omega * x) for omega in omega_list]
    return FunctionBasis(basis, inner_product), x, np.array(omega_list)


if __name__ == '__main__':
    from algorithms.funcapprox.func_plt import plot_functions


    def function_basis_test():
        f_basis, x, omega_list = uniform_fourier_basis(x_max=1.0, n_harmonics=4)
        gaussian = sp.exp(-x ** 2)
        opt_g, c, df = f_basis.l2_minimization(gaussian)
        plot_functions({'target': gaussian, 'approx': opt_g}, x)


    function_basis_test()

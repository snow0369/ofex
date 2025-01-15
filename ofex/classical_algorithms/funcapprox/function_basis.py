from itertools import product
from numbers import Number
from typing import List, Callable, Optional, Tuple, Union, Sequence

import numpy as np
import sympy as sp
from sympy import chebyshevt

from ofex.classical_algorithms.funcapprox.integrals import uniform_inner_product, monomial_fourier_integral, \
    first_chebyshev_inner_product, first_chebyshev_integral

__all__ = ['FunctionBasis', 'FourierBasis', 'FirstChebyshevBasis']


class FunctionBasis(object):
    """
    Represents a basis of functions for projection and fitting in function space.

    This class facilitates calculations such as the overlap matrix, projection of 
    functions, and function fitting based on L2-norm minimization with or without
    regularization using a custom inner product.
    """

    def __init__(self,
                 input_basis: List[sp.Expr],
                 inner_product: Optional[Callable[[sp.Expr, sp.Expr], complex]] = None, ):
        """
        Initializes the FunctionBasis object.
    
        Args:
            input_basis (List[sp.Expr]): The list of symbolic expressions forming the basis functions.
            inner_product (Optional[Callable[[sp.Expr, sp.Expr], complex]]): 
                Function to compute the inner product, defaults to a uniform inner product.
        """
        self.basis = input_basis
        if inner_product is None:
            self.inner_product = uniform_inner_product(x_max=1.0, numerical_integ=True)
        else:
            self.inner_product = inner_product
        self._overlap_matrix = None

    @property
    def n_basis(self) -> int:
        """
        Returns the total number of basis functions.
    
        Returns:
            int: The number of basis functions.
        """
        return len(self.basis)

    @property
    def overlap_matrix(self) -> np.ndarray:
        """
        Computes and caches the overlap matrix for the basis functions.
    
        Returns:
            np.ndarray: A matrix where each element represents the inner product 
                        of two basis functions.
        """
        if self._overlap_matrix is None:
            self._overlap_matrix = np.zeros((self.n_basis, self.n_basis), dtype=complex)
            for (idx1, b1), (idx2, b2) in product(enumerate(self.basis), repeat=2):
                self._overlap_matrix[idx1, idx2] = self.inner_product(b1, b2)
        return self._overlap_matrix

    def projection(self, func: sp.Expr) -> np.ndarray:
        """
        Projects a function onto the basis.
    
        Args:
            func (sp.Expr): The function to be projected.
    
        Returns:
            np.ndarray: The projection coefficients as a complex array.
        """
        return np.array([self.inner_product(b, func) for b in self.basis], dtype=complex)

    def l2_minimization(self, func: sp.Expr) -> Tuple[sp.Expr, np.ndarray, float]:
        """
        Performs L2-minimization to find an optimal linear combination of basis functions.
    
        Args:
            func (sp.Expr): The target function for minimization.
    
        Returns:
            Tuple[sp.Expr, np.ndarray, float]: 
                - The optimal function approximation.
                - Coefficients for the linear combination of basis functions.
                - Residual error norm of the approximation.
        """
        s = self.overlap_matrix
        mu = self.projection(func)
        c = np.linalg.solve(s, mu)
        opt_g: sp.Expr = sp.Add(*(cf * b for cf, b in zip(c, self.basis)))
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
        """
        Performs L2-minimization with a regularization term.
    
        Args:
            func (sp.Expr): The target function for minimization.
            reg_coeff (Union[Sequence[float], float]): Regularization coefficients.
            max_iter (int): Maximum number of iterations (default=1000).
            conv_atol (float): Convergence absolute tolerance (default=1e-8).
    
        Returns:
            Tuple[sp.Expr, np.ndarray, float]: 
                - The optimal function approximation.
                - Coefficients for the linear combination of basis functions.
                - Regularized residual error norm of the approximation.
        """
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
        opt_g = sp.Add(*(cf * b for cf, b in zip(c, self.basis)))
        return opt_g, c, np.sqrt(diff.real)


class FourierBasis(FunctionBasis):
    def __init__(self, n_harmonics: int, x_max: float = 1.0, deriv_order: int = 0,
                 numerical_integ: bool = False, sym_x: Optional[sp.Symbol] = None,
                 debug: bool = False):
        """
        Initializes the FourierBasis class.

        Args:
            n_harmonics (int): Number of harmonics for the Fourier basis.
            x_max (float): Maximum absolute value of the interval.
            deriv_order (int): Order of polynomial derivatives in the basis.
            numerical_integ (bool): Whether to use numerical integration.
            sym_x (Optional[sp.Symbol]): Symbol for the variable (defaults to 'x').
            debug (bool): Whether to enable debug mode for consistency check.
        """
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
                overlap_matrix[idx_b1, idx_b2] = monomial_fourier_integral(w2 - w1, d1 + d2, -x_max, x_max) / (
                            2 * x_max)

        for idx_b1, idx_b2 in product(range(len(basis)), repeat=2):
            if idx_b1 > idx_b2:
                overlap_matrix[idx_b1, idx_b2] = overlap_matrix[idx_b2, idx_b1].conjugate()

        super().__init__(basis, inner_product)
        if debug:
            assert np.allclose(self.overlap_matrix, overlap_matrix)
        else:
            self._overlap_matrix = overlap_matrix


class FirstChebyshevBasis(FunctionBasis):
    def __init__(self, n_cheby: int,
                 numerical_integ: bool = False, sym_x: Optional[sp.Symbol] = None,
                 debug: bool = False, ):
        """
        Initializes the FirstChebyshevBasis class.

        Args:
            n_cheby (int): Number of Chebyshev polynomials in the basis.
            numerical_integ (bool): Whether to use numerical integration for inner products.
            sym_x (Optional[sp.Symbol]): Symbol for the variable (defaults to 'x').
            debug (bool): Whether to enable debug mode for consistency check.
        """
        inner_product = first_chebyshev_inner_product(numerical_integ)
        if sym_x is None:
            sym_x = sp.Symbol('x')
        basis = [chebyshevt(n, sym_x) for n in range(n_cheby)]
        overlap_matrix = np.zeros((n_cheby, n_cheby), dtype=complex)
        for n, m in product(range(n_cheby), repeat=2):
            overlap_matrix[n, m] = first_chebyshev_integral(n, m)

        super().__init__(basis, inner_product)
        if debug:
            assert np.allclose(self.overlap_matrix, overlap_matrix)
        else:
            self._overlap_matrix = overlap_matrix


if __name__ == '__main__':
    from ofex.classical_algorithms.funcapprox.func_plt import plot_functions


    def function_basis_test():
        x = sp.Symbol('x')
        f_basis_0 = FourierBasis(x_max=1.0, n_harmonics=4, deriv_order=0,
                                 numerical_integ=True, debug=False)
        f_basis_1 = FourierBasis(x_max=1.0, n_harmonics=4, deriv_order=1,
                                 numerical_integ=True, sym_x=f_basis_0.basis[0].free_symbols.pop(),
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

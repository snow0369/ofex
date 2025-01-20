from itertools import product
from numbers import Number
from typing import List, Callable, Optional, Tuple, Union, Sequence, Dict

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

    def iterate_basis_with_index(self, *args, **kwargs):
        """
        Iterates over group of functions in the basis along with its index.
        By defaults, it yields a basis with one function at a time.
        
        Yields:
            List[int]: Indices of basis function.
        """
        for idx, func in enumerate(self.basis):
            yield [idx]

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

    def gradual_l2_minimization(self, func: sp.Expr, n_min=1, n_max=None, **kwargs) \
            -> Dict[int, Tuple[sp.Expr, np.ndarray, float]]:
        """
        Performs L2-minimization by gradually increasing the basis size.
    
        Args:
            func (sp.Expr): The target function for minimization.
            n_min (int): The minimum number of basis functions to start with.
            n_max (Optional[int]): The maximum number of basis functions to consider. 
                                  If not specified, defaults to the total number of iterations
                                  over basis groups.
           **kwargs: Additional parameters passed to `iterate_basis_with_index`.

       Returns:
           Dict[int, Tuple[sp.Expr, np.ndarray, float]]:
               A dictionary where keys are the number of basis functions used, and values are tuples containing:
               - The approximate function.
               - Coefficients for the basis functions.
               - Residual error norm of the approximation.

       Raises:
           ValueError: If `n_min` is less than 1 or `n_max` is less than `n_min`.
        """
        if n_max is None:
            n_max = sum(1 for _ in self.iterate_basis_with_index())
        if n_min < 1:
            raise ValueError(f"n_min must be greater than 0, got {n_min}")
        if n_max < n_min:
            raise ValueError(f"n_max must be greater than or equal to n_min, got {n_max} and {n_min}")
        s = self.overlap_matrix
        mu = self.projection(func)
        func_norm_2 = self.inner_product(func, func)

        basis_idx_list = list()
        opt_list = dict()

        basis_iterator = self.iterate_basis_with_index(**kwargs)
        if n_min > 1:
            for n, add_basis_idxs in enumerate(basis_iterator):
                basis_idx_list += add_basis_idxs
                if n == n_min - 1:
                    break

        for n, add_basis_idxs in enumerate(basis_iterator):
            n += n_min
            basis_idx_list += add_basis_idxs
            sorted_idx_list = sorted(basis_idx_list)
            sorted_basis_list = [self.basis[idx] for idx in sorted_idx_list]
            s_n = s[np.ix_(sorted_idx_list, sorted_idx_list)]
            mu_n = mu[sorted_idx_list]
            c_n = np.linalg.solve(s_n, mu_n)

            opt_g_n: sp.Expr = sp.Add(*(cf * b for cf, b in zip(c_n, sorted_basis_list)))

            diff = (func_norm_2 - np.dot(mu_n.conj(), c_n))
            assert np.isclose(diff.imag, 0.0), diff
            if diff < 0 and np.isclose(diff, 0.0):
                diff = 0.0

            opt_list[n] = (opt_g_n, c_n, diff)

            if n == n_max:
                break

        return opt_list

    def l2_minimization_regularized(self,
                                    func: sp.Expr,
                                    reg_coeff: Union[Sequence[float], float],
                                    max_iter: int = 1000,
                                    conv_atol: float = 1e-6) \
            -> Tuple[sp.Expr, np.ndarray, float]:
        """
        Performs L2-minimization with a regularization term.
        
        This method finds an optimal linear combination of basis functions with regularization. 
        Different methods are utilized for first and second-order minimization.
        
        .. math::
            \min_{\vec{c}} \left[\|f(x)\|^2 + \vec{c}^{\dagger}\mathbf{S}\vec{c}
            - (\vec{c}^{\dagger}\vec{\mu} + \vec{\mu}^{\dagger}\vec{c}) 
            + \vec{c}^{\dagger} \mathbf{E} \vec{c} \right]^{1/2},
        
        which leads to the linear equation:
        
        .. math::
            (\mathbf{S} + \mathbf{E}) \vec{c} = \vec{\mu}

        where :math:`\mathbf{E}` is a diagonal matrix with regularization coefficients.
        
        Args:
            func (sp.Expr): The target function for minimization.
            reg_coeff (Union[Sequence[float], float]): Regularization coefficients. 
                Positive values only (either a single value or a sequence matching the basis size).
            max_iter (int, optional): Maximum number of iterations for the 1st order optimization 
                (default is 1000).
            conv_atol (float, optional): Convergence absolute tolerance for the iterative method 
                (default is 1e-6).
        
        Returns:
            Tuple[sp.Expr, np.ndarray, float]: A tuple containing:
                - The optimal function approximation as a symbolic expression.
                - Coefficients for the linear combination of basis functions.
                - The residual error norm of the approximation.
        
        Raises:
            ValueError: If any regularization coefficient is non-positive.
            NotImplementedError: If the specified regularization order is not 1 or 2.
        """
        if isinstance(reg_coeff, Number):
            reg_coeff = np.array([reg_coeff for _ in range(self.n_basis)])
        if not np.all(np.array(reg_coeff) > 0):
            raise ValueError("All regularization coefficients (reg_coeff) must be positive.")
        func_norm_2 = self.inner_product(func, func)
        s = self.overlap_matrix
        mu = self.projection(func)

        order = 2
        if order == 1:  # Deprecated
            c = np.linalg.solve(s, mu)
            diff = (func_norm_2 - np.dot(mu.conj(), c))
            assert np.isclose(diff.imag, 0.0), diff
            if diff < 0 and np.isclose(diff, 0.0):
                diff = 0.0
            diff = np.sqrt(diff)

            c_phase = c / np.abs(c)

            mu_tilde = mu - diff * c_phase * reg_coeff

            for n_it in range(max_iter):
                c = np.linalg.solve(s, mu_tilde)
                diff = (func_norm_2 + c.conj().T @ s @ c - 2 * np.dot(c.conj(), mu).real) ** (1 / 2)
                c_phase = c / np.abs(c)

                # Update new mu_tilde and check the difference between Sc and mu_tilde
                mu_tilde = mu - diff * c_phase * reg_coeff
                delta = (s @ c) - mu_tilde
                print(n_it, np.linalg.norm(delta))
                if np.linalg.norm(delta) < conv_atol:
                    break
            else:
                print("Reached max iter")

        elif order == 2:
            e_mat = np.diag(reg_coeff ** 2)
            c = np.linalg.solve(s + e_mat, mu)
            diff = (func_norm_2 - np.dot(c.conj(), s @ c))

            if diff < 0 and np.isclose(diff, 0.0):
                diff = 0.0
            assert np.isclose(diff.imag, 0.0) and diff.real >= 0.0, diff
            diff = np.sqrt(diff.real)

        else:
            raise NotImplementedError("Only order 1 and 2 are supported.")

        assert np.isclose(diff.imag, 0.0), diff
        opt_g = sp.Add(*(cf * b for cf, b in zip(c, self.basis)))
        return opt_g, c, diff.real


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

        self.omega_list, self.n_harmonics, self.deriv_order = np.array(omega_list), n_harmonics, deriv_order

        super().__init__(basis, inner_product)
        if debug:
            assert np.allclose(self.overlap_matrix, overlap_matrix), (self.overlap_matrix, overlap_matrix)
        else:
            self._overlap_matrix = overlap_matrix

    def iterate_basis_with_index(self, max_deriv_order: Optional[int] = None):
        """
        Iterates over groups of functions in the Fourier basis along with their indices.
        
        Args:
            max_deriv_order (Optional[int]): The maximum derivative order 
                to include in the iteration. Defaults to the derivative 
                order of the Fourier basis.
        
        Yields:
            List[Tuple[int, sp.Expr]]: For each harmonic, a list of tuples 
            containing the index and the corresponding basis function, including
            derivatives up to the specified order if applicable.
        """
        if max_deriv_order is None:
            max_deriv_order = self.deriv_order
        n_omega = 2 * self.n_harmonics + 1
        for idx_w in range(self.n_harmonics + 1):
            idx_basis_pairs = list()
            if idx_w == 0:
                for d in range(max_deriv_order + 1):
                    idx = self.n_harmonics + d * n_omega
                    idx_basis_pairs.append(idx)
            else:
                for d in range(max_deriv_order + 1):
                    idx_neg = self.n_harmonics - idx_w + d * n_omega
                    assert idx_neg >= 0
                    idx_pos = self.n_harmonics + idx_w + d * n_omega
                    idx_basis_pairs += [idx_neg, idx_pos]
            yield idx_basis_pairs


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
            assert np.allclose(self.overlap_matrix, overlap_matrix), (self.overlap_matrix, overlap_matrix)
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
            opt_g_0, c_0, df_0 = f_basis_0.l2_minimization_regularized(gaussian, samp_err)
            opt_g_1, c_1, df_1 = f_basis_1.l2_minimization_regularized(gaussian, samp_err)
            plot_functions({'target': gaussian, 'approx_0': opt_g_0, 'qpprox_1': opt_g_1}, x, title=f"{samp_err}")
            print(c_0)
            print(c_1)
            print(df_0, df_1)


    function_basis_test()

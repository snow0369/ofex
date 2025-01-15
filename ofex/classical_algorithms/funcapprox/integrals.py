from functools import partial
from typing import Callable

import numpy as np
import sympy as sp
from scipy.integrate import quad

__all__ = ["uniform_inner_product", "first_chebyshev_inner_product",
           "monomial_fourier_integral", "first_chebyshev_integral"]

def _uniform_inner_product(x_max: float, numerical_integ: bool, a: sp.Expr, b: sp.Expr) -> complex:
    """
    Computes the inner product of two symbolic expressions over a uniform grid.

    Args:
        x_max (float): Maximum absolute value of the integration bounds.
        numerical_integ (bool): Whether to use numerical integration.
        a (sp.Expr): The first symbolic expression.
        b (sp.Expr): The second symbolic expression.

    Returns:
        complex: The computed inner product as a complex number.
    """
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
    """
    Creates a function for computing the uniform inner product.

    Args:
        x_max (float): Maximum absolute value of the integration bounds.
        numerical_integ (bool): Whether to use numerical integration.

    Returns:
        Callable[[sp.Expr, sp.Expr], complex]: A function for computing the inner product of two expressions.
    """
    return partial(_uniform_inner_product, x_max, numerical_integ)


def _first_chebyshev_inner_product(numerical_integ: bool, a: sp.Expr, b: sp.Expr) -> complex:
    """
    Computes the inner product of two symbolic expressions using the first Chebyshev polynomial weight.

    Args:
        numerical_integ (bool): Whether to use numerical integration.
        a (sp.Expr): The first symbolic expression.
        b (sp.Expr): The second symbolic expression.

    Returns:
        complex: The computed inner product as a complex number.
    """
    x_max = 1.0
    syms = a.free_symbols.union(b.free_symbols)
    if len(syms) == 0:
        x = sp.Symbol('x')
    elif len(syms) == 1:
        x = list(syms)[0]
    else:
        raise ValueError(f"There are redundant symbols {a}, {b}")
    b_norm = b / sp.sqrt(1 - x ** 2)
    return _uniform_inner_product(x_max, numerical_integ, a, b_norm)


def first_chebyshev_inner_product(numerical_integ: bool) -> Callable[[sp.Expr, sp.Expr], complex]:
    """
    Creates a function for computing the Chebyshev inner product of two symbolic expressions.

    Args:
        numerical_integ (bool): Whether to use numerical integration.

    Returns:
        Callable[[sp.Expr, sp.Expr], complex]: A function for computing the Chebyshev inner product of two expressions.
    """
    return partial(_first_chebyshev_inner_product, numerical_integ)


def monomial_fourier_integral(omega, n, a, b):
    """
    Computes the integral of a monomial multiplied by a complex exponential.

    Args:
        omega (float): The frequency of the exponential term.
        n (int): The degree of the monomial.
        a (float): The lower bound of the integral.
        b (float): The upper bound of the integral.

    Returns:
        complex: The result of the integral as a complex number.
    """
    # integrate x^n e^{iωx} from x=-a to b.
    if np.isclose(omega, 0):
        return (b ** (n + 1) - a ** (n + 1)) / (n + 1)
    elif n == 0:
        return (np.exp(1j * omega * b) - np.exp(1j * omega * a)) / (1j * omega)
    else:
        res = (np.exp(1j * omega * b) * b ** n - np.exp(1j * omega * a) * a ** n) / (1j * omega)
        return res - monomial_fourier_integral(omega, n - 1, a, b) * (n / (1j * omega))


def first_chebyshev_integral(n, m):
    """
    Computes the integral of the product of two Chebyshev polynomials of the first kind 
    over the interval [-1, 1] with the weight function (1 - x^2)^(-1/2).

    Args:
        n (int): The degree of the first Chebyshev polynomial.
        m (int): The degree of the second Chebyshev polynomial.

    Returns:
        float: The result of the integral. Returns:
            - 0.0 if n != m (orthogonality of Chebyshev polynomials).
            - π if n == m == 0.
            - π / 2 if n == m > 0.
    """
    if n != m:
        return 0.0
    elif n == 0:
        return np.pi
    else:
        return np.pi / 2

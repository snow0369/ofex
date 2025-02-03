from typing import Optional, Tuple

import numpy as np
import sympy as sp
from numpy.polynomial import Chebyshev
from scipy.special import chebyt

__all__ = ["chebyshev_find_n", "chebyshev_fluctuation", "chebyshev_filter_fourier"]


def chebyshev_find_n(width, period, fluctuation, max_n=1000):
    """
    Determines the minimum degree of a Chebyshev filter such that the fluctuation
    of the resulting filter satisfies a specified threshold.
    
    Args:
        width (float): Width parameter, defining the sharpness of the filter response.
        period (float): Period of the Fourier series or filter base frequency.
        fluctuation (float): Maximum allowable fluctuation of the filter response.
        max_n (int, optional): The maximum degree of the Chebyshev filter to consider. Defaults to 1000.
    
    Returns:
        int: The degree `n` of the Chebyshev filter that satisfies the fluctuation condition.
    
    Raises:
        ValueError: If no valid degree `n` is found within the given maximum limit.
    """
    for n in range(max_n):
        fluc_n = chebyshev_fluctuation(n, width, period)
        if fluc_n < fluctuation:
            break
    else:
        raise ValueError("No valid Chebyshev degree n found that satisfies the fluctuation condition"
                         "within the maximum limit.")
    return n


def chebyshev_fluctuation(n, width, period):
    """
    Calculates the fluctuation of a Chebyshev polynomial-based filter.
    
    Args:
        n (int): Degree of the Chebyshev polynomial.
        width (float): Width parameter, defining the sharpness of the filter response.
        period (float): Period of the Fourier series or filter base frequency.
    
    Returns:
        float: The fluctuation metric of the filter.
    """
    omega = 2 * np.pi / period
    y = 1 + 2 * (1 - np.cos(omega * width)) / (1 + np.cos(omega * width))
    tny = Chebyshev.basis(n)(y)
    return tny ** (-1)


def chebyshev_filter_fourier(n_fourier: int,
                             width: float,
                             center: float = 0.0,
                             period: float = 2.0,
                             peak_height: float = 1.0,
                             sym_x: Optional[sp.Symbol] = None,
                             high_order_repr: bool = False, ) \
        -> Tuple[sp.Expr, Optional[np.ndarray], Optional[np.ndarray]]:
    r"""
    Constructs a Chebyshev bandpass filter using its Fourier series representation.
    It returns the Fourier expansion of:

    .. math::

        f_n(x) = \mathcal{N}^{-1}T_n\left( 1 + 2\frac{\cos 2\pi (x-c)/p - \cos 2\pi w/p}{1 + \cos 2\pi w/p} \right),

    where :math:`\mathcal{N}` is the normalizer that ensures the peak value of the filter is equal to `peak_height`.
    The resulting Fourier series is represented as:

    .. math::

        f_n(x) = \sum_{k=-n}^n c_k e^{2\pi ikx/p}.

    Args:
        n_fourier (int): Number of Fourier coefficients per side of the spectrum (total = 2 * n_fourier + 1)
            or Chebyshev degree n.
        width (float): Width parameter defining the filter's sharpness.
        center (float, optional): Center position of the filter. Defaults to 0.0.
        period (float, optional): Period of the Fourier series affecting frequency spacing. Defaults to 2.0.
        peak_height (float, optional): Peak height (gain) of the filter. Defaults to 1.0.
        sym_x (Optional[sp.Symbol], optional): Symbol for the symbolic Fourier series representation. Defaults to None.
        high_order_repr (bool, optional): Whether to use high-order representation for large `n_fourier`.
    
    Returns:
        Tuple[sp.Expr, np.ndarray, np.ndarray]: A tuple containing:
            - `sp_series` (sp.Expr): Symbolic Fourier series representation.
            - `coeff` (np.ndarray): Fourier series coefficients as a complex-valued numpy array.
            - `freqs` (np.ndarray): Frequencies corresponding to the Fourier coefficients.
    
    Raises:
        ValueError: If the Chebyshev filter is ill-conditioned (e.g., when cos(a) is approximately -1).
        NotImplementedError: If the Chebyshev filter with more than 39 coefficients is requested without high_order_repr.
    """

    #   Tn(1+2[cos(2π(x-c)/p) - cos(2πw/p)]/[1 + cos(2πw/p)]) / Normalizer
    # = Tn(1+2[cos z - cos a]/[1 + cos a]) / Normalizer
    if sym_x is None:
        sym_x = sp.Symbol("x")

    d_omega = 2 * np.pi / period
    freqs = np.linspace(-n_fourier * d_omega, n_fourier * d_omega, 2 * n_fourier + 1)

    a = d_omega * width  #  * np.pi
    b = np.cos(a)
    if np.isclose(b, -1.0):
        raise ValueError("Ill-conditioned Chebyshev filter")

    if high_order_repr and n_fourier > 30:
        fluct = chebyshev_fluctuation(n_fourier, width, period)
        z = 1 + 2 * (sp.cos(d_omega * (sym_x - center)) - sp.cos(a)) / (1 + sp.cos(a))
        func = fluct * sp.Piecewise((sp.cos(n_fourier * sp.acos(z)), sp.Abs(z) < 1),
                                    (((z - sp.sqrt(z ** 2 - 1)) ** n_fourier + (
                                            z + sp.sqrt(z ** 2 - 1)) ** n_fourier) / 2,
                                     sp.Abs(z) >= 1))
        # func = fluct * sp.chebyshevt(n_fourier,
        #                              1 + 2 * (sp.cos(d_omega * (sym_x-center)) - sp.cos(a))/(1 + sp.cos(a)))
        coeff, freqs = None, None
        return func, coeff, freqs
    elif n_fourier > 39:
        raise NotImplementedError("Chebyshev filter with more than 39 coefficients is Numerically instable"
                                  "and not implemented.")

    # y = 1 + 2[cos z - cos a] / [1 + cos a]
    #   = [2cos z + (1 - b)] / [1 + b]
    y = np.poly1d(np.array([2, 1 - b]) / (1 + b))
    y0 = y(1)  # y₀ = 1 + 2[1 - cos a] / [1 + cos a]

    # Tn(y)
    tn = chebyt(n_fourier)
    tny = np.polyval(tn, y)
    tny0 = tny(y0)
    tny = tny / tny0  # normalize

    mon_c = tny.c[::-1]  # tn(y) = Σₖ mon_c[k] cosᵏ(z)
    mon_by_cheby = _monomial_by_chebyshev(n_fourier + 1)  # cosᵏ(z) = Σₗ d[k, l] Tₗ(cos z) =  Σₗ d[k, l] cos(lz)
    cheby_c = mon_c @ mon_by_cheby

    # Cosine to complex exponential coefficients
    coeff = np.zeros(freqs.shape, dtype=complex)
    coeff[:n_fourier] = cheby_c[:0:-1] / 2
    coeff[n_fourier] = cheby_c[0]
    coeff[n_fourier + 1:] = cheby_c[1:] / 2

    # Shift by c
    coeff *= np.exp(-1j * freqs * center)

    basis = [sp.exp(1j * f * sym_x) for f in freqs]
    sp_series = sp.Add(*(c * b for c, b in zip(coeff, basis)))
    normalizer = complex(sp_series.subs({sym_x: center}))

    sp_series = peak_height * sp_series / normalizer
    coeff *= peak_height / normalizer

    return sp_series, coeff, freqs


def _monomial_by_chebyshev(n):
    """
    Computes the transformation matrix for converting monomials to Chebyshev polynomials.
    
    The matrix maps powers of `cos(x)` to sums of Chebyshev polynomials.
    
    Args:
        n (int): The maximum degree of the Chebyshev polynomial.
    
    Returns:
        np.ndarray: A 2D array where each row represents coefficients of Chebyshev polynomials
            corresponding to a specific power of `cos(x)`.
    """
    # https://en.wikipedia.org/wiki/Chebyshev_polynomials#Explicit_expressions
    # -> x^n = ...
    cxk_to_ckx = np.zeros((n, n), dtype=np.complex128)  # cos^k x = Σj c[k, j] Tj(cos x) = Σj c[k, j] cos jx
    for k in range(n):
        # naive_cheby = chebyt(k).c
        for j in range(k + 1):
            if j % 2 != k % 2:
                continue
            # Evaluate coefficient
            c = 2.0 if j != 0 else 1.0
            for l in range(1, (k - j) // 2 + 1):
                c *= (0.5 + (k + j) / (4 * l))
            c *= 2 ** (-(j + k) / 2)
            cxk_to_ckx[k, j] = c  # * naive_cheby[j]
    return cxk_to_ckx

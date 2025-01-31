from typing import Optional, Tuple

import numpy as np
import scipy
import sympy as sp
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.polynomial import Polynomial
from scipy.optimize import minimize_scalar, minimize, fsolve
from scipy.special import chebyt, erf, i0, lambertw

__all__ = ["gaussian_function_fourier", "chebyshev_filter_fourier",
           "gaussian_width_fit_to_chebyshev", "gaussian_find_n", "chebyshev_find_n", "chebyshev_fluctuation"]

from sympy import diff, solve, lambdify, Abs, Piecewise


def gaussian_width_fit_to_chebyshev(n_cheby: int, width, period, precise=False):
    omega = 2 * np.pi / period
    y = 1 + 2 * (1 - np.cos(omega * width)) / (1 + np.cos(omega * width))
    tny = Chebyshev.basis(n_cheby)(y)
    new_width = width / np.sqrt(2 * np.log(tny))
    if not precise:
        return new_width

    x = np.exp(1j * np.linspace(-n_cheby * omega * width, n_cheby * omega * width, 2 * n_cheby + 1))

    def _diff_at_side(w):
        if isinstance(w, np.ndarray):
            w = w.item()
        _, c, _ = gaussian_function_fourier(n_cheby, w, center=0.0, period=period)
        return abs(np.dot(c, x) - tny ** (-1))

    return fsolve(_diff_at_side, new_width)[0]


def gaussian_find_n(width, period, fluctuation, max_n=300):
    c = width * 2 * np.pi / (np.sqrt(2) * period)
    for n in range(1, max_n):
        erf_val = erf(c * n)
        err = abs((1 - erf_val) / (c / (np.sqrt(np.pi) + erf_val)))  # TOO High
        if err < fluctuation:
            break
    else:
        raise ValueError("No valid number of Fourier coefficients 'n' found that satisfies the fluctuation condition"
                         "within the given maximum limit.")
    return n


def gaussian_find_n_numerical(width, period, fluctuation, side_edge, max_n=300):
    for n in range(max(1, int(period / (width * 2 * np.pi))), max_n):
        fluc_n = gaussian_fluctuation(n, width, period, side_edge)
        if fluc_n < fluctuation:
            break
    else:
        raise ValueError("No valid number of Fourier coefficients 'n' found that satisfies the fluctuation condition"
                         "within the given maximum limit.")
    return n, fluc_n


def chebyshev_find_n(width, period, fluctuation, max_n=1000):
    for n in range(max_n):
        fluc_n = chebyshev_fluctuation(n, width, period)
        if fluc_n < fluctuation:
            break
    else:
        raise ValueError("No valid Chebyshev polynomial degree n found that satisfies the fluctuation condition"
                         "within the maximum limit.")
    return n


def chebyshev_fluctuation(n, width, period):
    omega = 2 * np.pi / period
    y = 1 + 2 * (1 - np.cos(omega * width)) / (1 + np.cos(omega * width))
    tny = Chebyshev.basis(n)(y)
    return tny ** (-1)


def gaussian_fluctuation(n, width, period, side_edge):
    x = sp.Symbol("x")
    g, _, _ = gaussian_function_fourier(n, width, center=0.0, period=period, peak_height=1.0, sym_x=x)
    abs_g_numeric = lambdify(x, Abs(g), 'numpy')
    result = minimize_scalar(lambda x: -abs_g_numeric(x), bounds=(side_edge, side_edge + 2 * np.pi / (period * n)),
                             method="bounded")
    if result.success:
        return -result.fun
    else:
        raise ValueError("Optimization did not converged")


def gaussian_function_fourier(n_fourier: int,
                              width: float,
                              center: float = 0.0,
                              period: float = 2.0,
                              peak_height: float = 1.0,
                              sym_x: Optional[sp.Symbol] = None, ) \
        -> Tuple[sp.Expr, np.ndarray, np.ndarray]:
    r"""
    Computes the Fourier series representation of a Gaussian function.
    It returns the Fourier approximation of:

    .. math::

        f(x) = A\exp\left( -\frac{(x-c)^2}{2w^2} \right).

    The resulting Fourier series is represented as:

    .. math::

        \tilde{f}(x) = \sum_{k=-n}^n c_k e^{2\pi ikx/p}.

    Setting width_fit_to_chebyshev modifies the width so that the function coincides with the first sidelobe of the
    chebysev filter.

    .. math::

        \tilde{w} = w \left[ 2 / \log T_n\left( 1 + 2\frac{1-\cos 2\pi w/p}{1 + \cos 2\pi w/p} \right) \right]

    Parameters:
        n_fourier (int): The number of Fourier coefficients on each side of the frequency spectrum
            (total coefficients = 2 * n_fourier + 1).
        width (float): The width parameter for the Gaussian function.
        center (float, optional): The center position of the Gaussian. Defaults to 0.0.
        period (float, optional): The period of the Fourier series (affects the frequency spacing). Defaults to 2.0.
        peak_height (float, optional): The peak height (amplitude) of the Gaussian function. Defaults to 1.0.
        sym_x (Optional[sp.Symbol], optional): A symbolic variable for constructing the symbolic series representation
            using sympy. Defaults to None.

    Returns:
        Tuple[sp.Expr, np.ndarray, np.ndarray]: A tuple containing:
            - `sp_series` (sp.Expr): The symbolic representation of the Fourier series.
            - `coeff` (np.ndarray): The Fourier series coefficients as a complex-valued numpy array.
            - `freqs` (np.ndarray): The corresponding frequencies for the Fourier series coefficients.
    """

    if sym_x is None:
        sym_x = sp.Symbol("x")

    d_omega = 2 * np.pi / period
    freqs = np.linspace(-n_fourier * d_omega, n_fourier * d_omega, 2 * n_fourier + 1)

    coeff = np.zeros(freqs.shape, dtype=complex)
    for k in range(n_fourier + 1):
        erf_factor = (width / period) * (
                erf((period ** 2 - 4j * np.pi * k * width ** 2) / (np.sqrt(8) * period * width)) +
                erf((period ** 2 + 4j * np.pi * k * width ** 2) / (np.sqrt(8) * period * width)))
        coeff[n_fourier + k] = np.sqrt(np.pi / 2) * np.exp(-(d_omega * k * width) ** 2 / 2) * erf_factor
    coeff[:n_fourier] = coeff[n_fourier + 1:][::-1]
    coeff *= np.exp(-1j * freqs * center)
    basis = [sp.exp(1j * f * sym_x) for f in freqs]
    sp_series = sp.Add(*(c * b for c, b in zip(coeff, basis)))

    normalizer = complex(sp_series.subs({sym_x: center}))
    sp_series = peak_height * sp_series / normalizer
    coeff *= peak_height / normalizer

    return sp_series, coeff, freqs


def chebyshev_filter_fourier(n_fourier: int,
                             width: float,
                             center: float = 0.0,
                             period: float = 2.0,
                             peak_height: float = 1.0,
                             sym_x: Optional[sp.Symbol] = None,
                             high_order_approx: bool = False, ) \
        -> Tuple[sp.Expr, np.ndarray, np.ndarray]:
    r"""
    Constructs a Chebyshev bandpass filter using its Fourier series representation.
    It returns the Fourier expansion of:
    
    .. math::

        f_n(x) = \mathcal{N}^{-1}T_n\left( 1 + 2\frac{\cos 2\pi (x-c)/p - \cos 2\pi w/p}{1 + \cos 2\pi w/p} \right),

    where :math:`\mathcal{N}` is the normalizer that ensures the peak value of the filter is equal to `peak_height`.
    The resulting Fourier series is represented as:
    
    .. math::

        f_n(x) = \sum_{k=-n}^n c_k e^{2\pi ikx/p}.

    Parameters:
        n_fourier (int): Number of Fourier coefficients on each side of the frequency spectrum 
            (total coefficients = 2 * n_fourier + 1).
        width (float): Width parameter, defining the filter's sharpness.
        center (float, optional): Center position of the filter. Defaults to 0.0.
        period (float, optional): Period of the Fourier series (affects the frequency spacing). Defaults to 2.0.
        peak_height (float, optional): The peak height (gain) of the filter. Defaults to 1.0.
        sym_x (Optional[sp.Symbol], optional): A symbolic variable for constructing the symbolic series 
            representation using sympy. Defaults to None.

    Returns:
        Tuple[sp.Expr, np.ndarray, np.ndarray]: A tuple containing:
            - `sp_series` (sp.Expr): The symbolic representation of the Fourier series.
            - `coeff` (np.ndarray): The Fourier series coefficients as a complex-valued numpy array.
            - `freqs` (np.ndarray): The corresponding frequencies for the Fourier series coefficients.
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

    # if n_fourier > 39:
    #     raise NotImplementedError("Chebyshev filter with more than 39 coefficients is Numerically instable"
    #                               "and not implemented.")

    if high_order_approx and n_fourier > 30:
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


def kaiser_filter_fourier(n_fourier: int,
                          alpha: float,
                          center: float = 0.0,
                          period: float = 2.0,
                          peak_height: float = 1.0,
                          sym_x: Optional[sp.Symbol] = None, ):
    if sym_x is None:
        sym_x = sp.Symbol("x")

    d_omega = 2 * np.pi / period
    freqs = np.linspace(-n_fourier * d_omega, n_fourier * d_omega, 2 * n_fourier + 1, dtype=float)

    max_freq = float(n_fourier * d_omega)
    coeffs = i0(alpha * np.sqrt(1 - (freqs / max_freq) ** 2)) / i0(alpha)
    coeffs = coeffs.astype(complex)
    coeffs *= np.exp(-1j * freqs * center)
    basis = [sp.exp(1j * f * sym_x) for f in freqs]
    sp_series = sp.Add(*(c * b for c, b in zip(coeffs, basis)))

    normalizer = complex(sp_series.subs({sym_x: center}))
    sp_series = peak_height * sp_series / normalizer
    coeffs *= peak_height / normalizer

    return sp_series, coeffs, freqs


def kaiser_fluctuation(alpha):
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    cos_theta = 0.217234
    if np.isclose(alpha, 0.0):
        return cos_theta
    return alpha * cos_theta / np.sinh(alpha)


def kaiser_find_alpha_from_fluct(target_fluctuation):
    # Solve sinh(x)/x = k, where k = cos_theta/epsilon by newton method
    max_iter, alpha_atol, fluct_atol = 1000, 1e-6, 1e-6

    cos_theta = 0.217234
    k = cos_theta / target_fluctuation
    if k < 1.0:
        raise ValueError("Target fluctuation is too large")
    elif k > 1.5:
        x = -lambertw(- 1 / (2 * k))
    else:
        x = np.sqrt(6 * (k - 1))

    x = float(x)

    def _diff_kaiser_fluctuation(_x):
        return kaiser_fluctuation(_x) - target_fluctuation

    x = fsolve(_diff_kaiser_fluctuation, x)
    return float(x)


def kaiser_find_alpha_from_width(n_fourier, target_width, period):
    theta2 = 2.5535658
    return np.sqrt((2 * np.pi * n_fourier * target_width / period) ** 2 - theta2 ** 2)


def kaiser_width(n_fourier, alpha, period):
    # f2 in
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    theta2 = 2.5535658
    return period * np.sqrt(theta2 ** 2 + alpha ** 2) / (2 * np.pi * n_fourier)


def kaiser_find_n(alpha, target_width, period):
    theta2 = 2.5535658
    return int(period * np.sqrt(theta2 ** 2 + alpha ** 2) / (2 * np.pi * target_width))


def _chebyt_product(n):
    """
    Constructs the Chebyshev polynomial T_n(x) as a product of (x - x_k) where x_k are the roots.

    Parameters:
        n (int): Degree of the Chebyshev polynomial.

    Returns:
        np.ndarray: Coefficients of T_n(x) with the leading coefficient normalized to 1.
    """
    if n == 0:
        return np.array([1])  # T_0(x) = 1
    if n == 1:
        return np.array([1, 0])  # T_1(x) = x

    # Compute the roots
    roots = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))

    # Start with the constant polynomial 1
    poly = Polynomial([1.0])

    # Multiply (x - root) for each root
    for root in roots:
        poly = poly * Polynomial([-root, 1.0])  # (x - root)

    # Normalize the polynomial to ensure the leading coefficient is 1
    coefficients = poly.coef
    coefficients /= coefficients[-1]  # Normalize

    return coefficients


def _monomial_by_chebyshev(n):
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


def _monomial_by_chebyshev_high_precision(n):
    cxk_to_ckx = np.zeros((n, n), dtype=np.complex128)
    for k in range(n):
        for j in range(k + 1):
            if j % 2 != k % 2:
                continue
            log_c = 0.0
            if j != 0:
                log_c += np.log(2.0)
            for l in range(1, (k - j) // 2 + 1):
                log_c += np.log(0.5 + (k + j) / (4 * l))
            log_c -= ((j + k) / 2) * np.log(2)
            cxk_to_ckx[k, j] = np.exp(log_c)
    return cxk_to_ckx

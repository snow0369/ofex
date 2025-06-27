from typing import Optional, Tuple

import numpy as np
import sympy as sp
from numpy.polynomial import Chebyshev
from scipy.optimize import fsolve, minimize_scalar
from scipy.special import erf
from sympy import lambdify, Abs

__all__ = ["gaussian_width_fit_to_chebyshev", "gaussian_fluctuation", "gaussian_function_fourier"]


def gaussian_width_fit_to_chebyshev(n_cheby: int, width, period, precise=False):
    """
    Fits the width of a Gaussian function to Chebyshev filter.

    This function adjusts the width of the Gaussian blur so that its mainlobe
    matches the Chebyshev filter. The precise option refines
    the width to minimize error with the Chebyshev filter.

    Args:
        n_cheby (int): Degree of the Chebyshev polynomial.
        width (float): Initial width parameter of the Gaussian function.
        period (float): Period of the Gaussian function.
        precise (bool, optional): If True, performs precise fitting; otherwise,
            returns a quick approximate fit. Defaults to False.

    Returns:
        float: The adjusted width parameter ensuring proper fit with the
        Chebyshev polynomial.
    """
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


def gaussian_fluctuation(n, width, period, side_edge):
    """
    Computes the fluctuation amplitude of a Gaussian function.

    This function determines the maximum fluctuation of the Gaussian's
    absolute value when side constraints are applied.

    Args:
        n (int): Number of fluctuations to consider during evaluation.
        width (float): The width parameter of the Gaussian function.
        period (float): The periodicity of the Gaussian function.
        side_edge (float): The edge location where fluctuations are evaluated.

    Returns:
        float: The peak fluctuation amplitude of the Gaussian function.

    Raises:
        ValueError: If the optimization process fails to converge.
    """
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
                              sym_x: Optional[sp.Symbol] = None,
                              cutoff_coeff_atol: Optional[float] = None,) \
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
        n_fourier (int): Number of Fourier coefficients per side of the spectrum (total = 2 * n_fourier + 1).
        width (float): Width parameter defining the filter's sharpness.
        center (float, optional): Center position of the filter. Defaults to 0.0.
        period (float, optional): Period of the Fourier series affecting frequency spacing. Defaults to 2.0.
        peak_height (float, optional): Peak height (gain) of the filter. Defaults to 1.0.
        sym_x (Optional[sp.Symbol], optional): Symbol for the symbolic Fourier series representation. Defaults to None.
        cutoff_coeff_atol (Optional[float], optional): Cutoff coefficient tolerance. Defaults to None.

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

    if cutoff_coeff_atol is not None:
        keep_mask = np.abs(coeff) >= cutoff_coeff_atol
        coeff = coeff * keep_mask
        coeff_sp = coeff[keep_mask]
        freqs_sp = freqs[keep_mask]
        basis = [sp.exp(1j * f * sym_x) for f in freqs_sp]
        sp_series = sp.Add(*(c * b for c, b in zip(coeff_sp, basis)))
    else:
        basis = [sp.exp(1j * f * sym_x) for f in freqs]
        sp_series = sp.Add(*(c * b for c, b in zip(coeff, basis)))

    normalizer = complex(sp_series.subs({sym_x: center}))
    sp_series = peak_height * sp_series / normalizer
    coeff *= peak_height / normalizer

    return sp_series, coeff, freqs

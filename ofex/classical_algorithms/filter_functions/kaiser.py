from typing import Optional

import numpy as np
import sympy as sp
from scipy.optimize import fsolve
from scipy.special import lambertw, i0

__all__ = ["kaiser_find_n", "kaiser_find_alpha_from_fluct", "kaiser_find_alpha_from_width",
           "kaiser_width", "kaiser_fluctuation", "kaiser_filter_fourier"]


def kaiser_find_alpha_from_fluct(target_fluctuation):
    """Find the alpha parameter that matches the target fluctuation.

    Solves sinh(x)/x = k using the Newton-Raphson method, where k = cos_theta/epsilon.

    Args:
        target_fluctuation (float): The desired fluctuation level.

    Returns:
        float: The alpha parameter value.

    Raises:
        ValueError: If the target fluctuation is too large.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
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
    """Calculate the alpha parameter given the target width.

    Args:
        n_fourier (int): The number of Fourier basis functions.
        target_width (float): The desired width of the Kaiser window.
        period (float): The period of the signal.

    Returns:
        float: The alpha parameter value.

    Raises:
        ValueError: If the target width or Fourier basis is too large.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
    theta2 = 2.5535658
    alpha2 = (2 * np.pi * n_fourier * target_width / period) ** 2 - theta2 ** 2
    if alpha2 < 0:
        raise ValueError("Target width or fourier basis is too large")
    return np.sqrt(alpha2)


def kaiser_find_n(alpha, target_width, period):
    """Find the number of Fourier basis functions for the Kaiser filter.

    Args:
        alpha (float): The alpha parameter of the Kaiser window.
        target_width (float): The desired width of the Kaiser window.
        period (float): The period of the signal.

    Returns:
        int: The number of Fourier basis functions.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
    theta2 = 2.5535658
    return int(period * np.sqrt(theta2 ** 2 + alpha ** 2) / (2 * np.pi * target_width))


def kaiser_width(n_fourier, alpha, period):
    """Calculate the width of a Kaiser window.

    Args:
        n_fourier (int): The number of Fourier basis functions.
        alpha (float): The alpha parameter of the Kaiser window.
        period (float): The period of the signal.

    Returns:
        float: The width of the Kaiser window.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
    theta2 = 2.5535658
    return period * np.sqrt(theta2 ** 2 + alpha ** 2) / (2 * np.pi * n_fourier)


def kaiser_fluctuation(alpha):
    """Calculate the fluctuation level of a Kaiser window.

    Args:
        alpha (float): The alpha parameter of the Kaiser window.

    Returns:
        float: The fluctuation level.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
    cos_theta = 0.217234
    if np.isclose(alpha, 0.0):
        return cos_theta
    return alpha * cos_theta / np.sinh(alpha)


def kaiser_filter_fourier(n_fourier: int,
                          alpha: float,
                          center: float = 0.0,
                          period: float = 2.0,
                          peak_height: float = 1.0,
                          sym_x: Optional[sp.Symbol] = None, ):
    """Generate a Fourier-based Kaiser filter.

    Args:
        n_fourier (int): Number of Fourier coefficients per side of the spectrum (total = 2 * n_fourier + 1).
        width (float): Width parameter defining the filter's sharpness.
        center (float, optional): Center position of the filter. Defaults to 0.0.
        period (float, optional): Period of the Fourier series affecting frequency spacing. Defaults to 2.0.
        peak_height (float, optional): Peak height (gain) of the filter. Defaults to 1.0.
        sym_x (Optional[sp.Symbol], optional): Symbol for the symbolic Fourier series representation. Defaults to None.

    Returns:
        Tuple[sympy.Basic, numpy.ndarray, numpy.ndarray]: The symbolic Fourier series, coefficients, and frequencies.

    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1163349
    """
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

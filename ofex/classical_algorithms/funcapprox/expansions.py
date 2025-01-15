from typing import Optional, Tuple

import numpy as np
import sympy as sp
from scipy.special import chebyt


def gaussian_function_fourier(n_fourier: int,
                              width: float,
                              center: float = 0.0,
                              period: float = 2.0,
                              peak_height: float = 1.0,
                              sym_x: Optional[sp.Symbol] = None,)\
        -> Tuple[sp.Expr, np.ndarray, np.ndarray]:
    if sym_x is None:
        sym_x = sp.Symbol("x")

    d_omega = 2 * np.pi / period
    freqs = np.linspace(-n_fourier * d_omega, n_fourier * d_omega, 2 * n_fourier + 1)
    coeff = np.exp(-0.5 * (width * freqs) ** 2) * np.exp(-1j * freqs * center)
    basis = [sp.exp(1j * f * sym_x) for f in freqs]
    sp_series = sp.Add(*(c * b for c, b in zip(coeff, basis)))
    normalizer = sp_series.subs({sym_x: center})

    sp_series = peak_height * sp_series / normalizer
    coeff *= peak_height / normalizer

    return sp_series, coeff, freqs


def chebyshev_filter_fourier(n_fourier: int,
                             width: float,
                             center: float = 0.0,
                             period: float = 2.0,
                             peak_height: float = 1.0,
                             sym_x: Optional[sp.Symbol] = None,)\
        -> Tuple[sp.Expr, np.ndarray, np.ndarray]:
    #   Tn(1+2[cos(2π(x-c)/p) - cos(2πw/p)]/[1 + cos(2πw/p)]) / Normalizer
    # = Tn(1+2[cos z - cos a]/[1 + cos a]) / Normalizer
    if sym_x is None:
        sym_x = sp.Symbol("x")

    d_omega = 2 * np.pi / period
    freqs = np.linspace(-n_fourier * d_omega, n_fourier * d_omega, 2 * n_fourier + 1)

    a = d_omega * width * np.pi
    b = np.cos(a)
    if np.isclose(b, -1.0):
        raise ValueError("Ill-conditioned Chebyshev filter")

    # y = 1 + 2[cos z - cos a] / [1 + cos a]
    #   = [2cos z + (1 - b)] / [1 + b]
    y = np.poly1d(np.array([2, 1 - b])/(1 + b))
    y0 = y(1)  # y₀ = 1 + 2[1 - cos a] / [1 + cos a]

    # Tn(y)
    tn = chebyt(n_fourier)
    tn = np.polyval(tn, y)
    tn = peak_height * tn / tn(y0)  # normalize

    mon_c = tn.c[::-1]  # tn(y) = Σₖ mon_c[k] cosᵏ(z)
    mon_by_cheby = _monomial_by_chebyshev(n_fourier + 1)  # cosᵏ(z) = Σₗ d[k, l] Tₗ(cos z) =  Σₗ d[k, l] cos(lz)
    cheby_c = mon_c @ mon_by_cheby

    # Cosine to complex exponential coefficients
    coeff = np.zeros(freqs.shape, dtype=complex)
    coeff[:n_fourier] = cheby_c[:0:-1]/2
    coeff[n_fourier] = cheby_c[0]
    coeff[n_fourier + 1:] = cheby_c[1:]/2

    # Shift by c
    coeff *= np.exp(-1j * freqs * center)

    basis = [sp.exp(1j * f * sym_x) for f in freqs]
    sp_series = sp.Add(*(c * b for c, b in zip(coeff, basis)))

    return sp_series, coeff, freqs


def _monomial_by_chebyshev(n):
    # https://en.wikipedia.org/wiki/Chebyshev_polynomials#Explicit_expressions
    cxk_to_ckx = np.zeros((n, n), dtype=complex)  # cos x^k = Σj c[k, j] Tj(cos x) = Σj c[k, j] cos jx
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

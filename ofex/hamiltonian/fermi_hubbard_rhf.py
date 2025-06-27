from typing import Tuple

import numpy as np

from openfermion import up_index, down_index, FermionOperator
from openfermion.hamiltonians.hubbard import _right_neighbor, _bottom_neighbor


# ---------------------------------------------------------------------------
def build_spatial_hubbard_integrals(
        x_dimension: int,
        y_dimension: int = 1,
        tunneling: float = 1.0,
        coulomb: float = 1.0,
        chemical_potential: float = 0.0,
        periodic: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (h1, g2) spatial-orbital integrals for the Hubbard model.

    * h1[i,j]  = -t  if i,j nearest neighbours (hopping)
               += -mu if i == j          (chemical potential)
    * g2[p,q,r,s] = U  δ_{pq} δ_{pr} δ_{ps}   (on-site repulsion)
    """
    n_sites = x_dimension * y_dimension
    h1 = np.zeros((n_sites, n_sites), dtype=float)

    # chemical potential on-site term
    np.fill_diagonal(h1, -chemical_potential)

    # nearest-neighbour hopping (−t)
    for site in range(n_sites):
        right = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom = _bottom_neighbor(site, x_dimension, y_dimension, periodic)
        if right is not None:
            h1[site, right] = h1[right, site] = -tunneling
        if bottom is not None:
            h1[site, bottom] = h1[bottom, site] = -tunneling

    # two-electron integrals (chemist ordering) – only on-site U
    g2 = np.zeros((n_sites, n_sites, n_sites, n_sites), dtype=float)
    for i in range(n_sites):
        g2[i, i, i, i] = coulomb      #  (ii|ii) = U

    return h1, g2


# ---------------------------------------------------------------------------
def restricted_hf(h1, g2, n_elec,
                  conv_E=1e-10, conv_D=1e-8,
                  max_iter=64, mix=0.7):
    """
    Restricted Hartree–Fock for real, orthonormal AO/Hubbard site basis.

    Parameters
    ----------
    h1 : (n,n) ndarray
        One-electron matrix  h_{pq}.
    g2 : (n,n,n,n) ndarray
        Two-electron tensor  <pq|rs>  in chemists’ notation.
    n_elec : int
        Total number of electrons (must be even for RHF).
    conv_E : float
        Energy convergence threshold.
    conv_D : float
        Density-matrix convergence threshold (Frobenius norm).
    max_iter : int
        Maximum SCF iterations.
    mix : float in (0,1]
        Linear mixing for the density matrix; mix=1 disables mixing.

    Returns
    -------
    C : (n,n) ndarray
        Canonical HF orbital coefficients (columns).
    eps : (n,) ndarray
        Orbital energies.
    F : (n,n) ndarray
        Final Fock matrix.
    """
    n = h1.shape[0]
    occ = n_elec // 2
    # antisymmetrised two-electron integrals <pq||rs>
    eri = g2 - g2.transpose(0, 1, 3, 2)

    # ---- initial guess (AO/Site canonical) -------------------------------
    C = np.eye(n)
    D = 2.0 * (C[:, :occ] @ C[:, :occ].T)  # spin-paired density

    E_prev = None
    for it in range(1, max_iter + 1):
        # ---- Fock -------------------------------------------------------
        J = np.einsum('pqrs,rs->pq', eri, D, optimize=True)
        K = np.einsum('prqs,rs->pq', eri, D, optimize=True)
        F = h1 + J - 0.5 * K

        # ---- diagonalise & build new density ---------------------------
        eps, C_new = np.linalg.eigh(F)
        C_new, _ = np.linalg.qr(C_new)
        assert np.allclose(C_new.T.conj() @ C_new, np.eye(n), atol=1e-12)
        D_new = 2.0 * (C_new[:, :occ] @ C_new[:, :occ].T)

        # ---- mix density (simple damping) ------------------------------
        D_mixed = mix * D_new + (1.0 - mix) * D

        # ---- HF total energy ------------------------------------------
        E_one = np.sum(D_mixed * h1).real
        E_Coul = 0.5 * np.sum(D_mixed * (J - 0.5 * K)).real
        E_hf = E_one + E_Coul

        # ---- convergence checks ---------------------------------------
        dE = np.inf if E_prev is None else abs(E_hf - E_prev)
        dD = np.linalg.norm(D_mixed - D, 'fro')

        # print(f"[{it:2d}]  E = {E_hf:.12f}  ΔE = {dE:.3e}  ΔD = {dD:.3e}")

        if (dE < conv_E) and (dD < conv_D):
            # print("RHF converged.")
            return C_new, eps, E_hf

        # ---- update for next cycle -------------------------------------
        D = D_mixed
        C = C_new
        E_prev = E_hf

    raise RuntimeError("RHF failed to converge within max_iter")


# ---------------------------------------------------------------------------
def transform_integrals(C, h1, g2):
    """
    AO/site → MO transformation.

    Parameters
    ----------
    C  : (n,n) ndarray
        Orbital coefficient matrix (columns are MO’s).
    h1 : (n,n) ndarray
        One-electron integrals in the site/AO basis.
    g2 : (n,n,n,n) ndarray
        Two-electron integrals  ⟨ij|kl⟩  (chemist ordering).

    Returns
    -------
    h1_mo : (n,n) ndarray
        One-electron integrals in the MO basis.
    g2_mo : (n,n,n,n) ndarray
        Two-electron integrals in the MO basis, chemist order.
    """
    # One-electron part
    h1_mo = C.conj().T @ h1 @ C

    # Two-electron part  (pq|rs)
    g2_mo = np.einsum('ip,jq,kr,ls,ijkl->pqrs',
                      C.conj(), C.conj(), C, C, g2,
                      optimize=True)

    return h1_mo, g2_mo


# ---------------------------------------------------------------------------
def build_spin_orbital_ham(h1, g2, thresh=1e-8):
    """
    Promote spatial-orbital 1- & 2-electron integrals to the spin-orbital
    basis and return the corresponding FermionOperator.

    Parameters
    ----------
    h1 : (n,n) ndarray
        One-electron integrals  h_pq  in *spatial* basis.
    g2 : (n,n,n,n) ndarray
        Chemist-ordered two-electron integrals (pq|rs) in spatial basis.
        Antisymmetry is *not* assumed.
    thresh : float
        Terms with |coeff| < thresh are discarded.

    Returns
    -------
    H : FermionOperator
        Second-quantised Hamiltonian in the spin-orbital basis.
    """
    n_spatial = h1.shape[0]
    n_spin_orb = 2 * n_spatial

    H = FermionOperator()

    # ------------------------------------------------------------------
    # 1-electron part  :  h^{SO}_{P Q} = h_{p q} δ_{σ_P σ_Q}
    # ------------------------------------------------------------------
    for p in range(n_spatial):
        for q in range(n_spatial):
            coeff = h1[p, q]
            if abs(coeff) < thresh:
                continue
            for spin in (0, 1):           # α, β
                P = 2 * p + spin
                Q = 2 * q + spin
                H += FermionOperator(((P, 1), (Q, 0)), coeff)

    # ------------------------------------------------------------------
    # 2-electron part  :  (PQ|RS)^{SO} = (pq|rs) δ_{σ_P σ_R} δ_{σ_Q σ_S}
    # ------------------------------------------------------------------
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    coeff_spatial = 0.5 * g2[p, q, r, s]   # ½ already
                    if abs(coeff_spatial) < thresh:
                        continue
                    for spin_p in (0, 1):
                        for spin_q in (0, 1):
                            # enforce spin-selection rules
                            spin_r = spin_p   # σ_P = σ_R
                            spin_s = spin_q   # σ_Q = σ_S

                            P = 2 * p + spin_p
                            Q = 2 * q + spin_q
                            R = 2 * r + spin_r
                            S = 2 * s + spin_s

                            H += FermionOperator(
                                ((P, 1), (Q, 1), (S, 0), (R, 0)),
                                coeff_spatial
                            )

    return H


# ── demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # lattice & Hubbard parameters
    xdim, ydim   = 2, 2      # 2×2 square
    t_hop        = 1.0
    U_onsite     = 4.0
    mu_site      = 0.0
    periodic_xy  = True

    n_sites      = xdim * ydim
    n_elec       = n_sites     # half-filling (can be any even integer ≤ 2*n_sites)

    # (1) site-basis integrals
    h_site, g_site = build_spatial_hubbard_integrals(
        xdim, ydim, t_hop, U_onsite, mu_site, periodic_xy
    )

    # (2) RHF
    C_mo, eps_mo, Fock = restricted_hf(h_site, g_site, n_elec)

    # (3) MO integrals
    h_mo, g_mo = transform_integrals(C_mo, h_site, g_site)

    # (4) Hubbard Hamiltonian in HF orbital basis
    H_hf = build_spin_orbital_ham(h_mo, g_mo)

    # --- quick sanity print -------------------------------------------------
    print("Number of spin-orbitals :", h_site.shape[0])
    print("HF orbital energies     :", np.round(eps_mo, 6))
    print("Hamiltonian (HF basis)  :\n", H_hf)

import abc
import warnings
from itertools import product, chain
from typing import List, Tuple, Dict, Union, Optional

import numpy as np
from openfermion import FermionOperator, fermi_hubbard, up_index, down_index

from ofex.constant import EV_TO_HARTREE, DEG_TO_RADIAN
from ofex.hamiltonian.fermi_hubbard_rhf import build_spatial_hubbard_integrals, restricted_hf, transform_integrals, \
    build_spin_orbital_ham
from ofex.operators.fermion_operator_tools import one_body_excitation, one_body_number
from ofex.state import BinaryFockVector

__all__ = ["PolyacenePPP", "FermionicHubbard"]

from ofex.transforms import fermion_rotation_operator

SPIN_DOWN, SPIN_UP = 0, 1


class CustomElectronicStructure(abc.ABC):
    @property
    @abc.abstractmethod
    def n_qubits(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def n_electrons(self) -> int:
        pass

    @abc.abstractmethod
    def hf_state(self, *args) -> Dict[BinaryFockVector, complex]:
        pass

    @abc.abstractmethod
    def get_molecular_hamiltonian(self) -> FermionOperator:
        pass


class FermionicHubbard(CustomElectronicStructure):
    def __init__(self,
                 x_dimension: int,
                 y_dimension: int = 1,
                 tunneling: float = 1.0,
                 coulomb: float = 1.0,
                 chemical_potential: float = 0.0,
                 magnetic_field: float = 0.0,
                 periodic: bool = True,
                 n_electrons: Optional[int] = None,
                 run_rhf: bool = False):
        n_sites = x_dimension * y_dimension
        if n_electrons is None:  # Default = Half Filled
            n_electrons = n_sites
        self.x_dimension, self.y_dimension = x_dimension, y_dimension
        self.n_sites = n_sites
        self._n_electrons = n_electrons
        self._n_qubits = 2 * n_sites

        self.tunneling, self.coulomb = tunneling, coulomb
        self.chemical_potential, self.magnetic_field = chemical_potential, magnetic_field
        self.periodic = periodic

        self.run_rhf = run_rhf
        self.hf_energy = None

        if run_rhf:
            if self.magnetic_field != 0.0:
                raise ValueError
            h_site, g_site = build_spatial_hubbard_integrals(self.x_dimension, self.y_dimension,
                                                             self.tunneling, self.coulomb,
                                                             self.chemical_potential,
                                                             self.periodic)
            # self.hamiltonian = fermi_hubbard(self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
            #                                 self.chemical_potential, self.magnetic_field, self.periodic,
            #                                 spinless=False, particle_hole_symmetry=False)
            C_mo, eps_mo, hf_energy = restricted_hf(h_site, g_site, self.n_electrons)
            h_mo, g_mo = transform_integrals(C_mo, h_site, g_site)
            self.hamiltonian = build_spin_orbital_ham(h_mo, g_mo)
            self.hf_energy = hf_energy
            # self.hamiltonian = fermion_rotation_operator(self.hamiltonian, C_mo, spatial_v=True)
        else:
            self.hamiltonian = fermi_hubbard(self.x_dimension, self.y_dimension, self.tunneling, self.coulomb,
                                             self.chemical_potential, self.magnetic_field, self.periodic,
                                             spinless=False, particle_hole_symmetry=False)

    def xy_to_spatial_idx(self, x, y) -> int:
        return x + y * self.x_dimension

    def spatial_idx_to_xy(self, site_idx) -> Tuple[int, int]:
        return site_idx % self.x_dimension, site_idx // self.x_dimension

    @property
    def n_electrons(self) -> int:
        return self._n_electrons

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def get_molecular_hamiltonian(self) -> FermionOperator:
        return self.hamiltonian

    def hf_state(self):
        if not self.run_rhf:
            warnings.warn("HF state is not prepared. Neel state is returned instead.")
            return self.neel_state()
        occ = [1] * self.n_electrons + [0] * (self.n_qubits - self.n_electrons)
        return {BinaryFockVector(occ): 1.0}

    def neel_state(self):
        neel_pattern = []
        complement_pattern = []
        for x, y in product(range(self.x_dimension), range(self.y_dimension)):
            site = self.xy_to_spatial_idx(x, y)
            if (x + y) % 2 == 0:  # A-sublattice
                neel_pattern.append(up_index(site))
                complement_pattern.append(down_index(site))
            else:  # B-sublattice
                neel_pattern.append(down_index(site))
                complement_pattern.append(up_index(site))

        filling_order = list(chain(neel_pattern, complement_pattern))
        occ = [0] * self.n_qubits
        for orb_idx in filling_order[:self.n_electrons]:
            occ[orb_idx] = 1
        assert sum(occ) == self.n_electrons

        return {BinaryFockVector(occ): 1.0}


class PolyacenePPP(CustomElectronicStructure):
    """
    The PolyacenePPP class represents a linear polyacene molecule described using the Pariser-Parr-Pople (PPP) model
    for quantum chemistry calculations. It provides methods for computing molecular Hamiltonians, electron-electron
    repulsion, hopping terms, and other properties based on the PPP approximation.

    Reference:
        Chakraborty, H., & Shukla, A. (2013). Pariser–Parr–Pople Model Based Investigation of Ground and Low-Lying
        Excited States of Long Acenes. The Journal of Physical Chemistry A, 117(51), 14220–14229.
        doi:10.1021/jp408535u

    Attributes:
        n_ring (int): The number of aromatic rings in the polyacene molecule.
        t0 (float): Hopping strength value for horizontal bonds.
        t_ (float): Hopping strength value for vertical bonds.
        bond_length (float): The bond length between adjacent carbon atoms (in Angstroms).
    """
    # Unit = Angstrom, Hartree
    # n_ring: int = 3

    # t0: float = 2.4 * EV_TO_HARTREE
    # t_: float = 2.4 * EV_TO_HARTREE

    # bond_length: float = 1.4
    _bond_angle: float = 120 * DEG_TO_RADIAN

    # param_type: str = "standard"

    @property
    def atom_name(self) -> List[str]:
        """
        Returns:
            List[str]: A list of atom names in the polyacene molecule.
        """
        return list(self.coord_dict.keys())

    @property
    def num_spin_orbitals(self) -> int:
        """
        Returns:
            int: The total number of spin-orbitals in the polyacene molecule.
        """
        return 4 + 8 * self.n_ring

    @property
    def n_electrons(self) -> int:
        return self.num_spin_orbitals // 2

    @property
    def n_qubits(self) -> int:
        return self.num_spin_orbitals

    def __init__(self, n_ring: int,
                 param_type: str = "standard",
                 t0: float = 2.4 * EV_TO_HARTREE,
                 t_: float = 2.4 * EV_TO_HARTREE,
                 bond_length: float = 1.4, ):
        """
        Initializes the PolyacenePPP object and sets up molecular parameters.

        Args:
            n_ring (int): The number of aromatic rings in the polyacene molecule.
            param_type (str): Type of parameters to use ("standard" or "screened").
            t0 (float): Hopping strength value for horizontal bonds.
            t_ (float): Hopping strength value for vertical bonds.
            bond_length (float): The bond length between adjacent carbon atoms (in Angstroms).

        Attributes:
            coord_dict (Dict[str, Tuple[float, float]]): A dictionary containing the coordinates of each atom.
        """
        self.n_ring = n_ring
        self.param_type = param_type
        self.t0, self.t_ = t0, t_
        self.bond_length = bond_length
        self.coord_dict = self._polyacene_carbon_geometry()

    # <editor-fold desc="PPP Parameters">
    def kappa(self, i: str, j: str):
        """
        Computes the screening parameter based on the parameter type.

        Args:
            i (str): Label for the first atom.
            j (str): Label for the second atom.

        Returns:
            float: The screening parameter value.

        Raises:
            ValueError: If the parameter type is invalid.
        """
        if self.param_type == "standard":
            return 1.0
        elif self.param_type == "screened":
            return 1.0 if i == j else 2.0
        else:
            raise ValueError

    def u(self):
        """
        Computes the on-site repulsion term based on the parameter type.

        Returns:
            float: The on-site electron-electron repulsion in Hartree.

        Raises:
            ValueError: If the parameter type is invalid.
        """
        if self.param_type == "standard":
            return 11.13 * EV_TO_HARTREE
        elif self.param_type == "screened":
            return 8.0 * EV_TO_HARTREE
        else:
            raise ValueError

    def v(self, i: str, j: str):
        """
        Computes the Coulomb interaction between two atomic sites.

        Args:
            i (str): Label for the first atom.
            j (str): Label for the second atom.

        Returns:
            float: The Coulomb interaction value in Hartree, accounting for
            screening and distance-dependent effects between the atoms.
        """
        rij = self.atom_dist(i, j)
        return self.u() / (self.kappa(i, j) * (1 + 0.6117 * (rij ** 2)) ** 0.5)

    def _hopping_horizontal(self, part: str) -> FermionOperator:
        """
        Constructs the hopping operator for horizontal bonds in the specified part.

        Args:
            part (str): The part of the molecule, either 'L' (lower) or 'U' (upper).

        Returns:
            FermionOperator: The horizontal hopping operator for the specified part of the molecule.
        """
        half_num_carbons = 1 + 2 * self.n_ring
        hopping = FermionOperator.accumulate(
            [one_body_excitation(self.spin_idx(f"{part}{i}", spin=SPIN_DOWN),
                                 self.spin_idx(f"{part}{i + 1}", spin=SPIN_DOWN),
                                 spin_idx=True, hermitian=True)
             for i in range(half_num_carbons - 1)]
        )
        hopping += FermionOperator.accumulate(
            [one_body_excitation(self.spin_idx(f"{part}{i}", spin=SPIN_UP),
                                 self.spin_idx(f"{part}{i + 1}", spin=SPIN_UP),
                                 spin_idx=True, hermitian=True)
             for i in range(half_num_carbons - 1)]
        )
        return hopping

    def hopping_lower(self) -> FermionOperator:
        """
        Constructs the hopping operator for horizontal bonds in the lower part of the molecule.

        Returns:
            FermionOperator: The lower horizontal hopping operator scaled by -t0.
        """
        return self._hopping_horizontal("L") * -self.t0

    def hopping_upper(self) -> FermionOperator:
        """
        Constructs the hopping operator for horizontal bonds in the upper part of the molecule.

        Returns:
            FermionOperator: The upper horizontal hopping operator scaled by -t0.
        """
        return self._hopping_horizontal("U") * -self.t0

    def hopping_bridge(self) -> FermionOperator:
        """
        Constructs the hopping operator for vertical bonds between the upper and lower parts of the molecule.

        Returns:
            FermionOperator: The bridge hopping operator scaled by `-self.t_`.
        """
        half_num_carbons = 1 + 2 * self.n_ring
        hopping = FermionOperator.accumulate(
            [one_body_excitation(self.spin_idx(f"U{i}", spin=SPIN_DOWN),
                                 self.spin_idx(f"L{i}", spin=SPIN_DOWN),
                                 spin_idx=True, hermitian=True)
             for i in range(0, half_num_carbons, 2)]
        )
        hopping += FermionOperator.accumulate(
            [one_body_excitation(self.spin_idx(f"U{i}", spin=SPIN_UP),
                                 self.spin_idx(f"L{i}", spin=SPIN_UP),
                                 spin_idx=True, hermitian=True)
             for i in range(0, half_num_carbons, 2)]
        )
        return hopping * -self.t_

    def ee_repulsion(self) -> FermionOperator:
        """
        Constructs the electron-electron repulsion operator, including on-site and Ohno interaction terms.

        Returns:
            FermionOperator: The total electron-electron repulsion operator.
        """
        num_carbons = 2 + 4 * self.n_ring
        on_site = FermionOperator.accumulate(
            [one_body_number(2 * x + SPIN_DOWN, spin_idx=True) *
             one_body_number(2 * x + SPIN_UP, spin_idx=True)
             for x in range(num_carbons)]) * self.u()
        ohno_interaction = FermionOperator()
        for i in range(num_carbons):
            for j in range(i + 1, num_carbons):
                i_str, j_str = self.inv_spatial_idx(i), self.inv_spatial_idx(j)
                tmp = (one_body_number(2 * i + SPIN_DOWN, spin_idx=True) +
                       one_body_number(2 * i + SPIN_UP, spin_idx=True) - 1.0) * \
                      (one_body_number(2 * j + SPIN_DOWN, spin_idx=True) +
                       one_body_number(2 * j + SPIN_UP, spin_idx=True) - 1.0)
                ohno_interaction += tmp * self.v(i_str, j_str)
        return on_site + ohno_interaction

    def fermion_hamiltonian(self) -> FermionOperator:
        """
        Constructs the full fermionic Hamiltonian of the molecule, which includes
        contributions from horizontal hopping, vertical bridge hopping, and
        electron-electron repulsion.

        Returns:
            FermionOperator: The full fermionic Hamiltonian of the molecule.
        """
        return self.hopping_lower() + self.hopping_upper() + self.hopping_bridge() + self.ee_repulsion()

    def get_molecular_hamiltonian(self):
        """
        Retrieves the molecular Hamiltonian for the polyacene molecule.

        Returns:
            FermionOperator: The fermionic Hamiltonian of the molecule.
        """
        return self.fermion_hamiltonian()

    def _hf_fock(self, parity: bool = True):
        """
        Constructs a Hartree-Fock state based on the parity of orbitals.

        Args:
            parity (bool): Determines whether the alternating occupation starts with the 0th orbital or the 1st orbital.
                           If True, the occupation starts from the 1st orbital; if False, the occupation starts from the
                           0th orbital.

        Returns:
            List[int]: A list representing the Hartree-Fock state occupation numbers for each orbital.
        """
        fock = [0 for _ in range(self.num_spin_orbitals)]
        for an in self.atom_name:
            if (not parity) and (an[0] == "L" and int(an[1:]) % 2 == 0 or \
                                 an[0] == "U" and int(an[1:]) % 2 == 1):
                fock[self.spin_idx(an, SPIN_DOWN)] = 1
                fock[self.spin_idx(an, SPIN_UP)] = 1
            elif parity and (an[0] == "L" and int(an[1:]) % 2 == 1 or \
                             an[0] == "U" and int(an[1:]) % 2 == 0):
                fock[self.spin_idx(an, SPIN_DOWN)] = 1
                fock[self.spin_idx(an, SPIN_UP)] = 1
        assert sum(fock) == self.n_electrons
        return fock

    def hf_state(self, parity: bool = True):
        """
        Constructs a Hartree-Fock (HF) state for the molecule based on a given parity.

        Args:
            parity (bool): Determines whether the alternating occupation starts with the 0th orbital or the 1st orbital.
                           If True, the occupation starts from the 1st orbital; if False, the occupation starts from the
                           0th orbital.

        Returns:
            Dict[BinaryFockVector, float]: The Hartree-Fock state as a dictionary where the key is the
                                           BinaryFockVector representing the state and the value is its coefficient.
        """
        fock = self._hf_fock(parity=parity)
        return {BinaryFockVector(fock): 1.0}

    def ref_state(self):
        """
        Constructs the reference (singlet) state as a superposition of Hartree-Fock
        states with even and odd parities.

        Returns:
            Dict[BinaryFockVector, float]: The reference state as a dictionary of BinaryFockVector
                                           to their respective coefficients.
        """
        fock_even = self.hf_state(parity=False)
        fock_odd = self.hf_state(parity=True)
        return {BinaryFockVector(fock_odd): 1 / np.sqrt(2),
                BinaryFockVector(fock_even): 1 / np.sqrt(2)}

    # </editor-fold>

    def fermion_symmetries(self) -> List[List[Tuple[int, int]]]:
        """
        Generates symmetry transformations for the fermionic orbitals
        in the system based on spatial symmetry operations.

        Returns:
            List[List[Tuple[int, int]]]: A list of symmetry transformations,
                                         where each transformation is represented
                                         as a list of tuples. Each tuple defines a
                                         permutation of two orbitals (given by their indices).
                                         The transformations included are:
                                         - σ(x): Reflection across the x-axis.
                                         - σ(y): Reflection across the y-axis.
        """
        atoms = self.atom_name
        u_atoms = sorted([x for x in atoms if x.startswith("U")])
        l_atoms = sorted([x for x in atoms if x.startswith("L")])
        assert len(u_atoms) == len(l_atoms)
        perm_list = list()

        # σ(x)
        sigma_x_swaps = list()
        for ua, la in zip(u_atoms, l_atoms):
            sigma_x_swaps.append((self.spin_idx(ua, SPIN_DOWN), self.spin_idx(la, SPIN_DOWN)))
            sigma_x_swaps.append((self.spin_idx(ua, SPIN_UP), self.spin_idx(la, SPIN_UP)))
        perm_list.append(sigma_x_swaps)

        # σ(y)
        sigma_y_swaps = list()
        half_atom = len(u_atoms) // 2
        for ula, ura in zip(u_atoms[:half_atom], u_atoms[:half_atom:-1]):
            sigma_y_swaps.append((self.spin_idx(ula, SPIN_DOWN), self.spin_idx(ura, SPIN_DOWN)))
            sigma_y_swaps.append((self.spin_idx(ula, SPIN_UP), self.spin_idx(ura, SPIN_UP)))
        for lla, lra in zip(l_atoms[:half_atom], l_atoms[:half_atom:-1]):
            sigma_y_swaps.append((self.spin_idx(lla, SPIN_DOWN), self.spin_idx(lra, SPIN_DOWN)))
            sigma_y_swaps.append((self.spin_idx(lla, SPIN_UP), self.spin_idx(lra, SPIN_UP)))
        perm_list.append(sigma_y_swaps)

        # The i-operation is combination of σ(x) and σ(y)
        # i_swaps = list()
        # for ua, la in zip(u_atoms, l_atoms[::-1]):
        #     i_swaps.append((self.spin_idx(ua, SPIN_DOWN), self.spin_idx(la, SPIN_DOWN)))
        #     i_swaps.append((self.spin_idx(ua, SPIN_UP), self.spin_idx(la, SPIN_UP)))
        # perm_list.append(i_swaps)

        return perm_list

    # <editor-fold desc="Utilities">
    def spin_idx(self, atom: str, spin: Union[bool, int]):
        """
        Calculates the index of the spin orbital associated with a given atom and spin.

        Args:
            atom (str): The label of the atom, where "U" represents an upper atom
                        and "L" represents a lower atom, followed by a numerical index.
                        Example: "U0", "L1".
            spin (Union[bool, int]): The spin designation for the orbital. Acceptable
                                     values are either:
                                     - 0 or False for SPIN_DOWN
                                     - 1 or True for SPIN_UP

        Returns:
            int: The index of the spin orbital corresponding to the specified atom
                 and spin.

        Raises:
            ValueError: If the atom label does not start with "L" or "U".
        """
        spin = SPIN_UP if spin else SPIN_DOWN
        if atom[0] == "L":
            half_num_carbons = 1 + 2 * self.n_ring
            return 2 * (half_num_carbons + int(atom[1:])) + spin
        elif atom[0] == "U":
            return 2 * int(atom[1:]) + spin
        else:
            raise ValueError

    def inv_spatial_idx(self, i: int) -> str:
        """
        Converts a given spin-orbital index into its corresponding spatial atom label.

        Args:
            i (int): The index of the spin-orbital.

        Returns:
            str: The atom label corresponding to the provided spin-orbital index.
                 This is in the form "U<num>" for upper atoms or "L<num>" for lower atoms,
                 where <num> is the atom number.

        """
        half_num_carbons = 1 + 2 * self.n_ring
        if i < half_num_carbons:
            return f"U{i}"
        else:
            return f"L{i - half_num_carbons}"

    def atom_dist(self, i: str, j: str) -> float:
        """
        Calculates the Euclidean distance between two atoms based on their coordinates.

        Args:
            i (str): The label of the first atom (e.g., "U0" or "L1").
            j (str): The label of the second atom (e.g., "U1" or "L0").

        Returns:
            float: The Euclidean distance between the two atoms in Angstroms.
        """
        pi = self.coord_dict[i]
        pj = self.coord_dict[j]
        return sum([(a - b) ** 2 for a, b in zip(pi, pj)]) ** 0.5

    def _polyacene_carbon_geometry(self) -> Dict[str, Tuple[float, float]]:
        """
        Generates the spatial coordinates for all carbon atoms in the polyacene molecule
        based on the number of aromatic rings and geometric parameters.

        The geometry alternates between "upper" (U) and "lower" (L) layers of carbon atoms,
        with positions determined by the bond length and bond angle.

        Returns:
            Dict[str, Tuple[float, float]]: A dictionary mapping atom labels to their
                2D coordinates. The keys are strings in the format "Xn" where X is "U" (upper)
                or "L" (lower) and n is an integer. The values are tuples representing the
                (x, y) coordinates of each atom in Angstroms.
        """
        if self.n_ring < 1:
            raise ValueError
        # upper_carbons = [(0.0, bond_length) ]
        # lower_carbons = [(0.0, 0.0)]
        ret_coord = dict()
        ret_coord["L0"] = (0.0, 0.0)
        ret_coord["U0"] = (0.0, self.bond_length)
        bond_angle_sub = self._bond_angle - np.pi / 2

        top_y = self.bond_length * (1 + np.sin(bond_angle_sub))
        bot_y = -self.bond_length * np.sin(bond_angle_sub)
        inc_x = self.bond_length * np.cos(bond_angle_sub)
        i_l, i_u = 1, 1
        for i in range(self.n_ring):
            left_x = i * 2 * inc_x
            center_x = left_x + inc_x
            right_x = left_x + 2 * inc_x

            # upper_carbons += [
            ret_coord[f"U{i_u}"] = (center_x, top_y)
            ret_coord[f"U{i_u + 1}"] = (right_x, self.bond_length)
            # ]
            # lower_carbons += [
            ret_coord[f"L{i_l}"] = (center_x, bot_y)
            ret_coord[f"L{i_l + 1}"] = (right_x, 0.0)
            # ]
            i_u += 2
            i_l += 2
        return ret_coord
    # </editor-fold>

import math
from typing import Any, List, Tuple, Dict, Optional, Union

import numpy as np
from openfermion import FermionOperator

from ofex.constant import EV_TO_HARTREE, DEG_TO_RADIAN
from ofex.operators.fermion_operator_tools import one_body_excitation, one_body_number
from ofex.state.binary_fock import BinaryFockVector

SPIN_DOWN, SPIN_UP = 0, 1


class PolyacenePPP:
    # Unit = Angstrom, Hartree
    n_ring: int = 3

    t0: float = 2.4 * EV_TO_HARTREE
    t_: float = 2.4 * EV_TO_HARTREE

    bond_length: float = 1.4
    bond_angle: float = 120 * DEG_TO_RADIAN

    param_type: str = "standard"

    @property
    def atom_name(self) -> List[str]:
        return list(self.coord_dict.keys())

    @property
    def num_spin_orbitals(self) -> int:
        return 4 + 8 * self.n_ring

    @property
    def n_electrons(self) -> int:
        return self.num_spin_orbitals // 2

    def __init__(self, n_ring: int,
                 param_type: str = "standard",
                 t0: float = 2.4 * EV_TO_HARTREE,
                 t_: float = 2.4 * EV_TO_HARTREE,
                 bond_length: float = 1.4, ):
        self.n_ring = n_ring
        self.param_type = param_type
        self.t0, self.t_ = t0, t_
        self.bond_length = bond_length
        self.coord_dict = self._polyacene_carbon_geometry()

    # <editor-fold desc="PPP Parameters">
    def kappa(self, i: str, j: str):
        if self.param_type == "standard":
            return 1.0
        elif self.param_type == "screened":
            return 1.0 if i == j else 2.0
        else:
            raise ValueError

    def u(self):
        if self.param_type == "standard":
            return 11.13 * EV_TO_HARTREE
        elif self.param_type == "screened":
            return 8.0 * EV_TO_HARTREE
        else:
            raise ValueError

    def v(self, i: str, j: str):
        rij = self.atom_dist(i, j)
        return self.u() / (self.kappa(i, j) * (1 + 0.6117 * (rij ** 2)) ** 0.5)

    def _hopping_horizontal(self, part: str) -> FermionOperator:
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
        return self._hopping_horizontal("L") * -self.t0

    def hopping_upper(self) -> FermionOperator:
        return self._hopping_horizontal("U") * -self.t0

    def hopping_bridge(self) -> FermionOperator:
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
        return self.hopping_lower() + self.hopping_upper() + self.hopping_bridge() + self.ee_repulsion()

    def get_molecular_hamiltonian(self):
        return self.fermion_hamiltonian(), self.num_spin_orbitals

    def hf_state(self):
        fock = [0 for _ in range(self.num_spin_orbitals)]
        for an in self.atom_name:
            if an[0] == "L" and int(an[1:]) % 2 == 0 or \
                    an[0] == "U" and int(an[1:]) % 2 == 1:
                fock[self.spin_idx(an, SPIN_DOWN)] = 1
                fock[self.spin_idx(an, SPIN_UP)] = 1
        assert sum(fock) == self.n_electrons
        return {BinaryFockVector(fock): 1.0}

    # </editor-fold>

    def fermion_symmetries(self) -> List[List[Tuple[int, int]]]:
        """

        Returns:
            perm_list : Permutations of orbitals corresponding to the symmetry transform.

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

        Args:
            atom: The name of the atom
            spin: Up(0) or Down(1) spin

        Returns:
            tot_idx: The index of the orbital designated to the atom with the spin.

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
        half_num_carbons = 1 + 2 * self.n_ring
        if i < half_num_carbons:
            return f"U{i}"
        else:
            return f"L{i - half_num_carbons}"

    def atom_dist(self, i: str, j: str):
        pi = self.coord_dict[i]
        pj = self.coord_dict[j]
        return sum([(a - b) ** 2 for a, b in zip(pi, pj)]) ** 0.5

    def _polyacene_carbon_geometry(self) -> Dict[str, Tuple[float, float]]:
        """

        Returns:
            carbon_coordinate: dictionary of str to tuple of two floats, str="xi" x="L" or "U" i=integer

        """
        if self.n_ring < 1:
            raise ValueError
        # upper_carbons = [(0.0, bond_length) ]
        # lower_carbons = [(0.0, 0.0)]
        ret_coord = dict()
        ret_coord["L0"] = (0.0, 0.0)
        ret_coord["U0"] = (0.0, self.bond_length)
        bond_angle_sub = self.bond_angle - np.pi / 2

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ofex.state.state_tools import pretty_print_state


    def _test_coord(n=3):
        delta = (0.2, 0.2)

        polyacene = PolyacenePPP(n_ring=n)
        coord_dict = polyacene.coord_dict
        print(coord_dict)
        for name, point in coord_dict.items():
            # print(point, name)
            plt.scatter(point[0], point[1], color='k')
            plt.text(point[0] + delta[0], point[1] + delta[1],
                     name + f"({polyacene.spin_idx(name, 0)}, {polyacene.spin_idx(name, 1)})")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


    def _test_hamiltonian(n=3):
        polyacene = PolyacenePPP(n_ring=n)
        hc1 = polyacene.hopping_lower()
        print("hc1")
        print(hc1)
        print("")

        hc2 = polyacene.hopping_upper()
        print("hc2")
        print(hc2)
        print("")

        hc1c2 = polyacene.hopping_bridge()
        print("hc1c2")
        print(hc1c2)
        print("")

        hee = polyacene.ee_repulsion()
        print("hee")
        print(hee)
        print("")

        assert (hc1 + hc2 + hc1c2 + hee) == polyacene.fermion_hamiltonian()
        print(f"HF state = {pretty_print_state(polyacene.hf_state())}")


    _test_coord(n=2)
    _test_hamiltonian(n=2)

"""
This module provides tools and methods for working with molecular geometries 
and quantum chemistry calculations. It includes utilities for generating linear 
and bent molecular structures, defining example molecular geometries, and running 
quantum chemistry simulations using various drivers.

Main functionalities:
- Define linear and bent molecular geometries.
- Predefine common molecules with standard parameters.
- Run quantum chemistry calculations using drivers like PySCF and Psi4.
"""
import os
import subprocess
import warnings
from numbers import Number
from typing import List, Union, Tuple, Optional

import numpy as np
from openfermion import MolecularData

__all__ = ["MoleculeGeometry", "diatomic_molecules", "triatomic_molecules", "default_param",
           "linear_geometry", "bent_geometry", "molecule_example",
           "run_driver"]


MoleculeGeometry = List[Tuple[str, Tuple[float, float, float]]]

diatomic_molecules = ["H2", "HeH+", "LiH", ]
triatomic_molecules = ["BeH2", "H2O", ]

default_param = {
    "H2": 0.74,
    "H4": 0.74,
    "HeH+": 0.772,
    "LiH": 1.5,
    "BeH2": 1.413,
    "H2O": [0.957, 0.957, 104.5],
    "Li2O": 1.0
}


def linear_geometry(atoms: List[str],
                    distances: Union[float, List[float]]) -> MoleculeGeometry:
    """
    Generate a linear molecular geometry based on specified atoms and interatomic distances.

    This function positions a series of atoms in a one-dimensional linear arrangement,
    computing their coordinates such that each atom is separated from its neighbors
    by the specified distances.

    Args:
        atoms (List[str]): A list of atom symbols with length `n`. Example: ['H', 'Be', 'H'].
        distances (Union[float, List[float]]): A single numeric value or a list of `n-1` distances
            representing interatomic separations in angstrom. If a single value is provided, it will be
            uniformly applied between all atoms. Example: 1.5 or [1.5, 1.5].

    Returns:
        MoleculeGeometry: A list of tuples where each tuple contains the atomic symbol
            and its 3D coordinates (0.0, 0.0, z) in a linear arrangement. Example: [('H', (0.0, 0.0, 0.0)), 
            ('Be', (0.0, 0.0, 1.5)), ('H', (0.0, 0.0, 3.0))].

    Raises:
        ValueError: If the number of distances provided does not match the number of atoms - 1.
    """

    if isinstance(distances, Number):
        distances = [distances for _ in range(len(atoms) - 1)]
    if len(distances) != len(atoms) - 1:
        raise ValueError(f"The number of distances {len(distances)} is not equal to the number of atoms - 1"
                         f"{len(atoms) - 1}")
    position = [sum(distances[:i]) for i in range(len(atoms))]
    return [(a, (0.0, 0.0, p)) for a, p in zip(atoms, position)]


def bent_geometry(central: str,
                  peripheral: Union[str, List[str]],
                  distances: Union[float, List[float]],
                  angle: float) -> MoleculeGeometry:
    """
    Generate the 3D geometry of a bent molecule.
    
    Args:
        central (str): The central atom. Example: 'O'.
        peripheral (Union[str, List[str]]): The peripheral atoms. Example: 'H' or ['H', 'H'].
        distances (Union[float, List[float]]): Distances from the central atom to the peripheral atoms in angstrom.
            Example: 0.97 or [0.97, 0.97].
        angle (float): Angle between the bonds in degrees. Example: 104.5.
    
    Returns:
        MoleculeGeometry: A list of tuples representing atom names and their 3D coordinates.
            Example: [('O', (0, 0, 0)), ('H', (0, 0.97, 0)), ('H', (0, 0.825, 0.509))].
    
    Raises:
        ValueError: If less than two peripheral atoms or distances are specified.
    """

    if isinstance(peripheral, str):
        peripheral = [peripheral, peripheral]
    if isinstance(distances, Number):
        distances = [distances, distances]
    if len(peripheral) != 2 or len(distances) != 2:
        raise ValueError("Two peripheral atoms need to be specified.")
    x = distances[1] * np.cos(np.pi / 180 * angle)
    y = distances[1] * np.sin(np.pi / 180 * angle)
    return [(central, (0, 0, 0)),
            (peripheral[0], (0, distances[0], 0)),
            (peripheral[1], (0, x, y))]


def molecule_example(molecule_name: str,
                     param: Optional[Union[Number, List[Number]]] = None,
                     geometry: Optional[MoleculeGeometry] = None,
                     **kwargs) -> MolecularData:
    """
    Generate a `Molecule` object using a predefined molecular name and its geometry parameters.
    
    Args:
        molecule_name (str): Supported molecules include "H2", "H4", "HeH+", "LiH", "BeH2", "H2O", and "Li2O".
        param (Optional[Union[Number, List[Number]]]): Parameters for the geometry. Examples:
            - Single distance (e.g., 0.74 for "H2").
            - List of distances and angles (e.g., [0.957, 0.957, 104.5] for "H2O").
        geometry (Optional[MoleculeGeometry]): Custom molecular geometry as a list of tuples.
            If not provided, a default geometry based on `molecule_name` and `param` will be used.
        **kwargs: Additional keyword arguments passed to the `Molecule` object initialization, such as:
            - `basis` (str): Basis set to use. Default is "sto-3g".
            - `multiplicity` (int): Spin multiplicity. Default is 1.
    
    Returns:
        MolecularData: A `Molecule` object with the defined geometry and properties.
    
    Raises:
        ValueError: If the `molecule_name` is not recognized.
    """
    basis = 'sto-3g' if 'basis' not in kwargs else kwargs['basis']
    if 'basis' in kwargs:
        kwargs.pop('basis')
    multiplicity = 1 if 'multiplicity' not in kwargs else kwargs['multiplicity']
    if 'multiplicity' in kwargs:
        kwargs.pop('multiplicity')
    charge = 0

    if geometry is None:
        if param is None:
            param = default_param[molecule_name]
        if molecule_name == "H2":
            geometry = linear_geometry(['H', 'H'], param)
        elif molecule_name == "H4":
            geometry = linear_geometry(['H', 'H', 'H', 'H'], param)
        elif molecule_name == "HeH+":
            charge = 1
            geometry = linear_geometry(['He', 'H'], param)
        elif molecule_name == "LiH":
            geometry = linear_geometry(['Li', 'H'], param)
        elif molecule_name == "BeH2":
            if isinstance(param, Number):
                param = [param, param]
            geometry = linear_geometry(['H', 'Be', 'H'], param)
        elif molecule_name == "H2O":
            if isinstance(param, Number):
                param = [param, param, default_param["H2O"][2]]
            geometry = bent_geometry("O", "H", param[:2], param[2])
        elif molecule_name == 'Li2O':
            geometry = [['Li', [0, 0, -param]],
                        ['O', [0, 0, 0]],
                        ['Li', [0, 0, param]]]
        else:
            raise ValueError(f"{molecule_name} is not in the example list.")

    mol = MolecularData(geometry,
                        basis,
                        multiplicity,
                        charge,
                        description="tmp",
                        **kwargs)
    try:
        mol.load()
    except FileNotFoundError:
        pass
    return run_driver(mol)


def run_driver(molecule: MolecularData,
               run_scf: bool = True,
               run_mp2: bool = False,
               run_cisd: bool = False,
               run_ccsd: bool = False,
               run_fci: bool = False,
               driver: str = 'pyscf',
               **kwargs) -> MolecularData:
    """
    Execute quantum chemistry calculations for a molecule using a specified driver.

    This function supports running various levels of theory calculations, including SCF, MP2, CISD, CCSD, and FCI,
    for molecules represented as `MolecularData` objects. It works with multiple external drivers such as `pyscf` 
    and `psi4`.

    Args:
        molecule (MolecularData): A molecule object including its geometry, charge, and multiplicity.
        run_scf (bool): If True, perform a Self-Consistent Field (SCF) calculation. Default is True.
        run_mp2 (bool): If True, perform a 2nd order Møller–Plesset (MP2) calculation. Default is False.
        run_cisd (bool): If True, perform a Configuration Interaction with Single and Double Excitations
            (CISD) calculation. Default is False.
        run_ccsd (bool): If True, perform a Coupled Cluster with Single and Double Excitations (CCSD)
            calculation. Default is False.
        run_fci (bool): If True, perform a Full Configuration Interaction (FCI) calculation. Default is False.
        driver (str): The quantum chemistry driver to use ('pyscf' or 'psi4'). Default is 'pyscf'.
        **kwargs: Additional arguments specific to the quantum chemistry driver used.

    Returns:
        MolecularData: The input molecule object updated with the results of the calculations.

    Raises:
        ValueError: If an unsupported driver is specified.
        AttributeError: If a required class or updated module is not available for the driver.
    """
    if driver.lower() == 'psi4':
        return run_psi4(molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci, **kwargs)
    elif driver.lower() == 'pyscf':
        from openfermionpyscf import run_pyscf
        try:
            return run_pyscf(molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci, **kwargs)
        except AttributeError as e:
            msg = f"\nConsider updating the class PyscfMolecularData with"\
                  f"https://github.com/snow0369/OpenFermion-PySCF/blob/master/openfermionpyscf/_pyscf_molecular_data.py."
            raise AttributeError(str(e) + msg) from e
    else:
        raise ValueError(f"Driver {driver} not supported")


def run_psi4(molecule: MolecularData,
             run_scf: bool = True,
             run_mp2: bool = False,
             run_cisd: bool = False,
             run_ccsd: bool = False,
             run_fci: bool = False,
             verbose: bool = False,
             tolerate_error: bool = False,
             delete_input: bool = True,
             delete_output: bool = False,
             memory: int = 8000,
             template_file: Optional[str] = None) -> MolecularData:
    """
    This function runs a Psi4 calculation.
    Modified the original function to use psi4 in the environment variable

    Args:
        molecule: An instance of the MolecularData class.
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.
        tolerate_error: Optional boolean to warn or raise when Psi4 fails.
        delete_input: Optional boolean to delete psi4 input file.
        delete_output: Optional boolean to delete psi4 output file.
        memory: Optional int giving amount of memory to allocate in MB.
        template_file(str): Path to Psi4 template file

    Returns:
        molecule: The updated MolecularData object.

    Raises:
        psi4 errors: An error from psi4.
    """
    # Prepare input.

    from openfermionpsi4._run_psi4 import generate_psi4_input, clean_up
    input_file = generate_psi4_input(molecule,
                                     run_scf,
                                     run_mp2,
                                     run_cisd,
                                     run_ccsd,
                                     run_fci,
                                     verbose,
                                     tolerate_error,
                                     memory,
                                     template_file)

    # Run psi4.
    output_file = molecule.filename + '.out'
    process = None
    try:
        psi4_path = os.environ['PSI4PATH']
    except KeyError:
        raise RuntimeError("Specify the path to PSI4PATH environment variable.")
    print(psi4_path)
    try:
        process = subprocess.Popen([psi4_path, input_file, output_file])
        process.wait()
    except Exception as e:
        print(e)
        print('Psi4 calculation for {} has failed.'.format(molecule.name))
        process.kill()
        try:
            clean_up(molecule, delete_input, delete_output)
        except FileNotFoundError:
            pass
        if not tolerate_error:
            raise e
    else:
        try:
            clean_up(molecule, delete_input, delete_output)
        except FileNotFoundError:
            pass

    # Return updated molecule instance.
    try:
        molecule.load()
    except Exception as e:
        print(e)
        warnings.warn('No calculation saved. '
                      'Psi4 segmentation fault possible.',
                      Warning)
    return molecule

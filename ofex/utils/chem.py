import os
import subprocess
import warnings
from numbers import Number
from typing import List, Union, Tuple, Optional

import numpy as np
from openfermion import MolecularData

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
                    distances: Union[Number, List[Number]]) -> MoleculeGeometry:
    """Return geometry of atoms for linear geometry.

    Args:
        atoms: list of atoms with length of n. example) ['H', 'Be', 'H']
        distances: list of interatomic distances with length of n-1. example) [1.5, 1.5] or 1.5

    Returns:
        geometry: geometry fed to the ``Molecule`` object initialization.

    """

    if isinstance(distances, Number):
        distances = [distances for _ in range(len(atoms) - 1)]
    if len(distances) != len(atoms) - 1:
        raise ValueError(f"The number of distances {len(distances)} is not equal to the number of atoms - 1 {len(atoms) - 1}")
    position = [sum(distances[:i]) for i in range(len(atoms))]
    return [(a, (0.0, 0.0, p)) for a, p in zip(atoms, position)]


def bent_geometry(central: str,
                  peripheral: Union[str, List[str]],
                  distances: Union[Number, List[Number]],
                  angle: Number) -> MoleculeGeometry:
    """Return geometry of atoms for bent geometry.

    Args:
        central: The central atom. example) 'O'
        peripheral: The peripheral atoms. example) ['H', 'H']
        distances: Distances between central and peripheral atoms. example) [0.97, 0.97] or 0.97
        angle: angle between the bonds in degree. example) 104.5

    Returns:
        geometry: geometry fed to the ``Molecule`` object initialization.

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
                     **kwargs) -> MolecularData:
    """Return a ``Molecule`` object from molecule name and parameters for geometry configuration.

    Args:
        molecule_name: "H2", "H4", "HeH+", "LiH", "BeH2", "H2O" supported.
        param: [distance1, distance2, ... angle1, angle2] or just a real number.
        **kwargs: Keyword arguments for ``Molecule`` object initialization

    Returns:
        molecule: an example of ``Molecule`` object.

    """
    basis = 'sto-3g' if 'basis' not in kwargs else kwargs['basis']
    if 'basis' in kwargs:
        kwargs.pop('basis')
    multiplicity = 1 if 'multiplicity' not in kwargs else kwargs['multiplicity']
    if 'multiplicity' in kwargs:
        kwargs.pop('multiplicity')
    charge = 0

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
        raise NameError(f"{molecule_name} is not in the example list.")

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
    if driver.lower() == 'psi4':
        return run_psi4(molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci, **kwargs)
    elif driver.lower() == 'pyscf':
        from openfermionpyscf import run_pyscf
        try:
            return run_pyscf(molecule, run_scf, run_mp2, run_cisd, run_ccsd, run_fci, **kwargs)
        except AttributeError as e:
            msg = f"\nConsider updating the class PyscfMolecularData with"\
                  f"https://github.com/snow0369/OpenFermion-PySCF/blob/master/openfermionpyscf/_pyscf_molecular_data.py."
            raise AttributeError(str(e) + msg)
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
    """This function runs a Psi4 calculation.
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

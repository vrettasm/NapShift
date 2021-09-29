"""
This module includes the main class that handles the calculation
of the hydrogen bonds.
"""

# Python import(s).
from collections import namedtuple
import numpy as np

# BIO-python import(s).
from Bio.PDB.vectors import Vector, calc_angle

# Private import(s).
from src.chemical_shifts.auxiliaries import ACCEPTED_RES

# Add documentation to the NamedTuple.
__pdoc__ = {}

# Module level declaration,
BondInfo = namedtuple("BondInfo", ["res_name", "res_id", "H_", "O_"])

# Add documentation for the fields.
__pdoc__["BondInfo.res_name"] = "Residue name (three letter code) "\
                                "that contains the hydrogen bond."
__pdoc__["BondInfo.res_id"] = "Residue ID (from the PDB file)."
__pdoc__["BondInfo.H_"] = "Information about the NH-bond: "\
                          "(distance, cos(phi), cos(psi), 1.0)."
__pdoc__["BondInfo.O_"] = "Information about the CO-bond: "\
                          "(distance, cos(phi), cos(psi), 1.0)."


class HBond(object):
    """
    This class creates an object that handles hydrogen
    bonds formation between atoms (amino-acid chains).

        Note: Naming systems used (BMRB / PDB).
    """

    # Side-chain Hydrogen atoms: Asparagine (ASN):
    _ASN_H = {"HD21": "ND2", "HD22": "ND2",
              "1HD2": "ND2", "2HD2": "ND2"}

    # Side-chain Hydrogen atoms: Arginine (ARG):
    _ARG_H = {"HH11": "NH1", "HH12": "NH1",
              "HH21": "NH2", "HH22": "NH2",
              "1HH1": "NH1", "2HH1": "NH1",
              "1HH2": "NH2", "2HH2": "NH2"}

    # Side-chain Hydrogen atoms: Glutamine (GLN):
    _GLN_H = {"HE21": "NE2", "HE22": "NE2",
              "1HE2": "NE2", "2HE2": "NE2"}

    # Side-chain Hydrogen atoms: Lysine (LYS):
    _LYS_H = {"HZ1": "NZ", "HZ2": "NZ", "HZ3": "NZ",
              "1HZ": "NZ", "2HZ": "NZ", "3HZ": "NZ"}

    # Side-chain Oxygen atoms with their corresponding
    # Carbon atoms (used in hydrogen acceptor atoms).
    _ASN_O = {"OD1": "CG"}
    _ASP_O = {"OD1": "CG", "OD2": "CG"}
    _GLN_O = {"OE1": "CD"}
    _GLU_O = {"OE1": "CD", "OE2": "CD"}
    _SER_O = {"OG": "CB"}
    _THR_O = {"OG1": "CB"}
    _TYR_O = {"OH": "CZ"}

    # Object variables.
    __slots__ = ("back_H", "side_H", "side_O", "rule", "_verbose")

    def __init__(self, verbose=False):
        """
        Constructs an object that will handle the detections
        of hydrogen bonds in residue-chains.

        :param verbose: boolean flag. If true it will display
        more information while running.
        """

        # Backbone (NHx) Hydrogen names.
        self.back_H = ["H",
                       "H1", "H2", "H3",
                       "1H", "2H", "3H"]

        # Combine all side-chain (H) atoms.
        self.side_H = {**HBond._ASN_H, **HBond._ARG_H,
                       **HBond._GLN_H, **HBond._LYS_H}

        # Combine all side-chain (O) atoms.
        self.side_O = {**HBond._ASN_O, **HBond._ASP_O,
                       **HBond._GLN_O, **HBond._GLU_O,
                       **HBond._SER_O, **HBond._THR_O,
                       **HBond._TYR_O}

        # Default criteria for hydrogen bonds formation:
        # 1) Distances between atoms [L: Angstroms]:
        #  1.1)   donor  atom 'D' and acceptor atom 'A'.
        #  1.2) hydrogen atom 'H' and acceptor atom 'A'.
        #
        # 2) Angles between atoms (U: Degrees):
        #  2.1) (Base-Acceptor) B-A ... (Hydrogen)  H.
        #  2.2) (Acceptor) A ... (Hydrogen-Donor) H-D.
        self.rule = {"D-A": 3.90, "H-A": 2.50,
                     "PHI": 90.0, "PSI": 90.0}

        # Add the verbose flag.
        self._verbose = verbose
    # _end_def_

    @property
    def verbose(self):
        """
        Accessor (getter) of the verbose flag.

        :return: _verbose.
        """
        return self._verbose
    # _end_def_

    @verbose.setter
    def verbose(self, new_value):
        """
        Accessor (setter) of the verbose flag.

        :param new_value: (bool).
        """

        # Check for correct type.
        if isinstance(new_value, bool):

            # Assign the new value.
            self._verbose = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Verbose flag should be bool: {type(new_value)}.")
        # _end_if_
    # _end_def_

    def __call__(self, chain=None):
        """
        Compute the hydrogen bonds of the given input of
        amino-acid (residue) chain. These are determined
        with the internal criteria (rules) of the angles
        and distances between the (H+) and (O-) atoms.

        :param chain: Residue chain (amino-acids)

        :return: List with all the estimated hydrogen bonds.
        """

        # Return list with the h-bonds.
        h_bonds = []

        # Localize the append method.
        h_bonds_append = h_bonds.append

        # Check for empty input.
        if chain is None:
            return h_bonds
        # _end_if_

        # Process each residue in the chain.
        for res_i in chain.get_residues():

            # Get the residue name in three letter code.
            # Ensure there are no spaces before / after.
            res_name = str(res_i.get_resname()).strip()

            # Get the residue id from the PDB.
            res_id = res_i.get_id()[1]

            # Initialize the parameter lists.
            bond_H_vec = [0.0, 0.0, 0.0, 0.0]
            bond_O_vec = [0.0, 0.0, 0.0, 0.0]

            # Make sure the residue is in the accepted list.
            # This is to avoid parsing DNA or RNA molecules.
            if res_name not in ACCEPTED_RES:
                # Add an empty tuple to the list.
                h_bonds_append(BondInfo(res_name, res_id, bond_H_vec, bond_O_vec))

                # Skip to the next residue.
                continue
            # _end_if_

            # Get the backbone 'N' / 'C' / 'O' and 'H' atoms.
            atom_N, atom_C, atom_O, atom_H = None, None, None, []

            try:
                # Get the Nitrogen.
                atom_N = res_i["N"]

                # Get the Carbon.
                atom_C = res_i["C"]

                # Get the Oxygen.
                atom_O = res_i["O"]

                # We could have multiple hydrogen atoms
                # bounded with atom 'N' (i.e NH2, NH3):
                for atom in res_i.get_atoms():

                    # Find the backbone Hydrogen atoms.
                    if atom.name in self.back_H:
                        atom_H.append(atom)
                    # _end_if_

                # _end_for_
            except (AttributeError, KeyError):

                # Add an empty tuple to the list.
                h_bonds_append(BondInfo(res_name, res_id, bond_H_vec, bond_O_vec))

                # Skip to the next residue.
                continue
            # _end_try_

            # H-Bond auxiliaries (1/2): N-H ... O
            bond_NH = {"r_min": np.inf,
                       "N": None, "H": None,
                       "O": None, "C": None}

            # H-Bond auxiliaries (2/2): C=0 ... H
            bond_CO = {"r_min": np.inf,
                       "N": None, "H": None,
                       "O": None, "C": None}

            # Scan the whole residue chain from the start.
            for res_j in chain.get_residues():

                # Get the name of the "j-th" residue.
                RES_NAME_J = str(res_j.get_resname()).strip()

                # Make sure the residue is in the accepted list.
                if RES_NAME_J not in ACCEPTED_RES:
                    # Skip to the next residue.
                    continue
                # _end_if_

                # Get the 'O', 'C', 'N' atoms.
                try:
                    atom_N_j = res_j["N"]
                    atom_O_j = res_j["O"]
                    atom_C_j = res_j["C"]
                except (AttributeError, KeyError):
                    # Skip to the next residue.
                    continue
                # _end_try_

                # Store the H/O atoms.
                back_chain_H, side_chain_H, side_chain_O = [], [], []

                # Find the Hydrogen and Oxygen.
                for atom_j in res_j.get_atoms():

                    if atom_j.name in self.back_H:
                        back_chain_H.append(atom_j)
                    # _end_if_

                    if atom_j.name in self.side_H:
                        side_chain_H.append(atom_j)
                    # _end_if_

                    if atom_j.name in self.side_O:
                        side_chain_O.append(atom_j)
                    # _end_if_

                # _end_for_

                # Because the 'N-Hx' can have several 'H' atoms
                # we compare distances of the j-th 'O' with all
                # the i-th 'H' atoms.
                for atom_h in atom_H:

                    # Distance.
                    d_h = atom_h - atom_O_j

                    # Find the minimum.
                    if np.isfinite(d_h) and (d_h < bond_NH["r_min"]):
                        # Store the current minimum.
                        bond_NH["r_min"] = d_h

                        # Store the relevant atoms.
                        bond_NH["N"] = atom_N
                        bond_NH["H"] = atom_h

                        # Store the relevant atoms.
                        bond_NH["O"] = atom_O_j
                        bond_NH["C"] = atom_C_j
                    # _end_if_

                    # Search for side-chain 'O' atoms.
                    for atom_o in side_chain_O:

                        # Distance.
                        d_h = atom_o - atom_h

                        # Find the minimum.
                        if np.isfinite(d_h) and (d_h < bond_NH["r_min"]):

                            try:
                                # Get the connected 'C' atom.
                                side_C = res_j[self.side_O[atom_o.name]]

                                # Store the current minimum.
                                bond_NH["r_min"] = d_h

                                # Store the relevant atoms.
                                bond_NH["N"] = atom_N
                                bond_NH["H"] = atom_h

                                # Store the relevant atoms.
                                bond_NH["O"] = atom_o
                                bond_NH["C"] = side_C
                            except (AttributeError, KeyError):
                                # If the atom does not exist
                                # simply ignore it and pass.
                                pass
                            # _end_try_

                        # _end_if_

                    # _end_for_

                # _end_for_

                # First find the distances between the i-th 'O'
                # and the backbone j-th 'H' atoms.
                for atom_l in back_chain_H:

                    # Distance.
                    d_k = atom_l - atom_O

                    # Find the minimum.
                    if np.isfinite(d_k) and (d_k < bond_CO["r_min"]):
                        # Store the current minimum.
                        bond_CO["r_min"] = d_k

                        # Store the pair of atoms.
                        bond_CO["H"] = atom_l
                        bond_CO["N"] = atom_N_j

                        # Store the relevant atoms.
                        bond_CO["O"] = atom_O
                        bond_CO["C"] = atom_C
                    # _end_if_

                # _end_for_

                # For specific residues check the side-chains.
                if RES_NAME_J in {"ASN", "ARG", "GLN", "LYS"}:

                    # Go through all the side-chain 'H'.
                    for atom_l in side_chain_H:

                        try:
                            # Distance.
                            d_k = atom_l - atom_O

                            # Find the minimum.
                            if np.isfinite(d_k) and (d_k < bond_CO["r_min"]):
                                # Extract the side-chain -N-
                                side_N = res_j[self.side_H[atom_l.name]]

                                # Store the current minimum.
                                bond_CO["r_min"] = d_k

                                # Store the pair of atoms.
                                bond_CO["H"] = atom_l
                                bond_CO["N"] = side_N

                                # Store the relevant atoms.
                                bond_CO["O"] = atom_O
                                bond_CO["C"] = atom_C
                            # _end_if_
                        except (AttributeError, KeyError):
                            # If the atom does not exist
                            # simply ignore it and pass.
                            pass
                        # _end_try_

                    # _end_for_

                # _end_if_

            # _end_for_

            # Check if the criteria for the distances are satisfied.
            if (bond_NH["r_min"] < self.rule["H-A"]) \
                    and (atom_N - bond_NH["O"] < self.rule["D-A"]):

                # Get the "N-H" coordinates.
                vec_N = Vector(bond_NH["N"].get_coord())
                vec_H = Vector(bond_NH["H"].get_coord())

                # Get the "C-O" coordinates.
                vec_C = Vector(bond_NH["C"].get_coord())
                vec_O = Vector(bond_NH["O"].get_coord())

                # This angle is in radians.
                phi = calc_angle(vec_N, vec_H, vec_O)

                # This angle is in radians.
                psi = calc_angle(vec_C, vec_O, vec_H)
                
                # Check both angles.
                if (np.rad2deg(phi) > self.rule["PHI"]) and\
                        (np.rad2deg(psi) > self.rule["PSI"]):
                    bond_H_vec = [bond_NH["r_min"], *np.cos([phi, psi]), 1.0]
                # _end_if_

            # _end_if_

            # Check if the criteria for the distances are satisfied.
            if (bond_CO["r_min"] < self.rule["H-A"]) \
                    and (atom_C - bond_CO["H"] < self.rule["D-A"]):

                # Get the "N-H" coordinates.
                vec_N = Vector(bond_CO["N"].get_coord())
                vec_H = Vector(bond_CO["H"].get_coord())

                # Get the "C-O" coordinates.
                vec_C = Vector(bond_CO["C"].get_coord())
                vec_O = Vector(bond_CO["O"].get_coord())

                # This angle is in radians.
                phi = calc_angle(vec_N, vec_H, vec_O)

                # This angle is in radians.
                psi = calc_angle(vec_C, vec_O, vec_H)

                # Check both angles.
                if (np.rad2deg(phi) > self.rule["PHI"]) and\
                        (np.rad2deg(psi) > self.rule["PSI"]):
                    bond_O_vec = [bond_CO["r_min"], *np.cos([phi, psi]), 1.0]
                # _end_if_

            # _end_if_

            # Add the complete named-tuple to the list.
            h_bonds_append(BondInfo(res_name, res_id, bond_H_vec, bond_O_vec))
        # _end_for_

        # Return the list with all the bonds.
        return h_bonds
    # _end_def_

    @property
    def rules(self):
        """
        Accessor of the hydrogen bond rules.

        :return: Dictionary with the current
        distances and angles.
        """
        return self.rule
    # _end_def_

    def update_rule(self, new_value, code_str=None):
        """
        Update the values of the hydrogen bond rules.

        :param new_value: distance (Angstrom) or angle (degrees).

        :param code_str: Rule codes are ["D-A", "H-A", "PHI", "PSI"].

        :return: None.
        """

        # Check if the given code is correct.
        if code_str not in ["D-A", "H-A", "PHI", "PSI"]:
            raise ValueError(f"{self.__class__.__name__}:"
                             f" Unknown code string: {code_str}.")
        # _end_if_

        # Make sure the new_value is float.
        new_value = float(new_value)

        # Check if it is a "distance" rule.
        if (code_str == "D-A") or (code_str == "H-A"):
            # Accept only positive values.
            if new_value > 0.0:
                self.rule[code_str] = new_value
            else:
                # Check for verbose.
                if self.verbose:
                    print(f" Wrong update distance value: {new_value}")
                # _end_if_
            # _end_if_
        # _end_if_

        # Check if it is an "angle" rule.
        if (code_str == "PHI") or (code_str == "PSI"):
            # Accept only values in range [0.0 - 360.0].
            if 0.0 <= new_value <= 360.0:
                self.rule[code_str] = new_value
            else:
                # Check for verbose.
                if self.verbose:
                    print(f" Wrong update angle value: {new_value}")
                # _end_if_
        # _end_if_
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the
        object. This will include its id() along with its rules
        for forming h-bonds.

        :return: a string representation of a HBond object.
        """
        return f"HBond Id({id(self)}): Rules={self.rules}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: HBond().
        """
        return f"HBond(verbose={self._verbose})"
    # _end_def_

# _end_class_

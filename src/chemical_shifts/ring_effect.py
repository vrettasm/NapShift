"""
This module includes the main class that handles the detection
of the aromatic rings.
"""

# Python import(s).
from collections import namedtuple
import numpy as np

# Private import(s).
from src.chemical_shifts.auxiliaries import ACCEPTED_RES, RES_3_TO_1

# Add documentation to the NamedTuple.
__pdoc__ = {}

# Module level declaration,
RingInfo = namedtuple("RingInfo", ["res_name", "res_id", "coord"])

# Add documentation for the fields.
__pdoc__["RingInfo.res_name"] = "Residue name (one letter code) "\
                                "that contains the aromatic ring."
__pdoc__["RingInfo.res_id"] = "Residue ID (from the PDB file)."
__pdoc__["RingInfo.coord"] = "Coordinates (x,y,z) of the ring centroid."


class RingEffects(object):
    """
    This class creates an object that will identify the rings
    (aromatic amino acids) and will also detect the atoms,
    from a given input chain, that are affected.

        Note: Naming systems used (BMRB / PDB).
    """

    # Backbone (NHx) Hydrogen names.
    _back_H = ("H", "H1", "H2", "H3", "1H", "2H", "3H")

    # Aromatic Amino Acids (with 6 ring atoms).
    _AAA6 = {"PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
             "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
             "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2")}

    # Aromatic Amino Acids (with 5 ring atoms).
    _AAA5 = {"HIS": ("CG", "CE1", "CD2", "ND1", "NE2"),
             "TRP": ("CG", "CD1", "CD2", "CE2", "NE1")}

    # Object variables.
    __slots__ = ("d_ring", "five_atoms_rings", "rings", "_verbose")

    # Constructor.
    def __init__(self, d_ring=1.0, include_five_atoms=False, verbose=False):
        """
        Constructs an object that will handle the detections of rings in the
        chain and the atoms that are affected by them.

        :param d_ring: distance threshold [L: Angstrom].

        :param include_five_atoms: (bool) flag for the Tryptophan amino-acid
        ring (with the five atoms).

        :param verbose: (bool) if true it will display more information while
        running.
        """

        # Check for the correct type.
        if isinstance(d_ring, float):
            # Check for positive value.
            if d_ring > 0.0:
                # Assign the distance.
                self.d_ring = d_ring
            else:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Distance value should be positive: {d_ring}.")
            # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Distance value should be float: {type(d_ring)}.")
        # _end_if_

        # Check for the correct type.
        if isinstance(include_five_atoms, bool):
            # Assign the boolean flag.
            self.five_atoms_rings = include_five_atoms
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Five atoms ring flag should be bool: {type(include_five_atoms)}.")
        # _end_if_

        # Make an empty list of rings information.
        self.rings = []

        # Add the verbose flag.
        self._verbose = verbose
    # _end_def_

    def get_rings(self):
        """
        Accessor of the rings list.

        :return: the list with the ring information.
        """
        return self.rings
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
            self._verbose = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Verbose flag should be bool: {type(new_value)}.")
        # _end_if_
    # _end_def_

    @property
    def distance(self):
        """
        Accessor (getter) of the distance value that sets
        a threshold on the effect that the ring can have.

        :return: distance threshold value [L: Angstrom]
        """
        return self.d_ring
    # _end_def_

    @distance.setter
    def distance(self, new_value):
        """
        Accessor (setter) of the distance value that sets
        a threshold on the effect that the ring can have.

        :param new_value: (float) of the distance beyond
        which the ring effect is negligible.
        """

        # Check for the correct type.
        if isinstance(new_value, float):
            # Check for positive value.
            if new_value > 0.0:
                # Assign the distance.
                self.d_ring = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Distance value should be positive: {new_value}.")
            # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Distance value should be float: {type(new_value)}.")
        # _end_if_
    # _end_def_

    @property
    def include_five_atoms(self):
        """
        Accessor (getter) of the boolean flag that
        indicates if we consider both rings of the
        Tryptophan amino-acid.

        :return: boolean flag.
        """
        return self.five_atoms_rings
    # _end_def_

    @include_five_atoms.setter
    def include_five_atoms(self, new_value):
        """
        Accessor (setter) of the boolean flag that
        indicates if we consider both rings of the
        Tryptophan amino-acid.

        :param new_value: new boolean flag.
        """

        # Check for the correct type.
        if isinstance(new_value, bool):
            self.five_atoms_rings = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Five atoms ring flag should be bool: {type(new_value)}.")
        # _end_if_

    # _end_def_

    def find_rings_in_chain(self, chain=None):
        """
        Scan the input chain and look for aromatic rings.
        In the case of the Tryptophan, we can optionally
        consider both rings.

        :param chain: input amino-acid chain.

        :return: None.
        """

        # Check for empty input.
        if chain is None:
            return None
        # _end_if_

        # Make sure the list is empty.
        self.rings = []

        # Process each residue in the chain.
        for res_i in chain.get_residues():

            # Get the residue name in three letter code.
            res_name = str(res_i.get_resname()).strip()

            # Accepted list of residue names.
            if res_name not in ACCEPTED_RES:
                # Skip to the next residue.
                continue
            # _end_if_

            # Get the residue id from the PDB.
            res_id = res_i.get_id()[1]

            # Aromatic amino-acids.
            if res_name in RingEffects._AAA6:

                try:
                    # XYZ coordinates vector.
                    xyz_coord = np.zeros(3)

                    # Sum the coordinates of the ring atoms.
                    for atom in RingEffects._AAA6[res_name]:
                        xyz_coord += res_i[atom].get_coord()
                    # _end_for_

                    # Get the average.
                    xyz_coord /= 6.0

                    # Add the ring centre coordinates.
                    self.rings.append(RingInfo(RES_3_TO_1[res_name],
                                               res_id, xyz_coord))
                except KeyError as e6:

                    # Check for verbose.
                    if self.verbose:
                        print(f" Atom {e6} not found in {res_name}.")
                    # _end_if_

                    pass
                # _end_try_

            # _end_if_

            # Check for 5-atoms rings.
            if self.five_atoms_rings:

                if res_name in RingEffects._AAA5:

                    try:
                        # XYZ coordinates vector.
                        xyz_coord = np.zeros(3)

                        # Sum the coordinates of the ring atoms.
                        for atom in RingEffects._AAA5[res_name]:
                            xyz_coord += res_i[atom].get_coord()
                        # _end_for_

                        # Get the average.
                        xyz_coord /= 5.0

                        # Add the ring centre coordinates.
                        self.rings.append(RingInfo(RES_3_TO_1[res_name],
                                                   res_id, xyz_coord))
                    except KeyError as e5:

                        # Check for verbose.
                        if self.verbose:
                            print(f" Atom {e5} not found in {res_name}.")
                        # _end_if_

                        # Leave without doing nothing.
                        pass
                    # _end_try_

                # _end_if_

            # _end_if_

        # _end_for_

        # Print information (only if empty).
        if self.verbose and not self.rings:
            print(" Info: No aromatic rings were found in the input chain.")
        # _end_if_

    # _end_def_

    def check_effect(self, chain=None, find_rings=True, exclude_self=True):
        """
        Checks the input residue-chain to identify which atoms are affected
        by the aromatics rings.

        :param chain: input amino-acid chain.

        :param find_rings: if True, it will automatically find the rings of
        the same input chain.

        :param exclude_self: if True, it will automatically exclude all the
        CB atoms of its own ring. These atoms are usually very close to the
        center of the ring.

        :return: a list with atoms that are affected (given the threshold
        distance that we have already define).
        """

        # Check for empty input.
        if chain is None:
            return None
        # _end_if_

        # Check automatically for the rings.
        if find_rings:
            # This will override and clear
            # any previously detected rings.
            self.find_rings_in_chain(chain)
        # _end_if_

        # Affected atoms list.
        affected_atoms = []

        # Check if we have found rings.
        if not self.rings:

            # Avoid displaying the same message twice.
            if not find_rings:
                # Show a warning that there are no rings.
                print(" Info: No aromatic rings were found.")
            # _end_if_

            # Exit without scanning the chain.
            return affected_atoms
        # _end_if_

        # Localize the Euclidean norm.
        euclidean_norm = lambda x: np.sqrt(np.dot(x, x))

        # Process each residue in the chain.
        for res_i in chain.get_residues():

            # Get the residue name (three letter code).
            res_name = str(res_i.get_resname()).strip()

            # Accepted list of residue names.
            if res_name not in ACCEPTED_RES:
                # Skip to the next residue.
                continue
            # _end_if_

            # Get the residue id from the PDB.
            res_id = res_i.get_id()[1]

            # Atom (names) list (of interest).
            # Initialize with each residue, because
            # the number of hydrogen atoms may vary.
            atom_list = ["N", "C", "CA", "CB", "HA"]

            try:
                # We could have multiple hydrogen atoms
                # bounded with atom 'N' (i.e NH2, NH3):
                for atom in res_i.get_atoms():

                    # Find the backbone Hydrogen atoms.
                    if atom.name in RingEffects._back_H:
                        atom_list.append(atom.name)
                    # _end_if_

                # _end_for_

                # Check affected atoms.
                for atom in atom_list:

                    # Get the coordinates of the current atom.
                    atom_coordinates = res_i[atom].get_coord()

                    # Check against all the rings in the list.
                    for ring in self.rings:

                        # Skip "self ring effects" for the 'CB'
                        # atoms only.
                        if exclude_self and (atom == "CB") and\
                                (res_id == ring.res_id):
                            continue
                        # _end_if_

                        # Distance between coordinates.
                        diff = atom_coordinates - ring.coord

                        # Equivalent to Euclidean norm(diff).
                        dist = euclidean_norm(diff)

                        # Check the threshold.
                        if np.isfinite(dist) and dist <= self.d_ring:
                            # Include the following information:
                            # 1) residue name, 2) residue id,
                            # 3) atom name and 4) distance [Angstrom].
                            affected_atoms.append((RES_3_TO_1[res_name],
                                                   res_id, atom, dist))
                        # _end_if_

                    # _end_for_
                    
                # _end_for_
            except (AttributeError, KeyError):
                # Skip to the next residue.
                continue
            # _end_try_
        # _end_for_

        # Return the list with the affected atoms.
        return affected_atoms
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(.), along with its distance threshold
        value and five atom ring flag.

        :return: a string representation of a RingEffects object.
        """

        # Local import of new line.
        from os import linesep as new_line

        # Return the f-string.
        return f" RingEffects Id({id(self)}): {new_line}" \
               f" Distance={self.d_ring} {new_line}" \
               f" Include-Five-Atom-Rings={self.five_atoms_rings}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: RingEffects().
        """
        return f"RingEffects(d_ring={self.d_ring}," \
               f"include_five_atoms={self.five_atoms_rings}," \
               f"verbose={self.verbose})"
    # _end_def_

# _end_class_

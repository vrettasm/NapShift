"""
This module includes the main class that handles
the data generation from the (input) PDB files.
"""

# Python import(s).
from pathlib import Path
import numpy as np
from pandas import DataFrame
from numba import njit

# BIO-python import(s).
from Bio.PDB.PDBParser import PDBParser

# Private import(s).
from src.chemical_shifts.auxiliaries import (RES_3_TO_1, BLOSUM,
                                             TARGET_ATOMS, ACCEPTED_RES,
                                             get_sequence, modify_models_OX)

from src.chemical_shifts.hydrogen_bond import HBond
from src.chemical_shifts.ring_effect import RingEffects


class InputVector(object):
    """
    This class creates an input vector for the artificial neural network.
    The hydrogen bonds, and aromatic rings calculations are performed by
    class objects. We usually don't want to change these often once they
    are fixed.
    """

    # Hydrogen bond calculator (default settings):
    hydrogen_bond_calc = HBond()
    """
    The default setting for the hydrogen bond calculator are:

        > "D-A": 3.90, "H-A": 2.50
        > "PHI": 90.0, "PSI": 90.0

    Note: The distance units are in "Angstrom" and the angle
    units are in "degrees".
    """

    # Aromatic ring effects detector.
    aromatic_ring_calc = RingEffects(d_ring=3.0,
                                     include_five_atoms=True)
    """
    The default distance threshold is set equal to 3 Angstrom.
    It is also enabled to include five atom rings (default).
    """

    # Object variables.
    __slots__ = ("blosum_id", "blosum", "hydrogen_bonds", "data_type",
                 "aromatic_rings")

    # Constructor.
    def __init__(self, blosum_id=62, include_hydrogen_bonds=False,
                 check_aromatic_rings=True, data_type=np.float32):
        """
        Constructs an object that will create the input vectors from
        a given PDB (protein) file.

        :param blosum_id: (integer) value that defines the version of
        the block substitution matrix.

        :param include_hydrogen_bonds: (boolean) flag. If true it will
        detect and include all the hydrogen bonds information as part
        of the input vectors.

        :param check_aromatic_rings: (boolean) flag. If true it will
        detect the target atoms that are affected by an aromatic ring.
        This will flag them out so that we will not use them while
        training/testing of the ANN.

        :param data_type: This is the datatype of the input vector.
        This will affect the size (unit is MB) of the return vectors.
        Default here is set to float32.
        """

        # Check the version of the matrix.
        if str(blosum_id) in BLOSUM:
            # Accept the BLOSUM ID.
            self.blosum_id = blosum_id

            # Get the right version.
            self.blosum = BLOSUM[str(blosum_id)]
        else:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Available BLOSUM versions are {list(BLOSUM.keys())}: "
                             f"{blosum_id}.")
        # _end_if_

        # Boolean flag. If true the class will estimate
        # the hydrogen bonds of the given model, before
        # it creates the return vector.
        if isinstance(include_hydrogen_bonds, bool):
            # Assign the boolean flag.
            self.hydrogen_bonds = include_hydrogen_bonds
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Include hydrogen bonds flag should be bool: "
                            f"{type(include_hydrogen_bonds)}.")
        # _end_if_

        # Boolean flag. If true the class will detect which
        # of the model atoms are affected by aromatic rings
        # and will return a list of them.
        if isinstance(check_aromatic_rings, bool):
            # Assign the boolean flag.
            self.aromatic_rings = check_aromatic_rings
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Check aromatic rings flag should be bool: "
                            f"{type(check_aromatic_rings)}.")
        # _end_if_

        # Make sure "data_type" is type.
        if isinstance(data_type, type):
            # Accept only specific floating point types.
            # > np.half   == np.float16
            # > np.single == np.float32
            # > np.double == np.float64
            if data_type in {np.float16, np.float32, np.float64}:
                # Accept the datatype.
                self.data_type = data_type
            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"Datatype should be float(16/32/64): {data_type}.")
            # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Datatype should be a type: {type(data_type)}.")
        # _end_if_
    # _end_def_

    @property
    def blosum_version(self):
        """
        Accessor (getter) of the version of the BLOSUM.

        :return: integer value.
        """
        return self.blosum_id
    # _end_def_

    @blosum_version.setter
    def blosum_version(self, new_value):
        """
        Accessor (setter) of the BLOSUM.

        :param new_value: new integer version.
        """

        # Check the version of the accepted matrix.
        if str(new_value) in BLOSUM:
            # Update the BLOSUM ID.
            self.blosum_id = new_value

            # Update the BLOSUM.
            self.blosum = BLOSUM[str(new_value)]
        else:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Available BLOSUM versions {list(BLOSUM.keys())}:"
                             f"{new_value}.")
        # _end_if_
    # _end_def_

    @property
    def include_hydrogen_bonds(self):
        """
        Accessor (getter) of the boolean flag that
        indicates if we include the hydrogen bonds
        in the formulation of the input vector.

        :return: boolean flag.
        """
        return self.hydrogen_bonds
    # _end_def_

    @include_hydrogen_bonds.setter
    def include_hydrogen_bonds(self, new_value):
        """
        Accessor (setter) of the boolean flag.

        :param new_value: new boolean flag.
        """

        # Check for correct type.
        if isinstance(new_value, bool):
            # Assign the new value.
            self.hydrogen_bonds = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Hydrogen bonds flag should be bool: {type(new_value)}.")
        # _end_if_
    # _end_def_

    @property
    def check_aromatic_rings(self):
        """
        Accessor (getter) of the boolean flag that
        indicates if we detect the aromatic rings.

        :return: boolean flag.
        """
        return self.aromatic_rings
    # _end_def_

    @check_aromatic_rings.setter
    def check_aromatic_rings(self, new_value):
        """
        Accessor (setter) of the boolean flag.

        :param new_value: new boolean flag.
        """

        # Check for correct type.
        if isinstance(new_value, bool):
            # Assign the new value.
            self.aromatic_rings = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Aromatic rings flag should be bool: {type(new_value)}.")
        # _end_if_
    # _end_def_

    @staticmethod
    @njit(fastmath=True)
    def sine_cosine(angle):
        """
        Auxiliary function that returns the sin(x) / cos(x) for
        a given input angle 'x'. This method is using Numba for
        speed up. Upon testing the performance we found that we
        average a ~23x speed up comparing to using only numpy.

        NOTE:  We don't make any checks on the input angle as we
        assume that it has already been checked before this call.
        The whole point is to speed up the repeated calls to the
        trigonometric functions.

        :param angle: the angle (in degrees) that we want to get
        the sine / cosine.

        :return: sin(angle), cos(angle)
        """
        # First convert to radians.
        angle = np.deg2rad(angle)

        # Return the results.
        return np.sin(angle), np.cos(angle)
    # _end_def_

    @staticmethod
    def save_auxiliary(f_id, rec_data, kind=None, output=None):
        """
        This auxiliary (static)  function will save the auxiliary
        bi-products of the input vector construction, such as the
        hydrogen bonds, torsion angles and aromatic rings.

        :param f_id: File id. Usually the PDB-ID is ok to be used
        as a filename to identify the contents.

        :param rec_data: The data we want to save. Usually it is
        a list of things (tuples, dicts, etc.)

        :param kind: This is the type / kind of data that we are
        saving. It can only be of four types: 1) "t_peptides",
        2) "h_bonds", 3) "a_rings" and 4) "t_angles". Anything
        else will force the method to raise an exception.

        :param output: This is the main (parent) output directory
        where the data will be saved.

        :return: None.
        """

        # Set the local variables according to the
        # kind of data that we have to save.
        if kind == "t_peptides":
            f_ext, folder_name = "tripep", "Tri_Peptides"

        elif kind == "h_bonds":
            f_ext, folder_name = "bonds", "Hydrogen_Bonds"

        elif kind == "a_rings":
            f_ext, folder_name = "rings", "Aromatic_Rings"

        elif kind == "t_angles":
            f_ext, folder_name = "angles", "Torsion_Angles"

        else:
            raise ValueError(f" Unknown type of data: {kind}.")
        # _end_if_

        # Check for the output.
        if output is None:
            output = Path.cwd()
        # _end_if_

        # Create an output filename.
        output_path = Path(output / folder_name)

        # This will be true only once (for each kind).
        if not output_path.is_dir():
            output_path.mkdir(parents=True)
        # _end_if_

        # Convert to DataFrame.
        df = DataFrame(data=rec_data)

        # Save to csv file.
        df.to_csv(Path(output_path / f"{f_id}.{f_ext}"), header=False)
    # _end_def_

    # Auxiliary method.
    def _zero_padding(self, x_in, n_peptides, terminal="front"):
        """
        This auxiliary method is used to add the front and back
        terminal vectors, using zero padding. The method can be
        used for any (odd) number of peptides (3-, 5-, 7-, etc.)

        :param x_in: This is the input that will be used to make
        the front or back terminal poly-peptide entries.

        :param n_peptides: This is the number (int) of the poly -
        peptides that we are constructing. It should be strictly
        an odd number.

        :param terminal: This is the type of the terminal. It
        accepts only two (string) values "front" / "back".

        :return: a list with the front or back terminal entries.
        """

        # Quick sanity check.
        if terminal not in {"front", "back"}:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Zero padding type can be 'front' | 'back': {terminal}.")
        # _end_if_

        # Extract the data from the input.
        x_peptides = x_in["poly-peptides"]
        x_targets = x_in["targets"]
        x_vector = x_in["vector"]

        # Total length of the vector.
        K = x_vector.size

        # Number of elements (per peptide).
        L = K // int(n_peptides)

        # Declare the return list.
        list_out = []

        # Number of required padded elements.
        number_of_pads = int(n_peptides) // 2

        # Iterate to get all the zero padding
        # elements (front or back).
        for i in range(number_of_pads):
            # Create a temporary vector.
            t_vector = np.zeros_like(x_vector)

            # Make a temporary peptides copy.
            t_peptides = x_peptides.copy()

            # Compute the index where we want
            # to start copying elements from.
            kappa = L * (i + 1)

            # Switch according to the front/back side.
            if terminal == "front":
                # Copy the vector elements.
                t_vector[kappa:] = x_vector[:K-kappa].copy()

                # Rotate the elements forwards.
                t_peptides = t_peptides[-(i+1):] + t_peptides[:-(i+1)]

                # Mark the empty entries with None.
                for j in range(i+1):
                    t_peptides[j] = (None, '-', None)
                # _end_for_
            else:
                # Copy the vector elements.
                t_vector[:K-kappa] = x_vector[kappa:].copy()

                # Rotate the elements backwards.
                t_peptides = t_peptides[(i+1):] + t_peptides[:(i+1)]

                # Mark the empty entries with None.
                for j in range(i+1):
                    t_peptides[-(j+1)] = (None, '-', None)
                # _end_for_
            # _end_if_

            # Add it to the return list.
            list_out.append({"poly-peptides": t_peptides,
                             "targets": x_targets,
                             "vector": t_vector})
        # _end_for_

        # Return the list.
        return list_out

    # _end_def_

    # Main functionality.
    def get_data(self, f_path, n_peptides=3, all_models=False,
                 save_output_path=None, verbose=False):
        """
        This is the main function of the 'InputVector' class. It accepts
        as input a "Path(/path/to/the/PDB)" and returns a  list with all
        the information that is extracted.  This can be used as input to
        the ANN for predicting the chemical shift values of specific pre
        determined, atoms (e.g. "N", "C", "CA", "CB", "H", "HA").

        :param f_path: Path of the (protein) PDB file.

        :param n_peptides: Number of peptides to consider for the input
        vectors. By default, it considers tri-peptides.

        :param all_models: (bool) flag. If True the method processes
        all the models in the PDB file, otherwise only the first model.

        :param save_output_path: (Path/String) If given it will be the
        output path where all the auxiliary data will be saved.

        :param verbose: (bool) flag. If True it prints more info on the
        screen. The default is set to False.

        :return: A dictionary with (data + sequence) for each processed
        model from the PDB file. The data + sequence are given as:

            1) a list with the following information:
                1.1) poly-peptides that have been generated
                     (index + three-letter code)
                1.2) vector with the input values
                1.3) a list with the target atoms that are
                     available for each vector.

            2) the (modified) amino-acid sequence in string.
        """

        # Make sure peptides is int.
        n_peptides = int(n_peptides)

        # Make sure the input peptide is a positive number.
        if n_peptides <= 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input peptide should be greater than zero: {n_peptides}.")
        # _end_if_

        # Make sure the input peptide is an odd number.
        if np.mod(n_peptides, 2) == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input peptide should be odd number: {n_peptides}.")
        # _end_if_

        # Index of the middle element.
        # E.g.: tri-peptide: (0, 1, 2) -> '1'.
        mid_index = n_peptides // 2

        # Ensure the f_path is Path.
        f_path = Path(f_path)

        # Get the filename.
        f_name = f_path.stem

        # Get the structure(s) from the PDB file.
        structure = PDBParser(PERMISSIVE=True, QUIET=True).get_structure(f_name, f_path)

        # Sanity check.
        if not structure:
            raise ValueError(f"{self.__class__.__name__}:"
                             f" PDB file {f_path} is empty.")
        # _end_if_

        # Modify all models in the input structure.
        # N.B. : CYS -> CYO/CYR and PRO -> PRC/PRT.
        # N.B. : The modification is done in place.
        modify_models_OX(structure, all_models=all_models, all_chains=True, in_place=True)

        # Localize the static method.
        get_sin_cos = self.sine_cosine

        # Output dictionary.
        output_data = {}

        # Start processing all the structures in the list.
        for it, model in enumerate(structure, start=1):

            # Check if we want to process
            # all models, or only the 1st.
            if not all_models and (it > 1):
                break
            # _end_if_

            # Get the modified sequence(s).
            # Modified means that it contains the distinction
            # between: CYO / CYR and PRC / PRT.
            amino_acid_seq = get_sequence(model, modified=True)

            # Will hold the extracted data.
            x_out = {}

            # Compute the internal coordinates.
            model.atom_to_internal_coordinates()

            # We process every chain in the model.
            for ic, chain in enumerate(model, start=0):

                # Get the chain ID.
                chain_id = str(chain.get_id()).upper()

                # Create a new data entry
                # (for the current chain).
                x_out[chain_id] = []

                # Initially set these to "None".
                hd_bonds, ring_atoms = None, None

                # Stores all the poly-peptides / torsion angles.
                poly_peptides, torsion_angles = [], []

                # Localize the append method.
                poly_peptides_append = poly_peptides.append

                # Localize the append method.
                torsion_angles_append = torsion_angles.append

                # Get the h-bonds.
                if self.hydrogen_bonds:
                    hd_bonds = self.hydrogen_bond_calc(chain)
                # _end_if_

                # Atoms affected by aromatic rings.
                if self.aromatic_rings:
                    ring_atoms = self.aromatic_ring_calc.check_effect(chain,
                                                                      find_rings=True)
                # _end_if_

                # Extract all the residues in a list.
                residue_list = [*chain.get_residues()]

                # Get the length of the chain.
                chain_length = len(residue_list)

                # This should not happen.
                if verbose and self.hydrogen_bonds and len(hd_bonds) != chain_length:
                    # Display a warning rather than raising an exception.
                    print(f" {self.__class__.__name__}:"
                          f" WARNING! {f_name} : {len(hd_bonds)} : {chain_length}")
                # _end_if_

                # Amino-acid counter.
                aa_counter = 0

                # Go through all the residues in the chain.
                for i, res_i in enumerate(chain.get_residues(), start=0):

                    # Skip hetero-atoms.
                    if res_i.id[0] != " ":
                        continue
                    # _end_if_

                    # Check the index two residues ahead to
                    # ensure that we don't go out of bounds.
                    if (i + n_peptides - 1) >= chain_length:
                        break
                    # _end_if_

                    # Increase by one.
                    aa_counter += 1

                    # Auxiliary poly-peptide level lists.
                    vec_i, t_peptide, t_angles = [], [], []

                    # Localize the "extend" method.
                    vec_i_extend = vec_i.extend

                    # Boolean flag: Reset to "True"
                    # for every single poly-peptide.
                    all_good = True

                    # Get the poly-peptide information.
                    for k, res_k in enumerate(residue_list[i:i + n_peptides], start=0):

                        # Index (one-based).
                        INDEX = aa_counter + k

                        # Residue name (three-letter-code).
                        RES_NAME = str(res_k.get_resname()).strip()

                        # Accepted list of residue names.
                        if (RES_NAME not in ACCEPTED_RES) or \
                                (res_k.internal_coord is None):
                            # Change the flag value.
                            all_good = False

                            # Move on to the next poly-peptide.
                            break
                        # _end_if_

                        # Use this name to extract
                        # the info from the BLOSUM.
                        BLM_RES_NAME = RES_NAME

                        # Distinguish for CYS and PRO.
                        if RES_NAME in {"CYS", "PRO"}:

                            try:
                                # Get the value of the b-factor.
                                b_factor = res_k["C"].get_bfactor()

                                # Check for Cysteine.
                                if RES_NAME == "CYS":
                                    # Distinguish CYS in CYR/CYO.
                                    BLM_RES_NAME = "CYR" if b_factor == -1.0 else "CYO"
                                else:
                                    # In this case it's a Proline.
                                    # Distinguish PRO in PRC/PRT.
                                    BLM_RES_NAME = "PRC" if b_factor == -1.0 else "PRT"
                                # _end_if_

                            except KeyError as e0:
                                # Occasionally, some PDB files might have errors and
                                # missing atoms in their residues. In this case just
                                # ignore the error and keep the default residue name.
                                print(f"Residue {RES_NAME} has a missing {e0} atom.")
                            # _end_try_

                        # _end_if_

                        # N.B: The res_k.get_id() returns the ID directly from
                        # the PDB file, so it might not be linearly increasing.
                        RES_ID = res_k.get_id()[1]

                        # Append the residue info to a separate list. This
                        # built the poly-peptides information for later storage.
                        t_peptide.append((INDEX, BLM_RES_NAME, RES_ID))

                        # Add the BLOSUM score vector.
                        vec_i_extend(self.blosum[BLM_RES_NAME])

                        # Get the 'phi'/'psi' angles (in degrees).
                        phi_k = res_k.internal_coord.get_angle("phi")
                        psi_k = res_k.internal_coord.get_angle("psi")

                        # Get the 'chi1'/'chi2' angles (in degrees).
                        chi1_k = res_k.internal_coord.get_angle("chi1")
                        chi2_k = res_k.internal_coord.get_angle("chi2")

                        # Add all the angles (if they exist).
                        for angle in [phi_k, psi_k, chi1_k, chi2_k]:

                            # Check for "None" angles.
                            if angle is None:
                                # Since there is no angle in [0, 2pi]
                                # that satisfies both sine and cosine
                                # to be zero this will imply that the
                                # angle is missing (i.e. is None).
                                vec_i_extend([0.0, 0.0])
                            else:
                                # Add in the vector: [sin(x), cos(x)].
                                vec_i_extend(get_sin_cos(angle))
                            # _end_if_

                        # _end_for_

                        # Append the torsion angles (here in degrees).
                        t_angles.append((INDEX, RES_3_TO_1[RES_NAME],
                                         phi_k, psi_k, chi1_k, chi2_k))

                        # Compute only if requested.
                        if self.hydrogen_bonds and hd_bonds:

                            # Make sure we are on the same residue.
                            if hd_bonds[i + k].res_name == RES_NAME:

                                # First residue in the list.
                                if k == 0:
                                    # Add the 'O-' bond.
                                    vec_i_extend(hd_bonds[i + k].O_)

                                # Last residue in the list.
                                elif k == n_peptides - 1:
                                    # Add the 'H-' bond.
                                    vec_i_extend(hd_bonds[i + k].H_)

                                # Middle residues in the list.
                                else:
                                    # Add the 'H-' bond.
                                    vec_i_extend(hd_bonds[i + k].H_)

                                    # Add the 'O-' bond.
                                    vec_i_extend(hd_bonds[i + k].O_)
                                # _end_if_
                            else:
                                # Print if verbose is true.
                                if verbose:
                                    # Display a warning message.
                                    print(f" {self.__class__.__name__}:"
                                          f" WARNING! {hd_bonds[i + k].res_name} {RES_NAME}.")
                                # _end_if_

                            # _end_if_

                        # _end_if_

                    # _end_for_

                    # The total number of entries will vary according
                    # to what we will include in the data list vec_i.
                    if all_good:
                        # Holds the target atoms that can
                        # be used with that input vector.
                        unaffected_atoms = []

                        # If the flag is enabled we need to check if
                        # there are atoms affected by aromatic rings.
                        if self.aromatic_rings:

                            # Get the center residue info.
                            _, r_name, r_id = t_peptide[mid_index]

                            # Use the one-letter code.
                            r_name = RES_3_TO_1[r_name]

                            # Check all the target atoms.
                            for target in TARGET_ATOMS:

                                # We don't have yet the delta ring effect
                                # values. NOTE: we use only the FIRST TWO
                                # values from the center poly-peptide.
                                if (r_name, r_id, target, _) in ring_atoms:
                                    # Skip to the next.
                                    continue
                                # _end_if_

                                # Add the target atom in the "unaffected" list.
                                unaffected_atoms.append(target)
                            # _end_for_

                        # _end_if_

                        # Append all information to a final (return) list.
                        # The entries here are combined into a dictionary.
                        # This way we have easier access using the "keys",
                        # to access the right data for each record.
                        x_out[chain_id].append({"poly-peptides": t_peptide,
                                                "targets": unaffected_atoms,
                                                "vector": np.array(vec_i, dtype=self.data_type)})

                        # Update the list with the poly-peptides.
                        poly_peptides_append(t_peptide)

                        # Update the list with the torsion angles.
                        torsion_angles_append(t_angles)
                    # _end_if_

                # _end_for_

                # Check if the list is empty.
                if len(x_out[chain_id]) > 1:
                    # Copy the first entry.
                    x_front = x_out[chain_id][0].copy()

                    # Get the front end points.
                    front_end = self._zero_padding(x_front, n_peptides, terminal="front")

                    # Copy the last entry.
                    x_back = x_out[chain_id][-1].copy()

                    # Get the back end points.
                    back_end = self._zero_padding(x_back, n_peptides, terminal="back")

                    # Insert the front terminal points (in reverse order).
                    # This will place them in the right position in x_out.
                    for f, vec_f in enumerate(reversed(front_end)):
                        x_out[chain_id].insert(f, vec_f)
                    # _end_for_

                    # Append the back terminal points.
                    for vec_b in back_end:
                        x_out[chain_id].append(vec_b)
                    # _end_for_

                # _end_if_

                # Check if we want to save the data.
                if save_output_path:

                    # Make sure the save path exists.
                    if not Path(save_output_path).is_dir():
                        # This should run only the first time.
                        Path(save_output_path).mkdir(parents=True)
                    # _end_if_

                    # File ID includes:
                    # - 0) the filename "f_name"
                    # - 1) the model number "it"
                    # - 2) and chain number "ic"
                    file_id = f"{f_name}_{it}_{ic}"

                    # Poly-peptides.
                    if poly_peptides:
                        self.save_auxiliary(file_id, poly_peptides,
                                            "t_peptides", save_output_path)
                    # _end_if_

                    # Torsion angles.
                    if torsion_angles:
                        self.save_auxiliary(file_id, torsion_angles,
                                            "t_angles", save_output_path)
                    # _end_if_

                    # Aromatic rings.
                    if ring_atoms:
                        self.save_auxiliary(file_id, ring_atoms,
                                            "a_rings", save_output_path)
                    # _end_if_

                    # Hydrogen bonds.
                    if hd_bonds:
                        self.save_auxiliary(file_id, hd_bonds,
                                            "h_bonds", save_output_path)
                    # _end_if_

                # _end_if_

            # _end_for_model_

            # Add the {"data" + "sequence"} to the output dictionary.
            # NOTE: Both "data" and "sequence", are dictionaries too.
            output_data[f"model-{it}"] = {"data": x_out,
                                          "sequence": amino_acid_seq}
        # _end_for_structures_

        # Output data.
        return output_data
    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a "wrapper" method
        of the "get_data" method.
        """
        return self.get_data(*args, **kwargs)
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(), along with its values of its flags.

        :return: a string representation of a InputVector object.
        """

        # Local import of new line.
        from os import linesep as new_line

        # Return the f-string.
        return f" InputVector Id({id(self)}): {new_line}"\
               f" Include-HBonds={self.hydrogen_bonds} {new_line}"\
               f" Check-ARings={self.aromatic_rings} {new_line}"\
               f" Datatype={self.data_type} {new_line}"\
               f" BLOSUM={self.blosum_id}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: InputVector().
        """
        return f"InputVector(blosum_id={self.blosum_id}," \
               f"include_hydrogen_bonds={self.include_hydrogen_bonds}," \
               f"check_aromatic_rings={self.aromatic_rings}," \
               f"data_type={self.data_type})"
    # _end_def_

# _end_class_

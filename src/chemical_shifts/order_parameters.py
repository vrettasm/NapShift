"""
This module includes the main class that handles the computation
of the NH-order parameters.
"""

# Python import(s).
import numpy as np

# Private import(s).
from src.chemical_shifts.auxiliaries import ACCEPTED_RES


class NHOrderParameters(object):
    """
    This class computes and returns the "S^2" NH order
    parameters, of a given input chain of amino-acids.
    """

    # Backbone (NHx) Hydrogen names.
    _back_H = {"H",
               "H1", "H2", "H3",
               "1H", "2H", "3H"}

    # List of heavy atoms.
    # Carbon, Oxygen, Nitrogen, Sulfur.
    _heavy_atoms = {"C", "O", "N", "S"}

    # Object variables.
    __slots__ = ("r_eff",)

    # Constructor.
    def __init__(self, r_eff=1.0):
        """
        Constructs an object that will handle
        the calculation of order-parameters.

        :param r_eff (float): Distance defines the
        interaction range of steric contacts [L: Ang].
        """

        # Check for the correct type.
        if isinstance(r_eff, float):

            # Check for positive value.
            if r_eff > 0.0:
                # Assign the distance.
                self.r_eff = r_eff
            else:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Distance value should be positive: {r_eff}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Distance value should be float: {type(r_eff)}.")
        # _end_if_

    # _end_def_

    @property
    def distance(self):
        """
        Accessor (getter) of the distance value.

        :return: distance threshold value [L: Angstrom]
        """
        return self.r_eff

    # _end_def_

    @distance.setter
    def distance(self, new_value):
        """
        Accessor (setter) of the distance value.

        :param new_value: (float) of the distance that
        defines the interaction range of steric contacts.
        """

        # Check for the correct type.
        if isinstance(new_value, float):

            # Check for positive value.
            if new_value > 0.0:
                # Assign the distance.
                self.r_eff = new_value
            else:
                raise ValueError(f"{self.__class__.__name__}:"
                                 f" Distance value should be positive: {new_value}.")
            # _end_if_

        else:
            raise TypeError(f"{self.__class__.__name__}:"
                            f" Distance value should be float: {type(new_value)}.")
        # _end_if_

    # _end_def_

    def __call__(self, chain=None):
        """
        Accepts an amino-acid chain and computes the S^2
        NH order parameters.

        :param chain: Residue (amino-acids) chain.

        :return: a list with the NH-order-parameters.
        """

        # Return list with the NH-order-parameters.
        order_parameters = []

        # Localize append method.
        order_parameters_append = order_parameters.append

        # Get the list of all residues.
        residues = [*chain.get_residues()]

        # Get the length of the amino-acid chain.
        L = len(residues)

        # Process each residue in the input chain.
        for i, res_i in enumerate(residues, start=0):

            # Get the residue id from the PDB.
            res_id = res_i.get_id()[1]

            # Get the residue name in three-letter code.
            res_name = str(res_i.get_resname()).strip()

            # Skip the first residue.
            if (i == 0) or (res_name not in ACCEPTED_RES):
                # Add an 'NaN' at this position.
                order_parameters_append((res_id, res_name, np.nan))

                # Go to the next residue.
                continue
            # _end_if_

            try:
                # Get the oxygen atom 'O',
                # of the previous residue.
                atom_O = residues[i - 1]["O"]

                # Set the Hydrogen to None.
                atom_H = None

                # We could have multiple hydrogen atoms
                # bounded with atom 'N' (i.e. NH2, NH3):
                for atom in res_i.get_atoms():

                    # Find the backbone Hydrogen atoms.
                    if atom.get_name() in self._back_H:
                        # Get the Hydrogen atom.
                        atom_H = atom

                        # Break the loop.
                        break
                    # _end_if_

                # _end_for_

                # If there is not 'H' (e.g. in PRO) raise an error.
                if not atom_H:
                    raise KeyError(" Atom 'H' not found.")
                # _end_if_

            except (AttributeError, KeyError):
                # Add an 'NaN' at this position.
                order_parameters_append((res_id, res_name, np.nan))

                # Skip to the next residue.
                continue
            # _end_try_

            # Initialize the s^2 and the sum.
            res_s2, sum_of_atoms = 0.0, 0.0

            # Go through all amino-acids in the chain.
            for k in range(L):

                # Exclude the "i" and "i-1" positions.
                if (k == i - 1) or (k == i):
                    continue
                # _end_if_

                # Get the k-th residue.
                res_k = residues[k]

                # Skip hetero-atoms, or not valid names.
                if str(res_k.get_resname()).strip() not in ACCEPTED_RES:
                    continue
                # _end_if_

                # Go through all atoms in the "k-th" residue.
                for atom_k in res_k.get_atoms():

                    # Consider only the heavy atoms.
                    if str(atom_k.get_name()).strip() in self._heavy_atoms:
                        # Compute both distances.
                        r_O = atom_O - atom_k
                        r_H = atom_H - atom_k

                        # Sum the negative exponential values.
                        sum_of_atoms += np.exp(-r_O / self.r_eff) + 0.8 * np.exp(-r_H / self.r_eff)
                    # _end_if_

                # _end_for_

            # _end_for_

            # Compute the S^2.
            if np.isfinite(sum_of_atoms) and (sum_of_atoms != 0.0):
                res_s2 = np.tanh(0.8 * sum_of_atoms) - 0.1
            # _end_if_

            # Add the order-parameter S^2.
            order_parameters_append((res_id, res_name, res_s2))
        # _end_for_

        # S-squared NH order parameter.
        return order_parameters

    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(), along with its 'r_eff' distance.

        :return: a string representation of a NHOrderParameters object.
        """
        return f"NHOrderParameters Id({id(self)}): Distance={self.r_eff}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: NHOrderParameters().
        """
        return f"NHOrderParameters(r_eff={self.r_eff})"
    # _end_def_

# _end_class_

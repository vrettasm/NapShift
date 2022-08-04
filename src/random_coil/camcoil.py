"""
This module provides a "Python implementation" of the camcoil
program (originally written in C) to estimate the random coil
chemical shift values from a sequence (string) of amino-acids.

The work is described in detail at:

1.  Alfonso De Simone, Andrea Cavalli, Shang-Te Danny Hsu,
    Wim Vranken and Michele Vendruscolo (2009) (https://doi.org/10.1021/ja904937a).
    "Accurate Random Coil Chemical Shifts from an Analysis of Loop
    Regions in Native States of Proteins". Journal of the American
    Chemical Society (J.A.C.S.), 131 (45), 16332 - 16333.

NOTE:
    The txt files: 'corr_L1', 'corr_L2', 'corr_R1', 'corr_R2',
    are required for the estimation of the random coil values.
    They should be placed in the same directory with the module
    file. If they do not exist the code will exit with an error.

"""

import warnings
from pathlib import Path
from pandas import read_csv, DataFrame

from src.chemical_shifts.auxiliaries import TARGET_ATOMS
from src.random_coil.random_coil_properties import (MD5_HASH_CODES,
                                                    ACCEPTED_RES_ONE,
                                                    pH2_prop, pH7_prop,
                                                    weights, md5_checksum)


class CamCoil(object):
    """
    This class implements the CamCoil code in Python.
    """

    # Object variables.
    __slots__ = ("_pH", "_cs", "df")

    # Constructor.
    def __init__(self, pH=7.0):
        """
        Initializes the camcoil object. The pH is given as
        option during the initialization of the object even
        though only two actual implementations exist at the
        moment (i.e., pH=2 and pH=7).

        If the user selects another pH value, this will be
        set automatically to one of these two in the code.

        :param pH: (float) the default pH value is set to 7.0.
        """

        # Make sure the input is float.
        pH = float(pH)

        # Check for the correct range.
        if (pH < 0.0) or (pH > 14.0):
            raise ValueError(f"{self.__class__.__name__}: "
                             f"pH value should be in [0, 14]: {pH}.")
        # _end_if_

        # Load the right chemical shifts.
        if pH < 4.0:
            # Assign a fixed pH.
            self._pH = 2.0

            # Get the reference chemical shifts.
            self._cs = pH2_prop
        else:
            # Assign a fixed pH.
            self._pH = 6.1

            # Get the reference chemical shifts.
            self._cs = pH7_prop
        # _end_if_

        # Dictionary of (correction) dataframes.
        self.df = {}

        # Get the parent folder of the module.
        parent_dir = Path(__file__).resolve().parent

        # Load the correction files.
        for f_name in ["corr_L1", "corr_L2", "corr_R1", "corr_R2"]:

            # Initialize with None.
            self.df[f_name] = None

            # Make sure the input file is Path.
            f_path = Path(parent_dir / str(f_name + ".txt"))

            # Sanity check.
            if not f_path.is_file():
                raise FileNotFoundError(f"{self.__class__.__name__} : "
                                        f"File {f_path} doesn't exist.")
            # _end_if_

            # Check the checksum of the input file.
            if md5_checksum(f_path) != MD5_HASH_CODES[f_name]:
                # Create the warning message.
                msg = f"The MD5 checksum of {f_name} has changed."

                # Show the warning.
                warnings.warn(msg, UserWarning)
            # _end_if_

            # N.B.: We have to set the 'keep_default_na=False', because
            # the combination of residues 'N' + 'A' is interpreted here
            # by default as 'NaN', and that causes the indexing to fail
            # in that search.
            self.df[f_name] = read_csv(f_path, header=None, delim_whitespace=" ",
                                       keep_default_na=False,
                                       names=["RES", "ATOM", "CS", "UNKNOWN"])
            # This is to optimize search.
            self.df[f_name].set_index(["RES", "ATOM"], inplace=True)
        # _end_if_

    # _end_def_

    @property
    def pH(self):
        """
        Accessor (getter) of the pH parameter.

        :return: the pH value.
        """
        return self._pH
    # _end_def_

    @pH.setter
    def pH(self, new_value):
        """
        Accessor (setter) of the pH parameter.

        :param new_value: (float).
        """

        # Check for the correct type.
        if isinstance(new_value, float):

            # Check the range of the pH.
            if 0.0 <= new_value <= 14.0:

                # Re-load the right chemical shifts.
                if new_value < 4.0:
                    # Set the fixed pH.
                    self._pH = 2.0

                    # Update the chemical shifts.
                    self._cs = pH2_prop
                else:
                    # Set the fixed pH.
                    self._pH = 6.1

                    # Update the chemical shifts.
                    self._cs = pH7_prop
                # _end_if_

            else:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"pH value should be in [0, 14]: {new_value}.")
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"pH value should be float: {type(new_value)}.")
        # _end_if_
    # _end_def_

    # Main functionality.
    def predict(self, seq=None, verbose=False):
        """
        Accepts a string amino-acid sequence, and returns
        a prediction with the random coil chemical shifts.

        :param seq: (string) The input amino-acid sequence.

        :param verbose: (bool) If the flag is set to True
        it will print more information on the screen.

        :return: a pandas DataFrame, with the results.
        """

        # Sanity check.
        if seq is None:
            # Show a message.
            print(" No input sequence has been given.")

            # Return nothing.
            return None
        # _end_if_

        # Make sure there aren't empty spaces.
        seq = str(seq).strip().upper()

        # Get the length of the sequence.
        seq_length = len(seq)

        # Check the length of the input.
        if seq_length > 5000:
            raise ValueError(f"{self.__class__.__name__}: "
                             f" Sequence length is too long: {seq_length}")
        # _end_if_

        # Make a quick check for validity.
        for res in seq:

            # Valid residue check.
            if res not in ACCEPTED_RES_ONE:
                raise ValueError(f"{self.__class__.__name__}: "
                                 f" Input sequence is not valid: {res}")
            # _end_if_

        # _end_if_

        # Holds the output values.
        output = []

        # Localize the append method.
        output_append = output.append

        # Compute the random coil values.
        for i, res_i in enumerate(seq, start=0):

            # Create a new dictionary. This will
            # hold the chemical shift values for
            # all atoms of the "i-th" residue.
            cs_i = {"ID": int(i+1), "RES": str(res_i),
                    "CA": None, "CB": None, "C": None,
                    "H": None, "HA": None, "N": None}

            # Predict the chemical shifts.
            for atom in TARGET_ATOMS:

                # First get the "reference" chemical shift.
                cs_i[atom] = getattr(self._cs[res_i], atom)

                # Get the (LOWER) neighbourhood contributions.
                for j in [i - 2, i - 1]:

                    # Check (LOWER) bounds:
                    if j < 0:
                        continue
                    # _end_if_

                    # Backwards link with the i-th residue.
                    search_link = (seq[j] + res_i, atom)

                    if j == i - 1:
                        # Get the weight value.
                        alpha = getattr(weights["L1"], atom)

                        # Get the correction value from the dataframe.
                        corr_val = self.df["corr_L1"].loc[search_link, "CS"]

                        # Add the weighted correction.
                        cs_i[atom] += float(alpha * corr_val)
                    else:
                        # Get the weight value.
                        alpha = getattr(weights["L2"], atom)

                        # Get the correction value from the dataframe.
                        corr_val = self.df["corr_L2"].loc[search_link, "CS"]

                        # Add the weighted correction.
                        cs_i[atom] += float(alpha * corr_val)
                    # _end_if_

                # _end_for_

                # Get the (UPPER) neighbourhood contributions.
                for k in [i + 1, i + 2]:

                    # Check (UPPER) bounds:
                    if k > seq_length - 1:
                        break
                    # _end_if_

                    # Forward link with the i-th residue.
                    search_link = (res_i + seq[k], atom)

                    if k == i + 1:
                        # Get the weight value.
                        alpha = getattr(weights["R1"], atom)

                        # Get the correction value from the dataframe.
                        corr_val = self.df["corr_R1"].loc[search_link, "CS"]

                        # Add the weighted correction.
                        cs_i[atom] += float(alpha * corr_val)
                    else:
                        # Get the weight value.
                        alpha = getattr(weights["R2"], atom)

                        # Get the correction value from the dataframe.
                        corr_val = self.df["corr_R2"].loc[search_link, "CS"]

                        # Add the weighted correction.
                        cs_i[atom] += float(alpha * corr_val)
                    # _end_if_

                # _end_for_

            # _end_for_

            # Append the results.
            output_append(cs_i)
        # _end_for_

        # Check the flag.
        if verbose:
            # Size of the chunks.
            n = 20

            # Split the amino-acid sequence to chunk of size 'n'.
            chunks = [seq[i:i + n] for i in range(0, seq_length, n)]

            # Print message:
            print(f"SEQUENCE PROCESSED (pH={self.pH}):")

            # Print the sequence in chunks of 10 residues.
            for i, partial in enumerate(chunks, start=1):
                print(f"{i:>3}: {partial}")
            # _end_for_
        # _end_if_

        # Return the output in dataframe.
        return DataFrame(data=output)
    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is only a "wrapper" method
        of the "predict" method.
        """
        return self.predict(*args, **kwargs)
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of
        the object. This will include its id(.), along with
        its pH field value.

        :return: a string representation of a CamCoil object.
        """
        # Return the f-string.
        return f" CamCoil Id({id(self)}): pH={self._pH}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: CamCoil().
        """
        return f"CamCoil(pH={self._pH})"
    # _end_def_

# _end_class_

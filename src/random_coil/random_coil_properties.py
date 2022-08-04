"""
Properties for the CamCoil implementation. These include:

    1. ACCEPTED_RES_ONE
    2. pH2_prop
    3. pH7_prop
    4. weights
    5. weights_LFP

"""

# Main imports.
import hashlib
from functools import partial
from numpy import nan as np_nan
from src.chemical_shifts.auxiliaries import ChemShifts

# Accepted residues (one-letter-code).
ACCEPTED_RES_ONE = {'A', 'R', 'N', 'D', 'C', 'Q',
                    'E', 'G', 'H', 'I', 'L', 'K',
                    'M', 'F', 'P', 'S', 'T', 'W',
                    'Y', 'V', 'X', 'O'}

# MD5 Hash-codes of the files.
MD5_HASH_CODES = {"corr_L1": "88f13da5b38fb7385d5c23a5de753dc2",
                  "corr_L2": "1b12393292808938f1744b031352b3e6",
                  "corr_R1": "10ab22b9852d4319f61f9f9960c9a5bb",
                  "corr_R2": "f2dfaa4e62e9a24582872691458bae48"}

# These are the "target" atoms that are predicted.
# ChemShifts => ['N', 'C', 'CA', 'CB', 'H', 'HA'].

# Residue specific values for pH=2.0.
# NOTE: For the Gly (G) - 'Ha' atom we used the random coil chemical
# shift value from: https://pubs.acs.org/doi/10.1021/ja105656t
pH2_prop = {"A": ChemShifts(125.387, 177.505, 52.475, 19.374, 8.2680, 4.371),
            "R": ChemShifts(122.866, 176.180, 56.002, 30.998, 8.2530, 4.383),
            "N": ChemShifts(120.148, 175.280, 53.069, 39.128, 8.3620, 4.729),
            "D": ChemShifts(120.394, 175.640, 53.106, 38.300, 8.3160, 4.639),
            "C": ChemShifts(121.164, 174.872, 57.499, 30.633, 8.2740, 4.503),
            "Q": ChemShifts(121.363, 176.102, 55.789, 29.627, 8.2310, 4.353),
            "E": ChemShifts(121.595, 176.123, 55.659, 29.743, 8.3830, 4.346),
            "G": ChemShifts(110.642, 174.164, 45.764, np_nan, 8.3630, 3.980),
            "H": ChemShifts(120.600, 174.147, 55.027, 29.588, 8.2620, 4.674),
            "I": ChemShifts(122.534, 176.093, 60.578, 38.915, 8.2670, 4.244),
            "L": ChemShifts(123.372, 177.302, 55.029, 42.375, 8.0890, 4.378),
            "K": ChemShifts(123.179, 176.570, 56.080, 32.851, 8.3120, 4.390),
            "M": ChemShifts(122.168, 175.861, 55.562, 32.774, 8.2200, 4.499),
            "F": ChemShifts(121.828, 175.913, 57.315, 39.584, 8.2500, 4.626),
            "P": ChemShifts(137.180, 176.957, 62.655, 31.821, np_nan, 4.456),
            "S": ChemShifts(117.226, 174.796, 58.110, 63.796, 8.2850, 4.472),
            "T": ChemShifts(115.692, 174.672, 61.364, 69.862, 8.1840, 4.423),
            "W": ChemShifts(122.564, 176.444, 57.085, 29.442, 8.0600, 4.732),
            "Y": ChemShifts(121.267, 175.722, 57.521, 38.972, 8.2700, 4.629),
            "V": ChemShifts(121.369, 176.021, 61.798, 32.888, 8.0550, 4.216),
            "X": ChemShifts(120.611, 174.868, 55.161, 40.935, 8.4000, 4.895),
            "O": ChemShifts(139.068, 175.983, 62.633, 33.845, np_nan, 4.763)}

# Residue specific values for pH=7.0.
# NOTE:  For the Gly (G) - 'Ha' atom we used the random coil chemical shift
# value from: https://bmrb.io/published/Ikura_cs_study/part2_rc_aa_cs_stats
pH7_prop = {"A": ChemShifts(125.387, 177.505, 52.475, 19.374, 8.2680, 4.371),
            "R": ChemShifts(122.866, 176.180, 56.002, 30.998, 8.2530, 4.383),
            "N": ChemShifts(120.148, 175.280, 53.069, 39.128, 8.3620, 4.729),
            "D": ChemShifts(121.594, 176.340, 54.186, 41.300, 8.3160, 4.639),
            "C": ChemShifts(121.164, 174.872, 57.499, 30.633, 8.2740, 4.503),
            "Q": ChemShifts(121.363, 176.102, 55.789, 29.627, 8.2310, 4.353),
            "E": ChemShifts(122.095, 176.403, 56.339, 30.403, 8.3830, 4.346),
            "G": ChemShifts(110.642, 174.164, 45.764, np_nan, 8.3630, 4.120),
            "H": ChemShifts(120.600, 174.627, 55.447, 30.088, 8.2620, 4.674),
            "I": ChemShifts(122.534, 176.093, 60.578, 38.915, 8.2670, 4.244),
            "L": ChemShifts(123.372, 177.302, 55.029, 42.375, 8.0890, 4.378),
            "K": ChemShifts(123.179, 176.570, 56.080, 32.851, 8.3120, 4.390),
            "M": ChemShifts(122.168, 175.861, 55.562, 32.774, 8.2200, 4.499),
            "F": ChemShifts(121.828, 175.913, 57.315, 39.584, 8.2500, 4.626),
            "P": ChemShifts(137.180, 176.957, 62.655, 31.821, np_nan, 4.456),
            "S": ChemShifts(117.226, 174.796, 58.110, 63.796, 8.2850, 4.472),
            "T": ChemShifts(115.692, 174.672, 61.364, 69.862, 8.1840, 4.423),
            "W": ChemShifts(122.564, 176.444, 57.085, 29.442, 8.0600, 4.732),
            "Y": ChemShifts(121.267, 175.722, 57.521, 38.972, 8.2700, 4.629),
            "V": ChemShifts(121.369, 176.021, 61.798, 32.888, 8.0550, 4.216),
            "X": ChemShifts(120.611, 174.868, 55.161, 40.935, 8.4000, 4.895),
            "O": ChemShifts(139.068, 175.983, 62.633, 33.845, np_nan, 4.763)}

# Weight factors for the pairwise correction terms.
# N.B.: There are discrepancies compared to the SI.
weights = {"L2": ChemShifts(0.10, 0.00, 0.04, 0.08, 0.10, 0.06),
           "L1": ChemShifts(0.74, 0.00, 0.20, 0.20, 0.18, 0.28),
           "R1": ChemShifts(0.14, 0.60, 0.52, 0.38, 0.18, 0.28),
           "R2": ChemShifts(0.12, 0.26, 0.10, 0.10, 0.04, 0.02)}

# Weight factors for the pairwise correction terms (LFP).
# (LFP == Loops of Folded Proteins).
weights_LFP = {"L2": ChemShifts(0.54, 0.28, 0.64, 0.54, 0.06, 0.32),
               "L1": ChemShifts(0.66, 0.32, 0.78, 0.88, 0.16, 0.40),
               "R1": ChemShifts(0.58, 0.38, 0.92, 0.88, 0.16, 0.44),
               "R2": ChemShifts(0.42, 0.28, 0.74, 0.58, 0.08, 0.26)}

# Auxiliary function.
def md5_checksum(file_name: str):
    """
    Compute the MD5 checksum of an input file.

    :param file_name: String name of the text file.

    :return: the md5 hex-digest.
    """

    # Create an MD5 object.
    md5 = hashlib.md5()

    # Handle the contents in binary form.
    with open(file_name, mode="rb") as f:

        # Read the file in bytes.
        for buffer in iter(partial(f.read, 128), b''):

            # Update the MD5.
            md5.update(buffer)
        # _end_for_

    # _end_with_

    # Return the hex-digest.
    return md5.hexdigest()
# _end_def_

# _end_module_

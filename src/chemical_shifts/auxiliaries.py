"""
Constants that are used by all code are placed here for easy access.
These include:

    1. ACCEPTED_RES
    2. RES_3_TO_1
    3. TARGET_ATOMS
    4. RANDOM_COIL_TBL
    5. RANDOM_COIL_AVG
    6. RANDOM_COIL_STD
    7. ONEHOT (code: 10)
    8. BLOSUM (code: 45, 50, 62, 80, 90)

"""
from numpy import rad2deg as np_rad2deg
from numpy import isfinite as np_isfinite
from collections import namedtuple
from pandas import DataFrame
from Bio.PDB.vectors import calc_dihedral

# Add documentation to the NamedTuple.
__pdoc__ = {}

# Residue (amino-acid) names (three-letters code).
# This set is updated with: CYO /CYR and PRT /PRC.
ACCEPTED_RES = {"ALA", "ARG", "ASN", "ASP",
                "CYS", "GLN", "GLU", "GLY",
                "HIS", "ILE", "LEU", "LYS",
                "MET", "PHE", "PRO", "SER",
                "THR", "TRP", "TYR", "VAL",
                "CYO", "CYR", "PRT", "PRC"}

# Protein (amino-acid) 3-to-1 letters mapping.
# NOTE in the modified map:
# The CYO --> X, CYR --> C.
# The PRC --> O, PRT --> P.
RES_3_TO_1 = {"ALA": 'A', "CYS": 'C', "CYO": 'X', "CYR": 'C',
              "ASP": 'D', "GLU": 'E', "PHE": 'F', "GLY": 'G',
              "HIS": 'H', "ILE": 'I', "LYS": 'K', "LEU": 'L',
              "MET": 'M', "ASN": 'N', "PRO": 'P', "PRC": 'O',
              "PRT": 'P', "GLN": 'Q', "ARG": 'R', "SER": 'S',
              "THR": 'T', "VAL": 'V', "TRP": 'W', "TYR": 'Y'}

# These are the target atoms that are predicted.
# NOTE: The order of the entries matters, so we
#       can't change this to a set()!!
TARGET_ATOMS = ('N', 'C', 'CA', 'CB', 'H', 'HA')

# Module level declaration.
ChemShifts = namedtuple("ChemShifts", TARGET_ATOMS)

# Add documentation for the fields.
__pdoc__["ChemShifts.N"] = "Random coil chemical shift value for 'N'."
__pdoc__["ChemShifts.C"] = "Random coil chemical shift value for 'C'."
__pdoc__["ChemShifts.H"] = "Random coil chemical shift value for 'H'."

__pdoc__["ChemShifts.CA"] = "Random coil chemical shift value for 'CA'."
__pdoc__["ChemShifts.CB"] = "Random coil chemical shift value for 'CB'."
__pdoc__["ChemShifts.HA"] = "Random coil chemical shift value for 'HA'."

# Random-coil (corrections) chemical shift values.
# Source: https://pubs.acs.org/doi/10.1021/ja105656t
# -----------------------------------------------------------------------------------
RANDOM_COIL_TBL = {"ALA": ChemShifts(123.960, 178.418, 52.599, 19.102, 8.158, 4.224),
                   "CYO": ChemShifts(120.300, 173.500, 55.300, 40.500, 8.500, 4.850),
                   "CYR": ChemShifts(119.068, 174.927, 58.327, 28.085, 8.410, 4.447),
                   "ASP": ChemShifts(120.207, 176.987, 54.331, 41.089, 8.217, 4.537),
                   "GLU": ChemShifts(120.769, 177.125, 56.650, 30.225, 8.304, 4.222),
                   "PHE": ChemShifts(120.138, 176.368, 57.934, 39.660, 8.107, 4.573),
                   "GLY": ChemShifts(108.783, 174.630, 45.236, 00.000, 8.370, 3.980),
                   "HIS": ChemShifts(118.930, 175.349, 55.964, 29.719, 8.310, 4.585),
                   "ILE": ChemShifts(120.512, 176.897, 61.247, 38.563, 7.963, 4.076),
                   "LYS": ChemShifts(121.353, 177.224, 56.412, 32.921, 8.221, 4.237),
                   "LEU": ChemShifts(121.877, 178.037, 55.260, 42.212, 8.088, 4.260),
                   "MET": ChemShifts(120.002, 176.953, 55.591, 32.690, 8.209, 4.425),
                   "ASN": ChemShifts(118.668, 175.825, 53.231, 38.790, 8.366, 4.632),
                   "PRC": ChemShifts(136.612, 177.542, 63.180, 32.072, 0.000, 4.339),
                   "PRT": ChemShifts(136.612, 177.542, 63.180, 32.072, 0.000, 4.339),
                   "GLN": ChemShifts(120.224, 176.510, 55.840, 29.509, 8.258, 4.254),
                   "ARG": ChemShifts(121.288, 176.821, 56.088, 30.691, 8.232, 4.239),
                   "SER": ChemShifts(115.935, 175.236, 58.352, 63.766, 8.215, 4.392),
                   "THR": ChemShifts(114.024, 175.122, 61.926, 69.794, 8.047, 4.252),
                   "VAL": ChemShifts(120.403, 176.772, 62.347, 32.674, 8.037, 4.009),
                   "TRP": ChemShifts(120.733, 174.549, 57.500, 29.380, 7.725, 4.567),
                   "TYR": ChemShifts(120.228, 176.284, 57.761, 38.750, 8.026, 4.504)}

# Random-coil (avg) chemical shift values.
# Source: https://bmrb.io/published/Ikura_cs_study/part2_rc_aa_cs_stats.pdf
# -----------------------------------------------------------------------------------
RANDOM_COIL_AVG = {"ALA": ChemShifts(124.200, 176.800, 52.100, 19.300, 8.240, 4.390),
                   "CYO": ChemShifts(120.300, 173.500, 55.300, 40.500, 8.500, 4.850),
                   "CYR": ChemShifts(120.100, 174.700, 58.200, 29.400, 8.200, 4.960),
                   "ASP": ChemShifts(121.500, 175.700, 53.800, 41.200, 8.310, 4.710),
                   "GLU": ChemShifts(121.400, 175.900, 56.300, 30.300, 8.360, 4.390),
                   "PHE": ChemShifts(120.100, 175.000, 57.200, 40.200, 8.270, 4.650),
                   "GLY": ChemShifts(109.800, 173.900, 45.200, 00.000, 8.310, 4.120),
                   "HIS": ChemShifts(119.700, 174.400, 55.300, 30.100, 8.290, 4.730),
                   "ILE": ChemShifts(122.000, 174.900, 60.400, 38.700, 8.300, 4.310),
                   "LYS": ChemShifts(121.000, 176.000, 56.200, 32.800, 8.240, 4.360),
                   "LEU": ChemShifts(122.300, 176.400, 54.500, 42.500, 8.180, 4.470),
                   "MET": ChemShifts(121.200, 174.600, 55.400, 33.700, 8.350, 4.410),
                   "ASN": ChemShifts(119.400, 174.900, 53.000, 38.900, 8.420, 4.750),
                   "PRC": ChemShifts(135.355, 176.100, 62.600, 31.900, 8.756, 4.440),
                   "PRT": ChemShifts(135.355, 176.100, 62.600, 31.900, 8.756, 4.440),
                   "GLN": ChemShifts(120.200, 175.700, 55.500, 29.400, 8.210, 4.430),
                   "ARG": ChemShifts(121.700, 175.300, 55.900, 31.000, 8.240, 4.470),
                   "SER": ChemShifts(116.800, 174.200, 58.100, 64.100, 8.360, 4.550),
                   "THR": ChemShifts(114.600, 174.500, 60.900, 69.700, 8.270, 4.550),
                   "VAL": ChemShifts(121.800, 175.100, 61.400, 32.800, 8.320, 4.300),
                   "TRP": ChemShifts(121.700, 175.500, 57.300, 30.400, 8.190, 4.800),
                   "TYR": ChemShifts(120.000, 174.800, 57.600, 39.400, 8.240, 4.730)}

# Random-coil (std) chemical shift values.
# Source: https://bmrb.io/published/Ikura_cs_study/part2_rc_aa_cs_stats.pdf
# -----------------------------------------------------------------------------------
RANDOM_COIL_STD = {"ALA": ChemShifts(6.900, 2.100, 1.900, 2.000, 0.630, 0.430),
                   "CYO": ChemShifts(6.200, 1.700, 2.600, 2.100, 0.830, 0.840),
                   "CYR": ChemShifts(4.900, 1.700, 2.200, 4.000, 0.700, 0.340),
                   "ASP": ChemShifts(4.400, 1.600, 2.000, 1.600, 0.590, 0.360),
                   "GLU": ChemShifts(4.200, 1.800, 1.900, 1.800, 0.650, 0.440),
                   "PHE": ChemShifts(4.600, 2.100, 2.100, 1.700, 0.840, 0.470),
                   "GLY": ChemShifts(4.200, 1.600, 1.500, 1.000, 0.880, 2.290),
                   "HIS": ChemShifts(5.400, 1.900, 2.500, 2.300, 1.001, 0.810),
                   "ILE": ChemShifts(5.300, 1.800, 2.200, 1.700, 0.780, 0.450),
                   "LYS": ChemShifts(4.300, 1.700, 1.800, 1.700, 0.600, 0.420),
                   "LEU": ChemShifts(4.400, 1.900, 2.000, 2.400, 0.750, 0.430),
                   "MET": ChemShifts(4.600, 2.100, 1.700, 2.200, 0.620, 0.480),
                   "ASN": ChemShifts(4.400, 1.600, 1.900, 2.300, 0.720, 0.360),
                   "PRC": ChemShifts(5.450, 1.700, 1.300, 1.100, 0.710, 0.410),
                   "PRT": ChemShifts(5.450, 1.700, 1.300, 1.100, 0.710, 0.410),
                   "GLN": ChemShifts(4.400, 1.900, 2.000, 2.200, 0.730, 0.460),
                   "ARG": ChemShifts(4.500, 1.900, 2.000, 1.900, 0.700, 0.410),
                   "SER": ChemShifts(4.300, 1.700, 1.900, 1.600, 0.720, 0.410),
                   "THR": ChemShifts(5.000, 1.600, 2.200, 4.500, 0.660, 0.440),
                   "VAL": ChemShifts(5.100, 1.600, 2.200, 2.000, 0.700, 0.470),
                   "TRP": ChemShifts(4.300, 1.700, 2.300, 1.600, 0.740, 0.500),
                   "TYR": ChemShifts(4.500, 1.800, 2.400, 3.200, 0.750, 0.520)}

# ONEHOT10:
# "One hot encoding" is a process  of converting  categorical data variables so
# they can be provided to machine learning algorithms for improving predictions.
# One Letter Code:  A  R  N  D  X  C  Q  E  G  H  I  L  K  M  F  O  P  S  T  W  Y  V
# -----------------------------------------------------------------------------------
ONEHOT10 = {"ALA": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "ARG": (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "ASN": (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "ASP": (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "CYO": (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "CYR": (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "GLN": (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "GLU": (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "GLY": (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "HIS": (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "ILE": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "LEU": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "LYS": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            "MET": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            "PHE": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
            "PRC": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            "PRT": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
            "SER": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
            "THR": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            "TRP": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
            "TYR": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
            "VAL": (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)}

# BLOSUM45:
# Source: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM45
# One Letter Code:   A   R   N   D   X   C   Q   E   G   H   I   L   K   M   F   O   P   S   T   W   Y   V
# ---------------------------------------------------------------------------------------------------------
BLOSUM45 = {"ALA": (+5, -2, -1, -2, -1, -1, -1, -1, +0, -2, -1, -1, -1, -1, -2, -1, -1, +1, +0, -2, -2, +0),
            "ARG": (-2, +7, +0, -1, -3, -3, +1, +0, -2, +0, -3, -2, +3, -1, -2, -2, -2, -1, -1, -2, -1, -2),
            "ASN": (-1, +0, +6, +2, -2, -2, +0, +0, +0, +1, -2, -3, +0, -2, -2, -2, -2, +1, +0, -4, -2, -3),
            "ASP": (-2, -1, +2, +7, -3, -3, +0, +2, -1, +0, -4, -3, +0, -3, -4, -1, -1, +0, -1, -4, -2, -3),
            "CYO": (-1, -3, -2, -3, 12, 11, -3, -3, -3, -3, -3, -2, -3, -2, -2, -4, -4, -1, -1, -5, -3, -1),
            "CYR": (-1, -3, -2, -3, 11, 12, -3, -3, -3, -3, -3, -2, -3, -2, -2, -4, -4, -1, -1, -5, -3, -1),
            "GLN": (-1, +1, +0, +0, -3, -3, +6, +2, -2, +1, -2, -2, +1, +0, -4, -1, -1, +0, -1, -2, -1, -3),
            "GLU": (-1, +0, +0, +2, -3, -3, +2, +6, -2, +0, -3, -2, +1, -2, -3, +0, +0, +0, -1, -3, -2, -3),
            "GLY": (+0, -2, +0, -1, -3, -3, -2, -2, +7, -2, -4, -3, -2, -2, -3, -2, -2, +0, -2, -2, -3, -3),
            "HIS": (-2, +0, +1, +0, -3, -3, +1, +0, -2, 10, -3, -2, -1, +0, -2, -2, -2, -1, -2, -3, +2, -3),
            "ILE": (-1, -3, -2, -4, -3, -3, -2, -3, -4, -3, +5, +2, -3, +2, +0, -2, -2, -2, -1, -2, +0, +3),
            "LEU": (-1, -2, -3, -3, -2, -2, -2, -2, -3, -2, +2, +5, -3, +2, +1, -3, -3, -3, -1, -2, +0, +1),
            "LYS": (-1, +3, +0, +0, -3, -3, +1, +1, -2, -1, -3, -3, +5, -1, -3, -1, -1, -1, -1, -2, -1, -2),
            "MET": (-1, -1, -2, -3, -2, -2, +0, -2, -2, +0, +2, +2, -1, +6, +0, -2, -2, -2, -1, -2, +0, +1),
            "PHE": (-2, -2, -2, -4, -2, -2, -4, -3, -3, -2, +0, +1, -3, +0, +8, -3, -3, -2, -1, +1, +3, +0),
            "PRC": (-1, -2, -2, -1, -4, -4, -1, +0, -2, -2, -2, -3, -1, -2, -3, +9, +8, -1, -1, -3, -3, -3),
            "PRT": (-1, -2, -2, -1, -4, -4, -1, +0, -2, -2, -2, -3, -1, -2, -3, +8, +9, -1, -1, -3, -3, -3),
            "SER": (+1, -1, +1, +0, -1, -1, +0, +0, +0, -1, -2, -3, -1, -2, -2, -1, -1, +4, +2, -4, -2, -1),
            "THR": (+0, -1, +0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -1, -1, +2, +5, -3, -1, +0),
            "TRP": (-2, -2, -4, -4, -5, -5, -2, -3, -2, -3, -2, -2, -2, -2, +1, -3, -3, -4, -3, 15, +3, -3),
            "TYR": (-2, -1, -2, -2, -3, -3, -1, -2, -3, +2, +0, +0, -1, +0, +3, -3, -3, -2, -1, +3, +8, -1),
            "VAL": (+0, -2, -3, -3, -1, -1, -3, -3, -3, -3, +3, +1, -2, +1, +0, -3, -1, -1, +0, -3, -1, +5)}

# BLOSUM50:
# Source: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM50
# One Letter Code:   A   R   N   D   X   C   Q   E   G   H   I   L   K   M   F   O   P   S   T   W   Y   V
# ---------------------------------------------------------------------------------------------------------
BLOSUM50 = {"ALA": (+5, -2, -1, -2, -1, -1, -1, -1, +0, -2, -1, -2, -1, -1, -3, -1, -1, +1, +0, -3, -2, +0),
            "ARG": (-2, +7, -1, -2, -4, -4, +1, +0, -3, +0, -4, -3, +3, -2, -3, -3, -3, -1, -1, -3, -1, -3),
            "ASN": (-1, -1, +7, +2, -2, -2, +0, +0, +0, +1, -3, -4, +0, -2, -4, -2, -2, +1, +0, -4, -2, -3),
            "ASP": (-2, -2, +2, +8, -4, -4, +0, +2, -1, -1, -4, -4, -1, -4, -5, -1, -1, +0, -1, -5, -3, -4),
            "CYO": (-1, -4, -2, -4, 13, 12, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -4, -1, -1, -5, -3, -1),
            "CYR": (-1, -4, -2, -4, 12, 13, -3, -3, -3, -3, -2, -2, -3, -2, -2, -4, -4, -1, -1, -5, -3, -1),
            "GLN": (-1, +1, +0, +0, -3, -3, +7, +2, -2, +1, -3, -2, +2, +0, -4, -1, -1, +0, -1, -1, -1, -3),
            "GLU": (-1, +0, +0, +2, -3, -3, +2, +6, -3, +0, -4, -3, +1, -2, -3, -1, -1, -1, -1, -3, -2, -3),
            "GLY": (+0, -3, +0, -1, -3, -3, -2, -3, +8, -2, -4, -4, -2, -3, -4, -2, -2, +0, -2, -3, -3, -4),
            "HIS": (-2, +0, +1, -1, -3, -3, +1, +0, -2, 10, -4, -3, +0, -1, -1, -2, -2, -1, -2, -3, +2, -4),
            "ILE": (-1, -4, -3, -4, -2, -2, -3, -4, -4, -4, +5, +2, -3, +2, +0, -3, -3, -3, -1, -3, -1, +4),
            "LEU": (-2, -3, -4, -4, -2, -2, -2, -3, -4, -3, +2, +5, -3, +3, +1, -4, -4, -3, -1, -2, -1, +1),
            "LYS": (-1, +3, +0, -1, -3, -3, +2, +1, -2, +0, -3, -3, +6, -2, -4, -1, -1, +0, -1, -3, -2, -3),
            "MET": (-1, -2, -2, -4, -2, -2, +0, -2, -3, -1, +2, +3, -2, +7, +0, -3, -3, -2, -1, -1, +0, +1),
            "PHE": (-3, -3, -4, -5, -2, -2, -4, -3, -4, -1, +0, +1, -4, +0, +8, -4, -4, -3, -2, +1, +4, -1),
            "PRC": (-1, -3, -2, -1, -4, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, 10, +9, -1, -1, -4, -3, -3),
            "PRT": (-1, -3, -2, -1, -4, -4, -1, -1, -2, -2, -3, -4, -1, -3, -4, +9, 10, -1, -1, -4, -3, -3),
            "SER": (+1, -1, +1, +0, -1, -1, +0, -1, +0, -1, -3, -3, +0, -2, -3, -1, -1, +5, +2, -4, -2, -2),
            "THR": (+0, -1, +0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, -1, +2, +5, -3, -2, +0),
            "TRP": (-3, -3, -4, -5, -5, -5, -1, -3, -3, -3, -3, -2, -3, -1, +1, -4, -4, -4, -3, 15, +2, -3),
            "TYR": (-2, -1, -2, -3, -3, -3, -1, -2, -3, +2, -1, -1, -2, +0, +4, -3, -3, -2, -2, +2, +8, -1),
            "VAL": (+0, -3, -3, -4, -1, -1, -3, -3, -4, -4, +4, +1, -3, +1, -1, -3, -3, -2, +0, -3, -1, +5)}

# BLOSUM62:
# Source: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62
# One Letter Code:   A   R   N   D   X   C   Q   E   G   H   I   L   K   M   F   O   P   S   T   W   Y   V
# ---------------------------------------------------------------------------------------------------------
BLOSUM62 = {"ALA": (+4, -1, -2, -2, +0, +0, -1, -1, +0, -2, -1, -1, -1, -1, -2, -1, -1, +1, +0, -3, -2, +0),
            "ARG": (-1, +5, +0, -2, -3, -3, +1, +0, -2, +0, -3, -2, +2, -1, -3, -2, -2, -1, -1, -3, -2, -3),
            "ASN": (-2, +0, +6, +1, -3, -3, +0, +0, +0, +1, -3, -3, +0, -2, -3, -2, -2, +1, +0, -4, -2, -3),
            "ASP": (-2, -2, +1, +6, -3, -3, +0, +2, -1, -1, -3, -4, -1, -3, -3, -1, -1, +0, -1, -4, -3, -3),
            "CYO": (+0, -3, -3, -3, +9, +8, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1),
            "CYR": (+0, -3, -3, -3, +8, +9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -3, -1, -1, -2, -2, -1),
            "GLN": (-1, +1, +0, +0, -3, -3, +5, +2, -2, +0, -3, -2, +1, +0, -3, -1, -1, +0, -1, -2, -1, -2),
            "GLU": (-1, +0, +0, +2, -4, -4, +2, +5, -2, +0, -3, -3, +1, -2, -3, -1, -1, +0, -1, -3, -2, -2),
            "GLY": (+0, -2, +0, -1, -3, -3, -2, -2, +6, -2, -4, -4, -2, -3, -3, -2, -2, +0, -2, -2, -3, -3),
            "HIS": (-2, +0, +1, -1, -3, -3, +0, +0, -2, +8, -3, -3, -1, -2, -1, -2, -2, -1, -2, -2, +2, -3),
            "ILE": (-1, -3, -3, -3, -1, -1, -3, -3, -4, -3, +4, +2, -3, +1, +0, -3, -3, -2, -1, -3, -1, +3),
            "LEU": (-1, -2, -3, -4, -1, -1, -2, -3, -4, -3, +2, +4, -2, +2, +0, -3, -3, -2, -1, -2, -1, +1),
            "LYS": (-1, +2, +0, -1, -3, -3, +1, +1, -2, -1, -3, -2, +5, -1, -3, -1, -1, +0, -1, -3, -2, -2),
            "MET": (-1, -1, -2, -3, -1, -1, +0, -2, -3, -2, +1, +2, -1, +5, +0, -2, -2, -1, -1, -1, -1, +1),
            "PHE": (-2, -3, -3, -3, -2, -2, -3, -3, -3, -1, +0, +0, -3, +0, +6, -4, -4, -2, -2, +1, +3, -1),
            "PRC": (-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +7, +6, -1, -1, -4, -3, -2),
            "PRT": (-1, -2, -2, -1, -3, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, +6, +7, -1, -1, -4, -3, -2),
            "SER": (+1, -1, +1, +0, -1, -1, +0, +0, +0, -1, -2, -2, +0, -1, -2, -1, -1, +4, +1, -3, -2, -2),
            "THR": (+0, -1, +0, -1, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, -1, +1, +5, -2, -2, +0),
            "TRP": (-3, -3, -4, -4, -2, -2, -2, -3, -2, -2, -3, -2, -3, -1, +1, -4, -4, -3, -2, 11, +2, -3),
            "TYR": (-2, -2, -2, -3, -2, -2, -1, -2, -3, +2, -1, -1, -2, -1, +3, -3, -3, -2, -2, +2, +7, -1),
            "VAL": (+0, -3, -3, -3, -1, -1, -2, -2, -3, -3, +3, +1, -2, +1, -1, -2, -2, -2, +0, -3, -1, +4)}

# BLOSUM80:
# Source: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM80
# One Letter Code:   A   R   N   D   X   C   Q   E   G   H   I   L   K   M   F   O   P   S   T   W   Y   V
# ---------------------------------------------------------------------------------------------------------
BLOSUM80 = {"ALA": (+7, -3, -3, -3, -1, -1, -2, -2, +0, -3, -3, -3, -1, -2, -4, -1, -1, +2, +0, -5, -4, -1),
            "ARG": (-3, +9, -1, -3, -6, -6, +1, -1, -4, +0, -5, -4, +3, -3, -5, -3, -3, -2, -2, -5, -4, -4),
            "ASN": (-3, -1, +9, +2, -5, -5, +0, -1, -1, +1, -6, -6, +0, -4, -6, -4, -4, +1, +0, -7, -4, -5),
            "ASP": (-3, -3, +2, 10, -7, -7, -1, +2, -3, -2, -7, -7, -2, -6, -6, -3, -3, -1, -2, -8, -6, -6),
            "CYO": (-1, -6, -5, -7, 13, 12, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -6, -2, -2, -5, -5, -2),
            "CYR": (-1, -6, -5, -7, 12, 13, -5, -7, -6, -7, -2, -3, -6, -3, -4, -6, -6, -2, -2, -5, -5, -2),
            "GLN": (-2, +1, +0, -1, -5, -5, +9, +3, -4, +1, -5, -4, +2, -1, -5, -3, -3, -1, -1, -4, -3, -4),
            "GLU": (-2, -1, -1, +2, -7, -7, +3, +8, -4, +0, -6, -6, +1, -4, -6, -2, -2, -1, -2, -6, -5, -4),
            "GLY": (+0, -4, -1, -3, -6, -6, -4, -4, +9, -4, -7, -7, -3, -5, -6, -5, -5, -1, -3, -6, -6, -6),
            "HIS": (-3, +0, +1, -2, -7, -7, +1, +0, -4, 12, -6, -5, -1, -4, -2, -4, -4, -2, -3, -4, +3, -5),
            "ILE": (-3, -5, -6, -7, -2, -2, -5, -6, -7, -6, +7, +2, -5, +2, -1, -5, -5, -4, -2, -5, -3, +4),
            "LEU": (-3, -4, -6, -7, -3, -3, -4, -6, -7, -5, +2, +6, -4, +3, +0, -5, -5, -4, -3, -4, -2, +1),
            "LYS": (-1, +3, +0, -2, -6, -6, +2, +1, -3, -1, -5, -4, +8, -3, -5, -2, -2, -1, -1, -6, -4, -4),
            "MET": (-2, -3, -4, -6, -3, -3, -1, -4, -5, -4, +2, +3, -3, +9, +0, -4, -4, -3, -1, -3, -3, +1),
            "PHE": (-4, -5, -6, -6, -4, -4, -5, -6, -6, -2, -1, +0, -5, +0, 10, -6, -6, -4, -4, +0, +4, -2),
            "PRC": (-1, -3, -4, -3, -6, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 12, 10, -2, -3, -7, -6, -4),
            "PRT": (-1, -3, -4, -3, -6, -6, -3, -2, -5, -4, -5, -5, -2, -4, -6, 10, 12, -2, -3, -7, -6, -4),
            "SER": (+2, -2, +1, -1, -2, -2, -1, -1, -1, -2, -4, -4, -1, -3, -4, -2, -2, +7, +2, -6, -3, -3),
            "THR": (+0, -2, +0, -2, -2, -2, -1, -2, -3, -3, -2, -3, -1, -1, -4, -3, -3, +2, +8, -5, -3, +0),
            "TRP": (-5, -5, -7, -8, -5, -5, -4, -6, -6, -4, -5, -4, -6, -3, +0, -7, -7, -6, -5, 16, +3, -5),
            "TYR": (-4, -4, -4, -6, -5, -5, -3, -5, -6, +3, -3, -2, -4, -3, +4, -6, -6, -3, -3, +3, 11, -3),
            "VAL": (-1, -4, -5, -6, -2, -2, -4, -4, -6, -5, +4, +1, -4, +1, -2, -4, -4, -3, +0, -5, -3, +7)}

# BLOSUM90:
# Source: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM90
# One Letter Code:   A   R   N   D   X   C   Q   E   G   H   I   L   K   M   F   O   P   S   T   W   Y   V
# ---------------------------------------------------------------------------------------------------------
BLOSUM90 = {"ALA": (+5, -2, -2, -3, -1, -1, -1, -1, +0, -2, -2, -2, -1, -2, -3, -1, -1, +1, +0, -4, -3, -1),
            "ARG": (-2, +6, -1, -3, -5, -5, +1, -1, -3, +0, -4, -3, +2, -2, -4, -3, -3, -1, -2, -4, -3, -3),
            "ASN": (-2, -1, +7, +1, -4, -4, +0, -1, -1, +0, -4, -4, +0, -3, -4, -3, -3, +0, +0, -5, -3, -4),
            "ASP": (-3, -3, +1, +7, -5, -5, -1, +1, -2, -2, -5, -5, -1, -4, -5, -3, -3, -1, -2, -6, -4, -5),
            "CYO": (-1, -5, -4, -5, +9, +8, -4, -6, -4, -5, -2, -2, -4, -2, -3, -4, -4, -2, -2, -4, -4, -2),
            "CYR": (-1, -5, -4, -5, +8, +9, -4, -6, -4, -5, -2, -2, -4, -2, -3, -4, -4, -2, -2, -4, -4, -2),
            "GLN": (-1, +1, +0, -1, -4, -4, +7, +2, -3, +1, -4, -3, +1, +0, -4, -2, -2, -1, -1, -3, -3, -3),
            "GLU": (-1, -1, -1, +1, -6, -6, +2, +6, -3, -1, -4, -4, +0, -3, -5, -2, -2, -1, -1, -5, -4, -3),
            "GLY": (+0, -3, -1, -2, -4, -4, -3, -3, +6, -3, -5, -5, -2, -4, -5, -3, -3, -1, -3, -4, -5, -5),
            "HIS": (-2, +0, +0, -2, -5, -5, +1, -1, -3, +8, -4, -4, -1, -3, -2, -3, -3, -2, -2, -3, +1, -4),
            "ILE": (-2, -4, -4, -5, -2, -2, -4, -4, -5, -4, +5, +1, -4, +1, -1, -4, -4, -3, -1, -4, -2, +3),
            "LEU": (-2, -3, -4, -5, -2, -2, -3, -4, -5, -4, +1, +5, -3, +2, +0, -4, -4, -3, -2, -3, -2, +0),
            "LYS": (-1, +2, +0, -1, -4, -4, +1, +0, -2, -1, -4, -3, +6, -2, -4, -2, -2, -1, -1, -5, -3, -3),
            "MET": (-2, -2, -3, -4, -2, -2, +0, -3, -4, -3, +1, +2, -2, +7, -1, -3, -3, -2, -1, -2, -2, +0),
            "PHE": (-3, -4, -4, -5, -3, -3, -4, -5, -5, -2, -1, +0, -4, -1, +7, -4, -4, -3, -3, +0, +3, -2),
            "PRC": (-1, -3, -3, -3, -4, -4, -2, -2, -3, -3, -4, -4, -2, -3, -4, +8, +7, -2, -2, -5, -4, -3),
            "PRT": (-1, -3, -3, -3, -4, -4, -2, -2, -3, -3, -4, -4, -2, -3, -4, +7, +8, -2, -2, -5, -4, -3),
            "SER": (+1, -1, +0, -1, -2, -2, -1, -1, -1, -2, -3, -3, -1, -2, -3, -2, -2, +5, +1, -4, -3, -2),
            "THR": (+0, -2, +0, -2, -2, -2, -1, -1, -3, -2, -1, -2, -1, -1, -3, -2, -2, +1, +6, -4, -2, -1),
            "TRP": (-4, -4, -5, -6, -4, -4, -3, -5, -4, -3, -4, -3, -5, -2, +0, -5, -5, -4, -4, 11, +2, -3),
            "TYR": (-3, -3, -3, -4, -4, -4, -3, -4, -5, +1, -2, -2, -3, -2, +3, -4, -4, -3, -2, +2, +8, -3),
            "VAL": (-1, -3, -4, -5, -2, -2, -3, -3, -5, -4, +3, +0, -3, +0, -2, -3, -3, -2, -1, -3, -3, +5)}

# BLOSUM: Block substitution  matrices are used for  sequence alignment
# of proteins. They are used to score alignments between evolutionarily
# divergent protein sequences and are based on local alignments.
# NOTE: As special case we also include here the "one-hot-encoding".
BLOSUM = {"10": ONEHOT10, "45": BLOSUM45, "50": BLOSUM50,
          "62": BLOSUM62, "80": BLOSUM80, "90": BLOSUM90}

# Utility function.
def read_random_coil_to_df(f_path):
    """
    Read the Random Coil file and return the data in a DataFrame.
    The input file is generated from the PROSECCO program and has
    specific structure. E.g.:

        [ '#' 'ID' 'RES' 'Q3' 'CA' 'CB' 'C' 'H' 'HA' 'N' ]

    :param f_path: Input file with the random coil chemical shifts.

    :return: A dataframe (pandas) with the data from the file.
    """

    # Create a temporary list to hold the data.
    rows_list = []

    # Process the file.
    with open(f_path, "r") as f_in:
        # Localize the append method.
        row_append = rows_list.append

        # Line-by-line.
        for i, rec in enumerate(f_in.readlines(), start=1):

            # Skip the header line.
            if i == 1:
                continue
            # _end_if_

            # Get the items in a list.
            line_values = rec.split()

            # Sanity check.
            if line_values:

                # Skip if the line is a comment.
                if line_values[0] == '#':
                    continue
                # _end_if_

                # Unpack the values from the line.
                # N.B. Here we ignore the 'Q3' entry.
                id_, res_, _, ca_, cb_, c_, h_, ha_, n_ = line_values

                # Create a new row for the dataframe.
                new_rec = {"ID": int(id_), "RES": str(res_),
                           "CA": float(ca_), "CB": float(cb_), "C": float(c_),
                           "H": float(h_), "HA": float(ha_), "N": float(n_)}

                # Update the list.
                row_append(new_rec)

            # _end_if_

        # _end_for_

    # _end_with_

    # Return the dataframe.
    return DataFrame(data=rows_list)
# _end_def_

# Utility function.
def get_sequence(model, modified=False):
    """
    Returns the amino-acid sequence (one letter code) for
    the given input model. Optionally, we can request the
    "modified" sequence version where the CYS and PRO res
    are distinguished in:

        1) CYR (C) / CYO (X)

        2) PRT (P) / PRC (O)

    WARNING:   The modified version assumes that the input
    model has already been processed with modify_models_OX.
    This will ensure that the b-factors are updated to (-1
    / +1). Otherwise all CYS will be renamed to CYR and all
    PRO to PRT (which are the default values).

    :param model: amino-acid chain (model) from biopython.

    :param modified: (bool) Flag to indicate if we want to
    return the standard (20 : amino-acid), or the modified
    with the CYS: CYR / CYO and PRO: PRT / PRC distinction.

    :return: A dictionary with {"A": "seq_A", "B": "seq_B",
    "C": "seq_C",...} pairs in one letter code.
    """

    # Store the extracted sequences.
    chain_seq = {}

    # Start processing all
    # chains in the model.
    for chain in model:

        # Amino-acid sequence.
        seq = []

        # Localize the append method.
        seq_append = seq.append

        # Iterate through all residues.
        for res_k in chain.get_residues():

            # Get the name.
            RES_NAME = str(res_k.get_resname()).strip()

            # Accepted list of residue names.
            if RES_NAME not in ACCEPTED_RES:
                continue
            # _end_if_

            # Check for modified sequence.
            if modified and RES_NAME in {"CYS", "PRO"}:

                try:
                    # Get the value of the b-factor.
                    b_factor = res_k["C"].get_bfactor()

                    # Check for Cysteine.
                    if RES_NAME == "CYS":
                        # Distinguish CYS in CYR/CYO.
                        RES_NAME = "CYR" if b_factor == -1.0 else "CYO"
                    else:
                        # In this case its a Proline.
                        # Distinguish PRO in PRC/PRT.
                        RES_NAME = "PRC" if b_factor == -1.0 else "PRT"
                    # _end_if_

                except KeyError as e0:
                    # Occasionally, some PDB files might have errors and
                    # missing atoms in their residues. In this case just
                    # ignore the error and keep the default residue name.
                    print(f"Residue {RES_NAME} has a missing {e0} atom.")
                # _end_try_

            # _end_if_

            # Add the one letter residue in the list.
            seq_append(RES_3_TO_1[RES_NAME])

        # _end_for_

        # Get the ID of the chain. Make sure it
        # is in string type (and in upper-case).
        chain_id = str(chain.get_id()).upper()

        # Add the sequence to a dictionary.
        chain_seq[chain_id] = "".join(seq)
    # _end_for_

    # Return the dictionary.
    return chain_seq
# _end_def_

# Utility function.
def modify_models_OX(structure, all_models=False, all_chains=True, in_place=True):
    """
    This function gets a (list of) structures, from a PDB file, and modify the
    b-factor values of the CYS and PRO atoms to separate them in CYO / CYR and
    PRC / PRT. The values assigned in here are -1 / +1, but these are selected
    rather arbitrarily (without any physical meaning).

    :param structure: (list) of models directly from a PDB file.

    :param all_models: (bool) flag to indicate whether we want to process all
    models in the list, or only the first model. Default is "False".

    :param all_chains: (bool) flag to indicate whether we want to process all
    chains, or only the first. Default is "True".

    :param in_place: (bool) flag to indicate whether we want to modify the
    input structure itself, or return a copy of it.

    :return: the reference of the structure, with the correct modifications
    in the b-factors of CYS/PRO residues.
    """

    # Sanity check.
    if not len(structure):
        raise ValueError(" Input structure is empty.")
    # _end_if_

    # If the modification is
    # not happening in place.
    if not in_place:
        # Create a copy of the list.
        structure = structure.copy()
    # _end_if_

    # Start processing all models in the structure.
    for m1, model in enumerate(structure, start=1):

        # Check if we want only the first.
        if not all_models and (m1 > 1):
            # Skip the rest of the models.
            break
        # _end_if_

        # Declare an empty list / set.
        cys_list, ss_bond = [], set()

        # Start processing all chains in the model.
        for c1, chain in enumerate(model, start=1):

            # Check if we want only the first.
            if not all_chains and (c1 > 1):
                # Skip the rest of the chains.
                break
            # _end_if_

            # Extract all residues in a list.
            residues = [*chain.get_residues()]

            # Go through all the residues in the chain.
            for idx, res in enumerate(residues, start=0):

                # Get the residue name.
                res_name = str(res.get_resname()).strip()

                # Check for membership.
                if res_name in {"CYS", "PRO"}:

                    # Check for Cysteines.
                    if res_name == "CYS":

                        # Update the list.
                        cys_list.append(res)

                    else:
                        # Otherwise is a Proline!

                        # If this is the first residue we cannot
                        # used it because in this case the omega
                        # angle cannot be computed.
                        if idx == 0:
                            continue
                        # _end_if_

                        # Skip if the previous residue is an "ACE" residue.
                        if str(residues[idx - 1].get_resname()).strip() == "ACE":
                            continue
                        # _end_if_

                        # Get the coordinates of the four relevant atoms.
                        vector_1 = residues[idx - 1]["CA"].get_vector()
                        vector_2 = residues[idx - 1]["C"].get_vector()
                        vector_3 = res["N"].get_vector()
                        vector_4 = res["CA"].get_vector()

                        # Compute the "omega" torsion angle.
                        omega = calc_dihedral(vector_1, vector_2,
                                              vector_3, vector_4)

                        # Make sure the angle is finite.
                        if np_isfinite(omega):
                            # Set the default value
                            # for b-factor to "1.0".
                            b_value = 1.0

                            # Distinguish between "CIS" and "TRANS"
                            # conformations according to the value
                            # of the omega angle (in degrees).
                            if -10.0 < np_rad2deg(omega) < 10.0:
                                # Change value.
                                b_value = -1.0
                            # _end_if_

                            # Update the b-factor values
                            # for all atoms in the "res".
                            for atom in res.get_atoms():
                                atom.set_bfactor(b_value)
                            # _end_for_
                        # _end_if_
                    # _end_if_
                # _end_if_
            # _end_for_
        # _end_for_

        # Get the length of the cys_list.
        len_cys = len(cys_list)

        # If there are not enough "CYS" residues
        # (i.e. > 1) continue to the next model.
        if len_cys > 1:

            # Identify CYS involved in ss_bond.
            for i in range(0, len_cys):

                try:
                    # Get the i-th "SG" atom.
                    SG_i = cys_list[i]["SG"]

                    for j in range(i + 1, len_cys):
                        # Get the j-th "SG" atom.
                        SG_j = cys_list[j]["SG"]

                        # Compute the distance.
                        distance = SG_i - SG_j

                        # Distance units are assumed in [L:Angstrom].
                        if np_isfinite(distance) and distance <= 3.0:
                            ss_bond.add(cys_list[i])
                            ss_bond.add(cys_list[j])
                        # _end_if_

                    # _end_for_
                except KeyError:
                    # If there is an error
                    # continue to the next.
                    continue
                # _end_try_

            # _end_for_

            # Check all Cysteines in the list.
            for cys in cys_list:
                # Set the default value
                # for b-factor to "1.0".
                b_value = 1.0

                # All CYS residues NOT found in
                # ss_bond will have b-factor -1.
                if cys not in ss_bond:
                    # Change the value.
                    b_value = -1.0
                # _end_if_

                # Update the b-factor values.
                for atom in cys.get_atoms():
                    atom.set_bfactor(b_value)
                # _end_for_
            # _end_for_

        else:
            # Move on to the next model.
            continue
        # _end_if_

    # _end_for_

    # Return the modified structure(s).
    return None if in_place else structure
# _end_def_

# _end_module_

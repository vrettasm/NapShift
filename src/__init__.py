import warnings

from Bio import BiopythonDeprecationWarning

# Disable all the Deprecation warning from Bio.
warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)

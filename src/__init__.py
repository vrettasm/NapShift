import warnings

from Bio import BiopythonDeprecationWarning

# Disable all the Deprecation warnings from Bio module.
warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)

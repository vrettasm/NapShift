#!/usr/bin/env python3 -O

import sys

# Check the current python version before running.
if sys.version_info < (3, 7, 0):
    sys.exit("Error: NapShift program requires Python 3.7 or greater.")
# _end_if_

import os

# Disable some tensor-flow warning messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"

from pathlib import Path
from src.chemical_shifts.model_machine import ChemShiftPredictor

# INFO:
__version__ = '1.0.0'
__author__ = 'Michail Vrettas, PhD'
__email__ = 'vrettasm@duck.com'


# Main function.
def main(pdb_file=None, pH=None, output_path=None, random_coil_path=None,
         talos_fmt=True, all_models=False, verbose=False):
    """
    Main function that  wraps the call of  the predict method. First we create
    a ChemShiftPredictor object. We assume that the trained models are located
    in a folder named "/models/" that exists inside the parent directory (i.e.
    the same directory where the current script is stored).  The output of the
    prediction is stored in the "output_path/" directory using a predetermined
    name that includes the PDB-ID from the input file.

    :param pdb_file: This is the input (PDB) file that we want to predict the
    chemical shift values.

    :param pH: The pH value (default is set to 7).

    :param output_path: The directory (path) where we want to store the output
    file.

    :param random_coil_path: If we have available random coil values, for the
    same input file, we can use it here. If not (the default) we will generate
    automatically new values using the camcoil engine module.

    :param talos_fmt: This is the TALOS format for the output file. If not set
    to True, then we will use a tabular format, where each row corresponds to
    a single residue and each atom has each own column.

    :param all_models: If "True" the method will process all the models in the
    PDB file, otherwise only the first model.

    :param verbose: This is a boolean flag that determines whether we want to
    display additional information on the screen.

    :return: None.
    """

    try:
        # Get the parent folder of the module.
        parent_dir = Path(__file__).resolve().parent

        # Make sure the input file is Path.
        input_dir = Path(parent_dir/"models/")

        # Sanity check.
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory {input_dir} doesn't exist.")
        # _end_if_

        # Create a predictor object. NOTE: By default we
        # will overwrite the results file (if it exists).
        nn_predict = ChemShiftPredictor(dir_model=input_dir, dir_output=output_path,
                                        overwrite=True)

        # Check if we need to alter the pH value. This makes
        # sense only if we have not given a random coil file.
        if random_coil_path is None and pH:
            # Make sure its float.
            pH = float(pH)

            # Change the value of the random_coil object.
            nn_predict.random_coil.pH = pH
        # _end_if_

        # Count the successful predictions.
        count_success = 0

        # Process all input files.
        for f_in in pdb_file:

            # Make sure is a Path.
            f_path = Path(f_in)

            # If the file exists.
            if f_path.is_file():

                # Make the predictions.
                nn_predict(f_path, n_peptides=3, all_models=all_models,
                           random_coil_path=random_coil_path,
                           talos_fmt=talos_fmt, verbose=verbose)

                # Increase counter by one.
                count_success += 1
            else:
                raise FileNotFoundError(f"File {f_path} not found.")
            # _end_if_

        # _end_for_

        # Final message.
        if verbose:
            print(f" Successfully saved {count_success} result(s) to: {nn_predict.dir_output}")
        # _end_if_

    except Exception as e1:

        # Exit the program.
        sys.exit(f" Program ended with message: {e1}")
    # _end_try_

# _end_main_


# Run the main script.
if __name__ == "__main__":

    # Check if we have given input parameters.
    if len(sys.argv) > 1:
        # Local import.
        import argparse

        # Create a parser object.
        parser = argparse.ArgumentParser(description="Python chemical shift predictor (of NMR PDB files), "
                                                     "with the usage of Artificial Neural Networks (ANN). ")

        # Input (PDB) file with the residue coordinates.
        parser.add_argument("-f", "--file", type=str, nargs='+', required=True,
                            help="Input PDB file(s) (Path/String).")

        # Input pH values for the random coil shift values.
        parser.add_argument("--pH", type=float, default=7.0,
                            help="The pH value of reference chemical shifts. "
                                 "Default value is set to 7.0.")

        # Output path to save the predicted chemical shifts.
        parser.add_argument("-o", "--out", type=str, default=None,
                            help="Output path to save the predicted values. Note: "
                                 "The name of the file will be generated automatically,  "
                                 "e.g.: 'prediction_X_model_Y_chain_Z.tab', where 'X' is "
                                 "the input filename, 'Y' is the model number and 'Z' is "
                                 "the chain-id.")

        # Random coil path to load the pre-estimated values.
        parser.add_argument("--rc_path", type=str, default=None,
                            help="File with the random coil chemical shift values.")

        # Add the output format (True=Talos).
        parser.add_argument("--talos", dest="talos", action="store_true",
                            help="The default output is the 'TALOS' format.")

        # Add the output format (False=Tabular).
        parser.add_argument("--no-talos", dest="talos", action="store_false",
                            help="Alternatively we set a tabular (output) format.")

        # Enables verbosity.
        parser.add_argument("--verbose", dest="verbose", action="store_true",
                            help="Display information while running.")

        # Shows the version of the program.
        parser.add_argument("--version", action="version",
                            version=f" NapShift (c), version: {__version__}",
                            help="Print version information and exit.")

        # Enables prediction for all the models in the PDB file.
        parser.add_argument("--all-models", dest="all_models", action="store_true",
                            help="Allows prediction for all the models in the PDB file.")

        # Make sure the defaults are set.
        parser.set_defaults(talos=True, verose=False, all_models=False)

        # Parse the arguments.
        args = parser.parse_args()

        # Call the main function.
        main(pdb_file=args.file, pH=args.pH, output_path=args.out,
             random_coil_path=args.rc_path, talos_fmt=args.talos,
             all_models=args.all_models, verbose=args.verbose)

    else:
        # Display error message.
        sys.exit(f"Error: Not enough input parameters. {os.linesep}"
                 f" Run : {sys.argv[0]} -h/--help ")
    # _end_if_

# _end_program_

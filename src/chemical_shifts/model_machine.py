"""
This module includes the main classes that handle the training of the ANN models
as well as the predictions of the new chemical shifts.
"""

# Python import(s).
import logging
import sys
from datetime import date
from gc import collect as collect_mem_garbage
from pathlib import Path
from time import time
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.preprocessing import MaxAbsScaler

# Tensorflow version 2.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model, Sequential

# Neural networks imports.
from src.chemical_shifts.input_vector import InputVector
from src.chemical_shifts.auxiliaries import (RES_3_TO_1, TARGET_ATOMS,
                                             RANDOM_COIL_TBL)
# CamCoil random coil prediction engine.
from src.random_coil.camcoil import CamCoil

# Disables annoying TF warnings.
tf.get_logger().setLevel("ERROR")


# Base (abstract) class.
class ChemShiftBase(object):

    # Default directory.
    dir_default = Path.cwd()
    """
    The default directory is set to the "current working directory".
    """

    # Object variables.
    __slots__ = ("dir_input", "logger", "dir_output", "overwrite_flag")

    # Constructor.
    def __init__(self, dir_input=None, dir_output=None, overwrite=True,
                 file_logging=False):
        """
        Constructs an object that holds the basic functionality of the
        rest classes. It mostly handles input/output directories along
        with some commonly used variables/flags.

        :param dir_input: Input directory.

        :param dir_output: Output directory.

        :param overwrite: Overwrite file protection (flag). If is True
        then the output process WILL overwrite any pre-existing files.

        :param file_logging: If the flag is True it will start logging
        the activities of the object in files.
        """

        # Check if we have given explicitly
        # a new location for the input files.
        if dir_input is None:
            # This is the default input location.
            self.dir_input = ChemShiftBase.dir_default
        else:
            # This will be the new location.
            self.dir_input = Path(dir_input)
        # _end_if_

        # Make sure the input directory ALWAYS exists.
        if not self.dir_input.is_dir():
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Input directory doesn't exist: {self.dir_input}.")
        # _end_if_

        # Check if we have given explicitly a
        # new location for the output results.
        if dir_output is None:
            # This is the default output location.
            self.dir_output = ChemShiftBase.dir_default
        else:
            # This will be the new location.
            self.dir_output = Path(dir_output)

            # If the output directory doesn't exist
            # create it, along with all its parents.
            if not self.dir_output.is_dir():
                self.dir_output.mkdir(parents=True)
            # _end_if_
        # _end_if_

        # Boolean flag. If "True" the class will allow
        # the new results to overwrite old ones (if exist).
        if isinstance(overwrite, bool):
            # Copy the overwrite flag value.
            self.overwrite_flag = overwrite
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Overwrite protection flag should be bool: {type(overwrite)}.")
        # _end_if_

        # Create a new logger for each object.
        self.logger = logging.getLogger(self.__class__.__name__)

        # Remove all previous handlers.
        if self.logger.handlers:
            self.logger.handlers.clear()
        # _end_if_

        # Set the level to INFO.
        self.logger.setLevel(logging.INFO)

        # Enable logging only in the constructor.
        if isinstance(file_logging, bool) and file_logging:

            # Get the current date (in string).
            today = str(date.today()).replace('-', '_')

            # Creating an output (log) file handler.
            file_handler = logging.FileHandler(Path(self.dir_output /
                                                    f"Fid_{today}_{id(self)}.log"), mode='w')
            # Create a "formatter".
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")

            # Add it to the handler.
            file_handler.setFormatter(formatter)

            # Add the file handler to the logger.
            self.logger.addHandler(file_handler)
        # _end_if_

        # Creating a console handler.
        console_handler = logging.StreamHandler(sys.stdout)

        # Add the console handler.
        self.logger.addHandler(console_handler)

        # First message:
        self.logger.info(" -[Started logging]- ")

    # _end_def_

    # Destructor.
    def __del__(self):
        """
        Releases all the handlers before deleting
        the object from the memory.

        N.B. : I am not sure if this is actually
        helpful here.

        :return: None
        """

        # Sanity check.
        if self.logger.handlers:

            # Final message:
            self.logger.info(" -[Stopped logging]- ")

            # Release all handlers.
            for handler in self.logger.handlers:
                # First close the stream.
                handler.close()

                # Finally remove it from the list.
                self.logger.removeHandler(handler)
            # _end_for_

        # _end_if_

    # _end_def_

    @property
    def overwrite(self):
        """
        Accessor (getter) of the overwrite flag.

        :return: overwrite_flag.
        """
        return self.overwrite_flag
    # _end_def_

    @overwrite.setter
    def overwrite(self, new_value):
        """
        Accessor (setter) of the overwrite flag.

        :param new_value: (bool).
        """

        # Check for correct type.
        if isinstance(new_value, bool):
            # Update the flag value.
            self.overwrite_flag = new_value
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Overwrite protection flag should be bool: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def input_path(self):
        """
        Accessor (getter) of the input path.

        :return: dir_input.
        """
        return self.dir_input
    # _end_def_

    @input_path.setter
    def input_path(self, new_value):
        """
        Accessor (setter) of the input path.

        :param new_value: (Path / String).
        """

        # Check for correct type.
        if isinstance(new_value, (str, Path)):
            # Temporary path.
            tmp_path = Path(new_value)

            # Make sure the new input
            # always directory exists.
            if not tmp_path.is_dir():
                raise ValueError(f"{self.__class__.__name__}: "
                                 f"New input directory doesn't exist: {type(new_value)}.")
            # _end_if_

            # Update the path value.
            self.dir_input = tmp_path
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Input directory should be Path/String: {type(new_value)}.")
        # _end_if_

    # _end_def_

    @property
    def output_path(self):
        """
        Accessor (getter) of the output path.

        :return: dir_output.
        """
        return self.dir_output
    # _end_def_

    @output_path.setter
    def output_path(self, new_value):
        """
        Accessor (setter) of the output path.

        :param new_value: (Path / String).
        """

        # Check for correct type.
        if isinstance(new_value, (str, Path)):
            # Update the path value.
            self.dir_output = Path(new_value)

            # Make sure the output directory exists.
            if not self.dir_output.is_dir():
                self.dir_output.mkdir(parents=True)
            # _end_if_
        else:
            raise TypeError(f"{self.__class__.__name__}: "
                            f"Output directory should be Path/String: {type(new_value)}.")
        # _end_if_
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override to print a readable string presentation of the object.
        This will include its id(), along with its field values.

        NOTE: The overwrite protection is the "opposite" of the overwrite
        flag, so in the print version we show the "not overwrite_flag"!

        :return: a string representation of a ChemShiftBase object.
        """

        # Local import of new line.
        from os import linesep as new_line

        # Return the f-string.
        return f" ChemShiftBase Id({id(self)}): {new_line}" \
               f" Input  dir={self.dir_input} {new_line}" \
               f" Output dir={self.dir_output} {new_line}" \
               f" Overwrite protection={(not self.overwrite_flag)}"
    # _end_def_

    # Auxiliary.
    def __repr__(self):
        """
        Repr operator is called when a string representation
        is needed that can be evaluated.

        :return: ChemShiftBase().
        """
        return f"ChemShiftBase(dir_input={self.dir_input}," \
               f"dir_output={self.dir_input}," \
               f"overwrite={self.overwrite_flag}," \
               f"file_logging= {self.logger})"
    # _end_def_

# _end_class_


# Training Class
class ChemShiftTraining(ChemShiftBase):

    # Constructor.
    def __init__(self, dir_data=None, dir_output=None, overwrite=True):
        """
        Constructs an object that will perform the training of the artificial
        neural networks (ANNs), on predicting the chemical shift values from
        specific atoms.

        :param dir_data: Directory where the trained ANN models (one for each
        target atom) exist.

        :param dir_output: Directory where the output of the training will be
        saved (trained ANN models).

        :param overwrite: Overwrite file protection. If True, then the output
        process WILL overwrite any pre-existing output files.
        """
        # First call the base class constructor.
        super().__init__(dir_data, dir_output, overwrite, file_logging=False)
    # _end_def_

    # Auxiliary.
    def load_data(self, atom=None):
        """
        This method will load the training datasets for a
        given input atom (target).

        :param atom: Atom to be used as target during the
        training process by the ann.

        :return: The (X/y) datasets. Note that the method
        will look for NaN target values and will clean up
        these values.
        """

        # Check the atom.
        if atom not in TARGET_ATOMS:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Loading data: Unknown target '{atom}'.")
        # _end_if_

        # Open the train data file for read only.
        with h5py.File(Path(self.input_path/f"x_train_{atom}.h5"), 'r') as data_file:
            # Convert directly to numpy.
            data = np.array(data_file['x_mat'])
        # _end_with_

        # Total amount of data.
        num_data = data.shape[0]

        # Sanity check.
        if num_data == 0:
            raise ValueError(f"{self.__class__.__name__}: "
                             f"Loading data: Empty data set for target '{atom}'.")
        # _end_if_

        # Remove entries with NaN target values.
        clean_data = data[~np.isnan(data[:, -1])].copy()

        # X-Y (train):
        x_train = clean_data[:, 0:-1]
        y_train = clean_data[:, -1]

        # Percentage of NaN in the training set.
        nan_percent = 100.0 * (1.0 - np.round(clean_data.shape[0] / num_data, 3))

        # Print Info.
        self.logger.info(f" Train data shape: {data.shape},"
                         f" includes {nan_percent:.2f}% NaN.")

        # Return the (X/y) sets of data.
        return x_train, y_train
    # _end_def_

    def train_models(self, validation_split=0.10, save_plots=True, verbose=False):
        """
        The main purpose of this method is to use a pre-defined Artificial Neural
        Network and train it on the chemical shift data. Since we have six targets
        the method will train six networks separately and store its results.

        :param validation_split: (float) This value is used to keep a portion of the
        training data aside while training to validate the training process. This is
        not the test set hold out.

        :param save_plots: (bool) If "True" it will save the training errors, as
        function of time (epochs), for all the training targets (atoms).

        :param verbose: (bool) If "True" it will display more information during
        the training. The default is "False" to avoid cluttering the screen with
        information.

        :return: None.
        """

        # Make sure the neural network models are cleared.
        nn_model = {atom: None for atom in TARGET_ATOMS}

        # Make sure the neural network outputs are cleared.
        nn_output = {atom: None for atom in TARGET_ATOMS}

        # Default plots path.
        plots_path = None

        # Check if we want to save the training
        # figures and then create the directory.
        if save_plots:
            # Figure path.
            plots_path = Path(self.output_path/"plots")

            # Check if the models (output) directory exists.
            if not plots_path.is_dir():
                # Make the path.
                plots_path.mkdir(parents=True)

                # Display info (verbose mode).
                if verbose:
                    self.logger.info(f" Created plots directory at: {plots_path}")
                # _end_if_
            # _end_if_
        # _end_if_

        # Models path.
        models_path = Path(self.output_path/"models")

        # Check if the models (output) directory exists.
        if not models_path.is_dir():
            # Make the path.
            models_path.mkdir(parents=True)

            # Display info (verbose mode).
            if verbose:
                self.logger.info(f" Created models directory at: {models_path}")
            # _end_if_
        # _end_if_

        # Output path.
        results_path = Path(self.output_path/"results")

        # Check if the results (output) directory exists.
        if not results_path.is_dir():
            # Make the path.
            results_path.mkdir(parents=True)

            # Display info (verbose mode).
            if verbose:
                self.logger.info(f" Created results directory at: {results_path}")
            # _end_if_
        # _end_if_

        # Switch on/off the verbosity.
        nn_verbose_flag = tf.constant(1) if verbose else tf.constant(0)

        # Make batch size tf.constant.
        nn_batch_size = tf.constant(512, dtype=tf.int64, name="batch_size")

        # Make epochs tf.constant.
        nn_epochs = tf.constant(1000, dtype=tf.int64, name="epochs")

        # Make validation_split tf.constant.
        nn_validation_split = tf.constant(validation_split,
                                          dtype=tf.float32,
                                          name="val_split")

        # Localize the convert to tensor method.
        convert_to_tensor = tf.convert_to_tensor

        # Train for each target a separate ANN.
        for atom in TARGET_ATOMS:
            # Print info.
            self.logger.info(f" Training ANN for target '{atom}'.")

            # Prepare the data for the network.
            x_train, y_train = self.load_data(atom)

            # Reshape the target vector.
            y_train = y_train.reshape(-1, 1)

            # Fit the input scaler with the data.
            input_scaler = MaxAbsScaler().fit(x_train)

            try:
                # Try to save the x-Scaler because we
                # will need it in the prediction step.
                joblib.dump(input_scaler,
                            Path(models_path/f"data_scaler_{atom}.gz"))
            except RuntimeError as e0:
                self.logger.error(f" Error while saving input Scaler. {e0}")
            # _end_try_

            # Transform the input data.
            x_train = input_scaler.transform(x_train)

            # Weights initializer:
            #
            # 1) For the "hidden" units can also use:
            #    - RandomNormal(mean=0.0, stddev=0.1)
            #
            # 2) For the "output" units can also use:
            #    - RandomUniform(minval=-0.1, maxval=0.1)
            #

            # Setup the ANN (model).
            nn_model[atom] = Sequential([
                Dense(units=26,
                      name="Hidden_1",
                      activation="elu",
                      activity_regularizer=None,
                      kernel_initializer="glorot_normal",
                      input_shape=(x_train.shape[1],)),
                Dense(units=1,
                      name="Output_1",
                      activation='linear',
                      activity_regularizer=None,
                      kernel_initializer="glorot_uniform")
            ], name=f"model_{atom}")

            # Print info.
            if verbose:
                # Model summary.
                nn_model[atom].summary()
            # _end_if_

            # Monitor the 'val_loss' function and if it does not improve for
            # 'patience' epochs, then stop the training and restore the best
            # weights that have been found so far.
            early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                       patience=20, restore_best_weights=True)
            # Set the optimizer object.
            optimization_alg = SGD(learning_rate=0.01, momentum=0.8, nesterov=True)

            # Compile the model.
            nn_model[atom].compile(optimizer=optimization_alg, loss="mse")

            # First time instant.
            time_0 = time()

            # Fit the model.
            nn_output[atom] = nn_model[atom].fit(x=convert_to_tensor(x_train),
                                                 y=convert_to_tensor(y_train),
                                                 batch_size=nn_batch_size, epochs=nn_epochs,
                                                 validation_split=nn_validation_split, shuffle=True,
                                                 verbose=nn_verbose_flag, callbacks=[early_stop])
            # Final time instant.
            time_f = time()

            # Save the model to the pre-defined location.
            nn_model[atom].save(Path(models_path / f"ann_model_{atom}.h5"),
                                include_optimizer=False, save_format="h5")

            # Number of actual runs (epochs).
            n_epoch = len(nn_output[atom].history['loss'])

            # Timing message.
            self.logger.info(f" Finished {n_epoch} epochs"
                             f" in {time_f - time_0:.2f} seconds.")

            # Get the final (training and validation) error values.
            train_RMSE = np.sqrt(nn_output[atom].history["loss"][-1])
            valid_RMSE = np.sqrt(nn_output[atom].history["val_loss"][-1])

            # Error message.
            self.logger.info(f" RMSE = {train_RMSE:.3f},"
                             f" val-RMSE = {valid_RMSE:.3f}\n\n")

            # Convert the output history to a DataFrame.
            df_output = DataFrame(nn_output[atom].history)

            # Save to csv:
            with open(Path(results_path/f"nn_output_{atom}.csv"), mode='w') as f:
                df_output.to_csv(f)
            # _end_with_

            # Save the training figure.
            if save_plots:
                # Make a new figure.
                fig = plt.figure()

                # Plot training loss.
                plt.plot(np.sqrt(df_output['loss']), label="Training")

                # Plot validation loss.
                plt.plot(np.sqrt(df_output['val_loss']), label="Validation")

                # Add the x label.
                plt.xlabel("Epoch")

                # Add the y label.
                plt.ylabel("RMSE")

                # Finalize the plot.
                plt.title(f"Target: {atom}")
                plt.legend()
                plt.grid(True)

                # Maximize the space.
                fig.tight_layout()

                # Save the figure.
                fig.savefig(Path(plots_path/f"nn_training_vs_validation_{atom}.png"),
                            orientation="landscape", dpi=300)
            # _end_if_

            # Clean up the memory.
            collect_mem_garbage()
        # _end_for_

        # Print final info.
        self.logger.info(" Finished training models!")
    # _end_def_

    # Auxiliary.
    def __call__(self, *args, **kwargs):
        """
        This is a "wrapper" method of the "train_models"
        method to simplify the call.
        """
        return self.train_models(*args, **kwargs)
    # _end_def_

    # Auxiliary.
    def __str__(self):
        """
        Override method to print a readable string presentation of the
        object. This will include its id, along with its field values.

            NOTE: The overwrite protection is the opposite of the
            overwrite flag, so in the printed version we show the
            "not overwrite_flag"!

        :return: a string representation of a ChemShiftTraining object.
        """

        # Local import of new line.
        from os import linesep as new_line

        # Return the f-string.
        return f" ChemShiftTraining Id({id(self)}): {new_line}" \
               f" Data   dir={self.input_path} {new_line}" \
               f" Output dir={self.output_path} {new_line}" \
               f" Overwrite protection={(not self.overwrite)}"
    # _end_def_

# _end_class_


# Predictor Class.
class ChemShiftPredictor(ChemShiftBase):

    # InputVector.
    vec_in = InputVector(blosum_id=62, include_hydrogen_bonds=False,
                         check_aromatic_rings=True, data_type=np.float32)
    """
    This will be used to load the PDB file(s) before we make the prediction
    with the ANN model. Here we set (for brevity) all the input parameters.
    """

    # Convert the random coil (average) values to DataFrame.
    df_avg = DataFrame(RANDOM_COIL_TBL, index=TARGET_ATOMS)
    """
    This dataframe will be used in case we do not provide a random coil file.
    """

    # Camcoil predictor of random coil chemical shifts.
    random_coil = CamCoil(pH=7.0)
    """
    This object will be used to predict the random coil
    chemical shifts. The default value for the "pH" is
    set to '7.0'.
    """

    # Object variables.
    __slots__ = ("nn_model", "input_scaler")

    # Constructor.
    def __init__(self, dir_model=None, dir_output=None, overwrite=True):
        """
        Constructs an object that will perform the chemical shifts prediction.
        This is done by first constructing the necessary input values from the
        PDB file and subsequently calling the trained nn-models to perform the
        predictions on all atoms.

        :param dir_model: (Path) Directory where the trained ANN models exist,
        (one for each target atom).

        :param dir_output: (Path) Directory where the output of the predictions
        will be saved.

        :param overwrite: (bool) Overwrite file protection. If "True", then the
        output file WILL overwrite any pre-existing output.
        """

        # First call the base class constructor.
        super().__init__(dir_model, dir_output, overwrite, file_logging=False)

        # Neural network models. Set initial value to "None".
        self.nn_model = {atom: None for atom in TARGET_ATOMS}

        # Input data scaler. Set initial value to "None".
        self.input_scaler = {atom: None for atom in TARGET_ATOMS}

        # Load all the trained models (one for each atom).
        for atom in TARGET_ATOMS:

            # Make the file path.
            file_path = Path(self.input_path/f"ann_model_{atom}.h5")

            # Check if the file exists.
            if not file_path.is_file():
                # Show a message instead of raising an exception.
                self.logger.info(f" Model for target '{atom}', doesn't exist."
                                 f" Skipping ...")
                # Skip to the next atom.
                continue
            # _end_if_

            # If everything is OK load the model. We set the "compile"
            # flag to False, because we use the models for prediction.
            self.nn_model[atom] = load_model(file_path, compile=False)

            # Create the Scaler filename.
            scaler_path = Path(self.input_path/f"data_scaler_{atom}.gz")

            # Check if the scaler exists.
            if scaler_path.is_file():
                # Load the Scaler from the file.
                self.input_scaler[atom] = joblib.load(scaler_path)
            else:
                # Display what went wrong.
                self.logger.warning(f" {self.__class__.__name__}."
                                    f" WARNING: File {scaler_path} not found.")
            # _end_if_

        # _end_for_

    # _end_def_

    def save_to_file(self, pdb_id, aa_sequence, predictions, target_peptides, ref_peptides,
                     random_coil=None, model_id=None, chain_id=None, talos_format=True):
        """
        This method accepts the output of the __call__() method and writes all the
        information in a text file. Optionally we can save using the TALOS format.

        :param pdb_id: (string) This is the PDB-ID from the input file. It is used
        to provide information in the final output file.

        :param aa_sequence: (string) This is the sequence of the amino acids in the
        input file. It is used for information to the final output file.

        :param predictions: These are the predictions (numerical values) of the ANN.
        It is a dictionary where each entry (key) corresponds to an atom-target.

        :param target_peptides:  These are the poly-peptides that were predicted by
        the artificial neural network. Because of the "aromatic-rings effect" there
        could be poly-peptides that were not predicted for all the targets.

        :param ref_peptides: These are ALL the poly-peptides, as constructed by the
        InputVector class. They are used as reference with regards to the list of
        "poly_peptides".

        :param random_coil: This is a DataFrame with the random coil values. If it
        isn't given (default=None) we will use average values from a default table.

        :param model_id: This is the id of the model in the PDB file. We use it to
        distinguish the output results.

        :param chain_id: This is the id of the chain in the protein model.
        We use it to distinguish the output results.

        :param talos_format: (bool) Flag that defines the file format. If is set to
        "True" we will use the TALOS format. If it is set to "False" the file will
        be saved with a default tabular format.

        :return: None.
        """

        try:
            # Construct the model-id for the file name.
            model_id = "1" if model_id is None else model_id

            # Construct the chain-id for the file name.
            chain_id = "A" if chain_id is None else chain_id

            # Construct the output file name.
            f_name_out = Path(self.output_path/f"prediction_{pdb_id}_"
                                               f"model_{model_id}_chain_{chain_id}.tab")

            # Check if we have enabled the overwrite protection.
            if (not self.overwrite) and f_name_out.is_file():
                raise FileExistsError(f" Output: {f_name_out} already exists.")
            # _end_if_

            # Check there is a random coil dataframe.
            if random_coil is not None:
                # This will optimize the searches.
                random_coil.set_index(["ID", "RES"], inplace=True)
            # _end_if_

            # Size of chunks.
            n = 20

            # Split the amino-acid sequence to chucks of size 'n'.
            chunks = [aa_sequence[i:i + n] for i in range(0, len(aa_sequence), n)]

            # Write the prediction data to a text file.
            with open(f_name_out, "w") as f_out:

                # Localize the write function.
                file_write = f_out.write

                # In case we need to add comments. This is not mandatory but
                # it will let us know what the original file was coming from.
                file_write(f"REMARK Chemical Shift predictions for {pdb_id}. \n")

                # Model/Chain information
                file_write(f"REMARK Model {model_id} / Chain {chain_id}. \n")

                # Empty line.
                file_write("\n")

                # Default value is set to N/A (optional).
                file_write("DATA FIRST_RESID N/A \n")

                # Empty line.
                file_write("\n")

                # Write the whole sequence in chunks of size "n".
                for sub_k in chunks:
                    file_write("DATA SEQUENCE " + sub_k + "\n")
                # _end_for_

                # Empty line.
                file_write("\n")

                # Check the file format.
                if talos_format:
                    # Write the (TALOS) variable names.
                    file_write("VARS RESID RESNAME ATOMNAME SHIFT \n")

                    # Write the (TALOS) file format.
                    file_write("FORMAT %4d %1s %4s %8.3f \n")
                else:
                    # Declare a dictionary to group the data
                    # values according to their atom values.
                    record = {atom: np.nan for atom in TARGET_ATOMS}

                    # Tabular text format. This will preserve the same order
                    # (of atoms) as in the TARGET_ATOMS (tuple) declaration.
                    file_write("{:>4} {:>4} {:>8} {:>8} {:>8} {:>8} {:>8} "
                               "{:>8} \n".format("ID", "RES", *record.keys()))
                # _end_if_

                # Empty line.
                file_write("\n")

                # Extract the data and write them to the file.
                for n, peptide in enumerate(ref_peptides, start=1):

                    # Extract the information.
                    index, res_name, res_id = peptide

                    # Convert the name from 3 to 1 letters.
                    res_name_1 = RES_3_TO_1[res_name]

                    # Search link.
                    search_link = (index, res_name_1)

                    # Extract the predicted chemical shifts.
                    for atom in TARGET_ATOMS:

                        # Setting to NaN will indicate that we don't
                        # have a predicted "ss" value for this atom.
                        ss_value, rc_value = np.nan, np.nan

                        # Create a search-peptide tuple.
                        search_peptide = tuple(peptide)

                        # If the peptide is in the target list.
                        if search_peptide in target_peptides[atom]:

                            # Get its index.
                            idx = target_peptides[atom].index(search_peptide)

                            # Get the predicted (secondary structure)
                            # value that comes directly from the ANN.
                            ss_value = predictions[atom][idx].item()

                            # Get the random coil chemical shift.
                            if random_coil is not None:
                                # Get the value from the random coil file.
                                rc_value = random_coil.loc[search_link, atom]
                            else:
                                # Get the average value from a Table.
                                rc_value = ChemShiftPredictor.df_avg.loc[atom, res_name]
                            # _end_if_

                        # _end_if_

                        # Add the random coil value to
                        # the secondary structure value.
                        value = ss_value + rc_value

                        # Check the file format.
                        if talos_format:
                            # Rename the hydrogen from "H" to "HN".
                            atom = "HN" if atom == "H" else atom

                            # Put all the information together in one record.
                            file_write(f"{res_id:>4} {res_name_1:>3} {atom:>6} {value:>8.3f} \n")
                        else:
                            # Store it to the dictionary.
                            record[atom] = value
                        # _end_if_

                    # _end_for_

                    # Check the file format.
                    if not talos_format:
                        # NOTE: From Python "3.6" onwards, the standard dict type maintains
                        # insertion order by default!
                        file_write("{:>4} {:>4} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} "
                                   "{:>8.3f} \n".format(res_id, res_name_1, *record.values()))
                    # _end_if_

                # _end_for_

            # _end_with_
        except FileExistsError as e0:

            # Log the error message.
            self.logger.error(e0)
        # _end_try_

    # _end_def_

    def predict(self, f_path, n_peptides=3, all_models=False, random_coil_path=None,
                verbose=False, talos_fmt=True):
        """
        Primary method of a "ChemShiftPredictor" object. It accepts a PDB file as input,
        constructs the input to the trained NN and puts the results -(predicted chemical
        shifts)- in a new text file.

        :param f_path: (string) PDB file with the residue / atom coordinates.

        :param n_peptides: (int) Number of peptides to consider for the input vectors.
        By default it considers tri-peptides.

        :param all_models: (bool) flag. If "True" the method will process all the models
        in the PDB file, otherwise only the first model.

        :param random_coil_path: (string) file with the random coil chemical shift values.

        :param verbose: (bool) If "True" it will display more info during the prediction.
        The default set is "False" to avoid cluttering the screen with information.

        :param talos_fmt: (bool) If "True" (default) it will use the TALOS format to save
        the results. If it is set to "False" the output format will be tabular.

        :return: It will call the save method to write the results in a TALOS file format.
        """

        # Make sure the input file is a Path.
        f_path = Path(f_path)

        # Sanity check.
        if not f_path.is_file():
            raise FileNotFoundError(f"{self.__class__.__name__} : "
                                    f"File {f_path} doesn't exist.")
        # _end_if_

        # Call the "vector" method to create the input values.
        model_vectors = ChemShiftPredictor.vec_in(f_path,
                                                  all_models=all_models,
                                                  n_peptides=n_peptides)
        # Index of the middle element.
        mid_idx = int(n_peptides) // 2

        # Early exit if the input data is empty.
        # This shouldn't happen very frequently.
        if not model_vectors:
            # Display a warning message.
            self.logger.warning(f" File: {f_path} is empty.")

            # Exit from here.
            return None
        # _end_if_

        # Random coil shift values
        # (from an input file).
        random_coil_shifts = None

        # Check if a random coil file is given.
        if random_coil_path:
            # Make sure the input is a Path.
            rc_path = Path(random_coil_path)

            # Sanity check.
            if rc_path.is_file():
                # Extract the random coil chem-shifts.
                # The first row (id->0) is the header.
                random_coil_shifts = read_csv(rc_path, header=0)
            else:
                # Display a message.
                self.logger.info(f" Random coil file {rc_path} doesn't exist.")
            # _end_if_

        # _end_if_

        # Switch on/off the verbosity.
        nn_verbose_flag = tf.constant(1) if verbose else tf.constant(0)

        # Make batch size tf.constant.
        nn_batch_size = tf.constant(512, dtype=tf.int64, name="batch_size")

        # Localize the convert to tensor method.
        convert_to_tensor = tf.convert_to_tensor

        # Predict chemical shifts of all models.
        for model_id, model_value in enumerate(model_vectors.values(), start=0):

            # Unpack the contents.
            input_data = model_value["data"]
            amino_acid_seq = model_value["sequence"]

            # Sanity check.
            if len(input_data) != len(amino_acid_seq):
                raise RuntimeError(f"{self.__class__.__name__} : "
                                   f"Data / Sequence length mismatch.")
            # _end_if_

            # Process all chains in the model.
            for chain_id, chain_seq in amino_acid_seq.items():

                # Sanity check.
                if not input_data[chain_id]:

                    # Display a warning message.
                    if verbose:
                        # Display a warning message.
                        self.logger.warning(f" Model: {model_id} -"
                                            f" Chain: {chain_id} didn't produce any input data.")
                    # _end_if_

                    # Go to the next model.
                    continue
                # _end_if_

                # Check if a random coil file is given.
                if random_coil_shifts is not None:
                    # Assign the file shift values.
                    df_random_coil = random_coil_shifts
                else:
                    # Use Camcoil algorithm to predict the values.
                    df_random_coil = ChemShiftPredictor.random_coil(chain_seq)
                # _end_if_

                # Declare data dictionary.
                data = {atom: [] for atom in TARGET_ATOMS}

                # Declare peptides dictionary.
                y_peptide = {atom: [] for atom in TARGET_ATOMS}

                # Reference poly-peptide list.
                ref_peptide = []

                # Localize append method.
                ref_peptide_append = ref_peptide.append

                # Separate all the input data.
                for entry in input_data[chain_id]:

                    # Extract only the middle one.
                    peptide = entry["poly-peptides"][mid_idx]

                    # Check all target atoms.
                    for atom in TARGET_ATOMS:

                        # Check for membership.
                        if atom in entry["targets"]:
                            # Add the data (numpy) vector.
                            data[atom].append(entry["vector"])

                            # Add the peptide information.
                            y_peptide[atom].append(peptide)
                        # _end_if_

                    # _end_for_

                    # Here we add all the peptides, because
                    # these will be the "reference" peptides.
                    ref_peptide_append(peptide)
                # _end_for_

                # Model predictions: Initialize them with "None".
                y_predict = {atom: None for atom in TARGET_ATOMS}

                # Run through all atom-models.
                for atom in TARGET_ATOMS:

                    # Convert the data to a DataFrame.
                    df_data = DataFrame(data[atom])

                    # Check if we have to scale the data.
                    if self.input_scaler[atom]:
                        df_data = DataFrame(self.input_scaler[atom].transform(df_data))
                    # _end_if_

                    # Get the model predictions (secondary structure values).
                    y_predict[atom] = self.nn_model[atom].predict(x=convert_to_tensor(df_data),
                                                                  batch_size=nn_batch_size,
                                                                  verbose=nn_verbose_flag)
                # _end_for_

                # Send the predictions to the "save method".
                self.save_to_file(f_path.stem, chain_seq, y_predict, y_peptide, ref_peptide,
                                  df_random_coil, model_id, chain_id, talos_format=talos_fmt)
            # _end_for_

            # Clean up the memory.
            collect_mem_garbage()
        # _end_for_

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
        Override to print a readable string presentation of the object.
        This will include its id(), along with its field values.

            NOTE: The overwrite protection is the opposite of the
            overwrite flag, so in the printed version we show the
            "not overwrite_flag"!

        :return: a string representation of a ChemShiftPredictor object.
        """

        # Local import of new line.
        from os import linesep as new_line

        # Return the f-string.
        return f" ChemShiftPredictor Id({id(self)}): {new_line}" \
               f" Models dir={self.input_path} {new_line}" \
               f" Output dir={self.output_path} {new_line}" \
               f" Overwrite protection={(not self.overwrite)}"
    # _end_def_

# _end_class_

"""
    Verify that we use the correct trained models. This test should
    FAIL if we use re-trained models, and it will have to be updated.
"""
import warnings

from Bio import BiopythonDeprecationWarning

# Disable all the Deprecation warning from Bio.
warnings.simplefilter('ignore', category=BiopythonDeprecationWarning)

import unittest
import joblib
from pathlib import Path
from src.random_coil.camcoil import CamCoil
from tensorflow.keras.models import load_model

class TestTrainedModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestTrainedModels - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestTrainedModels - FINISH -")
    # _end_def_

    def test_correct_hash(self):
        """
        Test that the MD5 hash-codes of the loaded models
        are the ones that we have distributed.

        :return: None.
        """

        # Dictionary with all the md5 hash codes.
        md5_models = {"ann_model_C": {"L0": "fe754138e3b645195e3c91ffc1d2fa43",
                                      "L1": "a850923a04d53e9a3953f1d283ad63cd"},

                      "ann_model_CA": {"L0": "d21dfa02f6c2d4891784e3e97e3a5389",
                                       "L1": "d31c515716ed3be8fd118db4908600f0"},

                      "ann_model_CB": {"L0": "224cff671db87a28ca2b88daa8f97cc3",
                                       "L1": "f3b7438f298c81808b872a96d9791889"},

                      "ann_model_H": {"L0": "d36a2780cff3bad897dbb34d294210d7",
                                      "L1": "0abdc96089a9a3e3396d451299740ef9"},

                      "ann_model_HA": {"L0": "8d8655d3b6553e4745d51d3c0695364a",
                                       "L1": "725cd7a4a09199bc9525554181bd6a86"},

                      "ann_model_N": {"L0": "96ede85338b9fd1c498e5804d1533540",
                                      "L1": "6ebd81dd66fc25b219e3707d478872c9"}
                      }

        # Go through all the trained models.
        for i in md5_models:

            # Load the model from the disk.
            model = load_model(Path(f"../../models/{i}.h5"), compile=False)

            # Iterate through all the layers of the model.
            for n, layer in enumerate(model.layers, start=0):

                # Compute the hash code of the model layer.
                hash_code_n = joblib.hash(layer.get_weights(),
                                          hash_name='md5')

                # Check against the correct md5-hash.
                self.assertEqual(md5_models[i][f"L{n}"], hash_code_n,
                                 msg=f"MD5-hash code for model: {i} "
                                     f"and layer: L{n} is not correct.")
            # _end_for_

        # _end_for_

    # _end_def_

    def test_camcoil(self):
        """
        Test that the MD5 hash-codes of the camcoil object.

        :return: None.
        """

        # Create an object to test the MD5 checksums.
        _ = CamCoil(pH=7.0)

    # _end_def_


# _end_class_


if __name__ == '__main__':
    unittest.main()

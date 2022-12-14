"""
    Verify that we use the correct trained models. This test should
    FAIL if we use re-trained models, and it will have to be updated.
"""

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
        md5_models = {"ann_model_C": {"L0": "1cef868127c4dbad6a02064bae9c35b7",
                                      "L1": "a1cfde3be501ea09c4e41bac24186635"},

                      "ann_model_CA": {"L0": "abe776bf09434eac53c5c112b9218d9a",
                                       "L1": "057c1234508d495aed0fd98f09cbcaae"},

                      "ann_model_CB": {"L0": "d579f38c905e614e179bafd954d94847",
                                       "L1": "5b28039b23b5f99a64419eb21a20a4b0"},

                      "ann_model_H": {"L0": "6b1dc3a020393ccecdd407e9b67d68db",
                                      "L1": "fede3f623bfd705bb1c1a35611a1a788"},

                      "ann_model_HA": {"L0": "cbd432da3ad3bfcf4f5fc75b8fb4c731",
                                       "L1": "0e12c7814945aa2144dd2ec65c5a062f"},

                      "ann_model_N": {"L0": "0d63cf1e3734851b601cfeca1d5a78f9",
                                      "L1": "b0d920474291d2cfe693ce94478759aa"}
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

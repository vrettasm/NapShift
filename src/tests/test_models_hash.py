"""
    Verify that we use the correct trained models. This test should
    FAIL if we use re-trained models, and it will have to be updated.
"""

import unittest
import joblib
from pathlib import Path
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
        md5_models = {"ann_model_C": {"L0": "2962dc76e0c8b0f447132d448ec65f19",
                                      "L1": "1ba97cdfea220bb0a32d3bb17683423f"},

                      "ann_model_CA": {"L0": "b1351a2fa2bd74779a98e42c3155be26",
                                       "L1": "9c6626ca94de3d482e8c1a298d213ddc"},

                      "ann_model_CB": {"L0": "c4e83a8297cd5005d5a1f961e56457b9",
                                       "L1": "3e14922c4adf229d333c71c206816d15"},

                      "ann_model_H": {"L0": "a1f2b766ffa810657d50a0fd805840ca",
                                      "L1": "86aeeb96b481835502c1c089d6eb16a7"},

                      "ann_model_HA": {"L0": "72bb03231e4518ceb088e0efab07de64",
                                       "L1": "e878deac7748b3a30ee34782b7826724"},

                      "ann_model_N": {"L0": "3fc22cc9484889d26107b91b67e5a096",
                                      "L1": "5a09907f833c9aa305b69a43ae842360"}
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

# _end_class_


if __name__ == '__main__':
    unittest.main()

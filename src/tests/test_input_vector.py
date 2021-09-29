import unittest
from pathlib import Path

import numpy as np

from src.chemical_shifts.input_vector import InputVector


class TestInputVector(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestInputVector - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestInputVector - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default setting.

        :return: None.
        """
        # Default directory for the PDB file.
        self.f_path = Path.cwd()

        # Default setting.
        self.test_obj = InputVector(blosum_id=62,
                                    include_hydrogen_bonds=False,
                                    check_aromatic_rings=True,
                                    data_type=np.float16)
    # _end_def_

    def test_invalid_initializations(self) -> None:
        """
        Test the constructor method for wrong initial
        values.

        :return: None
        """

        # Create an object with wrong (invalid) values).
        with self.assertRaises(ValueError):
            # Blosum number is wrong.
            _ = InputVector(blosum_id=162,
                            include_hydrogen_bonds=False,
                            check_aromatic_rings=True,
                            data_type=np.float16)
        # _end_with_

        with self.assertRaises(ValueError):
            # Data type is wrong.
            _ = InputVector(blosum_id=62,
                            include_hydrogen_bonds=False,
                            check_aromatic_rings=True,
                            data_type=np.int)
        # _end_with_

    # _end_def_

    def test_blosum(self) -> None:
        """
        Test the BLOSUM get/set methods.

        :return: None.
        """
        # New BLOSUM value.
        new_blosum = 45

        # Change the old value.
        self.test_obj.blosum_version = new_blosum

        # The BLOSUM should have been updated to the new value.
        self.assertEqual(new_blosum, self.test_obj.blosum_version)
    # _end_def_

    def test_hydrogen_bond_flag(self) -> None:
        """
        Test the hydrogen bond flag get/set methods.

        :return: None.
        """
        # New h-bonds flag value.
        h_bonds_flag = True

        # Change the old value.
        self.test_obj.hydrogen_bonds = h_bonds_flag

        # The H-Bonds flag should have been updated to the new value.
        self.assertEqual(h_bonds_flag, self.test_obj.hydrogen_bonds)
    # _end_def_

    def test_aromatic_rings_flag(self) -> None:
        """
        Test the aromatic rings flag get/set methods.

        :return: None.
        """
        # New h-bonds flag value.
        a_rings_flag = True

        # Change the old value.
        self.test_obj.aromatic_rings = a_rings_flag

        # The A-Rings flag should have been updated to the new value.
        self.assertEqual(a_rings_flag, self.test_obj.aromatic_rings)
    # _end_def_

    def test_get_sin_cos_one(self) -> None:
        """
        Test the sin_cos_one method with several angles.

        :return: None.
        """

        # Test set of angles.
        for angle in [0.0, 45.0, 90.0, 180.0, 270.0, 360.0]:
            # Print info.
            print(f" Testing angle: {angle}")

            # Get the tuple from the method.
            x_out = self.test_obj.sine_cosine(angle)

            # Convert to radians.
            rad_angle = np.deg2rad(angle)

            # Get the triplet using numpy functions.
            z_out = (np.sin(rad_angle), np.cos(rad_angle))

            # Compare all items.
            for i, j in zip(x_out, z_out):
                self.assertAlmostEqual(i, j)
            # _end_for_

        # _end_for_

    # _end_def_

    def test_call_method(self) -> None:
        """
        Test the call method of the class.

        :return: None.
        """

        # Accept only ODD peptide numbers.
        with self.assertRaises(ValueError):
            _, _ = self.test_obj(Path(self.f_path / "6U7Q.pdb"), n_peptides=2)
        # _end_with_

        # Number of BLOSUM elements.
        n_blosum = 22

        # Number of angles elements.
        n_angles = 8

        # Get the data and the sequence.
        for peptide in [1, 3, 5, 7]:

            # Construct the dataset.
            data_in = self.test_obj(Path(self.f_path / "6U7Q.pdb"),
                                    n_peptides=peptide)
            # Unpack the data.
            x_in = data_in["model-1"]["data"]
            seq0 = data_in["model-1"]["sequence"]["A"]

            # The sequence must always be the same.
            self.assertEqual(seq0, "GRXTKSIOPRXFPD")

            # Get the length of the input vector.
            vector_length = len(x_in["A"][0]["vector"])

            # The length must be equal to: (BLOSUM + ANGLES) x PEPTIDES.
            self.assertEqual((n_blosum+n_angles) * peptide, vector_length)
        # _end_for_

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()

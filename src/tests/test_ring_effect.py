import unittest

from src.chemical_shifts.ring_effect import RingEffects


class TestRingEffects(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestRingEffects - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestRingEffects - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default parameters.
        :return: None.
        """

        # Create the test object.
        self.test_object = RingEffects()
    # _end_def_

    def test_invalid_initializations(self) -> None:
        """
        Test the constructor method for wrong initial values.
        :return: None
        """

        # Create an object with wrong (invalid) values).
        with self.assertRaises(TypeError):
            # The distance input should be float.
            _ = RingEffects(d_ring=1)
        # _end_with_

        with self.assertRaises(ValueError):
            # The distance input should be positive.
            _ = RingEffects(d_ring=-2.0)
        # _end_with_
    # _end_def_

    def test_distance_methods(self) -> None:
        """
        Test the distance accessor method.
        :return: None
        """

        # Create an object with wrong (invalid) values).
        with self.assertRaises(TypeError):
            # The distance input should be float.
            self.test_object.distance = 1
        # _end_with_

        with self.assertRaises(ValueError):
            # The distance input should be float.
            self.test_object.distance = -2.0
        # _end_with_

        # New distance value.
        new_distance = 3.0

        # Change the old value.
        self.test_object.distance = new_distance

        # The distance should have been updated to the new value.
        self.assertEqual(new_distance, self.test_object.distance)

        # New flag value.
        new_flag = True

        # Update to the new value.
        self.test_object.include_five_atoms = new_flag

        # The flag should have been updated to the new value.
        self.assertEqual(new_flag, self.test_object.include_five_atoms)
    # _end_def_


if __name__ == '__main__':
    unittest.main()

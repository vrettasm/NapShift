import unittest

from src.chemical_shifts.hydrogen_bond import HBond


class TestHBond(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestHBond - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestHBond - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default parameters.
        :return: None.
        """
        self.test_object = HBond()
    # _end_def_

    def test_update_rule_distances(self) -> None:
        """
        Test the "update_rule" method for distances.
        :return: None
        """

        # New test value.
        new_value = 1.0

        # Valid Rule codes are ["D-A", "H-A", "PHI", "PSI"].
        with self.assertRaises(ValueError):
            # Check an empty code.
            self.test_object.update_rule(new_value)

            # Check a mis-typed code.
            self.test_object.update_rule(new_value, code_str="D-S")
        # _end_with_

        # Update a rule "D-A" with a new distance value.
        self.test_object.update_rule(new_value, code_str="D-A")

        # The rule should have been updated to the new value.
        self.assertEqual(new_value, self.test_object.rules["D-A"])

        # Update the rule "D-A" with a wrong (negative) distance.
        self.test_object.update_rule(-new_value, code_str="D-A")

        # The rule should have been the same (not changed).
        self.assertEqual(new_value, self.test_object.rules["D-A"])
    # _end_def_

    def test_update_rule_angles(self) -> None:
        """
        Test the "update_rule" method for angles.
        :return: None
        """

        # New test value.
        new_value = 100.0

        # Valid Rule codes are ["D-A", "H-A", "PHI", "PSI"].
        with self.assertRaises(ValueError):
            # Check an empty code.
            self.test_object.update_rule(new_value)

            # Check a mis-typed code.
            self.test_object.update_rule(new_value, code_str="CHI")
        # _end_with_

        # Update the rule "PHI" with a new angle value.
        self.test_object.update_rule(new_value, code_str="PHI")

        # The rule should have been updated to the new value.
        self.assertEqual(new_value, self.test_object.rules["PHI"])

        # Update the rule "PHI" with a wrong (negative) angle.
        self.test_object.update_rule(-new_value, code_str="PHI")

        # The rule should have been the same (not changed).
        self.assertEqual(new_value, self.test_object.rules["PHI"])
    # _end_def_


if __name__ == '__main__':
    unittest.main()

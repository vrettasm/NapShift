import unittest
from pathlib import Path

from Bio.PDB.PDBParser import PDBParser

from src.chemical_shifts.auxiliaries import get_sequence


class TestAuxiliaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestAuxiliaries - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestAuxiliaries - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default directory.
        This is set to the "current working directory".

        :return: None.
        """
        self.f_path = Path.cwd()
    # _end_def_

    def test_get_sequence(self) -> None:
        """
        Test the get_sequence() function.

        :return: None.
        """

        # Get the structure from the first PDB file.
        structure = PDBParser(PERMISSIVE=True, QUIET=True).get_structure("6U7Q",
                                                                         Path(self.f_path/"6U7Q.pdb"))
        # Get the sequence UNMODIFIED.
        seq_0 = get_sequence(structure[0], modified=False)

        # Get the "TRUE" sequence from the "FASTA" file.
        self.assertEqual(seq_0["A"], "GRCTKSIPPRCFPD")

        # Get the sequence MODIFIED.
        seq_1 = get_sequence(structure[0], modified=True)

        # In this sequence all the "PRO" residues are converted
        # to "PRT" and all "CYS" residues are converted to "CYO".
        self.assertEqual(seq_1["A"], "GRXTKSIOPRXFPD")

        # They must NOT be the same.
        self.assertNotEqual(seq_0, seq_1)
    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()

import itertools
import random
import unittest

from impromptica import settings
from impromptica import utils


class TestNoteGeneration(unittest.TestCase):

    def setUp(self):
        random.seed(0)  # so tests run the same every time
        self.valid_notes = range(settings.MAX_NOTE + 1)  # all valid notes
        # `test_notes` contains a selection of important note values (e.g. edge
        # cases, some notes in the range [0, MAX_NOTE) for testing various
        # functions.
        self.test_notes = [
            0,
            1,
            settings.MAX_NOTE / 2,
            settings.MAX_NOTE - 1,
            settings.MAX_NOTE
        ]

    def test_generate_note(self):
        """Tests that `generate_note` returns a valid note.

        Executes `generate_note` multiple times with varying paramters.
        """

        # For parameters to `generate_note`, we use all permutations of length
        # three of the notes in `test_notes`.
        for args in itertools.permutations(self.test_notes, 3):
            note = utils.generate_note(args[0], args[1], [args[2]])
            assert note in self.valid_notes, "%d is not a valid note" % (note)

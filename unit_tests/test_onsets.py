import unittest

import numpy

from impromptica.utils import onsets
from impromptica.utils import sound


class TestOnsets(unittest.TestCase):

    def setUp(self):
        self.quick_note_length = 0.2
        self.quick_notes_frequencies = []
        for i in range(12):
            self.quick_notes_frequencies.append(
                sound.note_to_frequency(i + 60))

    def test_get_onsets_on_quick_onsets(self):
        """Tests that `get_onsets` returns the correct number of onsets for
        a series of quick notes.
        """
        samples = []
        for f in self.quick_notes_frequencies:
            note_samples = sound.generate_note(self.quick_note_length, 0.8, f)
            samples = numpy.append(samples, note_samples)
        onset_list = onsets.get_onsets(samples, 44100)[0]
        assert len(onset_list) == len(self.quick_notes_frequencies)

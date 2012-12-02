import unittest

from impromptica import settings
from impromptica.utils import sound


class TestNoteFrequencyConversion(unittest.TestCase):

    def setUp(self):
        self.notes = range(settings.MAX_NOTE + 1)

    def test_middle_c_maps_to_note_value_60(self):
        """Tests that C4 (middle C) maps to a note value of 60."""
        c4freq = 261.63  # in Hertz
        c4note = sound.frequency_to_note(c4freq)
        assert c4note == 60, "C4 maps to note value %d, not 60" % (c4note)

    def test_note_conversion_back_and_forth(self):
        """Tests that notes converted to frequencies and back are the same
        notes."""
        for note in self.notes:
            frequency = sound.note_to_frequency(note)
            result_note = sound.frequency_to_note(frequency)
            assert note == result_note, "note %d != %d" % (note, result_note)

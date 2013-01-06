import unittest

import numpy

from impromptica import settings
from impromptica.utils import keys
from impromptica.utils import sound


class TestKeyfinding(unittest.TestCase):

    def setUp(self):
        self.test_keys = []
        for tonic in range(12):
            # test major chord
            self.test_keys.append([[tonic, tonic + 4, tonic + 7], [tonic, 1]])
            # test minor chord
            self.test_keys.append([[tonic, tonic + 3, tonic + 7], [tonic, 0]])

    @staticmethod
    def error_message(found_key, correct_key):
        return "%s (found key) != %s (correct key)" % (found_key, correct_key)

    def test_keys(self):
        """Tests that `keys` correctly returns the key of simple arpeggiated
        notes.
        """
        for notes, correct_key in self.test_keys:
            # find frequencies for the notes (translated upward 5 octaves)
            frequencies = [sound.note_to_frequency(n + 60) for n in notes]
            samples = []
            for f in frequencies:
                note_samples = sound.generate_note(0.2, 0.8, f)
                samples = numpy.append(samples, note_samples)
            # Manually specify the onsets since our onset detection isn't
            # reliable enough yet to detect these every time.
            onset_list = [
                settings.SAMPLE_RATE / 5 * i +
                settings.SAMPLE_RATE / 10 for i in range(len(frequencies))]
            key = keys.get_keys(samples, onset_list,
                                samples_per_segment=len(samples))[0][1]
            assert key == correct_key, self.error_message(key, correct_key)

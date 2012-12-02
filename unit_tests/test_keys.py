import unittest

import numpy

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
            for i in range(4):
                for f in frequencies:
                    note_samples = sound.generate_note(0.2, 0.8, f)
                    samples = numpy.append(samples, note_samples)
            sound.write_wav(samples, "temp.wav")
            key = keys.get_keys(samples, [0], 44100)[0][1]
            assert key == correct_key, self.error_message(key, correct_key)

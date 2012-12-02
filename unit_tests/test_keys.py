import unittest

import numpy

from impromptica.utils import keys
from impromptica.utils import sound


class TestKeyfinding(unittest.TestCase):

    def setUp(self):
        self.test_keys = []
        for tonic in range(12):
            # translate note to a common octave
            note = tonic + sound._MIDDLE_OCTAVE * 12
            # test major chord
            self.test_keys.append([[note, note + 4, note + 7], [note, 1]])
            # test minor chord
            self.test_keys.append([[note, note + 3, note + 7], [note, 0]])

    @staticmethod
    def error_message(found_key, correct_key):
        return "%s (found key) != %s (correct key)" % (found_key, correct_key)

    def test_keys(self):
        """Tests that `keys` correctly returns the key of simple arpeggiated
        notes.
        """
        for notes, correct_key in self.test_keys:
            frequencies = [sound.note_to_frequency(note) for note in notes]
            samples = []
            for i in range(4):
                for f in frequencies:
                    note_samples = sound.generate_note(0.2, 0.8, f)
                    samples = numpy.append(samples, note_samples)
            sound.write_wav(samples, "temp.wav")
            key = keys.get_keys(samples, [0], 44100)[0]
            assert key == correct_key, self.error_message(key, correct_key)

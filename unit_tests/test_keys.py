import unittest

from impromptica import utils


class TestKeyfinding(unittest.TestCase):

    def setUp(self):
        self.test_keys = []
        for tonic in range(12):
            # test major chord
            self.test_keys.append([[tonic, tonic + 4, tonic + 7], [tonic, 1]])
            # test minor chord
            #self.test_keys.append([[tonic, tonic + 3, tonic + 7], [tonic, 0]])

    @staticmethod
    def error_message(found_key, correct_key):
        return "%s (found key) != %s (correct key)" % (found_key, correct_key)

    def test_keys(self):
        """Tests that `keys` correctly returns the key of simple chords."""

        for notes, correct_key in self.test_keys:
            frequencies = [utils.note_to_frequency(note) for note in notes]
            samples = utils.generate_chord(1.0, 0.8, frequencies)
            key = utils.get_keys(samples, [0], 44100)[0]
            assert key == correct_key, self.error_message(key, correct_key)

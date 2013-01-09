import random
import unittest

import numpy

from impromptica.utils import sound
from impromptica.utils import tempo


class TestTempo(unittest.TestCase):

    def setUp(self):
        frequencies = [sound.note_to_frequency(n + 60) for n in range(12)] * 2
        random.shuffle(frequencies)

        self.samples = []
        for f in frequencies:
            samples = sound.generate_note(0.25, 0.8, f)
            self.samples = numpy.append(self.samples, samples)
        self.sample_rate = 44100.

    def test_get_meter_returns_correct_number_of_arguments(self):
        """Tests that `get_meter` returns four lists."""
        pulses = tempo.get_meter(self.samples, self.sample_rate, verbose=True)
        assert len(pulses) == 4, "`get_meter` did not return four lists."

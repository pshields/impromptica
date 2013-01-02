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

    def test_pulses_returns_correct_number_of_arguments(self):
        """Tests that `pulses` returns three lists."""
        pulses = tempo.get_meter(self.samples, self.sample_rate, verbose=True)
        assert len(pulses) == 3, "Pulses did not return three lists."

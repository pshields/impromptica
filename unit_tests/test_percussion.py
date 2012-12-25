import unittest

import numpy

from impromptica import settings
from impromptica.utils import percussion


class TestPercussion(unittest.TestCase):

    def test_get_drumkit_samples(self):
        """Tests that `get_drumkit_samples` returns correct output."""
        samples_list = percussion.get_drumkit_samples(settings.DRUMKIT_DIR)
        assert len(samples_list) > 0, "Samples list is empty"
        for samples in samples_list:
            assert samples.dtype == numpy.double, (
                "Drumkit samples are not a numpy array of the correct dtype.")

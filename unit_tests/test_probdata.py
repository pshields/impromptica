import unittest

from impromptica import probdata


class TestProbabilitiesSumToOne(unittest.TestCase):

    def setUp(self):
        self.epsilon = 0.001
        self.l = 1.0 - self.epsilon
        self.r = 1.0 + self.epsilon

    def sum_distance_profile(self, profile):
        # Every non-zero index k in a distance profile is accessed on two
        # occasions, given note values i and j. The first occasion is when
        # k = i - j, and the second is when k = j - i. Therefore we include
        # each non-zero index in this profile twice when calculating the sum
        # of probabilities.
        return profile[0] + 2.0 * sum(profile[1:])

    def error_message(self, s):
        return "sum %f not in [%f, %f]" % (s, self.l, self.r)

    def test_major_key_profile(self):
        s = sum(probdata.MAJOR_KEY_PROFILE)
        assert self.l <= s <= self.r, self.error_message(s)

    def test_minor_key_profile(self):
        s = sum(probdata.MINOR_KEY_PROFILE)
        assert self.l <= s <= self.r, self.error_message(s)

    def test_proximity_profile(self):
        s = self.sum_distance_profile(probdata.PROXIMITY_PROFILE)
        assert self.l <= s <= self.r, self.error_message(s)

    def test_range_profile(self):
        s = self.sum_distance_profile(probdata.RANGE_PROFILE)
        assert self.l <= s <= self.r, self.error_message(s)

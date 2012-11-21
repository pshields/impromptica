"""Impromptica utilities related to probability."""
from scipy import stats


def build_distance_profile(standard_deviation):
    """Builds a distance profile using the given standard deviation (as a
    distance between note values.)

    The profile is built from a Gaussian distribution, which is an
    approximation for the actual data.

    The result is a table, where the probability of note value j given
    reference note value i is located at the index equal to the absolute
    value of j - i.
    """
    result = []
    dist = stats.norm(0, standard_deviation)
    for i in range(120):
        result.append(dist.pdf(i))
    return result

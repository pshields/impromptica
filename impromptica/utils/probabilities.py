"""Impromptica utilities related to probability."""
from scipy import stats


def build_proximity_profile(standard_deviation):
    """Builds a proximity profile using the given standard deviation (as a
    distance between note values.)

    The result is a table, where the probability of note value j given note
    value i is located at the index equal to the absolute value of j - i.
    """
    result = []
    dist = stats.norm(0, standard_deviation)
    for i in range(60):
        result.append(dist.pdf(i))
    return result

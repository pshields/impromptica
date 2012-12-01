from onsets import get_onsets
from itertools import combinations
from fractions import gcd


def get_tatum(filename):
    """
    Naive approach for computing the duration of a tatum for a music piece.
    A Tatum is the lowest level of the Metrical Structure of music.
    Implementation uses Inter-Onset Intervals of all pairs of onsets under a
    maximum limit, and returns the tatum as its gcd.
    The algorithm does not accommodate for tatum changes
    (e.g. accelerandos and ritardandos)
    Reference: http://www.cs.tut.fi/sgn/arg/music/jams/waspaa2001.pdf
    """
    onsets = get_onsets(filename)[0]
    ioi_cieling = max([(b - a) for a, b in zip(onsets, onsets[1:])])
    tatum = onsets[0]
    ioi_set = []
    for i in combinations(list(onsets), 2):
        tmp = abs(i[1] - i[0])
        if tmp <= ioi_cieling:
            ioi_set.append(tmp)
    for j in ioi_set:
        tatum = gcd(tatum, j)
    return tatum

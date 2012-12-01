"""Probability data from various sources.

In this module, the "Essen corpus" refers to a corpus of 6,217 European folk
songs from the Essen Folksong Collection. The songs are available at
http://kern.ccarh.org/cgi-bin/ksbrowse?l=/essen and the list of songs used to
train the monophonic key and meter programs is published at
http://theory.esm.rochester.edu/temperley/music-prob/data/essen-train-list.

The "Kostka-Payne corpus" refers to 46 excerpts from the common-practice
repertoire, appearing in the workbook for the textbook "Tonal Harmony" by
Stefan Kostka and Dorothy Payne. The list of of the songs in the corpus is
published at http://theory.esm.rochester.edu/temperley/music-prob/data/kp-list.

The source code and data for "Music and Probability" (Temperley 2007), which
we use for much of our probalistic data, is published at
http://theory.esm.rochester.edu/temperley/music-prob/materials.html.
"""

from scipy import stats

from impromptica import settings


def build_distance_profile_data(standard_deviation):
    """Builds distance profile data using the given standard deviation (as a
    distance between note values.)

    The profile data is built from a Gaussian distribution, which is an
    approximation for the actual data.

    The result is a table, where the probability of note value j given
    reference note value i is located at the index equal to the absolute
    value of j - i.
    """
    result = []
    dist = stats.norm(0, standard_deviation)
    for i in range(settings.MAX_NOTE + 1):
        result.append(dist.pdf(i))
    return result

# This monophonic key profile generated from the Essen corpus provides
# probabilities of the offset of a note from the tonic note of a major key.
# This profile sums to 1 because it represents the probability that the next
# monophonic note is the given index offset from the tonic note of the key.
# Source: David Temperley. Music and Probability (Figure 4.7).
ESSEN_MAJOR_KEY_PROFILE_DATA = [
    0.184,
    0.001,
    0.155,
    0.003,
    0.191,
    0.109,
    0.005,
    0.214,
    0.001,
    0.078,
    0.004,
    0.055,
]

# This monophonic key profile generated from the Essen corpus provides
# probabilities of the offset of a note from the tonic note of a minor key.
# This profile sums to 1 because it represents the probability that the next
# monophonic note is the given index offset from the tonic note of the key.
# Source: David Temperley. Music and Probability (Figure 4.7).
ESSEN_MINOR_KEY_PROFILE_DATA = [
    0.192,
    0.005,
    0.149,
    0.179,
    0.002,
    0.144,
    0.002,
    0.201,
    0.038,
    0.012,
    0.053,
    0.022,
]

# This polyphonic key profile generated from the Kostka-Payne corpus provides
# probabilities of the offset of a note from the tonic note of a major key.
# This profile doesn't sum to 1 because we view notes as independent variables
# representing whether that note is present in a segment of the given key.
# Source: David Temperley. Music and Probability (Figure 6.4).
KP_MAJOR_KEY_PROFILE_DATA = [
    0.748,
    0.060,
    0.488,
    0.082,
    0.670,
    0.460,
    0.096,
    0.715,
    0.104,
    0.366,
    0.057,
    0.400
]

# This polyphonic key profile generated from the Kostka-Payne corpus provides
# probabilities of the offset of a note from the tonic note of a minor key.
# This profile doesn't sum to 1 because we view notes as independent variables
# representing whether that note is present in a segment of the given key.
# Source: David Temperley. Music and Probability (Figure 6.4).
KP_MINOR_KEY_PROFILE_DATA = [
    0.712,
    0.084,
    0.474,
    0.618,
    0.049,
    0.460,
    0.105,
    0.747,
    0.404,
    0.067,
    0.133,
    0.330
]

# This proximity profile generated from the Essen corpus provides
# probabilities of the distance of a note from the previous note.
# Source: David Temperley. Music and Probability (Table 4.1).
PROXIMITY_PROFILE_DATA = build_distance_profile_data(7.2)

# This range profile generated from the Essen corpus provides probabilities
# of the distance of a note from the central pitch. The central pitch is
# essentially the mean note value of over a song.
# Source: David Temperley. Music and Probability (Table 4.1).
RANGE_PROFILE_DATA = build_distance_profile_data(29.0)

"""Probability data from various sources.

In this module, the "Essen corpus" refers to a corpus of 6,217 European folk
songs from the Essen Folksong Collection. The songs are available at
http://kern.ccarh.org/cgi-bin/ksbrowse?l=/essen and the list of songs used to
train the monophonic key and meter programs is published at
http://theory.esm.rochester.edu/temperley/music-prob/data/essen-train-list.
"""
from impromptica import utils

# This monophonic key profile generated from the Essen corpus provides
# probabilities of the offset of a note from the tonic note of a major key.
# Source: David Temperley. Music and Probability (Figure 4.7).
MAJOR_KEY_PROFILE = [
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
# Source: David Temperley. Music and Probability (Figure 4.7).
MINOR_KEY_PROFILE = [
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

# This proximity profile generated from the Essen corpus provides
# probabilities of the distance of a note from the previous note.
# Source: David Temperley. Music and Probability (Table 4.1).
PROXIMITY_PROFILE = utils.build_proximity_profile(7.2)

"""Utilities for analyzing the high-level structure of a musical piece.

References:

1. Paulus, Jouni, and Anssi Klapuri. "Music structure analysis using a
   probabilistic fitness measure and a greedy search algorithm." Audio, Speech,
   and Language Processing, IEEE Transactions on 17.6 (2009): 1159-1170.
"""
import numpy as np

from impromptica.utils import similarity
from impromptica.utils import visualization


def calculate_structure(
        samples, boundaries, pulse_salience, tempo_hop_size, visualize=False):
    """Calculates the structure of the musical piece.

    `boundaries` is an array of possible segment boundaries, as indices into
    `pulse_salience`.

    The structure is defined as a segmentation of the piece into segments at
    the tactus half-beat level, and partioning segments into groups by
    similarity.
    """
    # Remove boundaries that we don't have pulse salience information for.
    for i in range(boundaries.shape[0]):
        if boundaries[i] / tempo_hop_size + 1 >= pulse_salience.shape[0]:
            boundaries = boundaries[:i - 1]
            break
    # Calculate pulse salience vectors for each unit segment by averaging the
    # pulse salience frames between the beginning and end of each segment.
    rhythm = np.zeros((boundaries.shape[0] - 1, pulse_salience.shape[1]))
    for i in range(boundaries.shape[0] - 1):
        first, last = (boundaries[i] / tempo_hop_size,
                       boundaries[i + 1] / tempo_hop_size + 1)
        rhythm[i] = np.average(pulse_salience[first:last], axis=0)
    # Crop rhythm vectors for the last few beats, which don't have pulse
    # salience associated with them.
    rhythm_similarity = np.zeros(
        (boundaries.shape[0] - 1, boundaries.shape[0] - 1))
    for i in range(rhythm_similarity.shape[0]):
        for j in range(i, rhythm_similarity.shape[0]):
            rhythm_similarity[i][j] = rhythm_similarity[j][i] = similarity.l2(
                rhythm[i], rhythm[j])
    # Normalize the similarity matrix to have a maximum value of 1.
    max_value = np.max(np.max(rhythm_similarity, axis=1))
    if max_value > 0:
        rhythm_similarity /= max_value
    if visualize:
        # Visualize the self-similarity matrix of the rhythm feature.
        visualization.show_self_similarity_matrix(
            rhythm_similarity, "Rhythm similarity")
    return [[]]

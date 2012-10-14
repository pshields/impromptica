"""
Tempo extraction tools.
Different algorithms to evaluate the BPM of given input.
Works with audio formatted in 1-D numpy arrays.
"""

from numpy import copy


def map_pass(samples, low_bpm, high_bpm):
    """
    Estimates the consistent BPM for an audio sample based on
    the peaks in amplitude and the best fit bpm between the range
    (low_bpm, high_bpm).
    """
    #First pass through and zero out all but the
    #5% of samples with the largest amplitudes
    top_samples = copy(samples)
    top_samples.sort()
    top_samples = top_samples[-len(top_samples) * 0.05:]

    cut = min(top_samples)
    print(cut)

    filtered_samples = copy(samples)

    for pos, sample in enumerate(filtered_samples):
        if sample < cut:
            filtered_samples[pos] = 0

    return filtered_samples

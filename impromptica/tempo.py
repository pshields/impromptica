"""
Tempo extraction tools.
Different algorithms to evaluate the BPM of given input.
Works with audio formatted in 1-D numpy arrays.
"""

from numpy import copy


def map_pass(samples, frame_rate, low_bpm, high_bpm):
    """
    Estimates the consistent BPM for an audio sample based on
    the peaks in amplitude and the best fit bpm between the range
    (low_bpm, high_bpm).
    """
    #First pass through and zero out all but the
    #5% of samples with the largest amplitudes
    top_samples = copy(samples)
    top_samples.sort()
    top_samples = top_samples[-len(top_samples) * 0.10:]

    cut = min(top_samples)

    filtered_samples = copy(samples)

    for pos, sample in enumerate(filtered_samples):
        if abs(sample) < cut:
            filtered_samples[pos] = 0

    best_bpm = 0
    best_ratio = 0

    #Try to map every bpm in the range, and see which one best fits
    for bpm in range(low_bpm, high_bpm + 1):
        sample_rate_step = frame_rate * 60.0 / bpm
        cur_sample = sample_rate_step
        hits = 0.0
        steps = 0.0
        while cur_sample < len(filtered_samples):
            steps += 1
            if filtered_samples[cur_sample] != 0:
               hits += 1 
            cur_sample += sample_rate_step

        hit_ratio = hits/steps

        if(hit_ratio > best_ratio):
            best_bpm = bpm
            best_ratio = hit_ratio

    return best_bpm

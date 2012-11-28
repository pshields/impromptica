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
    NOTE: Not useful for audio where the tempo changes.
    """
    filtered_samples = copy(samples)
    # Square values to accentuate amplitudes

    # First pass through and zero out all but the
    # 5% of samples with the largest amplitudes
    top_samples = copy(filtered_samples)
    top_samples.sort()
    top_samples = top_samples[-len(top_samples) * 0.10:]

    cut = top_samples[0]

    for pos, sample in enumerate(filtered_samples):
        if abs(sample) < cut:
            filtered_samples[pos] = 0

    return map_best_beat(filtered_samples, low_bpm, high_bpm, frame_rate)


def map_best_beat(filtered_samples, low_bpm, high_bpm, frame_rate):
    # Try to map every bpm in the range, and see which one best fits
    best_bpm = 0
    most_hits = 0

    for bpm in range(low_bpm, high_bpm + 1):
        sample_rate_step = frame_rate * 60.0 / bpm
        cur_sample = sample_rate_step
        hits = 0.0
        while cur_sample < len(filtered_samples):
            if filtered_samples[cur_sample] != 0:
                hits += 1
            cur_sample += sample_rate_step

        if(hits > most_hits):
            best_bpm = bpm
            most_hits = hits

    # Check to make sure half or double the tempo isn't correct
    best_ratio = 0.0

    sample_rate_step = frame_rate * 60.0 / best_bpm
    sample_steps = [sample_rate_step, sample_rate_step / 2,
                    sample_rate_step * 2]

    for sample_step in sample_steps:
        hits = 0.0
        steps = 0.0
        cur_sample = sample_step
        while cur_sample < len(filtered_samples):
            if filtered_samples[cur_sample] != 0:
                hits += 1
            steps += 1
            cur_sample += sample_step

        if hits / steps > best_ratio:
            best_ratio = hits / steps
            best_bpm = (frame_rate * 60) / sample_step

    return best_bpm

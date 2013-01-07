"""Utilities for learning the tempo of a piece.

Works at the tatum, tactus and measure levels.

References:

* Ellis, Daniel PW. "Beat tracking by dynamic programming." Journal of New
  Music Research 36.1 (2007): 51-60.
* Klapuri, Anssi P., Antti J. Eronen, and Jaakko T. Astola. "Analysis of the
  meter of acoustic musical signals." Audio, Speech, and Language Processing,
  IEEE Transactions on 14.1 (2006): 342-355.
"""
import math

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from impromptica import probdata
from impromptica import settings
from impromptica.utils import novelty


# The beat placement algorithm assigns a penalty for placing a beat at a tempo
# other than the target one. The penalty is a function of the square of the
# logarithm of the change in tempo. `_BEAT_PLACEMENT_STRICTNESS` provides a
# scalar factor to balance this penalty with that of placing a beat at segment
# which has a low level of musical accentuation.
# found to work well.
_BEAT_PLACEMENT_STRICTNESS = 1000.


def calculate_pulse_salience(novelty_signal, frame_size, hop_size):
    """Returns salience information on period hypotheses.

    Period hypothesis are integers representing period length in segments.
    """
    # `frames_size` is the number of output frames.
    frames_size = (novelty_signal.shape[0] - frame_size) / hop_size + 1
    # `normalization_factor` is a denominator applied to each hypothesis term.
    normalization_factor = np.zeros(frame_size - 1)
    for i in range(frame_size - 1):
        normalization_factor[i] = 1 + frame_size - i
    # Allocate the resulting tempo salience signal.
    result = np.zeros((max(frames_size, 1), frame_size))
    if frames_size < 1:
        # Handle the special case where there is only a single frame.
        frame = np.zeros(frame_size)
        frame[:novelty_signal.shape[0]] = novelty_signal
        result[0][:frame_size - 1] = np.correlate(frame, frame, mode='full')[
            frame_size:2 * frame_size] / normalization_factor
    else:
        # Calculate the autocorrelation in each frame.
        for i in range(frames_size):
            frame = novelty_signal[i * hop_size:i * hop_size + frame_size]
            result[i][:frame_size - 1] = np.correlate(
                frame, frame, mode='full')[
                    frame_size:2 * frame_size] / normalization_factor
    return result


def calculate_periods(
        max_multiple, pulse_salience, prior, ptransition, hop_size,
        periods_size):
    """Calculates the most probable period at each segment.

    `prior` is an array of length `max_multiple` giving the prior probability
    of a period of the given length at the current metrical level.

    This function assigns a tempo, as a integer multiple of the segment
    duration, to each sample in the pulse salience signal.
    """
    # Allocate the dynamic programming table.
    tempo = np.zeros((pulse_salience.shape[0], max_multiple), dtype=np.int)
    scores = np.zeros((pulse_salience.shape[0], max_multiple))
    # Fill in the first row of the dynamic programming table.
    scores = prior * pulse_salience
    scores[0] /= scores[0].max()
    best = np.argmax(scores[0]) + 1
    for i in range(max_multiple):
        tempo[0][i] = best
    # Fill in the rest of the table. We also take into account the transition
    # probability between two tempos. At each potential tempo change segment,
    # calculate the lowest-cost route to each possible period using the `score`
    # temporary array.
    score = np.zeros(max_multiple)
    # At each index, only compute the score for a few relevant periods. Leave
    # the others at zero. This speeds up computation considerably. The relavant
    # periods are the top scores from the previous sample (with buffers to
    # account for minor tempo changes), and the top current scores at the
    # current index.
    for i in range(1, pulse_salience.shape[0]):
        score *= 0.
        # Figure out the relevant periods at this index.
        relevant = []
        buf = []
        for j in np.argsort(scores[i - 1])[scores.shape[1] - 5:]:
            relevant.append(j)
        for j in np.argsort(scores[i])[scores.shape[1] - 5:]:
            relevant.append(j)
        for j in relevant:
            for k in range(j - 1, j + 2):
                buf.append(k)
        relevant.extend(buf)
        relevant = sorted(list(set(relevant)))
        for j in relevant:
            # For each period in the previous possible tempo change segment,
            # calculate what the score would be at this segment and period if
            # transitioning from it.
            for k in relevant:
                score[k] = scores[i - 1][k] * ptransition[k][j]
            best = np.argmax(score)
            tempo[i][j] = best + 1
            scores[i][j] *= score[best]
        for j in range(max_multiple):
            if j not in relevant:
                scores[i][j] = 0.
        scores[i] /= scores[i].max()
    # Assign the best period to each segment in reverse order.
    periods = np.zeros(periods_size, dtype=np.int)
    best = tempo[-1][np.argmax(scores[-1])]
    for i in range(1, scores.shape[0]):
        periods[i * hop_size + pulse_salience.shape[1] / 2] = best
        best = tempo[-i][best - 1]
    # Fill in the segments that are before the first period estimate.
    for i in range(min(periods_size, pulse_salience.shape[1] / 2)):
        periods[i] = best
    # Fill in the gaps.
    for i in range(pulse_salience.shape[1] / 2, periods.shape[0]):
        if periods[i] == 0:
            periods[i] = periods[i - 1]
    # Calculate the change in tempo throughout the course of the song.
    changes = []
    current_period = -1
    for i in range(periods.shape[0]):
        if current_period != periods[i]:
            current_period = periods[i]
            changes.append((i, current_period))
    return (periods, scores, changes)


def calculate_beats(periods, novelty_signal, beat_placement_strictness):
    """Returns beat indices fitted to period estimates and accentuation.

    This function calculates beat times for any metrical level using dynamic
    programming. We want to minimize a cost function defined over a set
    of beat indices as the sum of the novelty at each index and a function at
    each beat index of the distance between the resulting tempo at that point
    and the target tempo, as measured in periods since the previous beat.

    For simplicity we assume beat indices lie at the beginning of segments,
    rather than calculating over the individual audio samples directly.

    `periods` is an array of period targets for the segments of the piece.

    `beat_placement_strictness` is a factor applied to the penalty for placing
    beats at distances other than the target distance
    """
    # The `beat_cost` array tracks the best cost of an optimal set of beat
    # indices ending with a beat at the given index.
    beat_cost = np.zeros(periods.shape[0])
    # The `previous_beat` array holds a reference to the previous beat index
    # in the optimal set of beat indices ending at the given index.
    previous_beat = np.ones(periods.shape[0], dtype=np.int) * -1
    # Allocate a temporary array for calculating the lowest-cost previous index
    # at each step. Rather than checking every previous cell in `beat_cost`, we
    # check only the previous 2*n cells, where n is the highest period tempo
    # found found anywhere in the piece. If the optimal previous beat were
    # further back than that, we'd be doing something wrong.
    costs = np.zeros(np.max(periods) * 2)
    # Fill in the `beat_cost` and `previous_beat` tables. For each segment
    # within the first segment's target period, the cost of beat placement is
    # only a function of the novelty at that segment. For all other segments,
    # we also account for how close the resulting tempo would be to the target
    # tempo if we placed a beat at that segment.
    for i in range(min(novelty_signal.shape[0], periods[0])):
        beat_cost[i] = (1. - novelty_signal[i])
    for i in range(periods[0], periods.shape[0]):
        target_period = periods[i]
        # Calculate the optimal previous beat index, assuming a beat is placed
        # at the current index. `furthest_back` is the maximum number of
        # previous indices to check.
        furthest_back = min(i, costs.shape[0])
        for j in range(1, furthest_back + 1):
            # Provide one segment of free leeway between the target and the
            # actual tempo, in case our tempo recognition was slightly off.
            actual_tempo = j
            if actual_tempo < target_period:
                actual_tempo += 1
            elif actual_tempo > target_period:
                actual_tempo -= 1
            costs[j - 1] = beat_cost[i - j] + beat_placement_strictness * (
                math.pow(math.log(float(actual_tempo) / target_period), 2.))
        best = np.argmin(costs[:furthest_back])
        previous_beat[i] = i - best - 1
        beat_cost[i] = costs[best] + (1. - novelty_signal[i])
    # After calculating the array values from beginning to end, the optimal set
    # of beat indices is the set found from backtracking through
    # `previous_beat` starting at the lowest-cost index in the last m indices,
    # where m is the target period of the last segment. Start by finding the
    # lowest-cost index from the the last m indices of `beat_cost`.
    i = np.argmin(beat_cost[periods.shape[0] - periods[-1]:]) + (
        periods.shape[0] - periods[-1])
    best_cost = beat_cost[i]
    # Accumulate the optimal beat indices into a list.
    beats = []
    while i >= 0:
        beats.append(i)
        i = previous_beat[i]
    # Reverse the list to get the in-order sequence of beat indices.
    beats.reverse()
    return (beats, best_cost)


def get_meter(
        samples, hop_size=settings.NOVELTY_HOP_SIZE,
        window_size=settings.NOVELTY_WINDOW_SIZE,
        interpolation_factor=settings.NOVELTY_INTERPOLATION_FACTOR,
        tempo_frame_size=settings.TEMPO_FRAME_SIZE,
        tempo_hop_size=settings.TEMPO_HOP_SIZE,
        beat_placement_strictness=_BEAT_PLACEMENT_STRICTNESS,
        sample_rate=settings.SAMPLE_RATE,
        verbose=False, visualize=False):
    """Returns (tatums, tactus, measures) pulse indices."""
    if verbose:
        print("Beginning tempo recognition...")
    # Calculate the novelty of a various frequency bands across segments.
    if verbose:
        print("Calculating novelty...")
    novelty_signal = novelty.calculate_novelty(
        samples, verbose, verbose=verbose)
    # Calculate the salience of various pulse period hypotheses at each
    # sample in the novelty signal by calculating the autocorrelation of
    # overlapping frames of `
    if verbose:
        print("Calculating pulse salience...")
    pulse_salience = calculate_pulse_salience(
        novelty_signal, tempo_frame_size, tempo_hop_size)
    if verbose:
        print("Searching for best meter...")
    # Calculate prior probability distributions for the metrical levels. Right
    # now these are all offset by one (e.g. the prior probability of n periods
    # is located at index n - 1 in each of the following lists.)
    ptatum = probdata.build_lognorm_tempo_profile_data(
        0.39, 0.18, hop_size / interpolation_factor / sample_rate,
        tempo_frame_size)
    ptactus = probdata.build_lognorm_tempo_profile_data(
        0.28, 0.55, hop_size / interpolation_factor / sample_rate,
        tempo_frame_size)
    pmeasure = probdata.build_lognorm_tempo_profile_data(
        0.26, 2.1, hop_size / interpolation_factor / sample_rate,
        tempo_frame_size)
    # Precompute the transition probabilities for all possible transitions
    # between hypotheses.
    ptransition = probdata.build_tempo_change_profile_data(tempo_frame_size)
    tactus_periods, tactus_scores, tactus_changes = calculate_periods(
        tempo_frame_size, pulse_salience, ptactus, ptransition,
        tempo_hop_size, novelty_signal.shape[0])
    tatum_periods, tatum_scores, tatum_changes = calculate_periods(
        tempo_frame_size, pulse_salience, ptatum, ptransition,
        tempo_hop_size, novelty_signal.shape[0])
    measure_periods, measure_scores, measure_changes = calculate_periods(
        tempo_frame_size, pulse_salience, pmeasure, ptransition,
        tempo_hop_size, novelty_signal.shape[0])
    if verbose:
        print("Tactus changes: %s" % (str(tactus_changes)))
        print("Tatum changes: %s" % (str(tatum_changes)))
        print("Measure changes: %s" % (str(measure_changes)))
        print("Finding beats...")
    tactus, best_tactus_cost = calculate_beats(
        tactus_periods, novelty_signal, beat_placement_strictness)
    tactus = [(i * hop_size + window_size / 2) / interpolation_factor
              for i in tactus]
    tatums, best_tatum_cost = calculate_beats(
        tatum_periods, novelty_signal, beat_placement_strictness)
    tatums = [(i * hop_size + window_size / 2) / interpolation_factor
              for i in tatums]
    measures, best_measure_cost = calculate_beats(
        measure_periods, novelty_signal, beat_placement_strictness)
    measures = [(i * hop_size + window_size / 2) / interpolation_factor
                for i in measures]
    if verbose:
        medians = [np.median(x, axis=0) for x in (
            measure_periods, tactus_periods, tatum_periods)]
        print("Median measure, tactus and tatum periods throughout the piece "
              "are %d, %d (ratio %f), and %d (ratio %f) respectively." % (
                  medians[0], medians[1], float(medians[0]) / medians[1],
                  medians[2], float(medians[1]) / medians[2]))
        print("Placing tatum, tactus, and meaure beats cost %f, %f, and %f, "
              "respectively." % (
                  best_tatum_cost, best_tactus_cost, best_measure_cost))
        print("Found %d measures, %d tactus beats, %d tatums" % (
            len(measures), len(tactus), len(tatums)))
        if len(measures) > 0:
            print("%f tactus beats per measure" % (
                float(len(tactus)) / len(measures)))
        if len(tactus) > 0:
            print("%f tatums per tactus beat" % (
                float(len(tatums)) / len(tactus)))
    if visualize:
        fig = plt.figure(figsize=(9, 10))
        # Plot the original waveform and the novelty levels.
        ax = fig.add_subplot(3, 1, 1)
        # Set the axes.
        plt.axis([0., samples.shape[0], -1., 1.])
        # Show the input audio waveform.
        plt.plot(samples, alpha=0.2, color='b')
        # Plot the novelty signals.
        plt.plot([(i * hop_size + window_size / 2) / interpolation_factor
                  for i in range(novelty_signal.shape[0])],
                 novelty_signal)
        # Add the locations of the identified tatums, tactus, and measures.
        for x in measures:
            plt.axvline(x, color='g', alpha=0.5)
        for x in tactus:
            plt.axvline(x, color='b', alpha=0.5)
        plt.xlabel('Sample #')
        plt.ylabel('Amplitude')
        plt.legend()
        # Visualize the pulse salience of each period hypothesis over time.
        ax = fig.add_subplot(3, 1, 2)
        ax.imshow(pulse_salience.swapaxes(0, 1), interpolation='nearest',
                  cmap=cm.binary)
        ax.set_aspect('auto')
        locs, labels = plt.xticks()
        plt.xticks(locs, [(i * hop_size + window_size / 2) /
                          interpolation_factor * tempo_hop_size +
                          tempo_frame_size / 2
                          / sample_rate for i in locs])
        locs, labels = plt.yticks()
        plt.yticks(locs, [(i * hop_size / interpolation_factor) / sample_rate
                          for i in locs])
        plt.axis([0, pulse_salience.shape[0] - 1, 0, tempo_frame_size - 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Period hypothesis (s)')
        ax = fig.add_subplot(3, 1, 3)
        scores = (tactus_scores + tatum_scores + measure_scores).swapaxes(0, 1)
        scores /= 3.
        ax.imshow(scores, cmap=cm.binary, interpolation='nearest')
        ax.set_aspect('auto')
        locs, labels = plt.xticks()
        plt.xticks(locs, [(
            (hop_size * i) /
            (sample_rate / hop_size / interpolation_factor)) for i in locs])
        locs, labels = plt.yticks()
        plt.yticks(locs, [(i * hop_size / interpolation_factor) / sample_rate
                          for i in locs])
        plt.axis([0 - tempo_frame_size / hop_size,
                 pulse_salience.shape[0] - tempo_frame_size /
                 hop_size, 0, tempo_frame_size - 1])
        plt.xlabel('Time (s)')
        plt.ylabel('Period hypothesis (s)')
        plt.show()
    return (tatums, tactus, measures)


def map_pass(samples, low_bpm, high_bpm, sample_rate=settings.SAMPLE_RATE):
    """
    Estimates the consistent BPM for an audio sample based on
    the peaks in amplitude and the best fit bpm between the range
    (low_bpm, high_bpm).
    NOTE: Not useful for audio where the tempo changes.
    """
    filtered_samples = np.copy(samples)
    # Square values to accentuate amplitudes

    # First pass through and zero out all but the
    # 5% of samples with the largest amplitudes
    top_samples = np.copy(filtered_samples)
    top_samples.sort()
    top_samples = top_samples[-len(top_samples) * 0.10:]

    cut = top_samples[0]

    for pos, sample in enumerate(filtered_samples):
        if abs(sample) < cut:
            filtered_samples[pos] = 0

    return map_best_beat(filtered_samples, low_bpm, high_bpm,
                         sample_rate=sample_rate)


def map_best_beat(filtered_samples, low_bpm, high_bpm,
                  sample_rate=settings.SAMPLE_RATE):
    # Try to map every bpm in the range, and see which one best fits
    best_bpm = 0
    most_hits = 0

    for bpm in range(low_bpm, high_bpm + 1):
        sample_rate_step = sample_rate * 60.0 / bpm
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

    sample_rate_step = sample_rate * 60.0 / best_bpm
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
            best_bpm = (sample_rate * 60) / sample_step

    return best_bpm

"""Utilities for learning the tempo of a piece.

Works at the tatum, tactus and measure levels.

References:

1. Klapuri, Anssi P., Antti J. Eronen, and Jaakko T. Astola. "Analysis of the
   meter of acoustic musical signals." Audio, Speech, and Language Processing,
   IEEE Transactions on 14.1 (2006): 342-355.
2. Ellis, Daniel PW. "Beat tracking by dynamic programming." Journal of New
   Music Research 36.1 (2007): 51-60.
3. Davies, Matthew EP, and Mark D. Plumbley. "Context-dependent beat tracking
   of musical audio." Audio, Speech, and Language Processing, IEEE Transactions
   on 15.3 (2007): 1009-1020.
"""
import math

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from impromptica import probdata
from impromptica import settings
from impromptica.utils import novelty
from impromptica.utils import structure
from impromptica.utils import visualization


# The beat placement algorithm assigns a penalty for placing a beat at a tempo
# other than the target one. The penalty is a function of the square of the
# logarithm of the change in tempo. `_BEAT_PLACEMENT_STRICTNESS` provides a
# scalar factor to balance this penalty with that of placing a beat at segment
# which has a low level of musical accentuation.
# found to work well.
_BEAT_PLACEMENT_STRICTNESS = 10000.
# The measure bar placement algorithm works similarly.
_MEASURE_PLACEMENT_STRICTNESS = 10000.


def calculate_measures(
        samples, tactus, prior, ptransition, measure_placement_strictness,
        max_multiple, verbose=False):
    """Returns the estimated indices of the measures of the piece.

    The measure indices are a subset of `tactus`, a provided list of beat
    indices.

    We formulate the measure-finding problem as follows: Given a set of beat
    indices at the tactus level, find the subset of beat indices which
    minimize a cost function constructed to penalize subsets that are poor
    candidates for measure boundaries.

    To solve the problem, we first find a target number of tactus beats per
    measure, which may vary throughout the piece. This processes uses
    similarity information, prior probabilities for potential measure periods
    (in terms of tactus beats per measure), and a likelihood function for
    changes in measure period.

    After finding the target measure period(s) for the piece, we attempt to
    place the beginnings of measures in such a fashion as to minimize a cost
    function which penalizes measure periods at periods other than the target
    as well as measures which poorly account for similarity information (e.g.
    measures which do not line up well against the segmentation of the piece
    that is derived from the similarity information.
    """
    # Consider only measure periods from one up to `max_multiple`.
    # TODO Calculate the salience of various measure period hypotheses at each
    # beat.
    measure_salience = np.ones((len(tactus), max_multiple))
    # Assign target measure periods using dynamic programming. `periods` is a
    # table of the best previous periods at each tactus beat and period
    # candidate.
    periods = np.zeros((len(tactus), max_multiple), dtype=np.int)
    scores = np.zeros((len(tactus), max_multiple))
    # Fill in the first row of the dynamic programming table and set the priors
    # at each beat.
    scores = prior * measure_salience
    scores[0] /= scores[0].max()
    best = np.argmax(scores[0]) + 1
    for i in range(max_multiple):
        periods[0][i] = best
    # Fill in the rest of the table. We also take into account the transition
    # probability between two measure periods. At each beat, calculate the
    # lowest-cost route to each possible period using the `score` array.
    score = np.zeros(max_multiple)
    for i in range(1, len(tactus)):
        score *= 0.
        for j in range(max_multiple):
            # For each measure period hypothesis at the previous beat,
            # calculate what the score would be at this beat and period if
            # transitioning from it.
            for k in range(max_multiple):
                score[k] = scores[i - 1][k] * ptransition[k][j]
            best = np.argmax(score)
            periods[i][j] = best + 1
            scores[i][j] *= score[best]
        scores[i] /= scores[i].max()
    # Assign the best period to each beat in reverse order.
    period = np.zeros(len(tactus), dtype=np.int)
    best = periods[-1][np.argmax(scores[-1])]
    for i in range(1, scores.shape[0] + 1):
        period[scores.shape[0] - i] = best
        best = periods[-i][best - 1]
    # Calculate the change in target beats per measure throughout the course of
    # the song.
    changes = []
    current_period = -1
    for i in range(period.shape[0]):
        if current_period != period[i]:
            current_period = period[i]
            changes.append((i, current_period))
    if verbose:
        print("Measure changes: %s" % (str(changes)))
    # Now calculate the actual measure bar locations.
    # The `measure_cost` array tracks the best cost of an optimal set of beat
    # indices ending with a measure bar at the given index.
    measure_cost = np.zeros(periods.shape[0])
    # The `previous_measure` array holds a reference to the previous beat index
    # in the optimal set of measure bars ending at the given index.
    previous_measure = np.ones(periods.shape[0], dtype=np.int) * -1
    # Allocate a temporary array for calculating the lowest-cost previous index
    # at each step. We check only the previous 2*n cells, where n is the
    # highest measure period found found anywhere in the piece.
    costs = np.zeros(np.max(period) * 2)
    # Fill in the `measure_cost` and `previous_measure` tables. For each beat
    # index within the first beat index's target period, the cost of measure
    # placement is zero. For all other indices,  we also account for how close
    # the resulting measure period would be to the target period if we placed a
    # measure bar at that index.
    for i in range(period[0]):
        measure_cost[i] = 0.
    for i in range(period[0], period.shape[0]):
        target_period = period[i]
        # Calculate the optimal previous measure bar, assuming a measure bar is
        # placed at the current index. `furthest_back` is the maximum number of
        # previous indices to check.
        furthest_back = min(i, costs.shape[0])
        for j in range(1, furthest_back + 1):
            costs[j - 1] = measure_cost[i - j] + (
                measure_placement_strictness *
                math.pow(math.log(float(j) / target_period), 2.))
        best = np.argmin(costs[:furthest_back])
        previous_measure[i] = i - best - 1
        # TODO Add in a dissimilarity penalty below.
        measure_cost[i] = costs[best]
    # After calculating the array values from beginning to end, the optimal set
    # of measure bar indices is the set found from backtracking through
    # `previous_measure` starting at the lowest-cost index in the last m
    # indices, where m is the target period of the last index. Start by finding
    # the lowest-cost index from the the last m indices of `measure_cost`.
    i = np.argmin(measure_cost[period.shape[0] - period[-1]:]) + (
        period.shape[0] - period[-1])
    if verbose:
        best_cost = measure_cost[i]
        print("Best cost for measure selection is %f" % (best_cost))
    # Accumulate the optimal measure bar indices into a list.
    beat_indices = []
    while i >= 0:
        beat_indices.append(i)
        i = previous_measure[i]
    # Reverse the list to get the in-order sequence of measure bar indices.
    beat_indices.reverse()
    # Get the actual indices into samples.
    result = [tactus[i] for i in beat_indices]
    return result


def calculate_pulse_salience(novelty_signal, frame_size, hop_size):
    """Returns salience information on period hypotheses.

    Period hypothesis are integers representing period length in segments.

    Only periods up to half of the frame size are considered, since quality
    of the autocorrelation has been found to degenerate as periods approach
    the size of the frame.
    """
    # `frames_size` is the number of output frames.
    frames_size = (novelty_signal.shape[0] - frame_size) / hop_size + 1
    # `normalization_factor` is a denominator applied to each hypothesis term.
    normalization_factor = np.linspace(frame_size - 1, 0, frame_size)[
        :frame_size / 2]
    # Allocate the resulting tempo salience signal.
    result = np.zeros((max(frames_size, 1), frame_size / 2))
    if frames_size < 1:
        # Handle the special case where there is only a single frame.
        frame = np.zeros(frame_size / 2)
        frame[:novelty_signal.shape[0]] = novelty_signal
        result[0] = np.correlate(frame, frame, mode='full')[
            frame_size:frame_size + frame_size / 2] / normalization_factor
    else:
        # Calculate the autocorrelation in each frame. Only copy over the
        # values for the lower half of all possible period hypotheses since the
        # values higher than that were found to degenerate and throw off the
        # results.
        for i in range(frames_size):
            frame = novelty_signal[i * hop_size:i * hop_size + frame_size]
            result[i] = np.correlate(frame, frame, mode='full')[
                frame_size:frame_size + frame_size / 2] / normalization_factor
    return result


def calculate_tactus_periods(
        max_multiple, pulse_salience, prior, ptransition, hop_size,
        periods_size):
    """Calculates the most probable period at each frame.

    `prior` is an array of length `max_multiple` giving the prior probability
    of a period of the given length at the current metrical level.

    This function assigns a tempo, as a integer multiple of the segment
    duration, to each sample in the pulse salience signal.
    """
    # Allocate the dynamic programming table.
    tempo = np.zeros((pulse_salience.shape[0], max_multiple), dtype=np.int)
    scores = np.zeros((pulse_salience.shape[0], max_multiple))
    # Fill in the first row of the dynamic programming table.
    scores = prior * pulse_salience.swapaxes(
        0, 1)[:max_multiple].swapaxes(0, 1)
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
        relevant = [x for x in sorted(list(set(relevant)))
                    if 0 <= x < scores.shape[1]]
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
        periods[(scores.shape[0] - i) * hop_size +
                pulse_salience.shape[1] / 2] = best
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
    # TODO Use knowledge about when measures begin to avoid half-phase errors.
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
        measure_placement_strictness=_MEASURE_PLACEMENT_STRICTNESS,
        max_beats_per_measure=settings.MAX_BEATS_PER_MEASURE,
        sample_rate=settings.SAMPLE_RATE,
        verbose=False, visualize=False):
    """Returns (measures, tactus, tatums, tatums_per_tactus) pulse information.

    `measures`, `tactus`, and `tatums` are each lists of beat indices for the
    associated metrical levels.

    `tatums_per_tactus` is an array containing the number of tatums per tactus
    for the period immediately following the tactus beat refered to by the
    index into the array.
    """
    if verbose:
        print("Beginning tempo recognition...")
    # Calculate the novelty of a various frequency bands across segments.
    if verbose:
        print("Calculating novelty...")
    novelty_signal = novelty.spectral_flux(samples, verbose=verbose)
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
        tempo_hop_size)
    ptactus = probdata.build_lognorm_tempo_profile_data(
        0.28, 0.55, hop_size / interpolation_factor / sample_rate,
        tempo_hop_size)
    # Precompute the transition probabilities for all possible transitions
    # between hypotheses.
    ptransition = probdata.build_tempo_change_profile_data(tempo_hop_size)
    # Calculate the tactus periods and beats.
    tactus_periods, tactus_scores, tactus_changes = calculate_tactus_periods(
        tempo_hop_size, pulse_salience, ptactus, ptransition,
        tempo_hop_size, novelty_signal.shape[0])
    tactus, best_tactus_cost = calculate_beats(
        tactus_periods, novelty_signal, beat_placement_strictness)
    # Calculate the boundaries for segmentation of the piece.
    boundaries = []
    for a, b in zip(tactus[:-1], tactus[1:]):
        boundaries.extend([a, (a + b) / 2])
    boundaries.append(tactus[-1])
    boundaries = np.array(boundaries)
    # Calculate the tatums. At a frame whose boundaries are defined by the
    # beats at the tactus level, calculate the pulse salience weighted by prior
    # probability.
    tatums_per_tactus = np.zeros(len(tactus), dtype=np.int)
    for i, first in enumerate(tactus):
        last = novelty_signal.shape[0]
        if i < len(tactus) - 2:
            last = min(novelty_signal.shape[0], tactus[i + 2])
        frame = novelty_signal[first:last]
        normalization_factor = np.zeros(frame.shape[0] - 1)
        for j in range(frame.shape[0] - 1):
            normalization_factor[j] = 1 + frame.shape[0] - j
        # Allocate the resulting tempo salience signal.
        salience = np.zeros(frame.shape[0])
        # Handle the special case where there is only a single frame.
        salience[:frame.shape[0] - 1] = np.correlate(
            frame, frame, mode='full')[
                frame.shape[0]:2 * frame.shape[0]] / normalization_factor
        # Crop results to the periods in question.
        salience = salience[:min(salience.shape[0], ptatum.shape[0])]
        salience *= ptatum[:min(salience.shape[0], ptatum.shape[0])]
        # Zero out candidates which are not factors or close to factors of the
        # tactus period.
        factors = set()
        for j in range(1, int(tactus_periods[first] ** 0.5) + 1):
            div, mod = divmod(tactus_periods[first], j)
            if mod == 0:
                factors |= {j - 1, j, j + 1, div - 1, div, div + 1}
        factors = sorted(list(factors))
        for j in range(salience.shape[0]):
            if j not in factors:
                salience[j] = 0.
        # Find the most likely number of tatums per tactus.
        best = np.argmax(salience) + 1
        tatums_per_tactus[i] = int(
            round(float(tactus_periods[first / tempo_hop_size]) / best))
    tactus = [(i * hop_size + window_size / 2) / interpolation_factor
              for i in tactus]
    # Calculate the beat indices of the tatums.
    tatums = []
    for i in range(tatums_per_tactus.shape[0]):
        # Calculate the first index of the span from the current tactus beat
        # to the next (or the end of the piece, if there are no future tactus
        # beats.)
        first = tactus[i]
        if i == tatums_per_tactus.shape[0] - 1:
            last = samples.shape[0]
        else:
            last = tactus[i + 1]
        width = last - first
        # Space the tatums equally apart in this period.
        for i in range(tatums_per_tactus[i]):
            tatums.append(first + float(i * width) / tatums_per_tactus[i])
    # Calculate the prior probability of n tactus beats per measure. The data
    # is based on Figure 8 from [1], with likelihoods for periods 10-13 made
    # up.
    pmeasure = np.array(
        [0.16, 0.18, 0.14, 0.22, 0.08, 0.16, 0.08, 0.16, 0.14, 0.04, 0.01,
         0.04, 0.01])
    measures = calculate_measures(
        samples, tactus, pmeasure, ptransition, measure_placement_strictness,
        max_beats_per_measure, verbose=verbose)
    # Calculate the high-level structure of the piece.
    piece_structure = structure.calculate_structure(
        samples, boundaries, pulse_salience, tempo_hop_size,
        visualize=visualize)
    if verbose:
        print("Tactus changes: %s" % (str(tactus_changes)))
        print("Finding beats...")
        median_tactus = np.median(tactus_periods, axis=0)
        median_tatums_per_tactus = np.median(tatums_per_tactus)
        print("Median tactus periods and tatum-to-tactus ratios "
              "are %d and %d respectively." % (
                  median_tactus, median_tatums_per_tactus))
        print("Placing tactus beats cost %f" % (best_tactus_cost))
        print("Found %d measures, %d tactus beats" % (
            len(measures), len(tactus)))
        if len(measures) > 0:
            print("%f tactus beats per measure" % (
                float(len(tactus)) / len(measures)))
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
        # Add the locations of the identified tactus and measures.
        for x in measures:
            plt.axvline(x, color='g', alpha=0.5)
        for x in tactus:
            plt.axvline(x, color='b', alpha=0.5)
        plt.xlabel('Sample #')
        plt.ylabel('Amplitude')
        plt.legend()
        # Visualize the rhythmogram of the piece.
        rate = (sample_rate / hop_size) * interpolation_factor
        visualization.show_rhythmogram(
            pulse_salience.swapaxes(0, 1),
            rate / tempo_hop_size,
            window_size / 2. / sample_rate,
            1. / rate)
        ax = fig.add_subplot(3, 1, 2)
        scores = tactus_scores.swapaxes(0, 1)
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
    return (measures, tactus, tatums, tatums_per_tactus)


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

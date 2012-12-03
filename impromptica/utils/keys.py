"""Impromptica utilities for key-finding."""
import math

from impromptica import probdata
from impromptica.utils import note_freq
from impromptica.utils import probabilities
from impromptica.utils import sound
from impromptica.utils import tempo


def get_keys(samples, onsets, frequency, samples_per_segment=None):
    """Returns a list of (index, key) tuples with the keys of the piece,
    where `index` is the index of `samples` where the `key` begins.

    This function uses part of the polyphonic keyfinding algorithm presented
    in "Music and Probability" (Temperley 2007.)

    TODO Implement calculations for the probability of changing key from
    segment to segment.
    """
    result = []

    # Get the frequencies of the piece, and their onsets.
    frequencies = note_freq.frequencies(onsets, samples, frequency)

    # Divide the piece into segments. A segment is 4 beats unless otherwise
    # specified.
    if not samples_per_segment:
        beats_per_minute = tempo.map_pass(samples, frequency, 1, 400)
        samples_per_segment = frequency * 60.0 / beats_per_minute * 4.0

    for i in range(int(math.ceil(len(samples) / samples_per_segment))):
        # Get the notes on this segment.
        segment_frequencies = []
        for index, frequency_list in frequencies.iteritems():
            if samples_per_segment * i <= index < (samples_per_segment *
                                                   (i + 1)):
                segment_frequencies.extend(frequency_list)
        notes = [sound.frequency_to_note(f) for f in segment_frequencies]
        # Get the pitch classes in this segment.
        pitch_classes = list(set([note_freq.pitch_class(n) for n in notes]))
        print('Pitch classes detected for keyfinding: %s' % (pitch_classes))
        # Calculate the modulation score of this segment.
        best_key = [0, 0]
        best_probability = 0.0
        for is_major in range(2):
            if is_major == 1:
                key_profile_data = probdata.KP_MAJOR_KEY_PROFILE_DATA
            else:
                key_profile_data = probdata.KP_MINOR_KEY_PROFILE_DATA
            for tonic in range(12):
                kp = probabilities.build_key_profile(key_profile_data,
                                                     [tonic, is_major])
                adjusted_pitch_classes = [(note - tonic) % 12 for note in
                                          pitch_classes]
                p = 1.0
                for scale_degree in range(12):
                    if scale_degree in adjusted_pitch_classes:
                        p *= kp(scale_degree + tonic)
                    else:
                        p *= 1.0 - kp(scale_degree + tonic)
                if p > best_probability:
                    best_key = [tonic, is_major]
                    best_probability = p
        result.append([i, best_key])
    return result


def is_major_key(key):
    """Returns whether or not the given key is a major key.

    `key` is a list of note values, where the first note is the base note
    of the key.
    """
    return True  # TODO Put some better logic here.

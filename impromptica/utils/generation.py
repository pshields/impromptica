"""Impromptica utilities for generating music."""
import random

from impromptica import settings
from impromptica import probdata

from impromptica.utils import probabilities
from impromptica.utils import keys


def generate_note(previous_note, central_note, key):
    """Returns a note value generated from range, key, and proximity profiles.

    `previous_note` is the note value of the note that the generated note
    will follow.

    `central_note` is the mean note value of the musical piece we are
    generating a note for.

    `key` is a (tonic, is_major) tuple where `tonic` is the base note of the
    key, and `is_major` is 1 if the key is a major key and 0 otherwise.
    """

    # Select the major or minor key profile based on the key.
    if keys.is_major_key(key):
        key_profile_data = probdata.ESSEN_MAJOR_KEY_PROFILE_DATA
    else:
        key_profile_data = probdata.ESSEN_MINOR_KEY_PROFILE_DATA

    kp = probabilities.build_key_profile(key_profile_data, key)

    # Get the range and proximity profiles from `probdata`.
    rp_data = probdata.RANGE_PROFILE_DATA
    rp = probabilities.build_distance_profile(rp_data, central_note)
    pp_data = probdata.PROXIMITY_PROFILE_DATA
    pp = probabilities.build_distance_profile(pp_data, previous_note)

    # Generate a list of all notes to consider.
    notes = range(settings.MAX_NOTE + 1)

    # Calculate the probability of each possible next note.
    # The resulting table is indexed by note value.
    weights = [rp(i) * pp(i) * kp(i) for i in notes]

    # Select a note randomly in proportion to its weight.
    result_note = None
    x = random.uniform(0.0, sum(weights))

    for note, weight in enumerate(weights):
        if x < weight:
            result_note = note
            break
        else:
            x = x - weight

    return result_note

"""Impromptica utilities related to probability."""


def build_distance_profile(distance_profile_data, base_note):
    """Builds an accessor to distance profile data.

    Returns a function which takes a note value and returns the value of the
    distance profile data at the distance between that note and the base note.
    """

    def accessor(note):
        return distance_profile_data[abs(note - base_note)]

    return accessor


def build_key_profile(key_profile_data, key):
    """Builds an accessor to key profile data."""

    def accessor(note):
        return key_profile_data[(note - key[0]) % 12]

    return accessor

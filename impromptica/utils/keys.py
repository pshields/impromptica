"""Impromptica utilities for working with the key of a piece."""


def is_major_key(key):
    """Returns whether or not the given key is a major key.

    `key` is a (tonic, is_major) tuple where `tonic` is the base note of the
    key, and `is_major` is 1 if the key is a major key and 0 otherwise.
    """
    return key[1] == 1  # TODO Make sure this works. If it does, document how.


def notes_in_key(key):
    """Returns the major or minor chord for the given key."""
    tonic = key[0]
    if key[1] == 1:
        notes = [60 + tonic, 64 + tonic, 67 + tonic]
    else:
        notes = [60 + tonic, 63 + tonic, 67 + tonic]
    return notes

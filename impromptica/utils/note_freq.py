"""Utilities for working with notes and frequencies."""
from impromptica.utils import sound


def pitch_class(note):
    """Returns the pitch class for a given note."""
    return note % 12


def equal_temperament_note(freq):
    return sound.note_to_frequency(sound.frequency_to_note(freq))

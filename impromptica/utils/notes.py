"""Structures for working with notes.

Impromptica represents notes as `Note` instances which are associated with
given tatums in the musical piece.
"""


class Note(object):
    """A specific note in a piece.
    
    A note has an instrument id, midi note value, and a duration measured in
    tatums.
    """

    def __init__(self, instrument, midi_note, duration):
        self.instrument = instrument
        self.midi_note = midi_note
        self.duration = duration

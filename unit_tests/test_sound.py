import unittest
import numpy
from copy import deepcopy

from impromptica.utils import sound


class TestSoundGeneration(unittest.TestCase):

    def test_merge_audio(self):
        #This merge should succeed, and do nothing
        note1 = sound.generate_note(1, 1, 440)
        note2 = numpy.zeros(len(note1))
        sound.merge_audio(note2, note1)

        assert (note2 == note1).all()

        #This merge should succeed, and do something
        note3 = sound.generate_note(1, 1, 100)
        old_note_3 = deepcopy(note3)
        sound.merge_audio(note3, note1)

        assert (note3 != old_note_3).all()

        #This merge should fail and do nothing
        note4 = sound.generate_note(0.5, 1, 220)
        old_note_4 = deepcopy(note4)
        sound.merge_audio(note4, note1)

        assert (note4 == old_note_4).all()

    def test_generate_note(self):
        note = sound.generate_note(2, 1, 440, 44100)
        assert len(note) == 44100 * 2

    def test_frequency_to_note(self):
        #Middle C
        semitone = sound.frequency_to_note(261.63)
        assert semitone == 60

    def test_seconds_to_samples(self):
        num_samples = sound.seconds_to_samples(2, 44100)
        assert num_samples == 2 * 44100

    def test_frequency_to_notestring(self):
        notestring = sound.frequency_to_notestring(440)
        assert notestring == "A4"

    def test_notestring_to_note(self):
        assert sound.notestring_to_note("C4") == 60

    def test_notestring_to_frequency(self):
        assert abs(sound.notestring_to_frequency("A4") - 440.0) < 0.5

    def test_note_to_notestring(self):
        assert sound.note_to_notestring(60) == "C4"

    def test_note_to_frequency(self):
        assert abs(sound.note_to_frequency(69) - 440) < 0.5

#!/usr/bin/env python
import time
import numpy
import fluidsynth
import math
import os
from scikits.audiolab import Format
from scikits.audiolab import Sndfile
from scipy import hamming

# Sampling rate in Hz
_sample_rate = 44100
# Default to mono sound
_num_channels = 1
# In Hz, used for Equal Temperament
_MIDDLE_C_SEMITONE = 60
_MIDDLE_C = 261.63
_MIDDLE_OCTAVE = 4

# Notes map to their semitones (there are 12)
NOTES = {
    "C":    0,
    "D":    2,
    "E":    4,
    "F":    5,
    "G":    7,
    "A":    9,
    "B":    11,
}


def notestring_to_note(note, octave=_MIDDLE_OCTAVE):
    """
    Converts a note string into a semitone.
    Notes can be formatted with sharps and octaves.
    e.g. A6, B4b
    """
    flats = note.count("b")
    sharps = note.count("#")

    potential_octave = filter(str.isdigit, note)
    if potential_octave:
        octave = int(potential_octave)

    note_val = NOTES.get(note[0])
    note_val += sharps - flats

    note = (octave - _MIDDLE_OCTAVE) * note_val + _MIDDLE_C_SEMITONE
    if octave == _MIDDLE_OCTAVE:
        note += note_val

    return note


def note_to_frequency(value):
    """Returns the frequency of the given note value.

    Our mapping of note values to frequencies matches the common convention
    that the note C4 (middle C) is represented by the note value 60.
    """
    frequency = _MIDDLE_C * 2 ** ((value - _MIDDLE_C_SEMITONE) / 12.0)
    return frequency


def notestring_to_frequency(note, octave=_MIDDLE_OCTAVE):
    """
    Converts a string note to a frequency.
    Notes can be formatted with sharps and octaves.
    e.g. A6, B4b
    """
    return note_to_frequency(notestring_to_note(note, octave))


def frequency_to_note(frequency):
    """Returns the closest note value of the given frequency (in Hz.)"""
    return (int(round(12.0 * math.log(frequency / _MIDDLE_C) / math.log(2))) +
            _MIDDLE_C_SEMITONE)


def note_to_notestring(semitone):
    """
    Takes a semitone value, returns a string note representation,
    i.e. something like A4b
    """
    octave = round((semitone - _MIDDLE_C_SEMITONE) / 12.0) + _MIDDLE_OCTAVE

    semitone = semitone % 12

    # Get the corresponding key, append the octave,
    # and adjust for flats / sharps
    # Default to flat if adjustment is necessary
    append_flat = False
    if semitone not in NOTES.values():
        semitone += 1
        append_flat = True

    semitone_to_note_dict = {value: key for key, value in NOTES.items()}
    note = str(semitone_to_note_dict[semitone])
    note += str(int(octave))
    if append_flat:
        note += "b"

    return note


def frequency_to_notestring(frequency):
    return note_to_notestring(frequency_to_note(frequency))


def generate_note(duration, amplitude, frequency, Fs=44100):
    """
    Returns a numpy array of samples.
    Duration: Time in seconds of clip
    Amplitude: Volume of the note, between 0 and 1
    Frequency: In Hz
    Fs: Sampling Rate. Defaults to 44100
    """
    samples = []

    if not Fs:
        Fs = _sample_rate

    if frequency == 0:
        return numpy.zeros(int(duration * Fs))

    num_samples = seconds_to_samples(duration)

    dampening = hamming(num_samples)
    period = Fs / frequency

    #Generate a dampened square wave
    for sample in range(num_samples):
        if sample % period < period / 2:
            samples.append(amplitude)
        else:
            samples.append(-amplitude)

    note = samples * dampening

    return note


def gen_midi_note(duration, amplitude, frequency, Fs=44100, instrument=0):
    """
    Generate a midi note.
    Duration: Time in seconds of clip
    Amplitude: Volume of the note, between 0 and 1
    Frequency: In Hz
    Fs: Sampling Rate. Defaults to 44100
    Instrument: 0-127 General Midi number
        Default to Acoustic Grand Piano
        http://en.wikipedia.org/wiki/General_MIDI
    """
    notenum = frequency_to_note(frequency)

    curdir = os.path.dirname(__file__)
    fs = fluidsynth.Synth()
    fs.start(driver="alsa")
    soundfont_id = fs.sfload(os.path.join(curdir, "soundfonts/FluidR3_GM.sf2"))
    fs.program_select(0, soundfont_id, 0, 0)

    #Generate at medium amplitude to avoid artifacts
    #Generate at medium amplitude to avoid artifacts
    fs.noteon(0, notenum, int(amplitude * 50))
    note = fs.get_samples(seconds_to_samples(duration, Fs))
    fs.noteoff(0, notenum)

    time.sleep(0.1)
    fs.delete()
    time.sleep(0.1)

    # Note is a stereo value by default, twice as long as it should be.
    # Make it mono. Values are interleaved, so sample every other one
    mono_note_indices = numpy.arange(0, len(note), 2)
    mono_note = note[mono_note_indices]
    mono_note = mono_note.astype(float)

    #Audiolab compliancy, amplitude adjustment
    mono_note /= numpy.max(mono_note)
    mono_note *= amplitude
    dampening = hamming(len(mono_note))
    mono_note *= dampening

    return mono_note


def merge_audio(to_samples, merge_samples):
    """Merges `merge_samples` into `to_samples`.

    The results are stored in `to_samples`.

    `merge_samples` must be of length less than or equal to the
    length of `to_samples`.

    If an amplitude in the merged array would be greater than 1.0,
    the merged array's amplitudes will be normalized so that the
    highest amplitude is 1.
    """
    diff = len(to_samples) - len(merge_samples)
    if diff > 0:
        merge_samples = numpy.append(merge_samples, numpy.zeros(diff))

    if diff < 0:
        print "Error: Can't merge clip longer than merge destination"
        return

    to_samples += merge_samples
    max_amplitude = numpy.max(to_samples)
    if max_amplitude > 1:
        to_samples /= max_amplitude


def generate_chord(duration, amplitude, frequencies, Fs=44100):
    """
    Duration: Time in seconds of clip
    Amplitude: Volume of the note, between 0 and 1
    Frequencies: List of frequencies in Hz
    Fs: Sampling rate. Defaults to 44100
    """
    samples = numpy.zeros(seconds_to_samples(duration))

    for frequency in frequencies:
        frequency_samples = generate_note(duration, amplitude, frequency, Fs)
        merge_audio(samples, frequency_samples)

    return samples


def seconds_to_samples(duration, Fs=44100):
    """
    Converts a duration in seconds to the corresponding number of samples.
    Defaults to a sampling frequency of 44100 Hz
    """
    return int(Fs * duration)


def samples_to_seconds(samples, Fs=44100):
    """
    Converts a duration in samples to the corresponding number of seconds.
    Defaults to a sampling frequency of 44100 Hz
    """
    return float(samples) / Fs


def write_wav(samples, filename, audio_format=Format(), Fs=44100,
              stereo=False):
    """
    Write the numpy samples to a wav file.
    Audio format expects an Sndfile.format object.
    One channel (mono) is default, 44100 Hz sampling frequency is default.
    """
    channels = 1
    if stereo:
        channels = 2

    f = Sndfile(filename, 'w', audio_format, channels, Fs)
    f.write_frames(samples)
    f.close()

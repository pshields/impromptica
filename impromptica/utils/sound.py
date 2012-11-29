#!/usr/bin/env python

import numpy
import math
from scikits.audiolab import Sndfile
from scipy import hamming

# Sampling rate in Hz
_sample_rate = 44100
# Default to mono sound
_num_channels = 1
# In Hz, used for Equal Temperament
_MIDDLE_C_SEMITONE = 60
_MIDDLE_C = 261.63
_CONCERT_OCTAVE = 5

# Notes map to their semitones (there are 12)
NOTES = {
    "A":    0,
    "B":    2,
    "C":    4,
    "D":    6,
    "E":    8,
    "F":    9,
    "G":    11,
}


def note_to_semitone(note, octave=_CONCERT_OCTAVE):
    """
    Converts a note string into a semitone.
    Notes can be formatted with sharps and octaves.
    e.g. A6, B4b
    """
    flats = note.count("b")
    sharps = note.count("#")

    potential_octave = filter(str.isdigit, note)
    if potential_octave:
        octave = potential_octave

    note_val = NOTES.get(note[0])
    note_val += sharps - flats

    return 12 * (octave - 5) + note_val + _MIDDLE_C_SEMITONE


def semitone_to_frequency(value):
    """
    Converts a semitone value to a frequency
    """
    frequency = _MIDDLE_C * 2 ** ((_MIDDLE_C_SEMITONE - value) / 12.0)
    return frequency


def note_to_frequency(note, octave=_CONCERT_OCTAVE):
    """
    Converts a string note to a frequency.
    Notes can be formatted with sharps and octaves.
    e.g. A6, B4b
    """
    return semitone_to_frequency(note_to_semitone(note, octave))


def frequency_to_semitone(frequency):
    """
    Takes a frequency in Hz, returns a semitone
    """
    semitone = -int(round(12.0 * math.log(frequency / _MIDDLE_C) /
                    math.log(2))) + _MIDDLE_C_SEMITONE
    return semitone


def semitone_to_note(semitone):
    """
    Takes a semitone value, returns a string note representation,
    i.e. something like A4b
    """
    octave = (semitone - _MIDDLE_C_SEMITONE) / 12 + _CONCERT_OCTAVE

    # Get the corresponding key, append the octave,
    # and adjust for flats / sharps
    # Default to flat if adjustment is necessary
    append_flat = False
    if semitone not in NOTES.values():
        semitone -= 1
        append_flat = True

    semitone_to_note_dict = {value: key for key, value in NOTES.items()}
    note = str(semitone_to_note_dict[semitone])
    note += str(octave - 5)
    if append_flat:
        note += "b"

    return note


def frequency_to_note(frequency):
    return semitone_to_note(frequency_to_semitone())


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

    return samples * dampening


def merge_audio(to_samples, merge_samples):
    """
    Merges the merge samples into the to_samples
    """
    diff = len(to_samples) - len(merge_samples)
    if diff > 0:
        merge_samples = numpy.append(merge_samples, numpy.zeros(diff))

    if diff < 0:
        print "Error: Can't merge clip longer than merge destination"
        return

    to_samples += merge_samples
    to_samples /= numpy.max(to_samples)


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


def write_wav(samples, filename, audio_format, Fs=44100, stereo=False):
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

"""Utilities for working with audio samples."""
import math
import time

import fluidsynth
import numpy
from scikits import audiolab
from scikits import samplerate
import scipy

from impromptica import settings


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


def get_samples(filename):
    """Returns samples from the given input audio file.

    Input audio consisting of multiple channels is condensed into a single
    channel.

    The sample rate of the returned samples is defined by the `SAMPLE_RATE`
    variable in Impromptica's settings module.
    """
    input_file = audiolab.Sndfile(filename, 'r')
    sample_rate = input_file.samplerate
    samples = input_file.read_frames(input_file.nframes)

    # Condense multiple tracks to a single track.
    if samples.ndim > 1:
        samples = numpy.average(samples, axis=1)

    # Resample the input audio to the target frequency.
    result = samplerate.resample(samples, settings.SAMPLE_RATE / sample_rate,
                                 'sinc_best')
    return result


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

    note = (octave + 1) * 12 + note_val

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
    """Returns the closest note value of the given frequency (in hertz.)"""
    return (int(round(12.0 * math.log(frequency / _MIDDLE_C) / math.log(2))) +
            _MIDDLE_C_SEMITONE)


def note_to_notestring(semitone):
    """
    Takes a semitone value, returns a string note representation,
    i.e. something like A4b
    """
    octave = math.floor((semitone - _MIDDLE_C_SEMITONE) / 12.0) + \
        _MIDDLE_OCTAVE

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


def generate_note(duration, amplitude, frequency,
                  sample_rate=settings.SAMPLE_RATE):
    """
    Returns a numpy array of samples representing the given note being played
    for the given duration.

    duration: Time in seconds of clip
    amplitude: Volume of the note, between 0 and 1
    frequency: Frequency of the tone, in hertz
    sample_rate: Sample rate of the output, in hertz
    """
    result = numpy.zeros(int(duration * sample_rate))
    period = sample_rate / frequency

    for i in range(result.shape[0]):
        if i % period < period / 2:
            result[i] = amplitude
        else:
            result[i] = -amplitude

    return result


def gen_midi_note(
        soundfont_filename, duration, amplitude, frequency,
        sample_rate=settings.SAMPLE_RATE, instrument=0):
    """Generate a midi note.

    duration: Time in seconds of clip
    amplitude: Volume of the note, between 0 and 1
    frequency: frequency of the note, in hertz
    sample_rate: Sample Rate of the output, in hertz
    Instrument: 0-127 General Midi number
        Default to Acoustic Grand Piano
        http://en.wikipedia.org/wiki/General_MIDI
    """
    notenum = frequency_to_note(frequency)

    fs = fluidsynth.Synth()
    fs.start(driver="alsa")
    soundfont_id = fs.sfload(soundfont_filename)
    fs.program_select(0, soundfont_id, 0, 0)

    # Generate at medium amplitude to avoid artifacts
    fs.noteon(0, notenum, int(amplitude * 50))
    note = fs.get_samples(seconds_to_samples(duration, sample_rate))
    fs.noteoff(0, notenum)

    time.sleep(0.1)
    fs.delete()
    time.sleep(0.1)

    # Note is a stereo value by default, twice as long as it should be.
    # Make it mono. Values are interleaved, so sample every other one
    mono_note_indices = numpy.arange(0, len(note), 2)
    mono_note = note[mono_note_indices]
    mono_note = mono_note.astype(float)

    # Audiolab compliancy, amplitude adjustment
    mono_note /= numpy.max(mono_note)
    mono_note *= amplitude
    dampening = scipy.hamming(len(mono_note))
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
    
    # Merge the samples using the technique described at
    # http://www.vttoth.com/CMS/index.php/technical-notes/68.
    product = to_samples * merge_samples
    to_samples += merge_samples
    to_samples -= (numpy.sign(merge_samples) * product)


def generate_chord(duration, amplitude, frequencies,
                   sample_rate=settings.SAMPLE_RATE):
    """
    duration: Time in seconds of clip
    amplitude: Volume of the note, between 0 and 1
    frequencies: List of frequencies, in hertz
    sample_rate: Sample rate of the output, in hertz
    """
    samples = numpy.zeros(seconds_to_samples(duration))

    for frequency in frequencies:
        frequency_samples = generate_note(duration, amplitude, frequency,
                                          sample_rate)
        merge_audio(samples, frequency_samples)

    return samples


def seconds_to_samples(duration, sample_rate=settings.SAMPLE_RATE):
    """Returns the number of samples of the given duration."""
    return int(sample_rate * duration)


def samples_to_seconds(samples, sample_rate=settings.SAMPLE_RATE):
    """Returns the duration of the given number of samples in seconds."""
    return float(samples) / sample_rate


def write_wav(samples, filename, audio_format=audiolab.Format(),
              sample_rate=settings.SAMPLE_RATE, stereo=False):
    """
    Write the numpy samples to a wav file.
    Audio format expects an Sndfile.format object.
    One channel (mono) is default.
    """
    channels = 1
    if stereo:
        channels = 2
    f = audiolab.Sndfile(filename, 'w', audio_format, channels, sample_rate)
    f.write_frames(samples)
    f.close()

def midival_note(midival):
    """Returns (note, octave) tuple for the midi_note"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    d = {}
    for i in range (12):
        d[i] = notes[i]
    s = (midival%12, midival/12)
    return s

def make_scales ():
    d = {}
    for i in range (12):
        s = str(i)+'m'
        d[str(i)] = [i, (i+2)%12,  (i+4)%12,  (i+5)%12,  (i+7)%12,  (i+9)%12,  (i+11)%12]
        d[s] =  [i, (i+2)%12,  (i+3)%12,  (i+5)%12,  (i+7)%12,  (i+8)%12,  (i+10)%12]
    return d

def get_key (tatum_grid):
    """Returns a naiive key for a set of notes"""
    notes = sum (tatum_grid, [])
    allnotes = map (lambda x: midival_note(x.midi_note)[0], notes)
    #print allnotes
    d = make_scales()
    max = 0
    scale = '0'
    for k,v in d.iteritems():
        num_notes = 0
        for j in allnotes:
            if j in v:
                num_notes = num_notes+1
        #print k, ':', num_notes
        if num_notes > max:
            max = num_notes
            scale = k
    return d[scale]
        
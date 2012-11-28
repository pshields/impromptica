"""Utilities for working with notes and frequencies."""
from scipy import fft
from pylab import arange, plot, show, subplot, title
import numpy
import math


def frequencies(onsets, samples, Fs=44100):
    """Returns a dict of lists of frequencies of onsets.

    Given note onset positions and a numpy array of samples (amplitudes), this
    function returns a dict containing lists of frequencies (in hz)
    corresponding to each onset.

    Onsets is a list of indices which point to onsets in samples,
    which is a list of amplitudes, where every Fs elements
    equations to 1 second of sound.

    Assumes samples is collapsed (mono)
    """
    # Arbitrary window size -
    # This is the number of samples we look at after each onset, and
    # also the number of samples in our window (We just use one).
    # The alternative would be to define this as the number of samples we
    # look at, but then analyze it using multiple windows
    # (Short time fourier transform))
    # Seems to work for now (roughly 20hz size bins)
    window = 1024

    # For smoothing the window, and better results
    hamming_window = numpy.hamming(window)

    # FFT returns frequency buckets.
    # The first index is 0Hz, and each subsequent one is i * Fs/window Hz
    freq_dist = Fs / window

    notes = {}

    for notenum, onset in enumerate(onsets):
        # Get the fft, use a hamming window
        transform = fft(hamming_window *
                        samples[onset: onset + window], window) / window

        # Get the magnitude of each FFT coefficient,
        # since it returns complex values
        fft_coeffs = [x.real ** 2 + x.imag ** 2 for x in transform]

        # We can only use the first half of the coeffs (See nyquist frequency).
        # The other half are duplicates
        fft_coeffs = fft_coeffs[:window / 2]

        # Slight misnomer.
        # Contains a list of frequencies associated with this note onset
        note = []

        # Go through each note and figure out the dominant frequencies
        # (find max, accept within a threshold?)
        max_coeff = max(fft_coeffs)
        prom_note = fft_coeffs.index(max_coeff) * freq_dist
        last_note = prom_note
        print "Prominent note freq for note %d is %d" % (notenum, prom_note)

        # Find the nearest note on the equal-tempered scale
        # that matches this frequency, round the note to it
        prom_note = equal_temperament_note(prom_note)

        note.append(prom_note)

        # Arbitrary threshold. Probably should make this less hardcoded
        threshold = .80 * max_coeff

        for i, val in enumerate(fft_coeffs):
            if val > threshold and val != max_coeff:
                this_note = equal_temperament_note(i * freq_dist)

                # Eliminate erroneous notes
                if last_note:
                    if abs(frequency_to_note(this_note) -
                           frequency_to_note(last_note)) <= 2:
                        # Take the average frequency,
                        # replace the last frequency with it
                        average_freq = (this_note + last_note) / 2
                        this_note = equal_temperament_note(average_freq)
                        note.pop()

                last_note = this_note
                note.append(this_note)

        notes[onset] = note

    for notenum, note in enumerate(sorted(notes)):
        print "Onset %d notes (Hz): " % notenum, notes[note]

    return notes


def frequency_to_note(freq):
    if freq < 1:
        freq = 1
    return round(12 * math.log(freq / 440.0) / math.log(2))


def note_to_frequency(n):
    """Returns the frequency of the given note value.
    
    Our mapping of note values to frequencies matches the common convention
    that the note C4 (middle C) is represented by the note value 60.
    """
    return 261.63 * 2 ** (n / 12.0)


def equal_temperament_note(freq):
    return note_to_frequency(frequency_to_note(freq))


def plot_note_frequencies(onset, samples, Fs, window, graph_title=""):
    """
    Visualize the frequency spectogram. Used for testing.
    """
    freq_dist = Fs / window

    note_samples = samples[onset: onset + window]
    x = fft(note_samples) / window

    # Can only use half the fft values. See above comment
    x = x[:len(x) / 2]
    t = arange(0.0, Fs / 2 - freq_dist, freq_dist)

    # Crop t to x in case it's the last note, and it's short
    t = t[:len(x)]

    title(graph_title)
    subplot(111)
    plot(t, abs(x))

    show()

"""Utilities for working with notes and frequencies."""
from scipy import fft
from scipy.signal import fftconvolve
from pylab import arange, diff, find, plot, show, subplot, title
from numpy import argmax
import numpy

from impromptica.utils.sound import note_to_frequency, frequency_to_note


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

    notes = {}

    window_cap = 1024 * 5

    for notenum, onset in enumerate(onsets):
        if notenum >= len(onsets) - 1:
            next_onset = len(samples)
        else:
            next_onset = onsets[notenum + 1]

        if next_onset - onset > window_cap:
            next_onset = window_cap

        windowed_samples = samples[onset: onset + next_onset]
        N = len(windowed_samples)

        # Autocorrelate the signal. Helps with harmonic errors
        # Same thing as convolution, but with
        # one input reversed in time
        windowed_samples = fftconvolve(windowed_samples,
                                       windowed_samples[::-1], mode='full')

        #Throw away negative values
        windowed_samples = windowed_samples[len(windowed_samples) / 2:]

        # Get the fft, use a hamming window
        windowed_samples *= numpy.hamming(len(windowed_samples))

        # Take advantage of the fact that we're using 1 dimensional real input
        rfft_vals = numpy.fft.rfft(windowed_samples)

        #Find the start of the first peak from the derivative
        derivative = diff(rfft_vals)
        start = find(derivative > 0)[0]

        # Find the local peak
        peak = argmax(rfft_vals[start:]) + start

        # Fit the curve to a parabola, find a better local maximum
        # (Successive Parabolic Interpolation)
        func = rfft_vals
        actual_peak = 1 / 2.0 * (func[peak - 1] - func[peak + 1]) /\
            (func[peak - 1] - 2 * func[peak] + func[peak + 1]) + peak

        guess_fundamental_freq = Fs * actual_peak / N

        # Slight misnomer.
        # Contains a list of frequencies associated with this note onset
        note = []

        # Go through each note and figure out the dominant frequencies
        # (find max, accept within a threshold?)
        prom_note = guess_fundamental_freq

        # Find the nearest note on the equal-tempered scale
        # that matches this frequency, round the note to it
        prom_note = equal_temperament_note(prom_note)
        note.append(prom_note)
        notes[onset] = note

    return notes


def pitch_class(note):
    """Returns the pitch class for a given note."""
    return note % 12


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

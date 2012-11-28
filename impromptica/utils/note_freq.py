"""
Given note onset positions and a numpy array of samples (amplitudes),
reports the note frequency (or frequencies) for each onset
"""

from scipy import fft
from pylab import arange, plot, show, subplot, title
import numpy
import math


def frequencies(onsets, samples, Fs=44100):
    """
    Returns a dict containing lists of frequencies (in hz) corresponding
    to each onset.

    Onsets is a list of indices which point to onsets in samples,
    which is a list of amplitudes, where every Fs elements
    equations to 1 second of sound.

    Assumes samples is collapsed (mono)
    """
    #Arbitrary window size -
    #This is the number of samples we look at after each onset, and
    #also the number of samples in our window (We just use one).
    #The alternative would be to define this as the number of samples we
    #look at, but then analyze it using multiple windows
    #(Short time fourier transform))
    #Seems to work for now (roughly 20hz size bins)
    NFFT = 1024

    #For smoothing the window, and better results
    hamming_window = numpy.hamming(NFFT)

    #FFT returns frequency buckets.
    #The first index is 0Hz, and each subsequent one is i * Fs/NFFT Hz
    freq_dist = Fs / NFFT

    notes = {}

    for notenum, onset in enumerate(onsets):
        #Get the fft, use a hamming window
        transform = fft(hamming_window *
                        samples[onset: onset + NFFT], NFFT) / NFFT

        #Get the magnitude of each FFT coefficient,
        #since it returns complex values
        fft_coeffs = [x.real ** 2 + x.imag ** 2 for x in transform]

        #We can only use the first half of the coeffs (See nyquist frequency).
        #The other half are duplicates
        fft_coeffs = fft_coeffs[:NFFT / 2]

        #Slight misnomer.
        #Contains a list of frequencies associated with this note onset
        note = []

        #Go through each note and figure out the dominant frequencies
        #(find max, accept within a threshold?)
        max_coeff = max(fft_coeffs)
        prom_note = fft_coeffs.index(max_coeff) * freq_dist
        last_note = prom_note
        print "Prominent note freq for note %d is %d" % (notenum, prom_note)

        #Find the nearest note on the equal-tempered scale
        #that matches this frequency, round the note to it
        prom_note = equalTemperamentNote(prom_note)

        note.append(prom_note)

        #Arbitrary threshold. Probably should make this less hardcoded
        threshold = .80 * max_coeff

        for i, val in enumerate(fft_coeffs):
            if val > threshold and val != max_coeff:
                this_note = equalTemperamentNote(i * freq_dist)

                #Eliminate erroneous notes
                if last_note:
                    if abs(frequencyToNote(this_note) -
                           frequencyToNote(last_note)) <= 2:
                        #Take the average frequency,
                        #replace the last frequency with it
                        average_freq = (this_note + last_note) / 2
                        this_note = equalTemperamentNote(average_freq)
                        note.pop()

                last_note = this_note
                note.append(this_note)

        notes[onset] = note

    for notenum, note in enumerate(sorted(notes)):
        print "Onset %d notes (Hz): " % notenum, notes[note]

    return notes


def frequencyToNote(freq):
    if freq < 1:
        freq = 1
    return round(12 * math.log(freq / 440.0) / math.log(2))


def noteToFrequency(n):
    return 440 * 2 ** (n / 12.0)


def equalTemperamentNote(freq):
    return noteToFrequency(frequencyToNote(freq))


def plotNoteFrequencies(onset, samples, Fs, NFFT, graph_title=""):
    """
    Visualize the frequency spectogram. Used for testing.
    """
    freq_dist = Fs / NFFT

    note_samples = samples[onset: onset + NFFT]
    x = fft(note_samples) / NFFT

    #Can only use half the fft values. See above comment
    x = x[:len(x) / 2]
    t = arange(0.0, Fs / 2 - freq_dist, freq_dist)

    #Crop t to x in case it's the last note, and it's short
    t = t[:len(x)]

    title(graph_title)
    subplot(111)
    plot(t, abs(x))

    show()

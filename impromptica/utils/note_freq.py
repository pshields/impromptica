"""
Given note onset positions and a numpy array of samples (amplitudes),
reports the note frequency (or frequencies) for each onset
"""

from scipy import fft
from pylab import *
import numpy

def frequencies(onsets, samples, Fs=44100):
    """
    Returns a dict containing lists of frequencies (in hz) corresponding to each onset
    Assumes samples is collapsed (mono)
    """
    #Arbitrary - change to something in terms of the sampling frequency?
    #Seems to work for now
    window_size = 1024

    #FFT returns frequency buckets. The first index is 0Hz, and each subsequent one is i * Fs/window_size Hz
    freq_dist = Fs/window_size

    notes = {}

    for notenum, onset in enumerate(onsets):
        #Get the fft
        transform = fft(samples[onset: onset + window_size])/window_size

        #Get the magnitude of each FFT coefficient, since it returns complex values
        fft_coeffs = [x.real**2 + x.imag**2 for x in transform]

        #We can only use the first half of the coeffs (See nyquist frequency). The other half are duplicates
        fft_coeffs = fft_coeffs[:window_size/2]

        #Slight misnomer. Contains a list of frequencies associated with this note onset
        note = []

        #Go through each note and figure out the dominant frequencies (find max, accept within a threshold?)
        max_coeff = max(fft_coeffs)
        prom_note = fft_coeffs.index(max_coeff) * freq_dist
        print "Prominent note freq for note %d is %d" % (notenum, prom_note) 

        note.append(prom_note)
        threshold = .40 * max_coeff #Arbitrary threshold. Probably should make this less hardcoded


        for i, val in enumerate(fft_coeffs):
            if val > threshold and val != max_coeff:
                note.append(i * freq_dist)

        notes[onset] = note

    #spam all the notes! (testing)
    #for notenum, onset in enumerate(onsets):
    #     graph_title = "Note %d" % notenum
    #     plotNoteFrequencies(onset, samples, 44100, window_size, graph_title)

    for notenum, note in enumerate(sorted(notes)):
        print "Onset %d notes (Hz): " % notenum, notes[note]

    return notes

def plotNoteFrequencies(onset, samples, Fs, NFFT, graph_title=""):
    freq_dist = Fs/NFFT

    note_samples = samples[onset: onset + NFFT]
    x = fft(note_samples)/NFFT
    x = x[:len(x)/2] #Can only use half the fft values (past that, values are duplicated)
    t = arange(0.0, Fs/2 - freq_dist, freq_dist)

    title(graph_title)
    subplt = subplot(111)
    plot(t, abs(x))
    
    show()


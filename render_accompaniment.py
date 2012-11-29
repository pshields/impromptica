#!/usr/bin/python2.7
"""
Render accompaniment for the given input audio file.

Uses Impromptica to add accompaniment to the given input audio file. Writes
the result to standard output.
"""
import argparse
from impromptica.utils import onsets, note_freq, sound
from scikits.audiolab import Sndfile
from copy import deepcopy


def gen_basic_accompaniment(audiofile):
    f = Sndfile(audiofile, 'r')
    format = f.format
    Fs = f.samplerate
    samples = f.read_frames(f.nframes)

    # Just support mono sound for now
    if samples.ndim > 1:
        samples = samples.sum(axis=1)

    #To help preserve relative amplitude
    samples_copy = deepcopy(samples)

    # NOTE: Make onsets accept samples instead of a filename
    note_onsets, _, _ = onsets.get_onsets(audiofile)
    note_frequencies = note_freq.frequencies(note_onsets, samples, Fs)

    # For each onset, match detected freqeuncies with square waves
    for onsetnum, onset in enumerate(sorted(note_frequencies)):
        notes = note_frequencies[onset]
        if onsetnum == len(note_onsets) - 1:
            next_onset = len(samples) - 1
        else:
            next_onset = note_onsets[onsetnum + 1]

        print "\033[0;36m",
        print "Note %d" % onsetnum,
        print "\033[0;0m"

        for frequency in notes:
            print "Frequency (Hz): ", frequency
            num_samples = next_onset - onset
            time_elapsed = sound.samples_to_seconds(num_samples, Fs)
            merged_note = sound.generate_note(time_elapsed, 0.5, frequency)

            next_onset = int(onset + len(merged_note))
            if next_onset > len(samples):
                next_onset = len(samples) - 1
                merged_note[:next_onset - onset]

            sound.merge_audio(samples[onset: next_onset], merged_note)

    #Pull up the amplitude of the original samples, since they've been lowered
    sound.merge_audio(samples, samples_copy)

    accompanied_name = audiofile[:-4] + "_accompanied"
    accompanied_file = Sndfile(accompanied_name + ".wav", "w", format, 1, Fs)
    accompanied_file.write_frames(samples)
    accompanied_file.close()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help=(
    'Input audio file. The format of the audio file must be one accepted by '
    'libsndfile. For a list of compatible formats, see '
    'http://www.mega-nerd.com/libsndfile/.'))

args = parser.parse_args()
gen_basic_accompaniment(args.input_file)

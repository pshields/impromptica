#!/usr/bin/python2.7
"""
Render accompaniment for the given input audio file.

Uses Impromptica to add accompaniment to the given input audio file. Writes
the result to standard output.
"""
import argparse
from impromptica.utils import onsets, note_freq, sound
from scikits.audiolab import Sndfile


def gen_basic_accompaniment(audiofile, use_midi):
    f = Sndfile(audiofile, 'r')
    format = f.format
    Fs = f.samplerate
    samples = f.read_frames(f.nframes)

    # Just support mono sound for now
    if samples.ndim > 1:
        samples = samples.sum(axis=1)

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
        print "\033[0;0m\t",

        for frequency in notes:
            print "\033[0;32m",
            print "Frequency (Hz): %f" % frequency,
            print "\033[0;0m\t",

            print "\033[0;33m",
            print "Note: %s" % sound.frequency_to_notestring(frequency),
            print "\033[0;0m"
            num_samples = next_onset - onset
            time_elapsed = sound.samples_to_seconds(num_samples, Fs)

            if use_midi:
                #Default to trumpet
                merged_note = sound.gen_midi_note(time_elapsed, 0.5, frequency,
                                                  Fs, 56)
            else:
                merged_note = sound.generate_note(time_elapsed, 0.5, frequency,
                                                  Fs)

            next_onset = int(onset + len(merged_note))
            if next_onset > len(samples):
                next_onset = len(samples) - 1
                merged_note[:next_onset - onset]

            sound.merge_audio(samples[onset: next_onset], merged_note)

    accompanied_name = audiofile[:-4] + "_accompanied"
    accompanied_file = Sndfile(accompanied_name + ".wav", "w", format, 1, Fs)
    accompanied_file.write_frames(samples)
    accompanied_file.close()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help=(
    'Input audio file. The format of the audio file must be one accepted by '
    'libsndfile. For a list of compatible formats, see '
    'http://www.mega-nerd.com/libsndfile/.'))

parser.add_argument('--use_midi', help=(
    'Generate midi notes instead of square waves'), action='store_true')

args = parser.parse_args()
gen_basic_accompaniment(args.input_file, args.use_midi)

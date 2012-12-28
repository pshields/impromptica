#!/usr/bin/python2.7
"""Render accompaniment for the given input audio file.

The resulting audio file is saved in the same folder as the input file with
'_accompanied' added to the name.
"""
import argparse

from scikits import audiolab

from impromptica.utils import keys
from impromptica.utils import onsets
from impromptica.utils import percussion
from impromptica.utils import note_freq
from impromptica.utils import sound
from impromptica.utils import tempo


def render_accompaniment(input_filename, use_midi, use_key_chords,
                         use_percussion):
    input_file = audiolab.Sndfile(input_filename, 'r')
    audio_format = input_file.format
    sample_rate = input_file.samplerate
    samples = input_file.read_frames(input_file.nframes)

    # If the input audio file has multiple tracks (e.g. stereo input), combine
    # them into a single track.
    if samples.ndim > 1:
        samples = samples.sum(axis=1)

    note_onsets, _, _ = onsets.get_onsets(samples, sample_rate)
    key_list = keys.get_keys(samples, note_onsets, sample_rate)

    note_frequencies = note_freq.frequencies(note_onsets, samples, sample_rate)

    # For each onset, match detected frequencies with square waves.
    for onsetnum, onset in enumerate(sorted(note_frequencies)):
        # Get the notes detected at this onset.
        notes = note_frequencies[onset]
        # Get the next onset.
        if onsetnum == len(note_onsets) - 1:
            next_onset = len(samples) - 1
        else:
            next_onset = note_onsets[onsetnum + 1]
        # Print information about the notes detected at this onset.
        note_strings = [sound.frequency_to_notestring(f) for f in notes]
        print("\033[0;36mOnset %-6d\033[0;33m%s\033[0;0m" % (
            onsetnum, ','.join(note_strings)))
        if use_key_chords:
            # Use the primary chord of the identified key rather than the
            # identified notes.
            upcoming_keys = [k for k in key_list if k[0] <= onset]
            if upcoming_keys:
                key = upcoming_keys[len(upcoming_keys) - 1][1]
                notes = [sound.note_to_frequency(n) for n in
                         keys.notes_in_key(key)]
        # Play the notes.
        for frequency in notes:
            num_samples = next_onset - onset
            time_elapsed = sound.samples_to_seconds(num_samples, sample_rate)
            if use_midi:
                # Use a trumpet sound by default.
                merged_note = sound.gen_midi_note(time_elapsed, 0.5, frequency,
                                                  sample_rate, 56)
            else:
                merged_note = sound.generate_note(time_elapsed, 0.5, frequency,
                                                  sample_rate)
            merged_note = merged_note[:next_onset - onset]
            sound.merge_audio(samples[onset: next_onset], merged_note)

    # Add percussion.
    if use_percussion:
        sounds = percussion.get_drumkit_samples()
        beats_per_minute = tempo.map_pass(samples, sample_rate, 1, 400)
        percussion.render_percussion(samples[note_onsets[0]:], sample_rate,
                                     beats_per_minute, sounds)

    output_filename_parts = input_filename.split('.')
    # Remove the old file extension.
    output_filename_parts.pop()
    # Construct the output file name.
    output_filename = '%s_accompanied.wav' % ('.'.join(output_filename_parts))
    output_file = audiolab.Sndfile(output_filename, 'w', audio_format, 1,
                                   sample_rate)
    output_file.write_frames(samples)
    output_file.close()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help=(
    'Input audio file. The format of the audio file must be one accepted by '
    'libsndfile. For a list of compatible formats, see '
    'http://www.mega-nerd.com/libsndfile/.'))

parser.add_argument('--use_key_chords', help=(
    'On onsets, play the primary chord of the identified key rather than the '
    'identified notes.'), action='store_true')
parser.add_argument('--use_midi', help=(
    'Generate midi notes instead of square waves'), action='store_true')
parser.add_argument('--use_percussion', help=(
    'Generate percussion'), action='store_true')
args = parser.parse_args()
render_accompaniment(args.input_file, args.use_midi, args.use_key_chords,
                     args.use_percussion)

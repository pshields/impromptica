#!/usr/bin/python2.7
"""Render accompaniment for the given input audio file.

The resulting audio file is saved in the same folder as the input file with
'_accompanied' added to the name.
"""
import argparse

import numpy
from scikits import audiolab

from impromptica import settings
from impromptica.utils import keys
from impromptica.utils import onsets
from impromptica.utils import percussion
from impromptica.utils import note_freq
from impromptica.utils import sound
from impromptica.utils import vamp


def render_accompaniment(
        input_filename, accompaniment_only, echo_notes, echo_key_chords,
        use_percussion, metronome, play_tatums, sounds_at_samples, soundfont,
        verbose=False, visualize=False):
    # Get the samples from the file.
    samples = sound.get_samples(input_filename)
    # Prepare the result samples array.
    result = numpy.zeros(samples.shape[0])

    if echo_notes:
        note_onsets, _, _ = onsets.get_onsets(samples)
        key_list = keys.get_keys(samples, note_onsets)
        note_frequencies = note_freq.frequencies(note_onsets, samples)
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
            if echo_key_chords:
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
                time_elapsed = sound.samples_to_seconds(
                    num_samples)
                if soundfont:
                    # Use a trumpet sound by default.
                    merged_note = sound.gen_midi_note(
                        soundfont, time_elapsed, 0.5, frequency, instrument=56)
                else:
                    merged_note = sound.generate_note(
                        time_elapsed, 0.5, frequency)
                merged_note = merged_note[:next_onset - onset]
                sound.merge_audio(samples[onset: next_onset], merged_note)
    soundbank = percussion.get_drumkit_samples()
    if metronome:
        beats = vamp.get_beats(input_filename)
        levels = [beats, beats, beats]
        percussion.render_metronome(result, levels, soundbank, play_tatums)
    if sounds_at_samples:
        hihat = soundbank[6]
        percussion.render_sounds(samples, sounds_at_samples, hihat)
    # Unless requested otherwise, add the original audio to the result track.
    if not accompaniment_only:
        sound.merge_audio(result, samples)
    # Calculate the filename of the output file.
    output_filename_parts = input_filename.split('.')
    # Remove the old file extension.
    output_filename_parts.pop()
    # Construct the output file name.
    output_filename = '%s_accompanied.wav' % ('.'.join(output_filename_parts))
    output_file = audiolab.Sndfile(output_filename, 'w', audiolab.Format(), 1,
                                   settings.SAMPLE_RATE)
    output_file.write_frames(result)
    output_file.close()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help=(
    'Input audio file. The format of the audio file must be one accepted by '
    'libsndfile. For a list of compatible formats, see '
    'http://www.mega-nerd.com/libsndfile/.'))
parser.add_argument(
    '--accompaniment-only', action='store_true', help=(
        'Output only the generated accompaniment; mute the original audio.'))
parser.add_argument(
    '--echo-notes', help='On onsets, play the identified notes.',
    action='store_true')
parser.add_argument(
    '--echo-key-chords', action='store_true', help=(
    'On onsets, play the primary chord of the identified key.'))
parser.add_argument(
    '--metronome', action='store_true', help=(
        'Play percussive sounds at the beats of the measure and tactus '
        'levels'))
parser.add_argument(
    '--play-tatums', action='store_true', help=(
        'If metronome sounds are activated, also play sounds at the tatum '
        'level.'))
parser.add_argument(
    '--sounds-at-samples', type=int, nargs='+', help=(
        'One or more samples to play sounds at'))
parser.add_argument(
    '--percussion', help='Generate percussion', action='store_true')
parser.add_argument(
    '--soundfont', action='store', help=(
        'The filename of a soundfont to use for note generation instead of '
        'the default synthesized note sounds.'))
parser.add_argument(
    '--verbose', help='Enable verbose output', action='store_true')
parser.add_argument(
    '--visualize', help='Enable all visualizations by default.',
    action='store_true')

args = parser.parse_args()
render_accompaniment(
    args.input_file, args.accompaniment_only, args.echo_notes,
    args.echo_key_chords, args.percussion, args.metronome, args.play_tatums,
    args.sounds_at_samples, args.soundfont, args.verbose, args.visualize)

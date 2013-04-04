#!/usr/bin/python2.7
"""Render accompaniment for the given input audio file.

The resulting audio file is saved in the same folder as the input file with
'_accompanied' added to the name.
"""
import argparse

import numpy
from scikits import audiolab

from impromptica import settings
from impromptica.utils import genetic
from impromptica.utils import percussion
from impromptica.utils import sound
from impromptica.utils import vamp


def render_accompaniment(
        input_filename, accompaniment_only, echo_notes, metronome,
        use_percussion, soundfont, use_cached_features, verbose=False,
        visualize=False):
    # Get the samples from the file.
    samples = sound.get_samples(input_filename)
    # Prepare the result samples array.
    result = numpy.zeros(samples.shape[0])
    # Get the features from the input audio.
    data = vamp.get_input_features(input_filename,
        use_cached_features=use_cached_features)
    if verbose:
        print(data)
    if echo_notes:
        # For each onset, match detected frequencies with square waves.
        for midi_note, start_index, duration in data['notes']:
            frequency = sound.note_to_frequency(midi_note)
            if verbose:
                # Print information about the detected note.
                print("\033[0;33m%s\033[0;0m" % (
                    sound.frequency_to_notestring(frequency)))
            # Play the notes.
            time_elapsed = sound.samples_to_seconds(duration)
            if soundfont:
                # Use a trumpet sound by default.
                merged_note = sound.gen_midi_note(
                    soundfont, time_elapsed, 0.5, frequency, instrument=56)
            else:
                merged_note = sound.generate_note(
                    time_elapsed, 0.1, frequency)
            merged_note = merged_note[:duration]
            sound.merge_audio(samples[start_index:start_index + duration],
                              merged_note)

    # Get a list of percussion samples for later use.
    soundbank = percussion.get_drumkit_samples()

    if metronome:
        # Render the metronome
        levels = [data['beats']] * 3
        percussion.render_metronome(result, levels, soundbank, False)

    accompaniment = genetic.evolve(data, 5000)
    # Play accompanied notes.
    print('Rendering accompaniment...')
    for tatum_number, midi_notes in enumerate(accompaniment['notes']):
        print("Tatum %d/%d" % (tatum_number, len(accompaniment['notes'])))
        this_beat_index = data['beats'][tatum_number / 4]
        if tatum_number / 4 + 1 >= len(data['beats']):
            break
        next_beat_index = data['beats'][tatum_number / 4 + 1]
        samples_this_beat = next_beat_index - this_beat_index
        samples_per_tatum = samples_this_beat / 4
        sample_offset = (tatum_number % 4) * samples_per_tatum
        for midi_note, tatum_length in midi_notes:
            note_duration = samples_per_tatum * tatum_length
            duration_time = note_duration / settings.SAMPLE_RATE
            frequency = sound.note_to_frequency(midi_note)
            merged_note = sound.generate_note(duration_time, 0.1, frequency)
            merged_note = merged_note[:note_duration]
            sound.merge_audio(
                samples[this_beat_index + sample_offset:this_beat_index + (
                    sample_offset + note_duration)], merged_note)

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
    '--metronome', action='store_true', help=(
        'Play percussive sounds at the beats of the measure and tactus '
        'levels'))
parser.add_argument(
    '--percussion', help='Generate percussion', action='store_true')
parser.add_argument(
    '--soundfont', action='store', help=(
        'The filename of a soundfont to use for note generation instead of '
        'the default synthesized note sounds.'))
parser.add_argument(
    '--use-cached-features', action='store_true', help=(
        'Use the cached features file instead of recalculating them.'))
parser.add_argument(
    '--verbose', help='Enable verbose output', action='store_true')
parser.add_argument(
    '--visualize', help='Enable all visualizations by default.',
    action='store_true')

args = parser.parse_args()
render_accompaniment(
    args.input_file, args.accompaniment_only, args.echo_notes,
    args.metronome, args.percussion, args.soundfont, args.use_cached_features,
    args.verbose, args.visualize)

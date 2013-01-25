#!/usr/bin/python2.7
"""Script for labeling the beats of an audio file."""
import argparse
import time

import pygame
from scikits import audiolab

from impromptica import settings
from impromptica.utils import percussion
from impromptica.utils import sound


def label_beats(input_filename):
    pygame.mixer.init()
    # Get the samples from the file.
    samples = sound.get_samples(input_filename)
    beats = []
    pygame.mixer.music.load(input_filename)
    pygame.mixer.music.play()
    song_length_in_s = samples.shape[0] / settings.SAMPLE_RATE
    while True:
        q = raw_input()
        # Quit labeling here if specified by the user.
        if q == 'q':
            break
        elapsed = pygame.mixer.music.get_pos() / 1000.  # in seconds
        elapsed -= 0.8  # to account for lag
        if elapsed > song_length_in_s:
            break
        else:
            beats.append(int(settings.SAMPLE_RATE * elapsed))
    print("Done")
    print(beats)
    # Add drum sounds at beats to verify.
    hihat = percussion.get_drumkit_samples()[6]
    percussion.render_sounds(samples, beats, hihat)
    output_filename_parts = input_filename.split('.')
    output_filename_parts.pop()
    output_filename = '%s_labeled.wav' % ('.'.join(output_filename_parts))
    output_file = audiolab.Sndfile(output_filename, 'w', audiolab.Format(), 1,
                                   settings.SAMPLE_RATE)
    output_file.write_frames(samples)
    output_file.close()
    pygame.mixer.music.load(output_filename)
    pygame.mixer.music.play()
    raw_input()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_filename', help=(
    'Input audio file for feature labeling.'))

args = parser.parse_args()
label_beats(args.input_filename)

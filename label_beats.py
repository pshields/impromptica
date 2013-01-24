#!/usr/bin/python2.7
"""Script for labeling the beats of an audio file."""
import argparse
import time

import pygame
from scikits import audiolab

from impromptica import settings
from impromptica.utils import sound


def label_beats(input_filename):
    pygame.mixer.init()
    # Get the samples from the file.
    samples = sound.get_samples(input_filename)
    beats = []
    song = pygame.mixer.Sound(input_filename)
    song.play()
    song_length_in_s = samples.shape[0] / settings.SAMPLE_RATE
    start = time.time()
    while True:
        wait = raw_input()
        elapsed = time.time() - start
        if elapsed > song_length_in_s:
            break
        else:
            beats.append(int(settings.SAMPLE_RATE * elapsed))
    print("Done")
    print(beats)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_filename', help=(
    'Input audio file for feature labeling.'))

args = parser.parse_args()
label_beats(args.input_filename)

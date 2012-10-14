#!/usr/bin/python2.7
"""
Render accompaniment for the given input audio file.

Uses Impromptica to add accompaniment to the given input audio file. Writes
the result to standard output.
"""
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help=(
    'Input audio file. The format of the audio file must be one accepted by '
    'libsndfile. For a list of compatible formats, see '
    'http://www.mega-nerd.com/libsndfile/.'))

args = parser.parse_args()

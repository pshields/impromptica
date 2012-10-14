#!/usr/bin/python2.7
"""
Render accompaniment for the given input audio file.

Uses Impromptica to add accompaniment to the given input audio file. Writes
the result to standard output.
"""
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input_file', help='an input audio file')

args = parser.parse_args()

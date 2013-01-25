#!/usr/bin/python2.7
"""Script for assistance in cleaning up manually-labeled beats."""
import argparse


def analyze_beats(indices):
    """Shows the tempo and change in tempo of given indices.
    
    The first derivative of the indices is the tempo.
    
    The second derivative is the change in tempo.
    """

    tempo = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
    changes = [tempo[i + 1] - tempo[i] for i in range(len(tempo) - 1)]

    for index, tempo, tempodelta in zip(indices, tempo, changes):
        print("%20d%20d%20d" % (index, tempo, tempodelta))


parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument('beats', type=int, nargs='+', help=(
    "Indices of the beats of a metrical level"))

args = parser.parse_args()

analyze_beats(args.beats)

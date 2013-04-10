===========
Impromptica
===========

Impromptu music accompaniment software

Installation
============

On Ubuntu 12.10, install the following dependencies:

    sudo apt-get install hydrogen libfftw3-dev libfluidsynth-dev libhdf5-serial-dev libsamplerate0-dev libsndfile1-dev swig python-numpy python-scipy

You may need to run

    sudo ln -s /usr/lib/i386-linux-gnu/libsndfile.so.1 /usr/local/lib/libsndfile.so

    sudo ln -s /usr/lib/i386-linux-gnu/libsamplerate.so.1 /usr/local/lib/libsamplerate.so

prior to installing `scikits.audiolab` and `scikits.samplerate` via pip.

Sonic Annotator is a dependency. Example installation code (for 64-bit Linux operating systems):

    wget http://code.soundsoftware.ac.uk/attachments/download/542/sonic-annotator-0.7-linux-amd64.tar.gz

    tar xzf sonic-annotator-0.7-linux-amd64.tar.gz

    sudo mv sonic-annotator-0.7-linux-amd64/sonic-annotator /usr/local/bin

    rm -r sonic-annotator-0.7-linux-amd64*

The Queen Mary set of Vamp plugins are required. See `this page <http://www.vamp-plugins.org/download.html>`_ for installation instructions.

Feature detection
=================

In order to accompany the input audio, Impromptica detects features, such as the notes and tempo of the piece. Here is an overview of the feature detection components.

Intermediate forms
------------------

Some features act as intermediaries between the input audio signal and the features we care about for accompaniment. For example, a novelty signal derived from the input audio can be used for tempo detection, beat tracking, and determining the onsets of musical notes and percussion.

Novelty signals
"""""""""""""""

Musical features
----------------

Note onsets
"""""""""""

Note onsets are indices into the input audio at which notes begin.

Note pitch values
"""""""""""""""""

Note pitch values are equal-temperament notes represented as integers. The note C4 (middle C) has the value 60.

Tempo
"""""

The tempo is the period of the piece at the tactus and measure levels. These may change throughout the piece.

Beats
"""""

Beats are the actual locations of onsets at the tactus, tactus and measure levels. The measure-level beats are a subset of the tactus-level beats, and the tactus-level beats are a subset of the tatum-level beats. At the tactus level, the locations of the piece selected as the beats are the locations that you would probably tap your foot to when listening to the piece. Tatums are calculated by finding the number of tatums per measure at each measure that seems most consistent with the novelty signal.

Key
"""

Fixed-size segments of the piece are classified as major or minor.

TODO Synchronize with the tempo algorithms.

Soundfonts
----------

You can plug in a soundfont for generated notes instead of using the default synthesized audio. Refer to the documentation of the `--soundfont` command-line argument.

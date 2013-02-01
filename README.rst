Impromptica
===========

Impromptu music accompaniment software

Installation
------------

On Ubuntu 12.10, install the following dependencies:

    sudo apt-get install hydrogen libfftw3-dev libfluidsynth-dev libhdf5-serial-dev libsamplerate0-dev libsndfile1-dev swig

You may need to run

    sudo ln -s /usr/lib/i386-linux-gnu/libsndfile.so.1 /usr/local/lib/libsndfile.so
    sudo ln -s /usr/lib/i386-linux-gnu/libsamplerate.so.1 /usr/local/lib/libsamplerate.so

prior to installing `scikits.audiolab` and `scikits.samplerate` via pip.

Feature detection
-----------------

In order to accompany the input audio, Impromptica detects features, such as the notes and tempo of the piece. Here is an overview of the feature detection components.

* Note onset times
* Note pitch values (as equal-temperament notes)
* Tempo (in beats per minute)
* Key (fixed-size segments of the piece are classified as major or minor)

Soundfonts
----------

You can plug in a soundfont for generated notes instead of using the default synthesized audio. Refer to the documentation of the `--soundfont` command-line argument.

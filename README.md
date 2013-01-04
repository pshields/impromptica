Impromptica
===========

Impromptu music accompaniment software

Installation
------------

On Ubuntu 12.10, install the following dependencies:

    sudo apt-get install hydrogen libfftw3-dev libfluidsynth-dev libhdf5-serial-dev libsndfile1-dev swig

You may need to run

    sudo ln -s /usr/lib/i386-linux-gnu/libsndfile.so.1 /usr/local/lib/libsndfile.so

prior to installing scikits.audiolab via pip.

Features
--------

Impromptica currently detects the following features:

* Note onset times
* Note pitch values (as equal-temperament notes)
* Tempo (in beats per minute)
* Key (fixed-size segments of the piece are classified as major or minor)

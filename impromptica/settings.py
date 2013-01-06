"""Impromptica configuration and settings."""

# `SAMPLE_RATE` is the sample rate that all input audio will be coerced to. By
# default we set it to 48 KHz, which is the recommended practice by the Audio
# Engineering Society for audio programs (AES5-2008). Impromptica expects this
# value to having a floating-point data type.
SAMPLE_RATE = 48000.

# `DRUMKIT_DIR` is the directory that contains a Hydrogen drumkit to use.
DRUMKIT_DIR = '/usr/share/hydrogen/data/drumkits/GMkit'

# `MAX_NOTE` is the highest note to generate probabilities for in probability
# tables. The minimum note is implicitly 0.
MAX_NOTE = 121

# `TRANSITION_CHANGE_STANDARD_DEVIATION` is the standard deviation of the
# Gaussian distribution used to predict the likelihood of a change in tempo
# as a function of the squared natural logarithm of the ratio of the tempos.
TEMPO_CHANGE_STANDARD_DEVIATION = 0.2

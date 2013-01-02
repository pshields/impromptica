"""Impromptica configuration and settings."""

# `DRUMKIT_DIR` is the directory that contains a Hydrogen drumkit to use.
DRUMKIT_DIR = '/usr/share/hydrogen/data/drumkits/GMkit'

# `MAX_NOTE` is the highest note to generate probabilities for in probability
# tables. The minimum note is implicitly 0.
MAX_NOTE = 121

# `TRANSITION_CHANGE_STANDARD_DEVIATION` is the standard deviation of the
# Gaussian distribution used to predict the likelihood of a change in tempo
# as a function of the squared natural logarithm of the ratio of the tempos.
TEMPO_CHANGE_STANDARD_DEVIATION = 0.2

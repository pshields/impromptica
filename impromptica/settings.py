"""Impromptica configuration and settings."""

# `SAMPLE_RATE` is the sample rate that all input audio will be coerced to. By
# default we set it to 48 KHz, which is the recommended practice by the Audio
# Engineering Society for audio programs (AES5-2008). Impromptica expects this
# value to having a floating-point data type.
SAMPLE_RATE = 48000.

# `NOVELTY_WINDOW_SIZE` is the number of original audio samples used to compute
# each sample of a novelty signal. Appropriate values appear to map to the 10
# to 30 millisecond time range. We use a power of two, since it might be more
# efficient. At a sample rate of 48KHz, a window size of 1024 corresponds to a
# duration of ~21.3ms.
NOVELTY_WINDOW_SIZE = 1024

# `NOVELTY_HOP_SIZE` is the number of original audio samples by which a novelty
# signal increments when calculating the value of the next sample in the
# novelty signal. At a sample rate of 48KHz, a hop size of 512 corresponds to a
# duration of ~10.7ms.
NOVELTY_HOP_SIZE = 512

# Once calculated, a novelty signal is interpolated to a new rate in order to
# be granular enough for tasks such as tempo induction. The appropriate rate
# for the interpolated signal appears to be in the range of 100 to 200 Hz.
NOVELTY_INTERPOLATION_FACTOR = 2

# `DRUMKIT_DIR` is the directory that contains a Hydrogen drumkit to use.
DRUMKIT_DIR = '/usr/share/hydrogen/data/drumkits/GMkit'

# `MAX_NOTE` is the highest note to generate probabilities for in probability
# tables. The minimum note is implicitly 0.
MAX_NOTE = 121

# `TRANSITION_CHANGE_STANDARD_DEVIATION` is the standard deviation of the
# Gaussian distribution used to predict the likelihood of a change in tempo
# as a function of the squared natural logarithm of the ratio of the tempos.
TEMPO_CHANGE_STANDARD_DEVIATION = 0.2

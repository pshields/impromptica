"""
Polyphonic note detection based on Klapuri's paper
[link paper here]
"""

import numpy as np
import math
from impromptica import settings
from impromptica.utils import novelty


def generate_bands(power_spectrum, bin_width):
    """
    power_spectrum: np array of frequencies
    bin_width: frequency range of a bin
    Returns the sub-bands of the power_spectrum split by 2/3 octaves
    """
    band_indices = []
    freq = settings.LOWEST_SUBBAND_FREQ
    band_indices.append(math.ceil(freq / bin_width))

    freq *= 4 / 3.0

    while freq < len(power_spectrum) * bin_width:
        band_indices.append(math.ceil(freq / bin_width))
        freq *= 4 / 3.0

    return band_indices


def subtract_noise(power_spectrum, bin_width):
    """
    Takes a logarithmically-scaled (magnitude warped) power spectrum
    and subtracts the noise linearly.

    Modifies Y(k) - the power spectrum - to Z(k)
    """
    band_indices = generate_bands(power_spectrum, bin_width)

    for first, last in zip(band_indices, band_indices[1:]):
        power_spectrum[first:last] -= np.average(power_spectrum[first:last])

    for i in range(power_spectrum.shape[0]):
        power_spectrum[i] = max(0, power_spectrum[i])


def preprocessing(window, samples):
    """
    Transform the signal from time spectrum to power spectrum.
    Perform magnitude warping and noise subtraction on the power spectrum.
    (Whitens it)
    Assumes the window is a hamming-windowed onset.
    """
    #X(K)
    power_spectrum = novelty.power_spectrum(window, settings.SAMPLE_RATE)
    freq_bin = settings.SAMPLE_RATE / len(window)

    k0 = 0
    k1 = 6000
    k1_index = 6000 / freq_bin

    # Magnitude-warping factor (Differentiates noise from harmonics)
    g = (sum(x ** (1 / 3.0) for x in power_spectrum[k0:k1_index])
         / (k1 - k0 + 1)) ** 3

    mag_warp_power_spectrum = np.log(1 + g * power_spectrum)

    subtract_noise(mag_warp_power_spectrum, freq_bin)

    return mag_warp_power_spectrum

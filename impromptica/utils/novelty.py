"""Utilities for generating and working with novelty signals.

Novelty signals model the novelty of audio over time. They are an intermediate
representation of audio used for feature extraction. Novelty signals are also
called accentuation signals, since they can measure the degree of accentuation
in the audio over time, or onset detection functions, since they can be used to
find note onsets in a musical piece. Novelty signals are used in onset
detection, beat tracking, and other facets of music information retrieval.
"""
import math

import numpy as np

from impromptica import settings


_TRIANGLE_FILTER_BANK_SIZE = 36


def get_bin_size(window_size):
    """Returns the number of bins resulting from applying a DFT to
    `window_size` samples.

    The number of bins is (window_size / 2) + 1 because we only want to
    retain the bins with non-negative frequencies.
    """
    return window_size / 2 + 1


def get_bin_frequencies(sample_size, sample_rate):
    """Returns an array of bin frequencies after application of FFT.

    This function assumes negative-frequency bins were filtered out.
    """
    result = np.fft.fftfreq(sample_size, d=1. / sample_rate)
    # Crop result to the appropriate size.
    return result[:get_bin_size(sample_size)]


def power_spectrum(samples, sample_rate):
    """Returns the power spectrum of the given samples.

    This function returns an array of frequency bins and associated power
    levels for each bin.

    The power is equal to the square of the absolute value of each non-negative
    frequency bin returned from applying a DFT to the input samples.

    The resulting power spectrum has (samples.size/2 + 1) bins.
    """
    bins = np.fft.rfft(samples)
    # Multiplying a complex number by its conjugate yields the same result as
    # squaring its absolute value.
    return np.real(bins * np.conjugate(bins))


def mel(frequency):
    """Returns the mel scale version of the given frequency."""
    result = 0.
    if frequency > 0:
        result = 2595. * math.log10(1. + frequency / 700.)
    return result


def inverse_mel(mel_frequency):
    """Returns the regular version of the given mel scale frequency."""
    return 700. * (math.pow(10., mel_frequency / 2595.) - 1.)


def create_triangle_filter_bank(
        window_size, bank_size=_TRIANGLE_FILTER_BANK_SIZE,
        sample_rate=settings.SAMPLE_RATE):
    """Returns a bank of triangle filters.

    The filters are spaced uniformly on the mel frequency scale in the range of
    human hearing so that they can provide the greatest coverage of features
    from the perspective of human hearing.

    Each triangle is normalized to have equal area so that analyses across
    bands compare accurately.
    """
    # Generate a uniform range of mel scale frequencies in the range of human
    # hearing (approximately 50 to 20,000 Hz.) We will try to center a triangle
    # filter as close to each interior frequency as possible.
    low_frequency, high_frequency = (mel(50.), mel(20000.))
    centers = np.linspace(low_frequency, high_frequency, bank_size + 2)
    # `bin_indices` maps filter indices to DFT bin indices.
    bin_indices = np.zeros(bank_size + 2, dtype=int)
    bin_frequencies = get_bin_frequencies(window_size, sample_rate=sample_rate)
    bin_mel_frequencies = [mel(f) for f in bin_frequencies]
    current_bin_index = 0
    for i in range(bank_size + 2):
        # While the target center frequency for this filter is closer to the
        # mel frequency of the current fourier bin index, increment the index.
        while (abs(centers[i] - bin_mel_frequencies[current_bin_index]) >
               abs(centers[i] - bin_mel_frequencies[current_bin_index + 1])):

            current_bin_index += 1
        bin_indices[i] = current_bin_index
        current_bin_index += 1
    # Create the filters.
    result = np.zeros((bank_size, get_bin_size(window_size)))
    for i in range(1, bank_size + 1):
        low_index, middle_index, high_index = bin_indices[i - 1:i + 2]
        # Normalize the height of this triangle filter to make it have unit
        # area.
        height = 2. / float(high_index - low_index)
        result[i - 1][low_index:middle_index] = np.linspace(
            0., height, middle_index - low_index)
        result[i - 1][middle_index:high_index + 1] = np.linspace(
            height, 0., high_index - middle_index + 1)
    return result


def get_segments(
        samples, window_size=settings.NOVELTY_WINDOW_SIZE,
        hop_size=settings.NOVELTY_HOP_SIZE,
        interpolation_factor=settings.NOVELTY_INTERPOLATION_FACTOR,
        sample_rate=settings.SAMPLE_RATE):
    """Divides the samples into Hanning-windowed segments with 50% overlap.

    Segments will be spaced `hop_size` samples apart.

    Up to window_size + hop_size samples might be clipped off the end of the
    original audio signal, but that shouldn't matter much since it is such a
    small duration.
    """
    number_of_segments = (samples.shape[0] - window_size) / hop_size + 1
    window = np.hanning(window_size)
    # Place the samples into each segment.
    result = np.zeros((number_of_segments, window_size))
    for i in range(number_of_segments):
        start = i * hop_size
        end = start + window_size
        result[i] = samples[start:end] * window
    return result


def spectral_flux(
        samples, window_size=settings.NOVELTY_WINDOW_SIZE,
        hop_size=settings.NOVELTY_HOP_SIZE,
        interpolation_factor=settings.NOVELTY_INTERPOLATION_FACTOR,
        sample_rate=settings.SAMPLE_RATE, verbose=False):
    """Returns a spectral flux-based novelty signal for the given audio.

    The novelty signal is our best guess at the level of meaningful
    musical accentuation in each of a few frequency ranges which form a
    partition of the usual human hearing range.

    The steps taken to calculate the differential are as follows:

    * We calculate the spectral power in each of several frequency bands at
      each segment.
    * We calculate the differential of spectral power in each band at each
      segment.
    * We half-wave rectify the differentials in each band at each segment and
      then sum them to yield the final value of the novelty signal.
    """
    # Divide the samples into segments for calculations.
    segments = get_segments(samples)
    if verbose:
        print("Original audio has been divided into %d segments." % (
            segments.shape[0]))
    bin_size = get_bin_size(segments.shape[1])
    # Create a triangle band-pass filter bank for separating the power spectrum
    # of each segment into an array of feature bands.
    filters = create_triangle_filter_bank(window_size)
    # Calculate the DFT power spectrum of each segment. Then, at each segment,
    # apply the power spectrum to a triangle band-pass filter bank and
    # calculate the power in each band.
    powers = np.zeros((segments.shape[0], bin_size))
    bands = np.zeros((segments.shape[0], filters.shape[0]))
    for i, segment in enumerate(segments):
        powers[i] = power_spectrum(segment, sample_rate=sample_rate)
    # To measure spectral change, we calculate the change in spectral power
    # from the previous segment to the current.
    differential = np.zeros((segments.shape[0], filters.shape[0]))
    # Allocate the storage for the novelty signal.
    result = np.zeros(segments.shape[0] * interpolation_factor)
    # Calculate the spectral change differential as previously described.
    for i in range(segments.shape[0]):
        for j in range(filters.shape[0]):
            bands[i][j] = min(1., np.sum(powers[i] * filters[j]))
            # Calculate the weighted differential.
            differential[i][j] = max(0., bands[i][j] - bands[i - 1][j])
            # Add this value to the novelty_signal.
            result[i * interpolation_factor] += differential[i][j]
        for j in range(1, interpolation_factor):
            result[
                i * interpolation_factor + j] = result[
                    i * interpolation_factor]
    # Normalize the novelty signal to have a maximum of one.
    result /= filters.shape[0]
    return result

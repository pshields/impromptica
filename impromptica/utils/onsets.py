import numpy
import matplotlib.pyplot as plt
import modal
import modal.onsetdetection as onsetdetection
import modal.ui.plot as trplot

from impromptica import settings


def get_onsets(samples, sample_rate=settings.SAMPLE_RATE):
    """
    Using Complex Domain Onset Detection from modal
    See Bello, et. al. - www.elec.qmul.ac.uk/dafx03/proceedings/pdfs/dafx81.pdf

    Returns note onset positions, the audio waveform after a low-pass filter,
    and modal classes necessary for plotting.
    """
    samples = numpy.asarray(samples, dtype=numpy.double)

    if samples.ndim > 1:
        samples = samples.sum(axis=1)

    # Use 20 frames per second, and a 50% overlap
    frame_size = int(sample_rate) / 20
    hop_size = frame_size / 2

    # Scale to -1 to 1 for easier processing
    max_ampl = numpy.max(samples)
    samples /= max_ampl

    # This can be substituted with several other
    # onset detection algorithms packaged with modal.
    # This one works pretty well.
    odf = modal.ComplexODF()

    odf.set_hop_size(hop_size)
    odf.set_frame_size(frame_size)
    odf.set_sampling_rate(int(sample_rate))
    odf_values = numpy.zeros(len(samples) / hop_size, dtype=numpy.double)
    odf.process(samples, odf_values)

    onset_det = onsetdetection.OnsetDetection()
    onset_det.peak_size = 3
    onsets = onset_det.find_onsets(odf_values) * odf.get_hop_size()

    filtered_samples = samples * max_ampl
    return onsets, filtered_samples, (onset_det, odf, frame_size, hop_size)


def plot_onsets(samples, sample_rate=settings.SAMPLE_RATE):
    onsets, filtered_samples, extras = get_onsets(samples,
                                                  sample_rate=sample_rate)
    onset_det, odf, frame_size, hop_size = extras

    # Scale to -1 to 1 to graph with onsets
    filtered_samples /= numpy.max(filtered_samples)

    plt.figure(1, figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.title("Onset detection with ComplexODF")
    plt.plot(filtered_samples, '0.4')
    plt.subplot(3, 1, 2)

    trplot.plot_detection_function(onset_det.odf, hop_size)
    trplot.plot_detection_function(onset_det.threshold, hop_size, "green")
    plt.subplot(3, 1, 3)

    trplot.plot_onsets(onsets, 1.0)
    plt.plot(filtered_samples, "0.4")

    plt.show()

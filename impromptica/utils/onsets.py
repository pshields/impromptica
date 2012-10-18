import numpy
import matplotlib.pyplot as plt
import modal
import modal.onsetdetection as onsetdetection
import modal.ui.plot as trplot
from scipy.io import wavfile

frame_size = 2048
hop_size = 512

def get_onsets(filename):
    """
    Using Complex Domain Onset Detection from modal
    See Bello, et. al. - www.elec.qmul.ac.uk/dafx03/proceedings/pdfs/dafx81.pdf

    Returns note onset positions, the audio waveform after a low-pass filter,
    and modal classes necessary for plotting.
    """
    frame_rate, samples = wavfile.read(filename)
    samples = numpy.asarray(samples, dtype=numpy.double)
    samples = samples.sum(axis=1)

    #Scale to -1 to 1 for easier processing
    max_ampl = numpy.max(samples)
    samples /= max_ampl
    
    odf = modal.ComplexODF()
    odf.set_hop_size(hop_size)
    odf.set_frame_size(frame_size)
    odf.set_sampling_rate(frame_rate)
    odf_values = numpy.zeros(len(samples) / hop_size, dtype=numpy.double)
    odf.process(samples, odf_values)

    onset_det = onsetdetection.OnsetDetection()
    onset_det.peak_size = 3
    onsets = onset_det.find_onsets(odf_values) * odf.get_hop_size()

    filtered_samples = samples * max_ampl
    return onsets, filtered_samples, onset_det, odf

def plot_onsets(filename):
    onsets, filtered_samples, onset_det, odf = get_onsets(filename)

    fig = plt.figure(1, figsize=(12, 12))
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

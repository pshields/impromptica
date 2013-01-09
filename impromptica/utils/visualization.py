"""Utilities for visualizing various accoustic, musical, or other data."""
import matplotlib.pyplot as plt
from matplotlib import cm


def show_rhythmogram(data, sample_rate, offset, base_period):
    """Show a rhythmogram of the given data.

    `data` is a table of pulse salience values over time. The table should be
    indexed first by period number minus one, and second by time value.

    `sample_rate` is the rate of the rhythmogram samples, in hertz.

    `offset` is the time value in seconds of the first sample of `data`.

    `base_period` is the duration of the base period used for pulse hypothesis
    binning, and is equal to the duration of a single sample of the novelty
    signal (in seconds.)
    """
    fig = plt.figure(figsize=(9, 9))
    plt.title("Rhythmogram content")
    ax = fig.add_subplot(1, 1, 1)
    # Show the data.
    ax.imshow(data, interpolation='nearest', cmap=cm.jet)
    # Allow the aspect ratio of the image to be more natural.
    ax.set_aspect('auto')
    # Set the x-axis to be the time into the piece in seconds.
    locs, labels = plt.xticks()
    plt.xticks(locs, ["%.1f" % (i / sample_rate + offset) for i in locs])
    # Set the y-axis to be the time of the corresponding period in seconds.
    locs, labels = plt.yticks()
    plt.yticks(locs, ["%.1f" % ((i + 1) * base_period) for i in locs])
    # Align the data to the boundaries of the image.
    plt.axis([0, data.shape[1] - 1, 0, data.shape[0] - 1])
    plt.xlabel("Time (s)")
    plt.ylabel("Period hypothesis (s)")
    plt.tight_layout()
    plt.show()


def show_self_similarity_matrix(data, title):
    """Show a self-similarity matrix."""
    fig = plt.figure(figsize=(9, 9))
    plt.title(title)
    ax = fig.add_subplot(1, 1, 1)
    # Show the data.
    ax.imshow(data, interpolation='nearest', cmap=cm.jet)
    plt.axis([0, data.shape[0] - 1, 0, data.shape[0] - 1])
    # Allow the aspect ratio of the image to be more natural.
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()
